#!/usr/bin/env python3
"""
web_app.py - Karaoke Agent v0.1 Web Application & Mobile-Ready REST API
Runs locally (`python3 web_app.py` -> http://localhost:8000) and scales cleanly
to custom domains and future Apple App Store (iOS/SwiftUI) clients via JSON REST endpoints.
"""

import os
import sys
import uuid
import json
import time
import threading
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from urllib.parse import quote

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import run_karaoke_pipeline
from audio_ingest import sanitize_filename, source_folder_name, is_url
from pipeline_logger import scan_all_song_runs, detect_and_recover_stale_run

WEB_DIR = os.path.join(BASE_DIR, "web")
DEFAULT_OUT_DIR = os.environ.get("KARAOKE_OUT_DIR", os.path.join(BASE_DIR, "out"))
DEFAULT_LOG_DIR = os.environ.get("KARAOKE_LOG_DIR", os.path.join(BASE_DIR, "logs"))
DEFAULT_UPLOAD_DIR = os.environ.get("KARAOKE_UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))

os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
os.makedirs(DEFAULT_UPLOAD_DIR, exist_ok=True)

STEP_PROGRESS_MAP = {
    "queued": (2, "Queued for processing"),
    "initializing": (5, "Initializing song workspace"),
    "step1_ingest": (15, "Step 1/6: Downloading & normalizing audio"),
    "step2_stem_separation": (38, "Step 2/6: Isolating singer vocals & clean instrumental (AI Demucs)"),
    "step3_lyrics_search": (52, "Step 3/6: Checking verified reference lyrics"),
    "step4_vocal_timing": (70, "Step 4/6: Listening to vocals & timing exact sung phrases"),
    "step5_phonetics_and_subtitles": (85, "Step 5/6: Generating strict phonetic English & 3-tier subtitles"),
    "step6_video_render": (94, "Step 6/6: Burning 1080p MP4 karaoke video"),
    "completed": (100, "Completed — 1080p Karaoke Video Ready!"),
}


def _media_url(song_folder: str, filename: str) -> str:
    return f"/media/{quote(song_folder)}/{quote(filename)}"


def _tail_file(path: str, max_lines: int = 45) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception:
        return ""


def _build_song_summary(out_root: str, song_folder: str, include_lyrics: bool = False) -> Optional[Dict[str, Any]]:
    song_dir = os.path.join(out_root, song_folder)
    if not os.path.isdir(song_dir):
        return None

    meta_path = os.path.join(song_dir, "metadata.json")
    lyrics_path = os.path.join(song_dir, "lyrics.json")
    status_path = os.path.join(song_dir, "run_status.json")
    log_path = os.path.join(song_dir, "pipeline.log")

    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    lyrics_data: Dict[str, Any] = {}
    if os.path.exists(lyrics_path):
        try:
            with open(lyrics_path, "r", encoding="utf-8") as f:
                lyrics_data = json.load(f)
        except Exception:
            pass

    run_status: Dict[str, Any] = {}
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                run_status = json.load(f)
        except Exception:
            pass

    files = os.listdir(song_dir)
    mp4_candidates = [f for f in files if f.endswith(".mp4") and f != "karaoke.mp4"]
    primary_mp4 = mp4_candidates[0] if mp4_candidates else ("karaoke.mp4" if "karaoke.mp4" in files else None)

    thumb_candidates = [f for f in files if f.startswith("raw_audio.") and f.endswith((".webp", ".jpg", ".jpeg", ".png"))]
    thumb_file = thumb_candidates[0] if thumb_candidates else None

    l_meta = lyrics_data.get("metadata", {})
    title = meta.get("title") or l_meta.get("title") or run_status.get("metadata", {}).get("title") or song_folder
    artist = meta.get("artist") or l_meta.get("artist") or run_status.get("metadata", {}).get("artist") or ""
    language = l_meta.get("language") or meta.get("language") or run_status.get("metadata", {}).get("language") or "auto"
    duration = float(meta.get("duration") or l_meta.get("duration_seconds") or 0.0)
    lines = lyrics_data.get("lines", [])

    status_str = run_status.get("status") or ("completed" if primary_mp4 else ("partial" if lines else "unknown"))

    # Determine newest modification timestamp for sorting
    mtime = os.path.getmtime(song_dir)
    for check_f in (primary_mp4, "lyrics.json", "run_status.json"):
        if check_f and os.path.exists(os.path.join(song_dir, check_f)):
            mtime = max(mtime, os.path.getmtime(os.path.join(song_dir, check_f)))

    summary: Dict[str, Any] = {
        "song_folder": song_folder,
        "title": title,
        "artist": artist,
        "language": language,
        "duration_seconds": round(duration, 2),
        "lines_count": len(lines),
        "status": status_str,
        "updated_epoch": round(mtime, 2),
        "source_input": meta.get("source_input") or run_status.get("source"),
        "urls": {
            "video_mp4": _media_url(song_folder, primary_mp4) if primary_mp4 else None,
            "video_filename": primary_mp4,
            "thumbnail": _media_url(song_folder, thumb_file) if thumb_file else None,
            "instrumental_wav": _media_url(song_folder, "instrumental.wav") if "instrumental.wav" in files else None,
            "vocals_wav": _media_url(song_folder, "vocals.wav") if "vocals.wav" in files else None,
            "karaoke_ass": _media_url(song_folder, "karaoke.ass") if "karaoke.ass" in files else None,
            "lyrics_lrc": _media_url(song_folder, "lyrics.lrc") if "lyrics.lrc" in files else None,
            "lyrics_json": _media_url(song_folder, "lyrics.json") if "lyrics.json" in files else None,
            "pipeline_log": _media_url(song_folder, "pipeline.log") if "pipeline.log" in files else None,
        },
    }

    if include_lyrics:
        summary["lines"] = lines
        summary["run_status"] = run_status
        summary["log_tail"] = _tail_file(log_path, max_lines=60)

    return summary


class ConversionJobManager:
    """Manages queued & running song conversion jobs with live watchdog status reads."""

    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="KaraokeWebWorker")
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def submit_job(
        self,
        source: str,
        song_folder: str,
        out_root: str,
        log_dir: str,
        language_hint: str = "auto",
        manual_lyrics: Optional[str] = None,
    ) -> Dict[str, Any]:
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        output_dir = os.path.join(out_root, song_folder)
        os.makedirs(output_dir, exist_ok=True)

        job_rec: Dict[str, Any] = {
            "job_id": job_id,
            "song_folder": song_folder,
            "source": source,
            "output_dir": output_dir,
            "language_hint": language_hint,
            "status": "queued",
            "created_epoch": round(time.time(), 2),
            "started_epoch": None,
            "finished_epoch": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job_rec

        self._executor.submit(self._run_worker, job_id, source, output_dir, log_dir, language_hint, manual_lyrics)
        return dict(job_rec)

    def _run_worker(
        self,
        job_id: str,
        source: str,
        output_dir: str,
        log_dir: str,
        language_hint: str,
        manual_lyrics: Optional[str],
    ) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["started_epoch"] = round(time.time(), 2)

        try:
            run_karaoke_pipeline(
                source=source,
                output_dir=output_dir,
                manual_lyrics=manual_lyrics,
                interactive=False,
                render_video=True,
                language_hint=language_hint,
                global_log_dir=log_dir,
            )
            with self._lock:
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["finished_epoch"] = round(time.time(), 2)
        except Exception as exc:
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["finished_epoch"] = round(time.time(), 2)
                self._jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"

    def get_job(self, job_id: str, out_root: str, log_dir: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            base = dict(self._jobs[job_id]) if job_id in self._jobs else None
        if not base:
            return None

        output_dir = base["output_dir"]
        status_path = os.path.join(output_dir, "run_status.json")
        log_path = os.path.join(output_dir, "pipeline.log")

        live_state: Dict[str, Any] = {}
        if os.path.exists(status_path):
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    live_state = json.load(f)
            except Exception:
                pass

        # Check watchdog freeze state if running
        watchdog_diag = "healthy"
        if base["status"] == "running":
            stale = detect_and_recover_stale_run(output_dir, global_log_dir=log_dir, freeze_threshold_sec=45.0)
            if stale and stale.get("diagnosis") in ("frozen", "crashed"):
                watchdog_diag = stale["diagnosis"]
            elif live_state.get("stall_detected"):
                watchdog_diag = "stall_warning"

        current_step = live_state.get("current_step") or ("completed" if base["status"] == "completed" else base["status"])
        pct, label = STEP_PROGRESS_MAP.get(current_step, (50, f"Processing ({current_step})"))
        if base["status"] == "completed":
            pct, label = 100, "Completed — 1080p Karaoke Video Ready!"
        elif base["status"] == "failed":
            label = f"Failed during {current_step}"

        elapsed = 0.0
        if base["started_epoch"]:
            end_t = base["finished_epoch"] or time.time()
            elapsed = round(max(0.0, end_t - base["started_epoch"]), 1)

        resp: Dict[str, Any] = {
            "job_id": job_id,
            "song_folder": base["song_folder"],
            "status": base["status"],
            "current_step": current_step,
            "progress_percent": pct,
            "progress_label": label,
            "elapsed_seconds": elapsed,
            "watchdog": watchdog_diag,
            "last_heartbeat_at": live_state.get("last_heartbeat_at"),
            "stage_durations_sec": live_state.get("stage_durations_sec", {}),
            "metadata": live_state.get("metadata", {}),
            "ml_features": live_state.get("ml_features", {}),
            "error": base["error"] or (live_state.get("error", {}) or {}).get("message"),
            "log_tail": _tail_file(log_path, max_lines=45),
        }
        if base["status"] == "completed":
            resp["song"] = _build_song_summary(out_root, base["song_folder"], include_lyrics=True)
        return resp

    def list_jobs(self, out_root: str, log_dir: str) -> List[Dict[str, Any]]:
        with self._lock:
            ids = list(self._jobs.keys())
        out = [self.get_job(jid, out_root, log_dir) for jid in reversed(ids)]
        return [j for j in out if j is not None]


def create_app(
    out_dir: str = DEFAULT_OUT_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
    upload_dir: str = DEFAULT_UPLOAD_DIR,
) -> FastAPI:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(upload_dir, exist_ok=True)

    app = FastAPI(
        title="Karaoke Agent API & Studio",
        version="0.1.0",
        description="Multilingual Phonetic Karaoke Creator (Web + iOS App Store Ready API)",
    )

    cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    job_manager = ConversionJobManager(max_workers=int(os.environ.get("KARAOKE_MAX_WORKERS", "1")))
    app.state.job_manager = job_manager
    app.state.out_dir = out_dir
    app.state.log_dir = log_dir
    app.state.upload_dir = upload_dir

    # Mount static web assets & range-enabled media output directory
    if os.path.isdir(WEB_DIR):
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    app.mount("/media", StaticFiles(directory=out_dir), name="media")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.get("/manifest.webmanifest", include_in_schema=False)
    async def serve_manifest():
        return FileResponse(
            os.path.join(WEB_DIR, "manifest.webmanifest"),
            media_type="application/manifest+json",
        )

    @app.get("/api/health")
    async def api_health():
        report = scan_all_song_runs(app.state.out_dir, global_log_dir=app.state.log_dir)
        return {
            "app": "Karaoke Agent",
            "version": "0.1.0",
            "status": "ok",
            "telemetry": report,
        }

    @app.get("/api/songs")
    async def api_list_songs():
        items: List[Dict[str, Any]] = []
        if os.path.isdir(app.state.out_dir):
            for entry in os.listdir(app.state.out_dir):
                summary = _build_song_summary(app.state.out_dir, entry, include_lyrics=False)
                if summary and (summary["urls"]["video_mp4"] or summary["lines_count"] > 0 or summary["status"] != "unknown"):
                    items.append(summary)
        items.sort(key=lambda x: x.get("updated_epoch", 0), reverse=True)
        return {"songs": items}

    @app.get("/api/songs/{song_folder}")
    async def api_get_song(song_folder: str):
        safe_folder = os.path.basename(song_folder)
        summary = _build_song_summary(app.state.out_dir, safe_folder, include_lyrics=True)
        if not summary:
            raise HTTPException(status_code=404, detail=f"Song '{safe_folder}' not found.")
        return summary

    @app.get("/api/jobs")
    async def api_list_jobs():
        return {"jobs": job_manager.list_jobs(app.state.out_dir, app.state.log_dir)}

    @app.get("/api/jobs/{job_id}")
    async def api_get_job(job_id: str):
        job = job_manager.get_job(job_id, app.state.out_dir, app.state.log_dir)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return job

    @app.post("/api/jobs", status_code=202)
    async def api_create_job(
        request: Request,
        source_type: Optional[str] = Form(default=None),
        source_url: Optional[str] = Form(default=None),
        language: str = Form(default="auto"),
        lyrics_mode: str = Form(default="auto"),
        manual_lyrics: Optional[str] = Form(default=None),
        audio_file: Optional[UploadFile] = File(default=None),
    ):
        # Support both HTML FormData (Web UI) and JSON bodies (future iOS / mobile clients)
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            source_type = body.get("source_type", "url")
            source_url = body.get("source_url") or body.get("source")
            language = body.get("language", "auto")
            lyrics_mode = body.get("lyrics_mode", "auto")
            manual_lyrics = body.get("manual_lyrics")

        lang_clean = language if language in ("auto", "fa", "hy", "ar") else "auto"
        lyrics_clean = manual_lyrics.strip() if (lyrics_mode == "manual" and manual_lyrics and manual_lyrics.strip()) else None

        resolved_source = ""
        song_folder = ""

        if source_type == "file" or (audio_file and audio_file.filename):
            if not audio_file or not audio_file.filename:
                raise HTTPException(status_code=400, detail="Please choose an audio file (.mp3, .wav, .m4a, .flac) to upload.")
            ext = Path(audio_file.filename).suffix.lower()
            if ext not in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".webm"):
                raise HTTPException(status_code=400, detail=f"Unsupported audio format '{ext}'.")
            stem_safe = sanitize_filename(Path(audio_file.filename).stem) or "uploaded_song"
            unique_tag = uuid.uuid4().hex[:6]
            saved_name = f"{stem_safe}_{unique_tag}{ext}"
            saved_path = os.path.join(app.state.upload_dir, saved_name)
            data = await audio_file.read()
            if len(data) < 256:
                raise HTTPException(status_code=400, detail="Uploaded audio file appears to be empty.")
            with open(saved_path, "wb") as f:
                f.write(data)
            resolved_source = saved_path
            song_folder = f"upload_{stem_safe}_{unique_tag}"
        else:
            url_val = (source_url or "").strip()
            if not url_val or not is_url(url_val):
                raise HTTPException(status_code=400, detail="Please enter a valid http:// or https:// song URL.")
            resolved_source = url_val
            song_folder = source_folder_name(url_val)

        job = job_manager.submit_job(
            source=resolved_source,
            song_folder=song_folder,
            out_root=app.state.out_dir,
            log_dir=app.state.log_dir,
            language_hint=lang_clean,
            manual_lyrics=lyrics_clean,
        )
        return JSONResponse(status_code=202, content=job)

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Start Karaoke Agent v0.1 Web UI & REST Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Bind host (use 0.0.0.0 for LAN/domain)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload during development")
    args = parser.parse_args()

    import uvicorn
    print("\n" + "=" * 68)
    print(" 🎤  KARAOKE AGENT v0.1 — WEB STUDIO & MOBILE-READY API")
    print("=" * 68)
    print(f" 🌐 Local Studio URL:   http://{args.host}:{args.port}")
    print(f" 📱 OpenAPI / iOS Spec: http://{args.host}:{args.port}/docs")
    print(f" 📂 Output Library:     {DEFAULT_OUT_DIR}")
    print("=" * 68 + "\n")

    uvicorn.run("web_app:app", host=args.host, port=args.port, reload=args.reload, app_dir=BASE_DIR)


if __name__ == "__main__":
    main()
