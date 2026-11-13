"""
modal_worker.py - Modal Serverless Worker for Karaoke Agent
Mirrors Aniden.ai spawn pattern:
- process_song_web: FastAPI endpoint that verifies auth and calls process_song_gpu.spawn()
- process_song_gpu: A10G GPU function executing stem separation, WhisperX, and transliteration
- render_mp4_web / render_mp4_cpu: CPU function for burning subtitles and encoding MP4 video
"""

import os
from pathlib import Path
import modal

# App Definition
app = modal.App("karaoke-render")

# Model weights cache volume
models_volume = modal.Volume.from_name("karaoke-model-cache", create_if_missing=True)

# Container Image with system tools & python packages
karaoke_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1", "libass-dev")
    .pip_install(
        "torch",
        "torchaudio",
        "yt-dlp",
        "audio-separator[gpu]",
        "whisperx",
        "google-genai",
        "beautifulsoup4",
        "pydantic",
        "supabase",
        "fastapi"
    )
)

# Resolve repo root so root modules can be bundled
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

@app.function(
    image=karaoke_image,
    gpu="A10G",
    timeout=600,
    volumes={"/root/.cache": models_volume},
    secrets=[modal.Secret.from_name("karaoke-secrets")],
    mounts=[modal.Mount.from_local_dir(REPO_ROOT, remote_path="/app")]
)
def process_song_gpu(song_id: str):
    """
    Background GPU worker executing the core AI pipeline:
    1. Downloads/reads audio
    2. Separates vocal & instrumental stems via Roformer
    3. Performs ASR / forced alignment via WhisperX
    4. Calls Gemini for strict phonetic English romanization
    5. Uploads stems & lyrics_json to Supabase
    6. Updates status to 'lyrics_ready'
    """
    import sys
    sys.path.insert(0, "/app")
    from supabase import create_client
    from pipeline import run_karaoke_pipeline

    # Connect to Supabase with service-role key
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase = create_client(supabase_url, supabase_key)

    # 1. Fetch song record
    song = supabase.table("songs").select("*").eq("id", song_id).single().execute().data
    if not song:
        print(f"[ERROR] Song {song_id} not found.")
        return

    # Check for early cancellation
    if song.get("status") == "failed":
        print(f"[CANCEL] Job {song_id} was marked failed/cancelled.")
        return

    try:
        # Update status
        supabase.table("songs").update({"status": "separating", "progress": 20}).eq("id", song_id).execute()

        # Run pipeline
        work_dir = f"/tmp/karaoke_{song_id}"
        os.makedirs(work_dir, exist_ok=True)
        source = song.get("source_url") or song.get("source_path")
        
        results = run_karaoke_pipeline(
            source=source,
            output_dir=work_dir,
            manual_lyrics=song.get("official_lyrics"),
            interactive=False,
            render_video=False, # Video is rendered during render step
            language_hint=song.get("language", "auto")
        )

        # Upload Stems & Lyrics JSON to Supabase Storage
        inst_path = f"{song_id}/instrumental.wav"
        voc_path = f"{song_id}/vocals.wav"
        
        with open(results["instrumental_path"], "rb") as f:
            supabase.storage.from_("stems").upload(inst_path, f, {"upsert": "true"})
        with open(results["vocals_path"], "rb") as f:
            supabase.storage.from_("stems").upload(voc_path, f, {"upsert": "true"})

        with open(results["lyrics_json_path"], "r", encoding="utf-8") as f:
            lyrics_data = f.read()

        # Update Song Record to 'lyrics_ready' for user review
        supabase.table("songs").update({
            "status": "lyrics_ready",
            "progress": 70,
            "instrumental_path": inst_path,
            "vocals_path": voc_path,
            "lyrics_json": lyrics_data,
            "duration_s": results["duration_seconds"]
        }).eq("id", song_id).execute()

        print(f"[COMPLETE] Song {song_id} processed successfully. Status: lyrics_ready.")

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        supabase.table("songs").update({
            "status": "failed",
            "error_message": str(e)
        }).eq("id", song_id).execute()
        # Refund credits if needed
        # /api/refund-credits handled in Next.js backend or here


@app.function(
    image=karaoke_image,
    cpu=2.0,
    timeout=300,
    secrets=[modal.Secret.from_name("karaoke-secrets")],
    mounts=[modal.Mount.from_local_dir(REPO_ROOT, remote_path="/app")]
)
def render_mp4_cpu(song_id: str):
    """
    Background CPU worker rendering the 1080p MP4 karaoke video.
    """
    import sys
    sys.path.insert(0, "/app")
    from supabase import create_client
    from subtitle_writer import export_ass_karaoke
    from karaoke_video_renderer import render_karaoke_video
    import json

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase = create_client(supabase_url, supabase_key)

    song = supabase.table("songs").select("*").eq("id", song_id).single().execute().data
    if not song:
        return

    work_dir = f"/tmp/render_{song_id}"
    os.makedirs(work_dir, exist_ok=True)

    try:
        supabase.table("songs").update({"status": "rendering", "progress": 85}).eq("id", song_id).execute()

        # Download instrumental stem
        inst_local = os.path.join(work_dir, "instrumental.wav")
        res = supabase.storage.from_("stems").download(song["instrumental_path"])
        with open(inst_local, "wb") as f:
            f.write(res)

        # Generate ASS subtitles from approved lyrics_json
        lyrics_data = json.loads(song["lyrics_json"]) if isinstance(song["lyrics_json"], str) else song["lyrics_json"]
        ass_path = os.path.join(work_dir, "karaoke.ass")
        export_ass_karaoke(lyrics_data.get("metadata", {}), lyrics_data.get("lines", []), ass_path)

        # Render MP4
        output_mp4 = os.path.join(work_dir, "karaoke.mp4")
        render_karaoke_video(inst_local, ass_path, output_mp4, duration_seconds=song.get("duration_s"))

        # Upload MP4 to karaoke-videos bucket
        video_storage_path = f"{song_id}/karaoke.mp4"
        with open(output_mp4, "rb") as f:
            supabase.storage.from_("karaoke-videos").upload(video_storage_path, f, {"upsert": "true"})

        supabase.table("songs").update({
            "status": "completed",
            "progress": 100,
            "video_path": video_storage_path
        }).eq("id", song_id).execute()

        print(f"[COMPLETE] MP4 rendered for song {song_id}.")

    except Exception as e:
        print(f"[ERROR] Render failed: {e}")
        supabase.table("songs").update({"status": "failed", "error_message": str(e)}).eq("id", song_id).execute()


# Web endpoints using Spawn Pattern (Return ~2s, avoid Vercel timeouts)
@app.function(image=karaoke_image, secrets=[modal.Secret.from_name("karaoke-secrets")])
@modal.fastapi_endpoint(method="POST")
def process_song_web(data: dict):
    """HTTP trigger for song processing."""
    song_id = data.get("song_id")
    if not song_id:
        return {"error": "song_id required"}, 400
    
    # Spawn background task immediately
    process_song_gpu.spawn(song_id)
    return {"status": "dispatched", "song_id": song_id}

@app.function(image=karaoke_image, secrets=[modal.Secret.from_name("karaoke-secrets")])
@modal.fastapi_endpoint(method="POST")
def render_mp4_web(data: dict):
    """HTTP trigger for MP4 rendering."""
    song_id = data.get("song_id")
    if not song_id:
        return {"error": "song_id required"}, 400

    render_mp4_cpu.spawn(song_id)
    return {"status": "dispatched", "song_id": song_id}
