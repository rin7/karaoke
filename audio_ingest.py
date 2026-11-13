"""
audio_ingest.py - Audio Ingestion & Normalization Module
Handles downloading audio from URLs (YouTube, SoundCloud, etc.) via yt-dlp
and processing local audio files into normalized 44.1kHz stereo PCM WAV.
Supports both native python `yt_dlp` library and CLI fallback.
"""

import os
import sys
import subprocess
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

def is_url(source: str) -> bool:
    """Check if the source string is an HTTP/HTTPS URL."""
    return bool(re.match(r"^https?://", source.strip()))

def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used safely as a filename."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")

def source_folder_name(url: str) -> str:
    """
    Build a stable, unique output folder name for a song URL.
    YouTube links (watch?v=, youtu.be/, shorts/) map to 'yt_<videoId>' so the same
    song always reuses its own folder and different songs never share one.
    """
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    if m:
        return f"yt_{m.group(1)}"
    import hashlib
    return "url_" + hashlib.sha1(url.strip().encode("utf-8")).hexdigest()[:12]

def check_ytdlp_available() -> bool:
    """Check if yt_dlp is installed as a python package or CLI tool."""
    try:
        import yt_dlp  # noqa
        return True
    except ImportError:
        pass
    return bool(shutil.which("yt-dlp"))

def download_with_python_ytdlp(source: str, raw_audio_target: str, metadata: Dict[str, Any]):
    """Download audio stream using yt_dlp python module directly."""
    import yt_dlp

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{raw_audio_target}.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "writethumbnail": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "0",
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        if info:
            metadata["title"] = info.get("title", metadata["title"])
            metadata["artist"] = info.get("artist") or info.get("creator") or info.get("uploader", metadata["artist"])
            metadata["duration"] = float(info.get("duration", 0.0))
            metadata["webpage_url"] = info.get("webpage_url", source)

def download_with_cli_ytdlp(source: str, raw_audio_target: str, metadata: Dict[str, Any]):
    """Download audio stream using yt-dlp CLI binary or python -m yt_dlp."""
    cmd_prefix = ["yt-dlp"] if shutil.which("yt-dlp") else [sys.executable, "-m", "yt_dlp"]
    
    # 1. Fetch info json
    info_proc = subprocess.run(
        cmd_prefix + ["--dump-json", "--no-playlist", source],
        capture_output=True, text=True
    )
    if info_proc.returncode == 0:
        try:
            info = json.loads(info_proc.stdout)
            metadata["title"] = info.get("title", metadata["title"])
            metadata["artist"] = info.get("artist") or info.get("creator") or info.get("uploader", metadata["artist"])
            metadata["duration"] = float(info.get("duration", 0.0))
            metadata["webpage_url"] = info.get("webpage_url", source)
        except Exception:
            pass

    # 2. Download audio
    download_cmd = cmd_prefix + [
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--write-thumbnail",
        "--no-playlist",
        "-o", f"{raw_audio_target}.%(ext)s",
        source
    ]
    subprocess.run(download_cmd, check=True)

def ingest_audio(source: str, output_dir: str) -> Dict[str, Any]:
    """
    Ingest audio from a local file path or online URL.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata: Dict[str, Any] = {
        "title": "Unknown Title",
        "artist": "Unknown Artist",
        "duration": 0.0,
        "thumbnail_path": None,
        "raw_audio_path": None,
        "normalized_wav": None,
        "source_type": "url" if is_url(source) else "file",
        "source_input": source
    }

    raw_audio_target = os.path.join(output_dir, "raw_audio")

    if is_url(source):
        if not check_ytdlp_available():
            raise RuntimeError(
                "\n\n[ERROR] 'yt-dlp' is required to download songs from YouTube links.\n"
                "Please install it in your terminal by running:\n\n"
                "    pip3 install yt-dlp\n"
                "or:\n"
                "    brew install yt-dlp\n"
            )

        # Remove leftover downloads so yt-dlp can never reuse a previous song's audio/thumbnail
        for f in os.listdir(output_dir):
            if f.startswith("raw_audio."):
                try:
                    os.remove(os.path.join(output_dir, f))
                except OSError:
                    pass

        try:
            download_with_python_ytdlp(source, raw_audio_target, metadata)
        except ImportError:
            download_with_cli_ytdlp(source, raw_audio_target, metadata)
        except Exception as e:
            # Fallback to CLI invocation
            download_with_cli_ytdlp(source, raw_audio_target, metadata)

        # Locate the downloaded audio file and thumbnail
        for f in os.listdir(output_dir):
            if f.startswith("raw_audio.") and not f.endswith((".webp", ".jpg", ".png", ".jpeg")):
                metadata["raw_audio_path"] = os.path.join(output_dir, f)
            elif f.startswith("raw_audio.") and f.endswith((".webp", ".jpg", ".png", ".jpeg")):
                metadata["thumbnail_path"] = os.path.join(output_dir, f)
    else:
        # Local file ingestion
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source file not found: {source}")
        
        metadata["raw_audio_path"] = os.path.abspath(source)
        metadata["title"] = Path(source).stem

    if not metadata["raw_audio_path"] or not os.path.exists(metadata["raw_audio_path"]):
        raise FileNotFoundError(f"Failed to locate downloaded or input audio file in: {output_dir}")

    # Probe duration and audio properties using ffprobe
    probe_cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        metadata["raw_audio_path"]
    ]
    probe_res = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe_res.returncode == 0:
        try:
            probe_data = json.loads(probe_res.stdout)
            fmt = probe_data.get("format", {})
            metadata["duration"] = float(fmt.get("duration", metadata["duration"]))
            tags = fmt.get("tags", {})
            if "title" in tags and metadata["title"] == "Unknown Title":
                metadata["title"] = tags["title"]
            if "artist" in tags and metadata["artist"] == "Unknown Artist":
                metadata["artist"] = tags["artist"]
        except Exception:
            pass

    # Normalize audio to standard 44.1 kHz, 16-bit stereo PCM WAV
    normalized_wav = os.path.join(output_dir, "audio_normalized.wav")
    norm_cmd = [
        "ffmpeg",
        "-y",
        "-i", metadata["raw_audio_path"],
        "-vn",
        "-ar", "44100",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        normalized_wav
    ]
    subprocess.run(norm_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    metadata["normalized_wav"] = normalized_wav

    # Save metadata JSON
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata

if __name__ == "__main__":
    if len(sys.argv) > 1:
        out = ingest_audio(sys.argv[1], "./test_ingest")
        print("Ingestion complete:", json.dumps(out, indent=2))
    else:
        print("Usage: python audio_ingest.py <URL_or_File>")
