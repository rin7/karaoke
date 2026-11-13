"""
karaoke_video_renderer.py - High-Quality Karaoke MP4 Video Renderer
Burns synchronized 3-line karaoke subtitles (.ass) into a 1080p MP4 video
using an FFmpeg binary compiled with libass, HarfBuzz, and FriBidi.
"""

import os
import sys
import shutil
import subprocess
from typing import Optional, Dict, Any

def check_binary_has_subtitles(exe_path: str) -> bool:
    """Check if a specific ffmpeg binary supports the 'subtitles' or 'ass' filter."""
    try:
        proc = subprocess.run([exe_path, "-filters"], capture_output=True, text=True)
        return " subtitles " in proc.stdout or " ass " in proc.stdout
    except Exception:
        return False

def get_ffmpeg_binary() -> str:
    """
    Resolve the best FFmpeg binary that supports hard burning subtitles via libass.
    Prioritizes imageio_ffmpeg (which bundles libass, HarfBuzz, and FriBidi)
    and falls back to system ffmpeg if it has libass enabled.
    """
    # 1. Try imageio_ffmpeg binary (standard on Python, includes libass + HarfBuzz + FriBidi)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe) and check_binary_has_subtitles(exe):
            return exe
    except Exception:
        pass

    # 2. Check system ffmpeg in PATH
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg and check_binary_has_subtitles(sys_ffmpeg):
        return sys_ffmpeg

    # 3. Fallback: try imageio_ffmpeg even if filter check was indeterminate
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return sys_ffmpeg or "ffmpeg"

def render_karaoke_video(
    instrumental_audio_path: str,
    ass_subtitle_path: str,
    output_mp4_path: str,
    background_image_path: Optional[str] = None,
    duration_seconds: Optional[float] = None
) -> str:
    """
    Render a 1080p MP4 karaoke video with HARD-BURNED subtitles.
    The lyrics are rendered directly onto the video frames so they are always visible
    on any player (QuickTime, YouTube, TikTok, iPhone, TV) without needing CC toggled.
    """
    if not os.path.exists(instrumental_audio_path):
        raise FileNotFoundError(f"Instrumental audio not found: {instrumental_audio_path}")
    if not os.path.exists(ass_subtitle_path):
        raise FileNotFoundError(f"ASS subtitle file not found: {ass_subtitle_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)
    ffmpeg_bin = get_ffmpeg_binary()

    # Escape path characters for FFmpeg filter syntax (escape colons, backslashes, and single quotes)
    ass_escaped = os.path.abspath(ass_subtitle_path).replace("\\", "/").replace(":", "\\:").replace("'", "'\\\\''")

    print(f"[RENDER] Using FFmpeg engine: {ffmpeg_bin}")
    print(f"[RENDER] Burning karaoke subtitles from: {os.path.basename(ass_subtitle_path)}")

    if background_image_path and os.path.exists(background_image_path):
        # Image background: scale to 1920x1080, gentle blur, darken for maximum text contrast
        filter_complex = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
            f"boxblur=10:5[bg];"
            f"color=c=black@0.45:s=1920x1080[dark];"
            f"[bg][dark]overlay=0:0,"
            f"subtitles='{ass_escaped}'[v]"
        )
        cmd = [
            ffmpeg_bin, "-y", "-loop", "1",
            "-i", os.path.abspath(background_image_path),
            "-i", os.path.abspath(instrumental_audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k",
            "-shortest", output_mp4_path
        ]
    else:
        # Sleek stage background with burned-in subtitles
        filter_complex = (
            f"color=c=#0b0b14:s=1920x1080:r=30,"
            f"drawbox=x=0:y=0:w=1920:h=1080:color=#141228@0.6:t=fill,"
            f"subtitles='{ass_escaped}'[v]"
        )
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "lavfi", "-i", "color=c=#0b0b14:s=1920x1080:r=30",
            "-i", os.path.abspath(instrumental_audio_path),
            "-vf", f"subtitles='{ass_escaped}'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k",
            "-shortest", output_mp4_path
        ]

    print(f"[RENDER] Encoding 1080p video with burned-in lyrics to: {output_mp4_path}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[ERROR] FFmpeg render failed:\n", proc.stderr[-800:])
        raise RuntimeError(f"FFmpeg render error: {proc.stderr[-400:]}")

    print(f"[RENDER] Successfully created karaoke video: {output_mp4_path}")
    return os.path.abspath(output_mp4_path)

if __name__ == "__main__":
    if len(sys.argv) > 3:
        bg = sys.argv[4] if len(sys.argv) > 4 else None
        render_karaoke_video(sys.argv[1], sys.argv[2], sys.argv[3], background_image_path=bg)
    else:
        print("Usage: python karaoke_video_renderer.py <instrumental.wav> <karaoke.ass> <output.mp4> [bg.jpg]")
