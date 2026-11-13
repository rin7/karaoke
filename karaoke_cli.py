#!/usr/bin/env python3
"""
karaoke_cli.py - Command Line Interface for AI Multilingual Karaoke Creator
Phase 0 Prototype: Takes a song link or audio file and produces:
1. Instrumental & Vocal stems
2. Standardized lyrics.json & lyrics.lrc
3. High-visibility 1080p Karaoke MP4 video with strictly phonetic English lyrics
4. Per-song crash/freeze logs & ML telemetry dataset (inspectable via --check-logs)
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add script directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run_karaoke_pipeline
from audio_ingest import sanitize_filename, source_folder_name
from pipeline_logger import scan_all_song_runs


def main():
    parser = argparse.ArgumentParser(
        description="Karaoke Agent CLI - Create phonetic karaoke songs & MP4 videos from Arabic, Persian, and Armenian songs."
    )
    parser.add_argument(
        "--source", "-s",
        required=False,
        default=None,
        help="Song URL (YouTube, SoundCloud) or path to local audio file (.mp3, .wav, .flac, .m4a)"
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Custom output directory (defaults to ./out/<song_name>)"
    )
    parser.add_argument(
        "--lyrics", "-l",
        default=None,
        help="Optional path to a text file containing official lyrics or raw lyric text"
    )
    parser.add_argument(
        "--lang",
        default="auto",
        choices=["auto", "ar", "fa", "hy"],
        help="Language hint: 'ar' (Arabic), 'fa' (Persian), 'hy' (Armenian), or 'auto'"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip rendering the final MP4 karaoke video (only output stems and subtitles)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Non-interactive mode (automatically transcribe/time from vocal audio)"
    )
    parser.add_argument(
        "--check-logs",
        action="store_true",
        help="Scan all song runs to detect frozen/crashed jobs and display ML telemetry summary"
    )

    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.check_logs:
        out_root = os.path.join(base_dir, "out")
        report = scan_all_song_runs(out_root)
        print("\n" + "="*65)
        print(" 📊  KARAOKE AGENT — HEALTH, CRASH/FREEZE & ML TELEMETRY REPORT")
        print("="*65)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("="*65 + "\n")
        return

    if not args.source:
        parser.error("the following arguments are required: --source/-s (or pass --check-logs)")

    # Determine lyrics content if passed as file or string
    manual_lyrics = None
    if args.lyrics:
        if os.path.isfile(args.lyrics):
            with open(args.lyrics, "r", encoding="utf-8") as f:
                manual_lyrics = f.read()
        else:
            manual_lyrics = args.lyrics

    # Determine output folder (one unique folder per song so runs never mix files)
    if args.out:
        output_dir = args.out
    else:
        if args.source.startswith("http"):
            base_name = source_folder_name(args.source)
        else:
            base_name = sanitize_filename(Path(args.source).stem)
        output_dir = os.path.join(base_dir, "out", base_name)

    print("\n" + "="*65)
    print(" 🎤  KARAOKE AGENT - AI MULTILINGUAL PHONETIC KARAOKE")
    print("="*65)
    print(f" Source:           {args.source}")
    print(f" Output Folder:    {output_dir}")
    print(f" Language Hint:    {args.lang}")
    print(f" Render MP4 Video: {not args.no_video}")
    print(f" Interactive Mode: {not args.non_interactive}")
    print("="*65)

    try:
        result = run_karaoke_pipeline(
            source=args.source,
            output_dir=output_dir,
            manual_lyrics=manual_lyrics,
            interactive=not args.non_interactive,
            render_video=not args.no_video,
            language_hint=args.lang
        )

        print("\n" + "✨"*30)
        print(" 🎉  KARAOKE TRACK CREATED SUCCESSFULLY!")
        print("✨"*30)
        print(f" 🎵 Instrumental Audio:  {result.get('instrumental_path')}")
        print(f" 🎙️  Isolated Vocals:     {result.get('vocals_path')}")
        print(f" 📄 Synchronized JSON:   {result.get('lyrics_json_path')}")
        print(f" 📝 Standard LRC File:   {result.get('lyrics_lrc_path')}")
        print(f" 🎨 Karaoke ASS Subs:    {result.get('karaoke_ass_path')}")
        if result.get("karaoke_video_path"):
            print(f" 🎬 1080p Karaoke Video: {result.get('karaoke_video_path')}")
        print(f" 📋 Song Execution Log:  {result.get('pipeline_log_path')}")
        print(f" 📈 Song Telemetry:      {result.get('telemetry_path')}")
        print(f" 🧠 Global ML Dataset:   {result.get('global_history_path')}")
        if result.get("karaoke_video_path"):
            print(f"\n 💡 You can now play '{result.get('karaoke_video_path')}' or upload it to YouTube/TikTok!")
        print("="*65 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}", file=sys.stderr)
        print(f"   Full error & traceback saved in: {os.path.join(output_dir, 'pipeline.log')}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
