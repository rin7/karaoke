"""
pipeline.py - Master Orchestrator for the Karaoke Agent Pipeline
Chains together audio ingestion, stem separation, vocal-first lyrics timing,
strict phonetic transliteration, 1080p MP4 rendering, and full per-song + ML
telemetry logging with watchdog freeze & crash detection.
"""

import os
import json
import time
import shutil
from typing import Dict, Any, Optional

from audio_ingest import ingest_audio, sanitize_filename
from lyrics_search import auto_search_lyrics, confirm_lyrics_interactive
from stem_separator import separate_stems
from lyrics_aligner import transcribe_and_align_vocals
from phonetic_transliterator import process_phonetics
from subtitle_writer import export_lyrics_json, export_lrc, export_ass_karaoke
from karaoke_video_renderer import render_karaoke_video
from pipeline_logger import SongRunLogger, compute_ml_quality_features


def run_karaoke_pipeline(
    source: str,
    output_dir: str,
    manual_lyrics: Optional[str] = None,
    interactive: bool = True,
    render_video: bool = True,
    language_hint: str = "auto",
    global_log_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute the full end-to-end karaoke creation pipeline with live logging,
    crash/freeze watchdog monitoring, and ML telemetry collection:
    1. Ingest original video/audio.
    2. Separate singer vocals (vocals.wav) and clean instrumental (instrumental.wav).
    3. Load optional verified reference lyrics (no blind LLM guessing).
    4. Listen directly to vocals.wav to extract & time exact sung lyrics on the video timeline.
    5. Generate strict phonetic English & translations while preserving exact vocal timestamps.
    6. Render 1080p MP4 using ONLY vocal-free instrumental.wav + burned-in timed subtitles.
    """
    os.makedirs(output_dir, exist_ok=True)

    with SongRunLogger(
        source=source,
        output_dir=output_dir,
        global_log_dir=global_log_dir
    ) as logger:
        start_time = time.time()
        print(f"\n========================================================")
        print(f"       AI MULTILINGUAL KARAOKE PIPELINE START           ")
        print(f"========================================================")

        # Step 1: Audio Ingestion & Normalization
        with logger.step("step1_ingest", stall_timeout_sec=180.0):
            print(f"[1/6] Ingesting and normalizing audio from: {source}")
            meta = ingest_audio(source, output_dir)
            logger.record_metadata(meta)
            print(f"      Title:  {meta.get('title')}")
            print(f"      Artist: {meta.get('artist')}")
            print(f"      Duration: {meta.get('duration'):.1f}s")

        # Step 2: Stem Separation (run FIRST so we have clean singer vocals for timing + vocal-free music for final video)
        had_cached_fp = os.path.exists(os.path.join(output_dir, "stems_source.txt"))
        t_stems = time.time()
        with logger.step("step2_stem_separation", stall_timeout_sec=900.0):
            print(f"\n[2/6] Separating singer vocals and clean instrumental tracks...")
            stems = separate_stems(meta["normalized_wav"], output_dir)
            inst_path = stems["instrumental"]
            voc_path = stems["vocals"]
            print(f"      Instrumental (for final video): {os.path.basename(inst_path)}")
            print(f"      Vocals (for lyric timing):      {os.path.basename(voc_path)}")
        stem_cache_hit = had_cached_fp and (time.time() - t_stems) < 5.0

        # Step 3: Optional Verified Reference Lyrics (never guesses from title alone)
        online_lyrics_found = False
        confirmed_lyrics = manual_lyrics
        confirmed_synced_lyrics = None
        with logger.step("step3_lyrics_search", stall_timeout_sec=60.0):
            if not confirmed_lyrics:
                print(f"\n[3/6] Checking verified online reference lyrics for: '{meta.get('title')}'...")
                lyrics_res = auto_search_lyrics(
                    meta.get("title", ""),
                    meta.get("artist", ""),
                    duration=float(meta.get("duration") or 0.0),
                )
                online_lyrics_found = bool(lyrics_res.get("found"))
                if online_lyrics_found:
                    print(f"      Found candidate reference lyrics from: {lyrics_res.get('source_url')}")
                    if interactive:
                        with logger.pause_watchdog_for_input():
                            confirmed_lyrics = confirm_lyrics_interactive(lyrics_res)
                            if confirmed_lyrics and confirmed_lyrics == lyrics_res.get("original_lyrics"):
                                confirmed_synced_lyrics = lyrics_res.get("synced_lyrics")
                    else:
                        confirmed_lyrics = lyrics_res.get("original_lyrics")
                        confirmed_synced_lyrics = lyrics_res.get("synced_lyrics")
                else:
                    print("      No verified text match online — will extract & time lyrics directly from singer's voice.")
                    if interactive:
                        with logger.pause_watchdog_for_input():
                            confirmed_lyrics = confirm_lyrics_interactive(lyrics_res)

        # Step 4: Audio-First Vocal Timing & Transcription on vocals.wav
        with logger.step("step4_vocal_timing", stall_timeout_sec=240.0):
            print(f"\n[4/6] Timing & extracting lyrics directly from singer's vocal audio...")
            align_res = transcribe_and_align_vocals(
                vocal_path=voc_path,
                reference_lyrics=confirmed_lyrics,
                language=language_hint,
                duration=meta["duration"],
                song_title=meta.get("title", ""),
                artist=meta.get("artist", ""),
                synced_lyrics=confirmed_synced_lyrics,
            )
            aligned_lines = align_res.get("lines", [])
            if align_res.get("language") and align_res["language"] not in ("auto", "none") and language_hint == "auto":
                language_hint = align_res["language"]
            meta["language"] = language_hint
            logger.record_metadata(meta)
            print(f"      Timed {len(aligned_lines)} sung lyric lines via '{align_res.get('method')}' (lang={language_hint}).")

            if not aligned_lines:
                logger.log_warning("No lyric lines were generated from vocal track; using fallback title line.")
                print("      [WARN] No lyric lines were generated. Creating fallback instrumental line.")
                aligned_lines = [{
                    "line_id": 0, "text": meta["title"], "start": 5.0, "end": max(10.0, meta["duration"] - 5.0),
                    "words": [{"word": meta["title"], "start": 5.0, "end": max(10.0, meta["duration"] - 5.0)}]
                }]

        # Step 5: Strict Phonetic Transliteration & English Translation (keeps exact vocal timestamps)
        with logger.step("step5_phonetics_and_subtitles", stall_timeout_sec=180.0):
            print(f"\n[5/6] Generating strict phonetic English lyrics (no Arabizi numbers / no diacritics)...")
            phonetic_lines = process_phonetics(aligned_lines, language_hint)
            print(f"      Transliterated {len(phonetic_lines)} lines (timestamps preserved).")

            json_path = os.path.join(output_dir, "lyrics.json")
            lrc_path = os.path.join(output_dir, "lyrics.lrc")
            ass_path = os.path.join(output_dir, "karaoke.ass")

            export_lyrics_json(meta, phonetic_lines, json_path)
            export_lrc(phonetic_lines, lrc_path)
            export_ass_karaoke(meta, phonetic_lines, ass_path)
            print(f"      Saved: lyrics.json, lyrics.lrc, karaoke.ass")

            ml_features = compute_ml_quality_features(
                phonetic_lines=phonetic_lines,
                duration_sec=meta["duration"],
                align_meta=align_res,
                stem_cache_hit=stem_cache_hit,
                online_lyrics_found=online_lyrics_found
            )
            logger.record_ml_features(ml_features)

        # Step 6: Render MP4 Karaoke Video
        video_path = None
        if render_video:
            with logger.step("step6_video_render", stall_timeout_sec=600.0):
                print(f"\n[6/6] Rendering 1080p Karaoke MP4 video with burned-in 3-line subtitles...")
                title = meta.get("title", "Karaoke_Track")
                artist = meta.get("artist", "")
                if artist and artist != "Unknown Artist" and artist.lower() not in title.lower():
                    song_label = f"{artist} - {title} (Karaoke)"
                else:
                    song_label = f"{title} (Karaoke)"

                video_filename = f"{sanitize_filename(song_label)}.mp4"
                video_path = os.path.join(output_dir, video_filename)

                bg_image = meta.get("thumbnail_path")
                render_karaoke_video(
                    instrumental_audio_path=inst_path,
                    ass_subtitle_path=ass_path,
                    output_mp4_path=video_path,
                    background_image_path=bg_image,
                    duration_seconds=meta["duration"]
                )

                # Maintain a karaoke.mp4 copy for generic references
                generic_path = os.path.join(output_dir, "karaoke.mp4")
                if os.path.abspath(video_path) != os.path.abspath(generic_path):
                    try:
                        shutil.copy2(video_path, generic_path)
                    except Exception:
                        pass

                print(f"      Karaoke video ready at: {video_path}")

        artifacts = {
            "instrumental": inst_path,
            "vocals": voc_path,
            "lyrics_json": json_path,
            "lyrics_lrc": lrc_path,
            "karaoke_ass": ass_path,
            "karaoke_video": video_path,
        }
        logger.record_artifacts(artifacts)

        elapsed = time.time() - start_time
        print(f"\n========================================================")
        print(f" PIPELINE COMPLETE in {elapsed:.1f}s")
        print(f" Output directory: {os.path.abspath(output_dir)}")
        print(f" Run Log & ML Telemetry: {logger.log_path}")
        print(f"========================================================\n")

        return {
            "run_id": logger.run_id,
            "metadata": meta,
            "instrumental_path": inst_path,
            "vocals_path": voc_path,
            "lyrics_json_path": json_path,
            "lyrics_lrc_path": lrc_path,
            "karaoke_ass_path": ass_path,
            "karaoke_video_path": video_path,
            "pipeline_log_path": logger.log_path,
            "telemetry_path": logger.telemetry_path,
            "run_status_path": logger.status_path,
            "global_history_path": logger.history_path,
            "ml_features": ml_features,
            "duration_seconds": meta["duration"],
            "elapsed_seconds": round(elapsed, 2)
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_karaoke_pipeline(sys.argv[1], "./test_run")
    else:
        print("Usage: python pipeline.py <URL_or_AudioFile>")
