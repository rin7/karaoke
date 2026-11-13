#!/usr/bin/env python3
"""
tests/test_pipeline_steps.py - Comprehensive Step-by-Step Verification Suite
Tests every stage of the Karaoke Agent pipeline:
  Step 1: Audio ingestion, normalization & per-song folder isolation
  Step 2: Vocal/instrumental stem separation & fingerprint-verified caching
  Step 3: Online query cleaning & anti-hallucination lyrics lookup
  Step 4: Vocal-first VAD segmentation, solo-skipping chunking & non-overlapping sync
  Step 5: Strict ASCII phonetic romanization & gap-accurate 3-tier ASS/LRC/JSON export
  Step 6: Vocal-free 1080p MP4 video rendering & full end-to-end pipeline orchestration
  Step 7: Per-song execution logging, ML telemetry dataset, freeze watchdog & hard-crash recovery
"""

import os
import re
import sys
import json
import math
import time
import wave
import struct
import shutil
import tempfile
import unittest
import subprocess
from unittest.mock import patch

# Ensure Karaoke_Agent root is importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from audio_ingest import ingest_audio, sanitize_filename, source_folder_name
from stem_separator import separate_stems, _audio_fingerprint, _write_stem_fingerprint
from lyrics_search import clean_song_query, auto_search_lyrics
from lyrics_aligner import (
    AUDIO_TIMING_SYSTEM_PROMPT,
    get_ffmpeg_exe,
    extract_vocal_intervals_vad,
    group_intervals_into_chunks,
    normalize_and_build_words,
    resolve_dominant_language_and_filter_outliers,
    transcribe_and_align_vocals,
)
from phonetic_transliterator import (
    PHONETIC_SYSTEM_PROMPT,
    sanitize_strict_ascii_phonetic,
    offline_phonetic_transliterator,
    process_phonetics,
)
from subtitle_writer import export_lyrics_json, export_lrc, export_ass_karaoke
from karaoke_video_renderer import render_karaoke_video
from pipeline import run_karaoke_pipeline
from pipeline_logger import (
    SongRunLogger,
    detect_and_recover_stale_run,
    compute_ml_quality_features,
    scan_all_song_runs,
)
from web_app import create_app


def generate_test_wav(
    path: str,
    duration_sec: float,
    vocal_windows: list,
    sample_rate: int = 44100,
    include_stereo_music: bool = False
) -> str:
    """
    Create a deterministic 16-bit stereo WAV file:
    - vocal_windows: list of (start_s, end_s) where a center-panned 440Hz vocal tone plays
      (identical on L and R channels, so center-channel cancellation removes it cleanly).
    - include_stereo_music: if True, adds a continuous out-of-phase 220Hz stereo backing tone
      (L = +sin, R = -sin) that survives center-channel cancellation as pure instrumental!
    """
    total_frames = int(duration_sec * sample_rate)
    frames = bytearray()

    for n in range(total_frames):
        t = n / sample_rate
        in_vocal = any(ws <= t <= we for ws, we in vocal_windows)
        voc_val = 0.45 * math.sin(2.0 * math.pi * 440.0 * t) if in_vocal else 0.0
        mus_val = 0.25 * math.sin(2.0 * math.pi * 220.0 * t) if include_stereo_music else 0.0

        left = max(-0.99, min(0.99, voc_val + mus_val))
        right = max(-0.99, min(0.99, voc_val - mus_val))
        frames.extend(struct.pack("<hh", int(left * 32767), int(right * 32767)))

    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))

    return path


def compute_wav_band_rms(wav_path: str, start_s: float, end_s: float) -> float:
    """Compute RMS amplitude of a WAV file inside [start_s, end_s]."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        start_f = int(start_s * sr)
        end_f = int(end_s * sr)
        wf.setpos(min(start_f, wf.getnframes()))
        raw = wf.readframes(max(0, end_f - start_f))

    if not raw:
        return 0.0
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    if not samples:
        return 0.0
    mean_sq = sum((s / 32768.0) ** 2 for s in samples) / len(samples)
    return math.sqrt(mean_sq)


class TestStep1AudioIngest(unittest.TestCase):
    """Step 1: Verify URL folder isolation, filename sanitization, and WAV normalization."""

    def test_folder_isolation_across_urls(self):
        sirvan = source_folder_name("https://www.youtube.com/watch?v=iKTnrJTj2CM")
        sirvan_playlist = source_folder_name("https://www.youtube.com/watch?v=iKTnrJTj2CM&list=RD")
        tata = source_folder_name("https://youtu.be/ykOUZYLM89I?si=abc123")

        self.assertEqual(sirvan, "yt_iKTnrJTj2CM")
        self.assertEqual(sirvan, sirvan_playlist)
        self.assertEqual(tata, "yt_ykOUZYLM89I")
        self.assertNotEqual(sirvan, tata)

    def test_sanitize_filename(self):
        raw = 'Tata Simonyan - Sirum em Qez // Aj te Dzakh: "Vol.3"? (Karaoke)'
        clean = sanitize_filename(raw)
        self.assertNotRegex(clean, r'[\\/*?:"<>| ]')
        self.assertTrue(clean.endswith("(Karaoke)"))

    def test_local_audio_ingest_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_wav = generate_test_wav(
                os.path.join(tmp, "input_song.wav"),
                duration_sec=4.0,
                vocal_windows=[(0.5, 2.5)],
                sample_rate=22050,
                include_stereo_music=True,
            )
            out_dir = os.path.join(tmp, "out")
            meta = ingest_audio(src_wav, out_dir)

            self.assertTrue(os.path.exists(meta["normalized_wav"]))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "metadata.json")))
            self.assertAlmostEqual(meta["duration"], 4.0, delta=0.15)

            with wave.open(meta["normalized_wav"], "rb") as wf:
                self.assertEqual(wf.getframerate(), 44100)
                self.assertEqual(wf.getnchannels(), 2)
                self.assertEqual(wf.getsampwidth(), 2)


class TestStep2StemSeparation(unittest.TestCase):
    """Step 2: Verify stem fingerprint cache safety and center-vocal suppression."""

    def test_fingerprint_prevents_cross_song_stem_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            song_a = generate_test_wav(os.path.join(tmp, "a.wav"), 3.0, [(0.5, 2.0)])
            song_b = generate_test_wav(os.path.join(tmp, "b.wav"), 3.0, [(1.0, 2.8)], include_stereo_music=True)

            fp_a = _audio_fingerprint(song_a)
            fp_b = _audio_fingerprint(song_b)
            self.assertNotEqual(fp_a, fp_b)

            # Seed fake cached Demucs stems belonging to song_a
            out_dir = os.path.join(tmp, "out")
            demucs_dir = os.path.join(out_dir, "htdemucs", "b")
            os.makedirs(demucs_dir, exist_ok=True)
            fake_inst = generate_test_wav(os.path.join(demucs_dir, "no_vocals.wav"), 3.0, [])
            fake_voc = generate_test_wav(os.path.join(demucs_dir, "vocals.wav"), 3.0, [(0.5, 2.0)])
            _write_stem_fingerprint(out_dir, fp_a)  # Mismatched fingerprint!

            # Calling separate_stems on song_b with prefer_gpu=False must NOT reuse song_a's stems
            stems_b = separate_stems(song_b, out_dir, prefer_gpu=False)
            self.assertTrue(os.path.exists(stems_b["instrumental"]))
            # Instrumental of song_b has stereo music (>0.1 RMS), whereas fake_inst was pure silence (0.0 RMS)
            rms_inst = compute_wav_band_rms(stems_b["instrumental"], 0.2, 0.8)
            self.assertGreater(rms_inst, 0.05)

            # Now write matching fingerprint for song_a and verify cache reuse succeeds
            demucs_dir_a = os.path.join(out_dir, "htdemucs", "a")
            os.makedirs(demucs_dir_a, exist_ok=True)
            shutil.copy2(fake_inst, os.path.join(demucs_dir_a, "no_vocals.wav"))
            shutil.copy2(fake_voc, os.path.join(demucs_dir_a, "vocals.wav"))
            _write_stem_fingerprint(out_dir, fp_a)
            stems_a = separate_stems(song_a, out_dir, prefer_gpu=False)
            self.assertEqual(compute_wav_band_rms(stems_a["instrumental"], 0.2, 0.8), 0.0)

    def test_center_vocal_removed_in_local_fallback_instrumental(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Pure center vocal (no stereo music): phase cancellation must strip the vocal completely!
            vocal_only = generate_test_wav(
                os.path.join(tmp, "vocal_center.wav"),
                duration_sec=3.0,
                vocal_windows=[(0.5, 2.5)],
                include_stereo_music=False,
            )
            stems = separate_stems(vocal_only, os.path.join(tmp, "stems"), prefer_gpu=False)
            inst_rms = compute_wav_band_rms(stems["instrumental"], 0.6, 2.4)
            voc_rms = compute_wav_band_rms(stems["vocals"], 0.6, 2.4)
            self.assertLess(inst_rms, 0.002, "Instrumental stem still contains center vocals!")
            self.assertGreater(voc_rms, 0.15, "Vocals stem lost the singer's voice!")


class TestStep3LyricsSearchAntiHallucination(unittest.TestCase):
    """Step 3: Verify metadata query cleaning and zero blind LLM lyric hallucinations."""

    def test_clean_song_query_strips_album_and_year(self):
        q = clean_song_query(
            "Tata Simonyan - Sirum em Qez // Aj te Dzakh - Vol.3 // 1998 [Official Video] 4K",
            "Tata Simonyan",
        )
        self.assertEqual(q, "Tata Simonyan Sirum em Qez")

    @patch("lyrics_search.search_lyricstranslate", return_value=[])
    def test_auto_search_never_hallucinates_when_unmatched(self, _mock_lt):
        res = auto_search_lyrics("Tata Simonyan - Sirum em Qez // Aj te Dzakh - Vol.3 // 1998", "Tata Simonyan")
        self.assertFalse(res["found"])
        self.assertEqual(res["original_lyrics"], "")


class TestStep4VocalFirstTimingAndAlignment(unittest.TestCase):
    """Step 4: Verify VAD early/late phrase capture, solo skipping, chunking, and non-overlapping sync."""

    def test_vad_captures_intro_and_ending_vocals_and_skips_solo(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 35s vocal track:
            # - Intro vocal at 0.8s - 3.2s (<5s!)
            # - Verse vocals at 6.0s - 9.5s and 10.5s - 14.0s
            # - 15-second silent guitar solo: 14.0s - 29.0s
            # - Ending chorus up to end of track: 29.0s - 34.8s
            voc_wav = generate_test_wav(
                os.path.join(tmp, "vocals.wav"),
                duration_sec=35.0,
                vocal_windows=[(0.8, 3.2), (6.0, 9.5), (10.5, 14.0), (29.0, 34.8)],
            )
            intervals = extract_vocal_intervals_vad(voc_wav, total_duration=35.0)
            self.assertEqual(len(intervals), 4, f"Expected 4 vocal phrases, got: {intervals}")
            self.assertAlmostEqual(intervals[0][0], 0.8, delta=0.15)
            self.assertAlmostEqual(intervals[0][1], 3.2, delta=0.15)
            self.assertAlmostEqual(intervals[-1][0], 29.0, delta=0.15)
            self.assertAlmostEqual(intervals[-1][1], 34.8, delta=0.25)

            # Verify chunk grouping skips the 15s solo gap (14.0 -> 29.0) and stays < 40s
            chunks = group_intervals_into_chunks(intervals, total_duration=35.0)
            self.assertEqual(len(chunks), 2, f"Expected 2 chunks split across 15s solo, got: {chunks}")
            self.assertLessEqual(chunks[0]["end"], 14.5)
            self.assertGreaterEqual(chunks[1]["start"], 28.5)
            for ch in chunks:
                self.assertLess(ch["duration"], 40.0)

    @patch("lyrics_aligner.transcribe_chunk_with_gemini")
    def test_multimodal_chunk_alignment_maps_global_times_and_prevents_overlaps(self, mock_gemini):
        with tempfile.TemporaryDirectory() as tmp:
            voc_wav = generate_test_wav(
                os.path.join(tmp, "vocals.wav"),
                duration_sec=35.0,
                vocal_windows=[(0.8, 3.2), (6.0, 9.5), (10.5, 14.0), (29.0, 34.8)],
            )

            # Mock Chunk 0 (0.55s - 14.25s) and Chunk 1 (28.75s - 35.0s) responses
            # Include a simultaneous backing-vocal duplicate + slight overlap to test normalization!
            mock_gemini.side_effect = [
                {
                    "language": "hy",
                    "lines": [
                        {"start": 0.25, "end": 2.65, "text": "Սիրում եմ քեզ, սիրում եմ քեզ"},
                        {"start": 5.45, "end": 9.10, "text": "Դու անուշիկ աղջիկ, անուշիկ աղջիկ ես"},
                        {"start": 5.55, "end": 8.95, "text": "Սիրում եմ քեզ"},  # simultaneous harmony duplicate
                        {"start": 9.00, "end": 13.40, "text": "Սև աչքերիդ գիշերում մոլորվել եմ ես"},  # slight overlap
                    ],
                },
                {
                    "language": "hy",
                    "lines": [
                        {"start": 0.25, "end": 5.80, "text": "Սիրում եմ աչքերը քո հեզ"},
                    ],
                },
            ]

            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                res = transcribe_and_align_vocals(voc_wav, language="auto", duration=35.0)

            self.assertEqual(res["method"], "gemini_audio")
            self.assertEqual(res["language"], "hy")
            lines = res["lines"]
            self.assertEqual(len(lines), 4)  # Duplicate harmony removed, 4 real sung lines kept!

            # Check zero lyrics placed in the 14.5s - 28.5s guitar solo
            for ln in lines:
                self.assertFalse(14.5 < ln["start"] < 28.5, f"Line leaked into guitar solo: {ln}")

            # Check strict monotonic non-overlapping line & word intervals
            for i in range(len(lines)):
                self.assertLess(lines[i]["start"], lines[i]["end"])
                if i + 1 < len(lines):
                    self.assertLess(
                        lines[i]["end"],
                        lines[i + 1]["start"],
                        f"Lines {i} and {i+1} overlap: {lines[i]} vs {lines[i+1]}",
                    )
                for w in lines[i]["words"]:
                    self.assertGreaterEqual(w["start"], lines[i]["start"])
                    self.assertLessEqual(w["end"], lines[i]["end"])

    @patch("lyrics_aligner.transcribe_chunk_with_gemini")
    def test_mohsen_yeganeh_cross_script_and_prompt_echo_prevention(self, mock_gemini):
        # 1. System prompts must NEVER contain hardcoded real song lyrics like Tata Simonyan or Amr Diab
        self.assertNotIn("Դու անուշիկ", AUDIO_TIMING_SYSTEM_PROMPT)
        self.assertNotIn("حبيبي يا نور العين", PHONETIC_SYSTEM_PROMPT)

        with tempfile.TemporaryDirectory() as tmp:
            # 3 chunks separated by >6s gaps:
            # Chunk 0 (1.0s-3.5s): Live concert crowd noise where Chunk 0 falsely returns Armenian 'hy'
            # Chunk 1 (12.0s-24.0s): Real Persian singing ('Behet Ghol Midam')
            # Chunk 2 (32.0s-38.0s): Guitar outro vocalization loop
            voc_wav = generate_test_wav(
                os.path.join(tmp, "vocals.wav"),
                duration_sec=40.0,
                vocal_windows=[(1.0, 3.5), (12.0, 24.0), (32.0, 38.0)],
            )

            mock_gemini.side_effect = [
                {
                    "language": "hy",
                    "lines": [
                        {"start": 0.2, "end": 1.8, "text": "Դու անուշիկ աղջիկ,"},
                    ],
                },
                {
                    "language": "fa",
                    "lines": [
                        {"start": 0.3, "end": 2.4, "text": "خودت می‌خوای بری خاطره شی"},
                        {"start": 2.6, "end": 5.0, "text": "بهت قول میدم سخت نیست"},
                        {"start": 5.3, "end": 8.2, "text": "لااقل برای تو راحت باش"},
                        {"start": 8.5, "end": 11.4, "text": "دورم از تو و دنیای تو"},
                    ],
                },
                {
                    "language": "fa",
                    "lines": [
                        {"start": 0.1, "end": 1.0, "text": "آه ای دل"},
                        {"start": 1.1, "end": 2.0, "text": "آه ای دل"},
                        {"start": 2.1, "end": 3.0, "text": "آه ای دل"},
                        {"start": 3.1, "end": 4.0, "text": "آه ای دل"},
                        {"start": 4.1, "end": 5.0, "text": "آه ای دل"},
                    ],
                },
            ]

            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                res = transcribe_and_align_vocals(
                    voc_wav,
                    language="auto",
                    duration=40.0,
                    song_title="Mohsen Yeganeh - Behet Ghol Midam",
                    artist="Mohsen Yeganeh",
                )

            # Verify Chunk 1 and Chunk 2 were NOT poisoned with language_hint='hy' from Chunk 0!
            for call_args in mock_gemini.call_args_list:
                self.assertEqual(call_args.kwargs["language_hint"], "auto")
                self.assertIn("Mohsen Yeganeh", call_args.kwargs["song_title"])

            # Dominant language must resolve to 'fa' (Persian), NOT 'hy'!
            self.assertEqual(res["language"], "fa")
            texts = [ln["text"] for ln in res["lines"]]
            self.assertEqual(len(texts), 4)
            self.assertTrue(any("بهت قول میدم" in t for t in texts))
            for t in texts:
                self.assertNotIn("Դու", t)
                self.assertNotIn("آه ای دل", t)


class TestStep5PhoneticsAndSubtitles(unittest.TestCase):
    """Step 5: Verify strict ASCII phonetics, timestamp preservation, and gap-accurate ASS sweeps."""

    def test_strict_ascii_phonetics_and_timestamp_preservation(self):
        dirty = "Ḥabībī 3al-7ob šams əl-8arām"
        clean = sanitize_strict_ascii_phonetic(dirty)
        self.assertRegex(clean, r"^[A-Za-z'\- ,.!?]+$")
        self.assertNotRegex(clean, r"\d")

        input_lines = [
            {
                "line_id": 0,
                "text": "Դու անուշիկ աղջիկ ես",
                "start": 13.10,
                "end": 16.40,
                "words": [
                    {"word": "Դու", "start": 13.10, "end": 13.70},
                    {"word": "անուշիկ", "start": 13.90, "end": 14.90},
                    {"word": "աղջիկ", "start": 15.00, "end": 15.80},
                    {"word": "ես", "start": 15.85, "end": 16.40},
                ],
            }
        ]
        out = offline_phonetic_transliterator(input_lines)
        self.assertEqual(out[0]["start"], 13.10)
        self.assertEqual(out[0]["end"], 16.40)
        for in_w, out_w in zip(input_lines[0]["words"], out[0]["words"]):
            self.assertEqual(in_w["start"], out_w["start"])
            self.assertEqual(in_w["end"], out_w["end"])
            self.assertRegex(out_w["phonetic"], r"^[A-Za-z'\-]+$")

    def test_three_tier_ass_and_gap_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = {"title": "Sirum em Qez", "artist": "Tata Simonyan", "language": "hy", "duration": 30.0}
            lines = [
                {
                    "line_id": 0,
                    "start": 10.00,
                    "end": 14.00,
                    "native": "Սիրում եմ քեզ, սիրում եմ քեզ",
                    "phonetic": "Seeroom em kez, seeroom em kez",
                    "translation": "I love you, I love you",
                    "words": [
                        {"native": "Սիրում եմ քեզ,", "phonetic": "Seeroom em kez,", "start": 10.00, "end": 11.20},
                        # Intentional 1.30s breath pause (11.20 -> 12.50 = 130 cs)
                        {"native": "սիրում եմ քեզ", "phonetic": "seeroom em kez", "start": 12.50, "end": 14.00},
                    ],
                }
            ]
            json_p = os.path.join(tmp, "lyrics.json")
            lrc_p = os.path.join(tmp, "lyrics.lrc")
            ass_p = os.path.join(tmp, "karaoke.ass")

            export_lyrics_json(meta, lines, json_p)
            lrc_str = export_lrc(lines, lrc_p)
            ass_str = export_ass_karaoke(meta, lines, ass_p)

            self.assertTrue(os.path.exists(json_p))
            self.assertIn("[00:10.00]", lrc_str)
            self.assertIn("Style: KaraokePhonetic, Montserrat, 62", ass_str)
            self.assertIn("Style: KaraokeNative, Arial, 56", ass_str)
            self.assertIn("KaraokeTranslation", ass_str)
            # Must contain {\k130} pause hold between word 1 and word 2!
            self.assertIn("{\\k130}", ass_str)


class TestStep6VideoRenderAndFullPipeline(unittest.TestCase):
    """Step 6 & End-to-End: Verify 1080p burned-in MP4 uses clean vocal-free instrumental."""

    @patch("lyrics_search.search_lyricstranslate", return_value=[])
    @patch("lyrics_aligner.transcribe_chunk_with_gemini")
    def test_full_pipeline_times_on_vocals_and_renders_vocal_free_mp4(self, mock_gemini, _mock_lt):
        mock_gemini.return_value = {
            "language": "hy",
            "lines": [
                {"start": 0.20, "end": 2.10, "text": "Դու անուշիկ աղջիկ ես"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            # Create 3.5s test song with:
            # - Loud center-panned 440Hz "singer vocal" at 0.5s - 2.5s
            # - Continuous stereo out-of-phase 220Hz "instrumental music" from 0.0s - 3.5s
            song_wav = generate_test_wav(
                os.path.join(tmp, "Tata_Test_-_Sirum_Test.wav"),
                duration_sec=3.5,
                vocal_windows=[(0.5, 2.5)],
                include_stereo_music=True,
            )
            out_dir = os.path.join(tmp, "run_out")
            log_dir = os.path.join(tmp, "global_logs")

            # Run full 6-step pipeline offline-safe (mocked Gemini audio + local stem fallback + real FFmpeg 1080p render)
            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
                with patch("stem_separator.prefer_gpu_default", False, create=True):
                    # Force local deterministic phase-cancellation stem separator for fast CI speed
                    with patch(
                        "pipeline.separate_stems",
                        side_effect=lambda wav, od: separate_stems(wav, od, prefer_gpu=False),
                    ):
                        with patch("phonetic_transliterator.transliterate_with_gemini", side_effect=RuntimeError("offline test")):
                            result = run_karaoke_pipeline(
                                source=song_wav,
                                output_dir=out_dir,
                                interactive=False,
                                render_video=True,
                                language_hint="auto",
                                global_log_dir=log_dir,
                            )

            mp4_path = result["karaoke_video_path"]
            self.assertIsNotNone(mp4_path)
            self.assertTrue(os.path.exists(mp4_path))
            self.assertGreater(os.path.getsize(mp4_path), 10000)

            # Verify per-song logs, telemetry, run_status.json, and global ML runs_history.jsonl
            self.assertTrue(os.path.exists(result["pipeline_log_path"]))
            self.assertTrue(os.path.exists(result["telemetry_path"]))
            self.assertTrue(os.path.exists(result["run_status_path"]))
            self.assertTrue(os.path.exists(result["global_history_path"]))

            with open(result["run_status_path"], encoding="utf-8") as f:
                status_data = json.load(f)
            self.assertEqual(status_data["status"], "completed")
            for step_key in (
                "step1_ingest",
                "step2_stem_separation",
                "step3_lyrics_search",
                "step4_vocal_timing",
                "step5_phonetics_and_subtitles",
                "step6_video_render",
            ):
                self.assertIn(step_key, status_data["stage_durations_sec"])
            self.assertEqual(status_data["ml_features"]["ascii_phonetic_compliance_rate"], 1.0)
            self.assertEqual(status_data["ml_features"]["overlap_violations_count"], 0)

            # Verify lyrics.json captured the vocal-timed Armenian line
            with open(result["lyrics_json_path"], encoding="utf-8") as f:
                ldata = json.load(f)
            self.assertEqual(ldata["metadata"]["language"], "hy")
            self.assertEqual(len(ldata["lines"]), 1)
            self.assertIn("անուշիկ", ldata["lines"][0]["native"])

            # Verify MP4 resolution (1920x1080) and verify singer's center 440Hz vocal is GONE from MP4 audio!
            ffmpeg_bin = get_ffmpeg_exe()
            extracted_wav = os.path.join(tmp, "mp4_audio_check.wav")
            subprocess.run(
                [ffmpeg_bin, "-y", "-i", mp4_path, "-vn", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", extracted_wav],
                check=True,
                capture_output=True,
            )

            # Sum L+R channels of MP4 audio: since instrumental is L=-R, any residual center vocal (L=+R) would appear in (L+R)/2!
            with wave.open(extracted_wav, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            samples = struct.unpack(f"<{len(raw) // 2}h", raw)
            mono_sum_sq = 0.0
            stereo_diff_sq = 0.0
            n_pairs = len(samples) // 2
            for idx in range(0, len(samples) - 1, 2):
                l_val = samples[idx] / 32768.0
                r_val = samples[idx + 1] / 32768.0
                mono_sum_sq += ((l_val + r_val) * 0.5) ** 2
                stereo_diff_sq += ((l_val - r_val) * 0.5) ** 2

            center_vocal_rms = math.sqrt(mono_sum_sq / max(1, n_pairs))
            instrumental_rms = math.sqrt(stereo_diff_sq / max(1, n_pairs))
            self.assertLess(center_vocal_rms, 0.01, "Final MP4 audio still contains singer center vocals!")
            self.assertGreater(instrumental_rms, 0.10, "Final MP4 audio is missing the instrumental backing music!")


class TestStep7TelemetryCrashAndFreezeDetection(unittest.TestCase):
    """Step 7: Verify error logging, watchdog freeze stack capture, dead-PID crash recovery & ML dataset."""

    def test_error_logging_captures_failed_step_and_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            song_dir = os.path.join(tmp, "out", "song_err")
            log_dir = os.path.join(tmp, "logs")

            with self.assertRaisesRegex(RuntimeError, "Simulated Gemini quota error"):
                with SongRunLogger(
                    source="https://youtu.be/test_err",
                    output_dir=song_dir,
                    global_log_dir=log_dir,
                    capture_stdio=False,
                ) as logger:
                    logger.record_metadata({"title": "Error Song", "artist": "Tester", "duration": 120.0})
                    with logger.step("step1_ingest"):
                        pass
                    with logger.step("step4_vocal_timing"):
                        raise RuntimeError("Simulated Gemini quota error")

            with open(os.path.join(song_dir, "run_status.json"), encoding="utf-8") as f:
                st = json.load(f)
            self.assertEqual(st["status"], "failed")
            self.assertEqual(st["error"]["failed_step"], "step4_vocal_timing")
            self.assertEqual(st["error"]["type"], "RuntimeError")

            with open(os.path.join(song_dir, "pipeline.log"), encoding="utf-8") as f:
                log_txt = f.read()
            self.assertIn("PIPELINE FAILED", log_txt)
            self.assertIn("Simulated Gemini quota error", log_txt)

            with open(os.path.join(log_dir, "runs_history.jsonl"), encoding="utf-8") as f:
                hist = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(hist), 1)
            self.assertEqual(hist[0]["status"], "failed")
            self.assertEqual(hist[0]["failed_step"], "step4_vocal_timing")

    def test_watchdog_freeze_detection_captures_live_main_thread_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            song_dir = os.path.join(tmp, "out", "song_hang")
            log_dir = os.path.join(tmp, "logs")

            with SongRunLogger(
                source="https://youtu.be/test_hang",
                output_dir=song_dir,
                global_log_dir=log_dir,
                heartbeat_interval_sec=0.04,
                capture_stdio=False,
            ) as logger:
                with logger.step("step2_stem_separation", stall_timeout_sec=0.10):
                    time.sleep(0.28)

            self.assertGreaterEqual(len(logger.stall_events), 1)
            self.assertEqual(logger.stall_events[0]["step"], "step2_stem_separation")
            self.assertIn(
                "test_watchdog_freeze_detection_captures_live_main_thread_stack",
                logger.stall_events[0]["main_thread_stack"],
            )

            with open(os.path.join(song_dir, "telemetry.jsonl"), encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            stall_evts = [e for e in events if e.get("event") == "stall_warning"]
            self.assertGreaterEqual(len(stall_evts), 1)

            with open(os.path.join(song_dir, "pipeline.log"), encoding="utf-8") as f:
                log_txt = f.read()
            self.assertIn("[WATCHDOG FREEZE ALERT]", log_txt)

    def test_dead_pid_hard_crash_recovery_and_health_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_root = os.path.join(tmp, "out")
            crashed_dir = os.path.join(out_root, "yt_crashed_song")
            log_dir = os.path.join(tmp, "logs")
            os.makedirs(crashed_dir, exist_ok=True)

            # Simulate a process that was SIGKILLed / OOM-crashed mid-run leaving status='running'
            stale_status = {
                "run_id": "run_simulated_crash",
                "status": "running",
                "pid": 99999999,
                "source": "https://youtu.be/crashed123",
                "current_step": "step2_stem_separation",
                "started_epoch": time.time() - 95.0,
                "last_heartbeat_epoch": time.time() - 80.0,
                "last_heartbeat_at": "2026-01-01T00:00:00.000Z",
                "stage_durations_sec": {"step1_ingest": 2.4},
                "metadata": {"title": "Crashed Track", "language": "hy"},
                "ml_features": {},
            }
            with open(os.path.join(crashed_dir, "run_status.json"), "w", encoding="utf-8") as f:
                json.dump(stale_status, f)

            report = scan_all_song_runs(out_root, global_log_dir=log_dir)
            self.assertEqual(report["crashed_runs"], 1)
            self.assertEqual(len(report["songs"]), 1)
            self.assertEqual(report["songs"][0]["diagnosis"], "crashed")

            with open(os.path.join(crashed_dir, "run_status.json"), encoding="utf-8") as f:
                recovered = json.load(f)
            self.assertEqual(recovered["status"], "crashed")
            self.assertEqual(recovered["error"]["type"], "UnexpectedProcessTermination")

    def test_frozen_live_pid_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen_dir = os.path.join(tmp, "yt_frozen_song")
            log_dir = os.path.join(tmp, "logs")
            os.makedirs(frozen_dir, exist_ok=True)

            # Use parent PID (guaranteed alive and != os.getpid()) with a stale heartbeat 120s old
            live_other_pid = os.getppid()
            frozen_status = {
                "run_id": "run_simulated_freeze",
                "status": "running",
                "pid": live_other_pid,
                "source": "https://youtu.be/frozen123",
                "current_step": "step6_video_render",
                "started_epoch": time.time() - 300.0,
                "last_heartbeat_epoch": time.time() - 120.0,
            }
            with open(os.path.join(frozen_dir, "run_status.json"), "w", encoding="utf-8") as f:
                json.dump(frozen_status, f)

            diag = detect_and_recover_stale_run(frozen_dir, global_log_dir=log_dir, freeze_threshold_sec=45.0)
            self.assertIsNotNone(diag)
            self.assertEqual(diag["diagnosis"], "frozen")
            self.assertGreaterEqual(diag["heartbeat_age_sec"], 100.0)


class TestStep8WebStudioAndAPI(unittest.TestCase):
    """Step 8: Verify v0.1 Web Studio UI, PWA manifest, REST API job submission, and HTTP Range streaming."""

    @patch("lyrics_search.search_lyricstranslate", return_value=[])
    @patch("lyrics_aligner.transcribe_chunk_with_gemini")
    def test_web_studio_ui_and_end_to_end_job_api(self, mock_gemini, _mock_lt):
        from fastapi.testclient import TestClient

        mock_gemini.return_value = {
            "language": "fa",
            "lines": [
                {"start": 0.25, "end": 2.20, "text": "بهت قول میدم"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            log_dir = os.path.join(tmp, "logs")
            upload_dir = os.path.join(tmp, "uploads")
            test_wav = generate_test_wav(
                os.path.join(tmp, "Mohsen_Sample.wav"),
                duration_sec=3.2,
                vocal_windows=[(0.5, 2.5)],
                include_stereo_music=True,
            )

            test_app = create_app(out_dir=out_dir, log_dir=log_dir, upload_dir=upload_dir)
            client = TestClient(test_app)

            # 1. Verify HTML UI, PWA manifest, and health endpoint
            r_home = client.get("/")
            self.assertEqual(r_home.status_code, 200)
            self.assertIn("Karaoke Agent Studio", r_home.text)

            r_manifest = client.get("/manifest.webmanifest")
            self.assertEqual(r_manifest.status_code, 200)
            self.assertEqual(r_manifest.json()["display"], "standalone")

            r_health = client.get("/api/health")
            self.assertEqual(r_health.status_code, 200)
            self.assertEqual(r_health.json()["status"], "ok")

            # 2. Submit a real file-upload conversion job and poll to completion
            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
                with patch(
                    "pipeline.separate_stems",
                    side_effect=lambda wav, od: separate_stems(wav, od, prefer_gpu=False),
                ):
                    with patch("phonetic_transliterator.transliterate_with_gemini", side_effect=RuntimeError("offline")):
                        with open(test_wav, "rb") as wf:
                            r_post = client.post(
                                "/api/jobs",
                                data={"source_type": "file", "language": "auto", "lyrics_mode": "auto"},
                                files={"audio_file": ("Mohsen_Sample.wav", wf, "audio/wav")},
                            )
                        self.assertEqual(r_post.status_code, 202)
                        job_id = r_post.json()["job_id"]

                        # Poll job status until completed (max 15s)
                        deadline = time.time() + 15.0
                        job_state = None
                        while time.time() < deadline:
                            job_state = client.get(f"/api/jobs/{job_id}").json()
                            if job_state["status"] in ("completed", "failed"):
                                break
                            time.sleep(0.2)

            self.assertIsNotNone(job_state)
            self.assertEqual(job_state["status"], "completed", f"Web job failed: {job_state}")
            self.assertEqual(job_state["progress_percent"], 100)

            song_folder = job_state["song_folder"]
            r_song = client.get(f"/api/songs/{song_folder}")
            self.assertEqual(r_song.status_code, 200)
            song_detail = r_song.json()
            self.assertEqual(song_detail["language"], "fa")
            self.assertEqual(len(song_detail["lines"]), 1)

            # 3. Verify HTTP Range request support on MP4 (required by macOS/iOS Safari video player)
            mp4_url = song_detail["urls"]["video_mp4"]
            self.assertIsNotNone(mp4_url)
            r_range = client.get(mp4_url, headers={"Range": "bytes=0-511"})
            self.assertEqual(r_range.status_code, 206)
            self.assertEqual(len(r_range.content), 512)


if __name__ == "__main__":
    unittest.main(verbosity=2)
