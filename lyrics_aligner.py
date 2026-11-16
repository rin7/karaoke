"""
lyrics_aligner.py - Audio-First Vocal Lyrics Extraction & Precision Timestamping
Listens directly to the isolated singer vocal stem (vocals.wav) BEFORE vocals are
discarded for the final video:
1. Detects active singing intervals via Voice Activity Detection (VAD), skipping
   pure instrumental intros and guitar/violin solos.
2. Groups active vocal intervals into drift-free <36s audio clips anchored to exact
   master video timestamps.
3. Uses Multimodal Gemini 2.5 Flash Audio to transcribe & time every sung phrase
   directly from the singer's voice (optionally guided by reference lyrics for spelling).
4. Enforces strictly monotonic, non-overlapping karaoke line & word timestamps.
"""

import os
import re
import json
import time
import shutil
import subprocess
from typing import List, Dict, Any, Optional, Tuple


def get_ffmpeg_exe() -> str:
    """Resolve an available ffmpeg binary (system PATH or imageio_ffmpeg)."""
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return "ffmpeg"


def probe_audio_duration(audio_path: str, fallback: float = 0.0) -> float:
    """Read exact audio duration in seconds using ffprobe or ffmpeg stderr."""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        proc = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
            capture_output=True, text=True
        )
        if proc.returncode == 0:
            try:
                data = json.loads(proc.stdout)
                return float(data.get("format", {}).get("duration", fallback))
            except Exception:
                pass

    ffmpeg_bin = get_ffmpeg_exe()
    proc = subprocess.run([ffmpeg_bin, "-i", audio_path], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", proc.stderr)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return fallback


def clean_lines(lyrics_text: str) -> List[str]:
    """Clean and filter non-lyric structural markers like [Chorus], [Verse], etc."""
    cleaned = []
    for line in lyrics_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        if re.match(r"^\((?:chorus|verse|intro|outro|bridge|solo|x\d+|\d+x).*?\)$", line, flags=re.IGNORECASE):
            continue
        cleaned.append(line)
    return cleaned


def _run_silencedetect(
    vocal_path: str,
    total_duration: float,
    noise_db: str = "-32dB",
    min_silence: float = 0.35,
    min_vocal_dur: float = 0.45
) -> List[List[float]]:
    """Run ffmpeg silencedetect and parse active (non-silent) vocal intervals."""
    ffmpeg_bin = get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin, "-y", "-i", vocal_path,
        "-af", f"silencedetect=noise={noise_db}:d={min_silence}",
        "-f", "null", "-"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    events: List[Tuple[str, float]] = []
    for line in proc.stderr.splitlines():
        m_start = re.search(r"silence_start:\s*([\d.]+)", line)
        if m_start:
            events.append(("start", float(m_start.group(1))))
        m_end = re.search(r"silence_end:\s*([\d.]+)", line)
        if m_end:
            events.append(("end", float(m_end.group(1))))

    intervals: List[List[float]] = []
    cur = 0.0
    in_silence = False

    for kind, ts in events:
        if kind == "start":
            if not in_silence and ts - cur >= min_vocal_dur:
                intervals.append([round(cur, 2), round(ts, 2)])
            in_silence = True
        elif kind == "end":
            cur = ts
            in_silence = False

    if not in_silence and total_duration > cur and (total_duration - cur) >= min_vocal_dur:
        intervals.append([round(cur, 2), round(total_duration, 2)])

    return intervals


def extract_vocal_intervals_vad(
    vocal_path: str,
    total_duration: Optional[float] = None,
    noise_db: str = "-32dB",
    min_silence: float = 0.35,
    max_single_interval: float = 34.0
) -> List[List[float]]:
    """
    Extract active vocal time intervals where the singer is vocalizing on vocals.wav,
    preserving early vocals (<5s) and end-of-song vocals while filtering out silent
    instrumental solos and breaks. Splits overly long continuous intervals cleanly.
    """
    dur = total_duration if (total_duration and total_duration > 0) else probe_audio_duration(vocal_path, 0.0)
    raw_intervals = _run_silencedetect(vocal_path, dur, noise_db=noise_db, min_silence=min_silence)

    # Ensure no single interval exceeds max_single_interval so chunks stay < 40s
    refined: List[List[float]] = []
    for s, e in raw_intervals:
        span = e - s
        if span <= max_single_interval:
            refined.append([s, e])
        else:
            # Split into balanced sub-segments <= max_single_interval
            parts = int((span // max_single_interval) + 1)
            step = span / parts
            for k in range(parts):
                sub_s = round(s + k * step, 2)
                sub_e = round(s + (k + 1) * step, 2) if k < parts - 1 else e
                refined.append([sub_s, sub_e])

    return refined


def group_intervals_into_chunks(
    intervals: List[List[float]],
    total_duration: float,
    max_chunk_sec: float = 36.0,
    solo_gap_sec: float = 5.5,
    pad_sec: float = 0.25
) -> List[Dict[str, Any]]:
    """
    Group consecutive active vocal intervals into drift-free audio chunks (<40s each).
    Breaks chunks across long instrumental solos (>= solo_gap_sec) so silent solos
    are never transcribed and timestamps never cross 60s or drift over long songs.
    """
    if not intervals:
        return []

    groups: List[List[List[float]]] = []
    current_grp: List[List[float]] = [intervals[0]]

    for iv in intervals[1:]:
        prev_end = current_grp[-1][1]
        gap = iv[0] - prev_end
        prospective_span = (iv[1] - current_grp[0][0]) + 2 * pad_sec

        if gap >= solo_gap_sec or prospective_span > max_chunk_sec:
            groups.append(current_grp)
            current_grp = [iv]
        else:
            current_grp.append(iv)

    if current_grp:
        groups.append(current_grp)

    chunks: List[Dict[str, Any]] = []
    for idx, grp in enumerate(groups):
        c_start = max(0.0, round(grp[0][0] - pad_sec, 2))
        c_end = round(grp[-1][1] + pad_sec, 2)
        if total_duration > 0:
            c_end = min(round(total_duration, 2), c_end)
        # Guarantee chunk duration stays strictly under 55s to avoid MM:SS ambiguity
        if c_end - c_start > 50.0:
            c_end = round(c_start + 50.0, 2)

        sub_ivs = [
            [round(max(0.0, s - c_start), 2), round(min(c_end - c_start, e - c_start), 2)]
            for s, e in grp
        ]
        chunks.append({
            "chunk_id": idx,
            "start": c_start,
            "end": c_end,
            "duration": round(c_end - c_start, 2),
            "sub_intervals": sub_ivs
        })

    return chunks


def extract_audio_chunk_bytes(audio_path: str, start_sec: float, end_sec: float) -> Tuple[bytes, str]:
    """
    Slice [start_sec, end_sec] from audio_path into compact mono MP3 bytes in memory
    (falling back to 16kHz mono WAV if libmp3lame is unavailable).
    """
    ffmpeg_bin = get_ffmpeg_exe()
    dur = max(0.2, round(end_sec - start_sec, 3))

    cmd_mp3 = [
        ffmpeg_bin, "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{dur:.3f}",
        "-i", audio_path,
        "-vn", "-ac", "1", "-ar", "24000",
        "-c:a", "libmp3lame", "-b:a", "96k",
        "-f", "mp3", "pipe:1"
    ]
    proc = subprocess.run(cmd_mp3, capture_output=True)
    if proc.returncode == 0 and len(proc.stdout) > 256:
        return proc.stdout, "audio/mp3"

    cmd_wav = [
        ffmpeg_bin, "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{dur:.3f}",
        "-i", audio_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "wav", "pipe:1"
    ]
    proc_wav = subprocess.run(cmd_wav, capture_output=True, check=True)
    return proc_wav.stdout, "audio/wav"


AUDIO_TIMING_SYSTEM_PROMPT = """You are an expert multilingual music transcription and forced-alignment engineer for Armenian (Հայերեն), Persian (فارسی), and Arabic (العربية) songs.

You will receive a short isolated vocal audio clip along with its exact duration in seconds and detected acoustic vocal activity windows.
Your task is to listen directly to the singing and return ONLY the intelligible words actually sung in THIS clip with precise start and end timestamps (in floating-point seconds relative to the start of THIS clip, from 0.00 up to clip_duration).

STRICT RULES:
1. AUDIO IS THE SOLE SOURCE OF TRUTH: Transcribe ONLY real, intelligible sung words that you genuinely hear in THIS specific audio clip. NEVER invent, guess, or copy placeholder text.
2. NATIVE SCRIPT: Write sung lyrics in the authentic native script of the language actually sung in the clip:
   - Persian (`fa`): Persian script (فارسی)
   - Arabic (`ar`): Arabic script (العربية)
   - Armenian (`hy`): Armenian script (Հայերեն)
3. INCLUDE REAL SUNG REPETITIONS: When a lyric verse or chorus phrase is genuinely sung multiple times, output a separate timed entry for each sung occurrence.
4. CONCERT, CROWD & WORDLESS VOCALIZATION GUARD (CRITICAL):
   - Live concert crowd cheering, applause, whistling, stage noise, guitar/violin/synth bleed, breaths, and wordless vocalizations ("aaah", "oooh", "la la", "na na", "hmm", "akh" sighs/moans) are NOT lyrics.
   - If a clip (or sub-window) contains ONLY instruments, crowd noise, humming, or wordless vocalizing with no real words, you MUST return `{"language": "none", "lines": []}`.
5. KARAOKE PHRASE SIZING & SINGLE TIMELINE:
   - Keep each line karaoke-sized (typically 3 to 8 words, lasting 1.2 to 5.5 seconds).
   - Start and end timestamps of consecutive lines MUST NOT overlap (line[i].end <= line[i+1].start).
   - Follow the primary lead vocal; include backing vocal call-and-response hooks only when sung in gaps between lead lines.
6. REFERENCE LYRICS & METADATA (IF PROVIDED):
   - Track title/artist metadata is provided ONLY to help disambiguate dialect/language when singing is present. NEVER generate lyrics from the title if no words are sung in the clip.
   - Use reference lyrics ONLY as a spelling aid for words genuinely heard in THIS clip.

When no intelligible words are sung in the clip, return:
{"language": "none", "lines": []}
"""


def transcribe_chunk_with_gemini(
    audio_bytes: bytes,
    mime_type: str,
    clip_duration: float,
    sub_intervals: List[List[float]],
    language_hint: str = "auto",
    reference_lyrics: Optional[str] = None,
    song_title: str = "",
    artist: str = ""
) -> Dict[str, Any]:
    """Send a short vocal clip to Gemini 2.5 Flash Multimodal Audio for transcription + timing."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for multimodal vocal timing.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    windows_str = ", ".join(f"[{s:.2f}s - {e:.2f}s]" for s, e in sub_intervals)
    prompt_parts = [
        f"Clip duration: {clip_duration:.2f} seconds (valid timestamp range: 0.00 to {clip_duration:.2f}).",
        f"Language hint: {language_hint}.",
        f"Detected acoustic activity windows inside this clip: {windows_str}."
    ]
    if song_title or (artist and artist != "Unknown Artist"):
        prompt_parts.append(
            f"Track metadata context (for language/dialect identification ONLY — return lines=[] if this clip is only instruments, crowd noise, or wordless humming): "
            f"Title='{song_title}', Artist='{artist}'."
        )
    if reference_lyrics and reference_lyrics.strip():
        prompt_parts.append(
            "Optional reference lyrics for the full song (ONLY use for spelling if heard in THIS clip):\n"
            + reference_lyrics.strip()[:3000]
        )
    prompt_parts.append(
        "Listen carefully to this vocal clip. If intelligible lyrics are sung, return their exact native-script text and clip-relative timestamps. "
        "If only crowd noise, instruments, or wordless vocalizations ('ah'/'oh') are heard, return an empty lines list."
    )

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    text_part = "\n".join(prompt_parts)

    response_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["language", "lines"],
        properties={
            "language": types.Schema(
                type=types.Type.STRING,
                description="ISO language code of sung words in this clip ('fa', 'ar', 'hy'), or 'none' if no words are sung.",
            ),
            "lines": types.Schema(
                type=types.Type.ARRAY,
                description="Ordered list of non-overlapping sung lyric phrases heard in this clip, or empty list if none.",
                items=types.Schema(
                    type=types.Type.OBJECT,
                    required=["start", "end", "text"],
                    properties={
                        "start": types.Schema(
                            type=types.Type.NUMBER,
                            description="Phrase start timestamp in seconds relative to clip start (0.00 to clip_duration).",
                        ),
                        "end": types.Schema(
                            type=types.Type.NUMBER,
                            description="Phrase end timestamp in seconds relative to clip start (must be > start and <= clip_duration).",
                        ),
                        "text": types.Schema(
                            type=types.Type.STRING,
                            description="Exact words sung in authentic native script.",
                        ),
                    },
                ),
            ),
        },
    )

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[audio_part, text_part],
                config=types.GenerateContentConfig(
                    system_instruction=AUDIO_TIMING_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.05
                )
            )
            parsed = json.loads(resp.text)
            if isinstance(parsed, dict) and "lines" in parsed:
                return parsed
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Gemini audio timing failed after retries: {last_err}")


def build_words_for_line(text: str, line_start: float, line_end: float) -> List[Dict[str, Any]]:
    """
    Distribute word timestamps proportionally within [line_start, line_end],
    giving punctuation (commas, pauses) natural breathing weight so karaoke
    color-sweeps track the singer's phrasing cleanly.
    """
    words = text.split()
    if not words:
        return []

    line_dur = max(0.25, line_end - line_start)
    weights = []
    for w in words:
        char_weight = max(1.0, float(len(re.sub(r"[^\w]", "", w))))
        if re.search(r"[,،՝;։:.!?—\-]$", w):
            char_weight += 1.4
        weights.append(char_weight)

    total_weight = sum(weights) or 1.0
    result_words = []
    cur_t = line_start

    for idx, (w, wt) in enumerate(zip(words, weights)):
        slot_dur = line_dur * (wt / total_weight)
        has_pause = bool(re.search(r"[,،՝;։:.!?]$", w)) and idx < len(words) - 1
        active_ratio = 0.85 if has_pause else 0.96
        w_start = round(cur_t, 2)
        w_end = round(min(line_end, cur_t + max(0.05, slot_dur * active_ratio)), 2)
        if w_end < w_start:
            w_end = w_start
        result_words.append({
            "word": w,
            "start": w_start,
            "end": w_end
        })
        cur_t += slot_dur

    # Ensure last word never exceeds line_end
    if result_words:
        result_words[-1]["end"] = round(min(line_end, max(result_words[-1]["start"], result_words[-1]["end"])), 2)

    return result_words


def normalize_and_build_words(
    raw_lines: List[Dict[str, Any]],
    total_duration: float,
    lead_in: float = 0.10,
    min_gap: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Clean, sort, clamp, deduplicate simultaneous overlaps, enforce strict
    non-overlapping display windows, and compute word-level karaoke timings.
    """
    cleaned: List[Dict[str, Any]] = []
    max_dur = total_duration if total_duration > 0 else 3600.0

    for item in raw_lines:
        txt = str(item.get("text", "")).strip()
        if not txt or (txt.startswith("[") and txt.endswith("]")):
            continue
        try:
            s = float(item.get("start", 0.0))
            e = float(item.get("end", s + 2.0))
        except (TypeError, ValueError):
            continue

        s = max(0.0, min(max_dur - 0.2, s - lead_in))
        e = max(s + 0.35, min(max_dur, e))
        cleaned.append({"text": txt, "start": s, "end": e})

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    # Filter out simultaneous overlapping harmony duplicates that start at virtually the same instant
    deduped: List[Dict[str, Any]] = []
    for item in cleaned:
        if deduped:
            prev = deduped[-1]
            overlap = min(prev["end"], item["end"]) - max(prev["start"], item["start"])
            min_len = max(0.1, min(prev["end"] - prev["start"], item["end"] - item["start"]))
            if (item["start"] - prev["start"]) < 0.40 and (overlap / min_len) > 0.60:
                # Keep the richer/longer lyric phrase
                if len(item["text"]) > len(prev["text"]):
                    deduped[-1] = item
                continue
        deduped.append(item)

    # Enforce strict non-overlapping windows: line[i].end <= line[i+1].start - min_gap
    for i in range(len(deduped) - 1):
        nxt_start = deduped[i + 1]["start"]
        if deduped[i]["end"] > nxt_start - min_gap:
            new_end = round(nxt_start - min_gap, 2)
            if new_end >= deduped[i]["start"] + 0.25:
                deduped[i]["end"] = new_end
            else:
                # Push next line start slightly forward so both stay readable
                deduped[i]["end"] = round(deduped[i]["start"] + 0.25, 2)
                deduped[i + 1]["start"] = round(deduped[i]["end"] + min_gap, 2)
                if deduped[i + 1]["end"] < deduped[i + 1]["start"] + 0.25:
                    deduped[i + 1]["end"] = round(min(max_dur, deduped[i + 1]["start"] + 0.35), 2)

    final_lines: List[Dict[str, Any]] = []
    for idx, item in enumerate(deduped):
        l_start = round(item["start"], 2)
        l_end = round(max(l_start + 0.25, min(max_dur, item["end"])), 2)
        words = build_words_for_line(item["text"], l_start, l_end)
        final_lines.append({
            "line_id": idx,
            "text": item["text"],
            "start": l_start,
            "end": l_end,
            "words": words
        })

    return final_lines


def align_lines_to_vocal_intervals(
    lines: List[str],
    intervals: List[List[float]],
    total_duration: float = 0.0,
    lead_in: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Offline deterministic fallback when API/ASR is unavailable and manual lyrics exist:
    maps provided lines onto detected VAD intervals without overlapping windows.
    """
    if not lines or not intervals:
        return []

    N = len(lines)
    curr = [list(x) for x in intervals]

    while len(curr) < N and curr:
        longest_idx = max(range(len(curr)), key=lambda i: curr[i][1] - curr[i][0])
        s, e = curr[longest_idx]
        mid = round((s + e) / 2.0, 2)
        curr[longest_idx] = [s, mid]
        curr.insert(longest_idx + 1, [mid, e])

    while len(curr) > N:
        smallest_gap_idx = min(range(len(curr) - 1), key=lambda i: curr[i + 1][0] - curr[i][1])
        curr[smallest_gap_idx][1] = curr[smallest_gap_idx + 1][1]
        curr.pop(smallest_gap_idx + 1)

    dur = total_duration if total_duration > 0 else (curr[-1][1] + 5.0)
    raw = [
        {"text": line_text, "start": curr[i][0], "end": curr[i][1]}
        for i, line_text in enumerate(lines)
    ]
    return normalize_and_build_words(raw, total_duration=dur, lead_in=lead_in)


def align_fallback_heuristic(lyrics_text: str, total_duration: float) -> List[Dict[str, Any]]:
    """Last-resort mathematical fallback if neither audio VAD nor AI ASR is available."""
    lines = clean_lines(lyrics_text)
    if not lines:
        return []

    intro = min(6.0, total_duration * 0.05)
    usable_time = max(10.0, total_duration - intro - 4.0)
    total_words = sum(len(l.split()) for l in lines) or 1
    sec_per_word = usable_time / total_words

    current_t = intro
    raw = []
    for line_text in lines:
        words = line_text.split()
        if not words:
            continue
        l_dur = max(1.0, len(words) * sec_per_word * 0.9)
        raw.append({"text": line_text, "start": current_t, "end": current_t + l_dur})
        current_t += max(l_dur + 0.2, len(words) * sec_per_word)

    return normalize_and_build_words(raw, total_duration=total_duration, lead_in=0.0)


def detect_line_script(text: str) -> Tuple[str, int, int]:
    """
    Inspect Unicode characters in a lyric line and return:
    (script_family, fa_marker_count, ar_marker_count)
    where script_family is 'armenian', 'arabic_persian', 'latin', or 'other'.
    """
    arm_cnt = len(re.findall(r"[\u0530-\u058F]", text))
    ar_fa_cnt = len(re.findall(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]", text))
    lat_cnt = len(re.findall(r"[A-Za-z]", text))

    # Persian-distinct Unicode codepoints (Peh, Tcheh, Jeh, Gaf, Keheh \u06A9, Farsi Yeh \u06CC, ZWNJ)
    fa_markers = len(re.findall(r"[پچژگکیۀ\u200c]", text))
    # Arabic-distinct Unicode codepoints (Teh Marbuta, Alef Maksura, Hamza forms, Arabic Kaf \u0643, Arabic Yeh \u064A)
    ar_markers = len(re.findall(r"[ةىأإؤكي]", text))

    if arm_cnt > ar_fa_cnt and arm_cnt >= lat_cnt and arm_cnt > 0:
        return "armenian", 0, 0
    if ar_fa_cnt >= arm_cnt and ar_fa_cnt >= lat_cnt and ar_fa_cnt > 0:
        return "arabic_persian", fa_markers, ar_markers
    if lat_cnt > 0:
        return "latin", 0, 0
    return "other", 0, 0


_VOCALIZATION_INTERJECTION_RE = re.compile(
    r"^(?:آه|اوه|آخ|لالا|لا\s+لا|ها\s+ها|اه|Ա՜խ|Ախ|Օ՜|Վա՜յ)\b",
    flags=re.IGNORECASE,
)


def resolve_dominant_language_and_filter_outliers(
    raw_lines: List[Dict[str, Any]],
    chunk_lang_votes: List[str],
    explicit_language: str = "auto"
) -> Tuple[str, List[Dict[str, Any]], int]:
    """
    Prevent first-chunk prompt/noise poisoning across multi-chunk songs:
    1. Votes on dominant Unicode script & language across ALL chunks of the song.
    2. Strips cross-script noise hallucinations (e.g., Armenian lines hallucinated during
       live concert crowd noise or guitar outros inside a Persian/Arabic song, or vice versa).
    3. Strips rapid repetitive wordless-vocalization loops (e.g., 5+ identical <=1.8s 'Akh...' / 'Aaah...' lines).
    """
    if not raw_lines:
        fallback_lang = explicit_language if explicit_language != "auto" else "auto"
        return fallback_lang, [], 0

    annotated = []
    script_word_totals = {"armenian": 0, "arabic_persian": 0, "latin": 0, "other": 0}
    total_fa_markers = 0
    total_ar_markers = 0

    for ln in raw_lines:
        txt = str(ln.get("text", "")).strip()
        if not txt or txt.lower() in ("none", "null") or (txt.startswith("<") and txt.endswith(">")):
            continue
        script, fa_m, ar_m = detect_line_script(txt)
        w_cnt = max(1, len(txt.split()))
        script_word_totals[script] = script_word_totals.get(script, 0) + w_cnt
        total_fa_markers += fa_m
        total_ar_markers += ar_m
        annotated.append((ln, script, w_cnt))

    if not annotated:
        return (explicit_language if explicit_language != "auto" else "auto"), [], len(raw_lines)

    # 1. Determine authoritative script family & ISO language across the whole song
    if explicit_language == "hy":
        dominant_script = "armenian"
        resolved_lang = "hy"
    elif explicit_language in ("fa", "ar"):
        dominant_script = "arabic_persian"
        resolved_lang = explicit_language
    else:
        arm_w = script_word_totals["armenian"]
        ar_fa_w = script_word_totals["arabic_persian"]
        if arm_w > ar_fa_w and arm_w > 0:
            dominant_script = "armenian"
            resolved_lang = "hy"
        elif ar_fa_w > 0:
            dominant_script = "arabic_persian"
            fa_votes = sum(1 for v in chunk_lang_votes if v == "fa")
            ar_votes = sum(1 for v in chunk_lang_votes if v == "ar")
            fa_score = total_fa_markers + (fa_votes * 4)
            ar_score = total_ar_markers + (ar_votes * 4)
            resolved_lang = "fa" if fa_score >= ar_score else "ar"
        else:
            dominant_script = "latin"
            valid_votes = [v for v in chunk_lang_votes if v in ("fa", "ar", "hy")]
            resolved_lang = max(set(valid_votes), key=valid_votes.count) if valid_votes else "auto"

    # 2. Strip cross-script noise hallucinations when dominant script is clear (>= 65% of native words or explicit)
    non_latin_total = script_word_totals["armenian"] + script_word_totals["arabic_persian"]
    dominant_ratio = (
        script_word_totals.get(dominant_script, 0) / max(1, non_latin_total)
        if dominant_script in ("armenian", "arabic_persian") else 0.0
    )

    script_filtered: List[Dict[str, Any]] = []
    removed_count = len(raw_lines) - len(annotated)
    enforce_script = (explicit_language in ("fa", "ar", "hy")) or (dominant_ratio >= 0.65)

    for ln, script, _w_cnt in annotated:
        if enforce_script and dominant_script in ("armenian", "arabic_persian"):
            if script in ("armenian", "arabic_persian") and script != dominant_script:
                removed_count += 1
                continue
        script_filtered.append(ln)

    # 3. Strip rapid repetitive vocalization loops (e.g. >=4 consecutive identical short interjection phrases)
    final_filtered: List[Dict[str, Any]] = []
    i = 0
    n = len(script_filtered)
    while i < n:
        j = i + 1
        base_txt = re.sub(r"[^\w\u0530-\u058F\u0600-\u06FF]+", " ", script_filtered[i]["text"]).strip().lower()
        while j < n:
            nxt_txt = re.sub(r"[^\w\u0530-\u058F\u0600-\u06FF]+", " ", script_filtered[j]["text"]).strip().lower()
            gap = float(script_filtered[j]["start"]) - float(script_filtered[j - 1]["end"])
            if nxt_txt == base_txt and gap <= 1.0:
                j += 1
            else:
                break

        run_len = j - i
        word_count = len(script_filtered[i]["text"].split())
        avg_dur = sum(float(script_filtered[k]["end"]) - float(script_filtered[k]["start"]) for k in range(i, j)) / run_len
        is_interjection = bool(_VOCALIZATION_INTERJECTION_RE.search(script_filtered[i]["text"].strip()))

        if run_len >= 4 and word_count <= 3 and avg_dur <= 1.8 and is_interjection:
            removed_count += run_len
        elif run_len >= 2 and word_count >= 3 and avg_dur <= 1.25:
            # Coalesce ultra-rushed identical repetitions (< 1.25s for 3+ words, > 4.5 words/s)
            # in Pairs so outro choruses match natural ~2.3s singing tempo instead of flashing 2x speed
            k = i
            while k < j:
                if k + 1 < j and (float(script_filtered[k + 1]["end"]) - float(script_filtered[k]["start"])) <= 2.8:
                    merged_item = dict(script_filtered[k])
                    merged_item["end"] = script_filtered[k + 1]["end"]
                    final_filtered.append(merged_item)
                    removed_count += 1
                    k += 2
                else:
                    final_filtered.append(script_filtered[k])
                    k += 1
        else:
            final_filtered.extend(script_filtered[i:j])
        i = j

    return resolved_lang, final_filtered, removed_count


def parse_synced_lrc_lines(
    synced_lrc: str,
    total_duration: float,
    vad_intervals: Optional[List[List[float]]] = None,
    explicit_language: str = "auto",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse timestamp-synced LRC lines ([MM:SS.xx] text) and anchor their end timestamps
    against detected VAD vocal intervals so lines never linger across instrumental solos.
    """
    entries: List[Tuple[float, str]] = []
    for raw_ln in (synced_lrc or "").splitlines():
        m = re.match(r"^\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$", raw_ln.strip())
        if m:
            ts = int(m.group(1)) * 60 + float(m.group(2))
            txt = m.group(3).strip()
            entries.append((ts, txt))

    raw_lines: List[Dict[str, Any]] = []
    for i, (ts, txt) in enumerate(entries):
        if not txt:
            continue
        nxt_ts = entries[i + 1][0] if i + 1 < len(entries) else min(total_duration, ts + 4.5)
        w_cnt = max(1, len(txt.split()))
        natural_max = max(3.6, min(6.2, w_cnt * 0.68))
        end_ts = min(nxt_ts - 0.08, ts + natural_max)

        # If VAD intervals exist and there is a long solo gap after this line, clamp to active singing end
        if vad_intervals and (nxt_ts - ts) > 6.5:
            overlapping_ends = [iv[1] for iv in vad_intervals if iv[0] <= ts + 2.5 and iv[1] >= ts + 0.8]
            if overlapping_ends:
                vad_end = min(overlapping_ends, key=lambda e: abs(e - (ts + natural_max)))
                if ts + 1.2 <= vad_end <= nxt_ts - 0.08:
                    end_ts = min(end_ts, round(vad_end, 2))

        if end_ts <= ts + 0.3:
            end_ts = min(total_duration, ts + 1.8)
        raw_lines.append({"text": txt, "start": round(ts, 2), "end": round(end_ts, 2)})

    resolved_lang, filtered_raw, _ = resolve_dominant_language_and_filter_outliers(
        raw_lines, chunk_lang_votes=[], explicit_language=explicit_language
    )
    aligned = normalize_and_build_words(filtered_raw, total_duration=total_duration, lead_in=0.04)
    return resolved_lang, aligned


def transcribe_and_align_vocals(
    vocal_path: str,
    reference_lyrics: Optional[str] = None,
    language: str = "auto",
    duration: float = 180.0,
    song_title: str = "",
    artist: str = "",
    synced_lyrics: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Primary audio-first timing engine:
    1. Scans vocals.wav with VAD to locate real singing and skip instrumental solos.
    2. If duration-matched synced_lyrics (.lrc) are available, anchors them directly
       to the vocal VAD envelope for millisecond-accurate line & word sync.
    3. Otherwise slices active singing into <22s chunks and listens with Gemini 2.5 Flash Audio
       to capture exact sung lyrics + exact master timestamps.
    4. Resolves true song language across ALL chunks via Unicode script majority voting
       and strips cross-script / crowd / vocalization hallucinations.
    5. Falls back cleanly to local Whisper or offline VAD alignment if offline.
    """
    actual_dur = duration if duration > 0 else probe_audio_duration(vocal_path, 180.0)
    explicit_lang_hint = language if language in ("fa", "ar", "hy") else "auto"
    detected_lang = explicit_lang_hint

    intervals: List[List[float]] = []
    if vocal_path and os.path.exists(vocal_path):
        try:
            print("[ALIGN] Scanning isolated singer vocals with Voice Activity Detection (VAD)...")
            intervals = extract_vocal_intervals_vad(vocal_path, total_duration=actual_dur, max_single_interval=20.0)
            print(f"[ALIGN] Detected {len(intervals)} active singing phrases (skipping silent instrumental breaks).")
        except Exception as e:
            print(f"[WARN] VAD scan failed ({e}).")

    # 0. Duration-matched Synced LRC + VAD envelope verification when available
    if synced_lyrics and synced_lyrics.strip():
        resolved_lang, synced_aligned = parse_synced_lrc_lines(
            synced_lyrics,
            total_duration=actual_dur,
            vad_intervals=intervals,
            explicit_language=explicit_lang_hint,
        )
        if synced_aligned:
            print(f"[ALIGN] Anchored {len(synced_aligned)} duration-matched synced LRC lines to vocal waveform.")
            return {
                "language": resolved_lang,
                "lines": synced_aligned,
                "chunks_processed": len(intervals),
                "outliers_filtered": 0,
                "method": "synced_lrc_vad",
            }

    # 1. Multimodal Gemini Audio Transcription & Timing on chunked vocal clips
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if vocal_path and os.path.exists(vocal_path) and api_key and intervals:
        chunks = group_intervals_into_chunks(intervals, total_duration=actual_dur, max_chunk_sec=22.0)
        print(f"[ALIGN] Timing lyrics directly from singer's voice across {len(chunks)} vocal segment(s)...")
        collected_raw_lines: List[Dict[str, Any]] = []
        chunk_lang_votes: List[str] = []
        chunk_failures = 0

        for ch in chunks:
            try:
                audio_bytes, mime_type = extract_audio_chunk_bytes(vocal_path, ch["start"], ch["end"])
                # IMPORTANT: Never lock chunk N+1's language_hint to chunk 1's guess!
                res = transcribe_chunk_with_gemini(
                    audio_bytes=audio_bytes,
                    mime_type=mime_type,
                    clip_duration=ch["duration"],
                    sub_intervals=ch["sub_intervals"],
                    language_hint=explicit_lang_hint,
                    reference_lyrics=reference_lyrics,
                    song_title=song_title,
                    artist=artist
                )
                ch_lines = res.get("lines", [])
                ch_lang = str(res.get("language", "")).strip().lower()
                if ch_lines and ch_lang in ("fa", "ar", "hy"):
                    chunk_lang_votes.append(ch_lang)

                for item in ch_lines:
                    txt = str(item.get("text", "")).strip()
                    if not txt:
                        continue
                    rel_s = max(0.0, min(ch["duration"], float(item.get("start", 0.0))))
                    rel_e = max(rel_s + 0.3, min(ch["duration"], float(item.get("end", rel_s + 2.0))))
                    collected_raw_lines.append({
                        "text": txt,
                        "start": round(ch["start"] + rel_s, 2),
                        "end": round(ch["start"] + rel_e, 2)
                    })
                print(
                    f"      Chunk {ch['chunk_id'] + 1}/{len(chunks)} "
                    f"[{ch['start']:.1f}s - {ch['end']:.1f}s]: timed {len(ch_lines)} sung line(s)"
                )
            except Exception as e:
                chunk_failures += 1
                print(f"[WARN] Vocal chunk {ch['chunk_id'] + 1} audio timing error: {e}")

        if collected_raw_lines:
            detected_lang, clean_raw_lines, outliers_removed = resolve_dominant_language_and_filter_outliers(
                collected_raw_lines,
                chunk_lang_votes=chunk_lang_votes,
                explicit_language=explicit_lang_hint
            )
            if outliers_removed > 0:
                print(
                    f"[ALIGN] Filtered {outliers_removed} cross-script / non-lyric noise outlier line(s) "
                    f"(resolved language='{detected_lang}')."
                )
            aligned = normalize_and_build_words(clean_raw_lines, total_duration=actual_dur)
            if aligned:
                return {
                    "language": detected_lang,
                    "lines": aligned,
                    "chunks_processed": len(chunks) - chunk_failures,
                    "outliers_filtered": outliers_removed,
                    "method": "gemini_audio"
                }

    # 2. Local Whisper fallback if installed
    if vocal_path and os.path.exists(vocal_path):
        try:
            from lyrics_transcriber import transcribe_vocals
            asr_res = transcribe_vocals(vocal_path)
            if asr_res.get("lines"):
                asr_lang = str(asr_res.get("language", "")).strip().lower()
                detected_lang, clean_asr, _ = resolve_dominant_language_and_filter_outliers(
                    asr_res["lines"],
                    chunk_lang_votes=[asr_lang] if asr_lang else [],
                    explicit_language=explicit_lang_hint
                )
                aligned = normalize_and_build_words(clean_asr, total_duration=actual_dur, lead_in=0.05)
                return {
                    "language": detected_lang,
                    "lines": aligned,
                    "chunks_processed": 1,
                    "method": "whisper_audio"
                }
        except Exception:
            pass

    # 3. Offline VAD interval mapping if reference lyrics were supplied
    ref_lines = clean_lines(reference_lyrics) if reference_lyrics else []
    if ref_lines and intervals:
        print("[ALIGN] Offline fallback: mapping reference lyrics to VAD vocal intervals...")
        aligned = align_lines_to_vocal_intervals(ref_lines, intervals, total_duration=actual_dur)
        return {
            "language": detected_lang,
            "lines": aligned,
            "chunks_processed": 0,
            "method": "vad_reference"
        }

    # 4. Last-resort heuristic
    if ref_lines:
        return {
            "language": detected_lang,
            "lines": align_fallback_heuristic(reference_lyrics or "", actual_dur),
            "chunks_processed": 0,
            "method": "heuristic"
        }

    return {"language": detected_lang, "lines": [], "chunks_processed": 0, "method": "none"}


def align_lyrics(
    vocal_path: str,
    lyrics_text: str = "",
    language: str = "auto",
    duration: float = 180.0
) -> List[Dict[str, Any]]:
    """Backwards-compatible wrapper returning timed line dicts."""
    res = transcribe_and_align_vocals(
        vocal_path=vocal_path,
        reference_lyrics=lyrics_text,
        language=language,
        duration=duration
    )
    return res.get("lines", [])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        out = transcribe_and_align_vocals(sys.argv[1])
        print(json.dumps(out, indent=2, ensure_ascii=False))
