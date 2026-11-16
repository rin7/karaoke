"""
subtitle_writer.py - Advanced Subtitle & Karaoke Script Generator
Generates:
1. lyrics.json: Full structured data format
2. lyrics.lrc: Standard synchronized lyric format
3. karaoke.ass: Advanced SubStation Alpha with real-time word-level karaoke sweep animations
"""

import json
import math
from typing import Dict, Any, List

def format_timestamp_lrc(seconds: float) -> str:
    """Format seconds into [mm:ss.xx] for LRC."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"[{m:02d}:{s:05.2f}]"

def format_timestamp_ass(seconds: float) -> str:
    """Format seconds into h:mm:ss.cc for ASS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def export_lyrics_json(metadata: Dict[str, Any], lines: List[Dict[str, Any]], output_path: str) -> Dict[str, Any]:
    """Write standardized lyrics.json."""
    data = {
        "metadata": {
            "title": metadata.get("title", ""),
            "artist": metadata.get("artist", ""),
            "language": metadata.get("language", "auto"),
            "duration_seconds": metadata.get("duration", 0.0)
        },
        "lines": lines
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data

def export_lrc(lines: List[Dict[str, Any]], output_path: str) -> str:
    """Write standard & word-enhanced LRC file."""
    lrc_content = []
    for line in lines:
        start_ts = format_timestamp_lrc(line["start"])
        words = line.get("words", [])
        if words:
            # Enhanced LRC with word-level tags
            word_str = "".join([f"<{w['start']:.2f}>{w['phonetic']} " for w in words])
            lrc_content.append(f"{start_ts}{word_str}")
        else:
            lrc_content.append(f"{start_ts}{line['phonetic']}")
    
    content = "\n".join(lrc_content)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content

def export_ass_karaoke(metadata: Dict[str, Any], lines: List[Dict[str, Any]], output_path: str) -> str:
    """
    Generate an Advanced SubStation Alpha (.ass) file configured with:
    - 3-Slot Center-Focused Rolling Karaoke Stage on 1920x1080:
        * Top Slot (Previous Line i-1, when gap <= 4.0s):
            Drifts upward by 25px and smoothly fades out over ~1.8s (\\move + \\fad(0,1100)):
            1. Prev Native Script (48pt Bold, y=195 -> 170)
            2. Prev Phonetic English (54pt Bold, y=265 -> 240)
            3. Prev English Translation (32pt Italic, y=325 -> 300)
        * Center Slot (Main Active Line i — Full Brightness + \\K Sweep):
            1. Native Script (68pt Bold, y=465 — slightly smaller than 74pt Latin)
            2. Phonetic English (74pt Bold, y=555, with \\K color-sweep karaoke effect)
            3. English Meaning Translation (42pt Italic, y=635)
        * Bottom Slot (Upcoming Next Line i+1, when gap <= 6.0s):
            Rises gently by 25px into place and fades in (\\move + \\fad(320,120)):
            1. Next Native Script (52pt Bold, y=785 -> 760)
            2. Next Phonetic English (58pt Bold, y=860 -> 835)
            3. Next English Meaning Translation (34pt Italic, y=925 -> 900)
    """
    # ASS Header and Styles
    # Colors in ASS are &HAABBGGRR (Hex format: Alpha, Blue, Green, Red)
    ass_template = """[Script Info]
Title: {title} - AI Phonetic Karaoke
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KaraokePhonetic, Montserrat, 74, &H00FFFFFF, &H0000E5FF, &H00100C09, &H80000000, 1, 0, 0, 0, 100, 100, 1, 0, 1, 3.8, 2.2, 5, 40, 40, 0, 1
Style: KaraokeNative, Arial, 68, &H00F8F8F8, &H00FFFFFF, &H00100C09, &H80000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, 3.2, 2.0, 5, 40, 40, 0, 1
Style: KaraokeTranslation, Arial, 42, &H00D8DEE9, &H00FFFFFF, &H00100C09, &H80000000, 0, 1, 0, 0, 100, 100, 0, 0, 1, 2.2, 1.2, 5, 40, 40, 0, 1
Style: KaraokePrevPhonetic, Montserrat, 54, &H70B8C4D4, &H70B8C4D4, &H70100C09, &H80000000, 1, 0, 0, 0, 100, 100, 1, 0, 1, 2.4, 1.2, 5, 40, 40, 0, 1
Style: KaraokePrevNative, Arial, 48, &H75B0BCCB, &H75B0BCCB, &H70100C09, &H80000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, 2.2, 1.2, 5, 40, 40, 0, 1
Style: KaraokePrevTranslation, Arial, 32, &H8094A3B8, &H8094A3B8, &H75100C09, &H80000000, 0, 1, 0, 0, 100, 100, 0, 0, 1, 1.6, 0.8, 5, 40, 40, 0, 1
Style: KaraokeNextPhonetic, Montserrat, 58, &H25DCE6F2, &H00FFFFFF, &H30100C09, &H80000000, 1, 0, 0, 0, 100, 100, 1, 0, 1, 2.8, 1.5, 5, 40, 40, 0, 1
Style: KaraokeNextNative, Arial, 52, &H30C8D4E3, &H00FFFFFF, &H30100C09, &H80000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, 2.5, 1.5, 5, 40, 40, 0, 1
Style: KaraokeNextTranslation, Arial, 34, &H40A6B4C8, &H00FFFFFF, &H30100C09, &H80000000, 0, 1, 0, 0, 100, 100, 0, 0, 1, 1.8, 1.0, 5, 40, 40, 0, 1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    for idx, line in enumerate(lines):
        l_start = float(line["start"])
        l_end = float(line["end"])
        display_end = l_end

        if idx + 1 < len(lines):
            next_start = float(lines[idx + 1]["start"])
            if l_end > next_start - 0.04:
                l_end = max(l_start + 0.20, round(next_start - 0.04, 2))
                display_end = l_end
            elif (next_start - l_end) <= 1.20:
                # Hold active display window cleanly up to the next line's start so the roll is seamless
                display_end = max(l_end, round(next_start - 0.02, 2))

        start_str = format_timestamp_ass(l_start)
        end_str = format_timestamp_ass(display_end)

        # --- TOP SLOT: Previous Line (Line idx - 1) Fading Out & Drifting Upward ---
        if idx > 0:
            prev_line = lines[idx - 1]
            gap_from_prev = l_start - float(prev_line["end"])
            if gap_from_prev <= 4.0:
                prev_fade_end = min(display_end, round(l_start + 1.80, 2))
                if prev_fade_end > l_start + 0.30:
                    prev_end_str = format_timestamp_ass(prev_fade_end)
                    fade_ms = max(350, int(round((prev_fade_end - l_start) * 650)))
                    move_ms = max(400, int(round((prev_fade_end - l_start) * 900)))
                    prev_native = prev_line.get("native", "").replace("\n", " ").strip()
                    prev_phonetic = prev_line.get("phonetic", "").replace("\n", " ").strip()
                    prev_translation = prev_line.get("translation", "").replace("\n", " ").strip()

                    if prev_native:
                        events.append(
                            f"Dialogue: 0,{start_str},{prev_end_str},KaraokePrevNative,,0,0,0,,"
                            f"{{\\an8\\move(960,195,960,170,0,{move_ms})\\fad(0,{fade_ms})}}{prev_native}"
                        )
                    if prev_phonetic:
                        events.append(
                            f"Dialogue: 0,{start_str},{prev_end_str},KaraokePrevPhonetic,,0,0,0,,"
                            f"{{\\an5\\move(960,265,960,240,0,{move_ms})\\fad(0,{fade_ms})}}{prev_phonetic}"
                        )
                    if prev_translation:
                        events.append(
                            f"Dialogue: 0,{start_str},{prev_end_str},KaraokePrevTranslation,,0,0,0,,"
                            f"{{\\an2\\move(960,325,960,300,0,{move_ms})\\fad(0,{fade_ms})}}\"{prev_translation}\""
                        )

        # --- CENTER SLOT: Main Active Line (Line idx) ---
        # 1. Native script line (68pt bold, centered at y=465 above 74pt phonetic line)
        native_text = line.get("native", "").replace("\n", " ").strip()
        if native_text:
            events.append(
                f"Dialogue: 2,{start_str},{end_str},KaraokeNative,,0,0,0,,{{\\an8\\pos(960,465)\\fad(140,140)}}{native_text}"
            )

        # 2. Phonetic line with karaoke \K tags (centiseconds duration + exact gap holds, centered at y=555)
        words = line.get("words", [])
        karaoke_parts = []
        if words:
            cursor_t = l_start
            for w_idx, w in enumerate(words):
                w_s = max(cursor_t, float(w.get("start", cursor_t)))
                w_e = min(l_end, max(w_s + 0.05, float(w.get("end", w_s + 0.2))))
                gap_cs = int(round((w_s - cursor_t) * 100))
                if gap_cs > 0:
                    karaoke_parts.append(f"{{\\k{gap_cs}}}")
                dur_cs = max(1, int(round((w_e - w_s) * 100)))
                suffix = " " if w_idx < len(words) - 1 else ""
                karaoke_parts.append(f"{{\\K{dur_cs}}}{w['phonetic']}{suffix}")
                cursor_t = w_e
            phonetic_karaoke_text = "".join(karaoke_parts).strip()
        else:
            total_cs = max(1, int(round((l_end - l_start) * 100)))
            phonetic_karaoke_text = f"{{\\K{total_cs}}}{line.get('phonetic', '')}"

        events.append(
            f"Dialogue: 3,{start_str},{end_str},KaraokePhonetic,,0,0,0,,{{\\an5\\pos(960,555)\\fad(140,140)}}{phonetic_karaoke_text}"
        )

        # 3. Translation line (42pt italic, centered at y=635 below phonetic line)
        translation_text = line.get("translation", "").replace("\n", " ").strip()
        if translation_text:
            events.append(
                f"Dialogue: 2,{start_str},{end_str},KaraokeTranslation,,0,0,0,,{{\\an2\\pos(960,635)\\fad(140,140)}}\"{translation_text}\""
            )

        # --- BOTTOM SLOT: Upcoming Next Line (Line idx + 1) Rising & Fading In ---
        if idx + 1 < len(lines):
            nxt_line = lines[idx + 1]
            gap_to_next = float(nxt_line["start"]) - l_end
            # Show look-ahead preview when next line starts within 6.0s (skip across long instrumental solos)
            if gap_to_next <= 6.0:
                nxt_native = nxt_line.get("native", "").replace("\n", " ").strip()
                nxt_phonetic = nxt_line.get("phonetic", "").replace("\n", " ").strip()
                nxt_translation = nxt_line.get("translation", "").replace("\n", " ").strip()

                if nxt_native:
                    events.append(
                        f"Dialogue: 1,{start_str},{end_str},KaraokeNextNative,,0,0,0,,"
                        f"{{\\an8\\move(960,785,960,760,0,350)\\fad(320,120)}}{nxt_native}"
                    )
                if nxt_phonetic:
                    events.append(
                        f"Dialogue: 1,{start_str},{end_str},KaraokeNextPhonetic,,0,0,0,,"
                        f"{{\\an5\\move(960,860,960,835,0,350)\\fad(320,120)}}{nxt_phonetic}"
                    )
                if nxt_translation:
                    events.append(
                        f"Dialogue: 1,{start_str},{end_str},KaraokeNextTranslation,,0,0,0,,"
                        f"{{\\an2\\move(960,925,960,900,0,350)\\fad(320,120)}}\"{nxt_translation}\""
                    )

    full_ass = ass_template.format(title=metadata.get("title", "Karaoke Track")) + "\n".join(events) + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_ass)

    return full_ass

if __name__ == "__main__":
    sample_meta = {"title": "Nour El Ein", "artist": "Amr Diab", "duration": 180.0}
    sample_lines = [
        {
            "start": 10.0, "end": 14.5,
            "native": "حبيبي يا نور العين",
            "phonetic": "Habeebee ya noor el-ein",
            "translation": "My love, you are the light of my eyes",
            "words": [
                {"native": "حبيبي", "phonetic": "Habeebee", "start": 10.0, "end": 11.2},
                {"native": "يا", "phonetic": "ya", "start": 11.3, "end": 11.7},
                {"native": "نور", "phonetic": "noor", "start": 11.8, "end": 12.8},
                {"native": "العين", "phonetic": "el-ein", "start": 12.9, "end": 14.5}
            ]
        }
    ]
    export_ass_karaoke(sample_meta, sample_lines, "./test_karaoke.ass")
    print("ASS export test successful.")
