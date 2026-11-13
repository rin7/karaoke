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
    - 3-tier layout:
        1. Native Script (Arabic / Persian / Armenian)
        2. Phonetic English (Large, Centered, with \\k color-sweep karaoke effect)
        3. English Meaning Translation (Italic, subtle contrast)
    """
    # ASS Header and Styles
    # Colors in ASS are &HAABBGGRR (Hex format: Alpha, Blue, Green, Red)
    # Primary: White (&H00FFFFFF), Secondary/Karaoke highlight: Electric Cyan/Yellow (&H0000E5FF)
    # Outline: Deep Navy/Black (&H00110B08), Shadow: Black with alpha (&H80000000)
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
Style: KaraokePhonetic, Montserrat, 62, &H00FFFFFF, &H0000E5FF, &H00100C09, &H80000000, 1, 0, 0, 0, 100, 100, 1, 0, 1, 3.5, 2.0, 5, 40, 40, 0, 1
Style: KaraokeNative, Arial, 56, &H00F8F8F8, &H00FFFFFF, &H00100C09, &H80000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, 3.0, 1.8, 5, 40, 40, 80, 1
Style: KaraokeTranslation, Arial, 34, &H00B0B0B0, &H00FFFFFF, &H00100C09, &H80000000, 0, 1, 0, 0, 100, 100, 0, 0, 1, 2.0, 1.0, 5, 40, 40, 160, 1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    for idx, line in enumerate(lines):
        l_start = float(line["start"])
        l_end = float(line["end"])
        if idx + 1 < len(lines):
            next_start = float(lines[idx + 1]["start"])
            if l_end > next_start - 0.04:
                l_end = max(l_start + 0.20, round(next_start - 0.04, 2))

        start_str = format_timestamp_ass(l_start)
        end_str = format_timestamp_ass(l_end)

        # 1. Native script line (placed above phonetic line, slightly smaller than 62pt Latin phonetic)
        native_text = line.get("native", "").replace("\n", " ")
        if native_text:
            events.append(
                f"Dialogue: 0,{start_str},{end_str},KaraokeNative,,0,0,180,,{{\\an8\\pos(960,635)}}{native_text}"
            )

        # 2. Phonetic line with karaoke \K tags (centiseconds duration + exact gap holds)
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
            f"Dialogue: 1,{start_str},{end_str},KaraokePhonetic,,0,0,0,,{{\\an5\\pos(960,760)}}{phonetic_karaoke_text}"
        )

        # 3. Translation line (placed below phonetic line)
        translation_text = line.get("translation", "").replace("\n", " ")
        if translation_text:
            events.append(
                f"Dialogue: 0,{start_str},{end_str},KaraokeTranslation,,0,0,180,,{{\\an2\\pos(960,860)}}\"{translation_text}\""
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
