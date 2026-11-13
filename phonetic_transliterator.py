"""
phonetic_transliterator.py - Strict Phonetic Romanizer & Translator
Converts Arabic, Persian, and Armenian lyrics into strictly phonetic English letters
(plain ASCII, no Arabizi numbers 3/7, no diacritics) matching word-for-word timestamps.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional

# System prompt embodying strict phonetic rules from the architectural blueprint
PHONETIC_SYSTEM_PROMPT = """You are an expert ethnomusicologist, vocal coach, and phonetic linguist specializing in musical romanization.
Your goal is to transcribe non-Latin lyrics (Arabic, Persian, Armenian) into strictly phonetic English letters so that ANY native English speaker with zero knowledge of the source language can read and sing along accurately to the music.

STRICT PHONETIC RULES:
1. PLAIN ASCII ONLY: Use only standard English alphabet letters (a-z, A-Z), hyphens (-), and apostrophes (').
2. NO ARABIZI NUMBERS: NEVER use numbers (do NOT use 3, 7, 5, 2, 8, etc.).
3. NO ACADEMIC DIACRITICS: NEVER use macrons or dots (do NOT use ā, ī, ū, ḥ, ṣ, ḍ, ṭ, etc.).
4. DIGRAPHS & SOUNDS:
   - Guttural 'kh' (خ, Խ): write as 'kh' (like Scottish 'loch').
   - French/gargled 'gh' (غ, ق, Ղ): write as 'gh'.
   - 'sh' (ش, Շ): write as 'sh'.
   - 'ch' (چ, Չ, Ճ): write as 'ch'.
   - 'zh' (ژ, Ժ, or Levantine ج): write as 'zh' (like the 's' in 'measure').
   - Soft/guttural 'h' (ح): write as simple 'h'.
   - 'ayn (ع): write as 'a' or soft pause '''.
   - Glottal stop / hamza (ء): write as '''.
   - Armenian 'ts' (Ց, Ծ) / 'dz' (Ձ): write as 'ts' / 'dz'.
   - Armenian rolled 'r' (Ռ): write as 'rr'.
5. VOWEL CONVENTIONS FOR NATURAL ENGLISH PRONUNCIATION:
   - Long 'a' sound (father): write as 'aa' (e.g. 'salaam', 'janaam').
   - Long 'e' sound (see): write as 'ee' (e.g. 'habeebee', 'deleem').
   - Long 'u' sound (moon): write as 'oo' (e.g. 'noor', 'doost').
   - Short vowels: 'a', 'e', 'i', 'o', 'u'.
6. 1:1 WORD MAPPING CONTRACT:
   - For every line, the array of phonetic words ('phonetic_words') MUST have the exact same number of items as the input native words array ('native_words').
   - The joined phonetic line must match 'phonetic_words'.
7. ENGLISH TRANSLATION:
   - Provide a clear, natural English meaning translation for context.

Input is provided as a JSON list of lines. Return ONLY valid JSON matching this schema (do NOT alter or replace the input native words):
{
  "lines": [
    {
      "line_id": 0,
      "phonetic": "<Plain ASCII phonetic English line>",
      "translation": "<Natural English meaning translation>",
      "phonetic_words": ["<word1>", "<word2>"]
    }
  ]
}
"""

ARABIZI_DIGIT_MAP = {
    "2": "'", "3": "a", "5": "kh", "6": "t",
    "7": "h", "8": "gh", "9": "s", "0": "o", "1": "l", "4": "dh"
}

DIACRITIC_FOLD_MAP = {
    "ā": "aa", "Ā": "Aa", "ī": "ee", "Ī": "Ee", "ū": "oo", "Ū": "Oo",
    "ē": "e", "Ē": "E", "ō": "o", "Ō": "O", "ə": "uh", "Ə": "Uh",
    "ḥ": "h", "Ḥ": "H", "ṣ": "s", "Ṣ": "S", "ḍ": "d", "Ḍ": "D",
    "ṭ": "t", "Ṭ": "T", "ẓ": "z", "Ẓ": "Z", "ġ": "gh", "Ġ": "Gh",
    "ḫ": "kh", "Ḫ": "Kh", "š": "sh", "Š": "Sh", "č": "ch", "Č": "Ch",
    "ž": "zh", "Ž": "Zh", "ʿ": "'", "ʾ": "'"
}


def sanitize_strict_ascii_phonetic(text: str) -> str:
    """Guarantee plain ASCII English phonetics with zero Arabizi digits or diacritics."""
    import unicodedata
    out_chars = []
    for ch in text:
        if ch in DIACRITIC_FOLD_MAP:
            out_chars.append(DIACRITIC_FOLD_MAP[ch])
        elif ch in ARABIZI_DIGIT_MAP:
            out_chars.append(ARABIZI_DIGIT_MAP[ch])
        elif ord(ch) < 128:
            out_chars.append(ch)
        else:
            norm = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
            out_chars.append(norm)
    cleaned = "".join(out_chars)
    return re.sub(r"[^A-Za-z'\- ,.!?]", "", cleaned).strip()


def _call_gemini_phonetic_batch(batch_lines: List[Dict[str, Any]], language: str, api_key: str) -> List[Dict[str, Any]]:
    input_payload = {
        "language_hint": language,
        "lines": [
            {
                "line_id": i,
                "native_line": line.get("text", ""),
                "native_words": [w.get("word", "") for w in line.get("words", [])] if "words" in line else line.get("text", "").split()
            }
            for i, line in enumerate(batch_lines)
        ]
    }
    user_prompt = f"Perform strict phonetic romanization and English translation for these lyrics:\n{json.dumps(input_payload, ensure_ascii=False)}"

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=PHONETIC_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        data = json.loads(response.text)
    except Exception:
        import urllib.request
        import ssl
        try:
            import certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ssl_ctx = ssl._create_unverified_context()

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        body = {
            "system_instruction": {"parts": [{"text": PHONETIC_SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=45, context=ssl_ctx) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            text = raw["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(text)

    return data.get("lines", [])


def transliterate_with_gemini(lines: List[Dict[str, Any]], language: str = "auto", batch_size: int = 25) -> List[Dict[str, Any]]:
    """
    Call Gemini API in batches to perform strict phonetic transliteration while
    preserving exact 1:1 word and line timestamps from the vocal audio alignment.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required for AI transliteration.")

    output_lines: List[Dict[str, Any]] = []
    for start_idx in range(0, len(lines), batch_size):
        batch = lines[start_idx:start_idx + batch_size]
        batch_out = _call_gemini_phonetic_batch(batch, language, api_key)
        # Pad if fewer returned than sent
        while len(batch_out) < len(batch):
            batch_out.append({})
        output_lines.extend(batch_out[:len(batch)])

    # Merge back exact timestamps from input lines into output
    aligned_results = []
    for i, orig_line in enumerate(lines):
        out_line = output_lines[i] if i < len(output_lines) else {}
        raw_pwords = out_line.get("phonetic_words", [])
        phonetic_words = [sanitize_strict_ascii_phonetic(str(pw)) or deterministic_phonetic_word(orig_line.get("text", "")) for pw in raw_pwords]
        native_words = orig_line.get("words", [])

        merged_words = []
        if native_words:
            for j, nw in enumerate(native_words):
                fallback_pw = sanitize_strict_ascii_phonetic(deterministic_phonetic_word(nw.get("word", "")))
                pw = phonetic_words[j] if (j < len(phonetic_words) and phonetic_words[j]) else fallback_pw
                merged_words.append({
                    "native": nw.get("word", ""),
                    "phonetic": pw,
                    "start": nw.get("start", orig_line.get("start", 0.0)),
                    "end": nw.get("end", orig_line.get("end", 0.0))
                })
        else:
            merged_words = [
                {
                    "native": w,
                    "phonetic": pw,
                    "start": orig_line.get("start", 0.0),
                    "end": orig_line.get("end", 0.0)
                }
                for w, pw in zip(orig_line.get("text", "").split(), phonetic_words)
            ]

        raw_line_phonetic = out_line.get("phonetic", "")
        clean_line_phonetic = sanitize_strict_ascii_phonetic(raw_line_phonetic) if raw_line_phonetic else ""
        if not clean_line_phonetic:
            clean_line_phonetic = " ".join(mw["phonetic"] for mw in merged_words)

        aligned_results.append({
            "line_id": i,
            "start": orig_line.get("start", 0.0),
            "end": orig_line.get("end", 0.0),
            "native": orig_line.get("text", ""),
            "phonetic": clean_line_phonetic,
            "translation": out_line.get("translation", ""),
            "words": merged_words
        })

    return aligned_results

# Offline deterministic transliteration table for Arabic, Persian, and Armenian
CHAR_MAP = {
    # Arabic & Persian common
    "ا": "a", "أ": "a", "إ": "e", "آ": "aa", "ء": "'",
    "ب": "b", "پ": "p", "ت": "t", "ث": "s",
    "ج": "j", "چ": "ch", "ح": "h", "خ": "kh",
    "د": "d", "ذ": "z", "ر": "r", "ز": "z", "ژ": "zh",
    "س": "s", "ش": "sh", "ص": "s", "ض": "z",
    "ط": "t", "ظ": "z", "ع": "a", "غ": "gh",
    "ف": "f", "ق": "gh", "ك": "k", "ک": "k", "گ": "g",
    "ل": "l", "م": "m", "ن": "n", "ه": "h", "ة": "a",
    "و": "oo", "ؤ": "'", "ي": "ee", "ی": "ee", "ئ": "'",
    "ى": "a", "ً": "an", "ٌ": "un", "ٍ": "in",
    "َ": "a", "ُ": "o", "ِ": "e", "ّ": "",
    # Armenian uppercase & lowercase
    "Ա": "A", "ա": "a", "Բ": "B", "բ": "b", "Գ": "G", "գ": "g",
    "Դ": "D", "դ": "d", "Ե": "Ye", "ե": "e", "Զ": "Z", "զ": "z",
    "Է": "E", "է": "e", "Ը": "U", "ը": "u", "Թ": "T", "թ": "t",
    "Ժ": "Zh", "ժ": "zh", "Ի": "Ee", "ի": "ee", "Լ": "L", "լ": "l",
    "Խ": "Kh", "խ": "kh", "Ծ": "Ts", "ծ": "ts", "Կ": "K", "կ": "k",
    "Հ": "H", "հ": "h", "Ձ": "Dz", "ձ": "dz", "Ղ": "Gh", "ղ": "gh",
    "Ճ": "Ch", "ճ": "ch", "Մ": "M", "մ": "m", "Յ": "Y", "յ": "y",
    "Ն": "N", "ն": "n", "Շ": "Sh", "շ": "sh", "Ո": "Vo", "ո": "o",
    "Չ": "Ch", "չ": "ch", "Պ": "P", "պ": "p", "Ջ": "J", "ջ": "j",
    "Ռ": "Rr", "ռ": "rr", "Ս": "S", "ս": "s", "Վ": "V", "վ": "v",
    "Տ": "T", "տ": "t", "Ր": "R", "ր": "r", "Ց": "Ts", "ց": "ts",
    "Ու": "Oo", "ու": "oo", "Փ": "P", "փ": "p", "Ք": "K", "ք": "k",
    "Օ": "O", "օ": "o", "Ֆ": "F", "ֆ": "f"
}

def deterministic_phonetic_word(word: str) -> str:
    """Convert a word into phonetic Latin letters using the strict rule table."""
    out = []
    # Check for two-character combos like 'ու'
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i:i+2] in CHAR_MAP:
            out.append(CHAR_MAP[word[i:i+2]])
            i += 2
        elif word[i] in CHAR_MAP:
            out.append(CHAR_MAP[word[i]])
            i += 1
        elif word[i].isalnum():
            out.append(word[i])
            i += 1
        else:
            i += 1
    res = "".join(out).strip()
    return res.capitalize() if res else word

def offline_phonetic_transliterator(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    High-fidelity offline phonetic transliterator applying strict phonetic rules
    when Gemini API is unavailable or offline.
    """
    results = []
    for i, line in enumerate(lines):
        words = line.get("words", [])
        native_text = line.get("text", "")
        phonetic_words = []
        for w in words:
            pw = deterministic_phonetic_word(w.get("word", ""))
            phonetic_words.append({
                "native": w.get("word", ""),
                "phonetic": pw,
                "start": w.get("start", line.get("start", 0.0)),
                "end": w.get("end", line.get("end", 0.0))
            })
        
        phonetic_line_str = " ".join(pw["phonetic"] for pw in phonetic_words)
        results.append({
            "line_id": i,
            "start": line.get("start", 0.0),
            "end": line.get("end", 0.0),
            "native": native_text,
            "phonetic": phonetic_line_str if phonetic_line_str else native_text,
            "translation": "(Sing-along phonetic track)",
            "words": phonetic_words
        })
    return results

def process_phonetics(lines: List[Dict[str, Any]], language: str = "auto") -> List[Dict[str, Any]]:
    """
    Unified entry point: tries Gemini API first, seamlessly falls back to
    deterministic offline phonetic rules if network or API policy blocks.
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        try:
            return transliterate_with_gemini(lines, language)
        except Exception as e:
            print(f"[WARN] Gemini API transliteration call failed ({e}). Using offline phonetic rules.")
            return offline_phonetic_transliterator(lines)
    else:
        print("[INFO] Using offline deterministic phonetic transliterator.")
        return offline_phonetic_transliterator(lines)

if __name__ == "__main__":
    sample_lines = [
        {"start": 12.0, "end": 15.0, "text": "حبيبي يا نور العين", "words": [
            {"word": "حبيبي", "start": 12.0, "end": 13.0},
            {"word": "يا", "start": 13.1, "end": 13.5},
            {"word": "نور", "start": 13.6, "end": 14.2},
            {"word": "العين", "start": 14.3, "end": 15.0}
        ]}
    ]
    res = process_phonetics(sample_lines, "ar")
    print(json.dumps(res, indent=2, ensure_ascii=False))
