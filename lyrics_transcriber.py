"""
lyrics_transcriber.py - Speech-to-Text Transcription for Vocals
Transcribes isolated vocal stems using WhisperX / faster-whisper when official
lyrics cannot be found online or provided by the user.
"""

import os
from typing import Dict, Any, List

def transcribe_vocals_whisperx(vocals_path: str, model_size: str = "large-v3") -> Dict[str, Any]:
    """Transcribe vocal stem using WhisperX with automatic language detection."""
    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    # 1. Load Whisper model
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(vocals_path)
    
    # 2. Transcribe
    result = model.transcribe(audio, batch_size=16)
    detected_lang = result.get("language", "en")
    
    # 3. Align Whisper output using wav2vec2 / MMS
    align_model, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
    aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
    
    # 4. Format into unified line structure
    formatted_lines = []
    for i, seg in enumerate(aligned_result.get("segments", [])):
        words = []
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "word": w.get("word", ""),
                    "start": round(w["start"], 2),
                    "end": round(w["end"], 2)
                })
        formatted_lines.append({
            "line_id": i,
            "text": seg.get("text", "").strip(),
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2),
            "words": words
        })
        
    return {
        "language": detected_lang,
        "lines": formatted_lines,
        "full_text": "\n".join(l["text"] for l in formatted_lines)
    }

def transcribe_vocals_openai_whisper(vocals_path: str, model_size: str = "base") -> Dict[str, Any]:
    """Transcribe vocal stem using standard openai-whisper (lightweight, runs on Mac MPS/CPU)."""
    import whisper
    print(f"[ASR] Loading Whisper ({model_size}) model for vocal transcription...")
    model = whisper.load_model(model_size)
    result = model.transcribe(vocals_path)
    detected_lang = result.get("language", "auto")
    
    formatted_lines = []
    for i, seg in enumerate(result.get("segments", [])):
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue
        seg_words = seg_text.split()
        dur = max(0.1, seg.get("end", 0.0) - seg.get("start", 0.0))
        w_dur = dur / max(1, len(seg_words))
        cur_t = seg.get("start", 0.0)
        words = []
        for w in seg_words:
            words.append({
                "word": w,
                "start": round(cur_t, 2),
                "end": round(cur_t + (w_dur * 0.9), 2)
            })
            cur_t += w_dur
            
        formatted_lines.append({
            "line_id": i,
            "text": seg_text,
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2),
            "words": words
        })
        
    return {
        "language": detected_lang,
        "lines": formatted_lines,
        "full_text": "\n".join(l["text"] for l in formatted_lines)
    }

def transcribe_vocals(vocals_path: str) -> Dict[str, Any]:
    """Unified transcription entry point with graceful multi-engine fallback."""
    # 1. Try WhisperX (best for forced word alignment)
    try:
        return transcribe_vocals_whisperx(vocals_path)
    except Exception:
        pass

    # 2. Try standard openai-whisper (pip install openai-whisper)
    try:
        return transcribe_vocals_openai_whisper(vocals_path)
    except Exception:
        pass

    print("[WARN] Neither WhisperX nor openai-whisper are installed.")
    print("       To enable automatic speech-to-text on singing vocals, run:")
    print("           pip3 install openai-whisper")
    return {
        "language": "auto",
        "lines": [],
        "full_text": ""
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = transcribe_vocals(sys.argv[1])
        print("Transcription complete:", res)
