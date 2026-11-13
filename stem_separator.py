"""
stem_separator.py - AI Vocal Isolation & Instrumental Separation
Supports MelBand-Roformer / BS-Roformer (via audio-separator) and HTDemucs v4.
Includes fallback for local testing without heavy GPU CUDA dependencies.
"""

import os
import subprocess
import shutil
from typing import Dict, Any

def separate_stems_roformer(audio_path: str, output_dir: str) -> Dict[str, str]:
    """Separate vocals using audio-separator (MelBand-Roformer)."""
    from audio_separator.separator import Separator
    separator = Separator(
        output_dir=output_dir,
        output_format="WAV",
        model_file_dir="/tmp/audio-separator-models"
    )
    separator.load_model(model_filename="model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt")
    output_files = separator.separate(audio_path)
    # audio-separator returns list of generated filenames
    vocals_path = None
    inst_path = None
    for f in output_files:
        full_p = os.path.join(output_dir, f)
        if "Vocals" in f or "vocals" in f:
            vocals_path = full_p
        elif "Instrumental" in f or "instrumental" in f:
            inst_path = full_p
    return {"vocals": vocals_path, "instrumental": inst_path}

def separate_stems_demucs(audio_path: str, output_dir: str) -> Dict[str, str]:
    """Separate vocals using HTDemucs v4 via CLI or python -m demucs."""
    import shutil
    import sys
    demucs_bin = shutil.which("demucs")
    cmd_prefix = [demucs_bin] if demucs_bin else [sys.executable, "-m", "demucs"]
    cmd = cmd_prefix + [
        "--two-stems=vocals",
        "-n", "htdemucs",
        "-o", output_dir,
        audio_path
    ]
    print("[STEMS] Running AI vocal separation with Demucs...")
    subprocess.run(cmd, check=True)
    
    song_folder = os.path.splitext(os.path.basename(audio_path))[0]
    target_dir = os.path.join(output_dir, "htdemucs", song_folder)
    demucs_inst = os.path.join(target_dir, "no_vocals.wav")
    demucs_voc = os.path.join(target_dir, "vocals.wav")
    
    # Overwrite root instrumental.wav with the clean Demucs no_vocals.wav
    final_inst = os.path.join(output_dir, "instrumental.wav")
    final_voc = os.path.join(output_dir, "vocals.wav")
    shutil.copy2(demucs_inst, final_inst)
    shutil.copy2(demucs_voc, final_voc)

    return {
        "vocals": final_voc,
        "instrumental": final_inst
    }

def separate_stems_mock_local(audio_path: str, output_dir: str) -> Dict[str, str]:
    """
    Fallback when demucs/audio-separator are not installed.
    Uses center-channel phase inversion cancellation.
    """
    os.makedirs(output_dir, exist_ok=True)
    inst_path = os.path.join(output_dir, "instrumental.wav")
    vocals_path = os.path.join(output_dir, "vocals.wav")

    print("\n" + "!"*68)
    print(" [ATTENTION: AI VOCAL REMOVER NOT INSTALLED]")
    print(" To cleanly remove the singer's voice and create a studio instrumental,")
    print(" please install Demucs in your terminal by running:")
    print("     pip3 install -U demucs torchaudio")
    print(" Proceeding with basic phase cancellation filter for now...")
    print("!"*68 + "\n")

    cmd_inst = [
        "ffmpeg", "-y", "-i", audio_path,
        "-af", "pan=stereo|c0=c0-c1|c1=c1-c0",
        "-ar", "44100", "-ac", "2",
        inst_path
    ]
    subprocess.run(cmd_inst, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cmd_voc = [
        "ffmpeg", "-y", "-i", audio_path,
        "-af", "highpass=f=180,lowpass=f=3400",
        "-ar", "44100", "-ac", "2",
        vocals_path
    ]
    subprocess.run(cmd_voc, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return {"instrumental": inst_path, "vocals": vocals_path}

def _audio_fingerprint(audio_path: str) -> str:
    """Fingerprint of the input audio (file size + hash of first 4 MB)."""
    import hashlib
    h = hashlib.sha1()
    with open(audio_path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return f"{os.path.getsize(audio_path)}-{h.hexdigest()}"

def _write_stem_fingerprint(output_dir: str, fingerprint: str) -> None:
    with open(os.path.join(output_dir, "stems_source.txt"), "w") as f:
        f.write(fingerprint)

def _read_stem_fingerprint(output_dir: str) -> str:
    p = os.path.join(output_dir, "stems_source.txt")
    if os.path.exists(p):
        with open(p) as f:
            return f.read().strip()
    return ""

def separate_stems(audio_path: str, output_dir: str, prefer_gpu: bool = True) -> Dict[str, str]:
    """
    Unified entry point for vocal isolation:
    Checks for cached Demucs stems (only if they belong to THIS audio) -> Roformer -> Demucs -> Local fallback.
    """
    fingerprint = _audio_fingerprint(audio_path)

    # 0. Reuse cached Demucs stems only when they were produced from this exact audio
    song_folder = os.path.splitext(os.path.basename(audio_path))[0]
    demucs_inst = os.path.join(output_dir, "htdemucs", song_folder, "no_vocals.wav")
    demucs_voc = os.path.join(output_dir, "htdemucs", song_folder, "vocals.wav")
    if (os.path.exists(demucs_inst) and os.path.exists(demucs_voc)
            and os.path.getsize(demucs_inst) > 10000
            and _read_stem_fingerprint(output_dir) == fingerprint):
        print("[STEMS] Found Demucs vocal-free stems for this song. Reusing clean instrumental.")
        final_inst = os.path.join(output_dir, "instrumental.wav")
        final_voc = os.path.join(output_dir, "vocals.wav")
        shutil.copy2(demucs_inst, final_inst)
        shutil.copy2(demucs_voc, final_voc)
        return {"instrumental": final_inst, "vocals": final_voc}
    elif os.path.exists(demucs_inst):
        print("[STEMS] Cached stems belong to a different audio file. Re-separating...")

    result = None
    if prefer_gpu:
        try:
            result = separate_stems_roformer(audio_path, output_dir)
        except Exception:
            result = None
        if not result or not result.get("instrumental"):
            try:
                result = separate_stems_demucs(audio_path, output_dir)
            except Exception:
                result = None

    if result and result.get("instrumental"):
        _write_stem_fingerprint(output_dir, fingerprint)
        return result

    return separate_stems_mock_local(audio_path, output_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        res = separate_stems(sys.argv[1], sys.argv[2])
        print("Stems:", res)
