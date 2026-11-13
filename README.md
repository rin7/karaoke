# 🎤 Karaoke Agent: Multilingual AI Phonetic Karaoke Platform

> Automatically transforms **Armenian, Persian, and Arabic** songs into instrumental karaoke tracks with **strictly phonetic English lyrics** that any English speaker can read and sing along to, packaged as ready-to-share **1080p MP4 videos**.

---

## 🌟 Key Features

1. **Strictly Phonetic English Romanization**
   - No Arabizi numbers (`3`, `7`, `5`).
   - No academic diacritics (`ā`, `ḥ`, `ṣ`).
   - Plain ASCII letters with intuitive digraphs (`kh`, `gh`, `sh`, `ch`, `zh`) and musical vowel extensions (`aa`, `ee`, `oo`).
   - Strict 1:1 word-for-word alignment with millisecond timestamps.

2. **Automated Lyrics Search & Confirmation**
   - Automatically queries **LyricsTranslate.com**, Genius, and web search using the song's title and artist.
   - User can confirm, edit, or paste official lyrics before audio alignment.
   - Falls back to vocal Speech-to-Text (WhisperX) if online lyrics are unavailable.

3. **High-Definition 1080p Karaoke MP4 Video**
   - Burns in a **3-tier subtitle layout**:
     - **Top:** Original script (Arabic / Persian / Armenian)
     - **Center (Large):** Strictly Phonetic English with real-time glowing color-fill animation as words are sung
     - **Bottom:** English meaning translation
   - High-contrast, stage-ready background with 320kbps clean instrumental audio.
   - Direct download link ready for YouTube, TikTok, or party screens.

4. **Credits & Serverless Architecture (Mirrors Aniden.ai)**
   - **Signup Bonus:** 500 free credits upon user signup.
   - **Usage:** 100 credits per song processed (all-inclusive: stem separation, transliteration, player preview, and 1080p MP4 download).
   - **Cache Deduplication:** Instant reuse of previously processed songs (`song_cache`).
   - **Stack:** Next.js (Vercel) + Supabase (Postgres, Storage, Auth) + Modal (A10G GPU worker via spawn pattern).

---

## 📁 Project Structure

```
Karaoke_Agent/
├── SOFTWARE_BLUEPRINT.md        # Complete software blueprint, flowcharts & module logic
├── web_app.py                   # v0.1 FastAPI Web Studio & Apple App Store-ready REST API
├── web/                         # Responsive light/dark PWA UI (index.html, styles.css, app.js)
├── audio_ingest.py              # Download audio via yt-dlp & normalize to 44.1kHz stereo WAV
├── stem_separator.py            # Roformer / HTDemucs vocal & instrumental stem separation + SHA-1 cache
├── lyrics_search.py             # Clean metadata query & verified LyricsTranslate lookup
├── lyrics_aligner.py            # Vocal-first VAD chunker, Gemini 2.5 Flash audio aligner & script voter
├── lyrics_transcriber.py        # Optional local Whisper / WhisperX fallback ASR
├── phonetic_transliterator.py   # Batched Gemini + offline strict ASCII romanizer & translator
├── subtitle_writer.py           # Exports lyrics.json, lyrics.lrc, and gap-accurate 3-tier karaoke.ass
├── karaoke_video_renderer.py    # FFmpeg + libass 1080p MP4 vocal-free karaoke video burner
├── pipeline_logger.py           # Per-song logs, freeze/crash watchdog & ML telemetry collector
├── pipeline.py                  # Master 6-step pipeline orchestrator
├── karaoke_cli.py               # CLI runner (--source, --check-logs)
├── requirements.txt             # Python dependencies
├── logs/
│   └── runs_history.jsonl       # Cross-song ML performance, quality & crash dataset
├── tests/
│   └── test_pipeline_steps.py   # 18 automated verification tests across Steps 1-8
└── Serverless_implementation/
    ├── migrations/
    │   └── 001_initial_schema.sql # Supabase schema (profiles, songs, song_cache)
    └── modal/
        └── modal_worker.py      # Modal GPU & CPU workers using the Aniden spawn pattern
```

---

## 🌐 v0.1 Web Studio & Mobile-Ready API Quickstart

Start the local web application and REST API server:
```bash
python3 web_app.py
```
Then open **`http://127.0.0.1:8000`** in your browser (or inspect the mobile/iOS OpenAPI spec at **`http://127.0.0.1:8000/docs`**).

- **Paste a YouTube URL or upload an audio file** (`.mp3`, `.wav`, `.m4a`, `.flac`).
- **Live 6-step progress & watchdog health badge** with expandable real-time `pipeline.log` console.
- **Built-in 1080p Karaoke Player + Click-to-Seek 3-Tier Lyrics** and one-click downloads (`MP4`, `Instrumental WAV`, `.ass`, `.lrc`, `lyrics.json`).
- **Custom Domain / LAN Hosting:**
  ```bash
  CORS_ORIGINS="https://yourdomain.com" python3 web_app.py --host 0.0.0.0 --port 8000
  ```

---

## 🚀 CLI Quickstart

### 1. Prerequisites & Installation
Ensure you have `ffmpeg` installed on your machine:
```bash
# macOS
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt update && sudo apt install -y ffmpeg libass-dev
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Set your Gemini API Key for strict phonetic transliteration:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 2. Run the Karaoke CLI

#### From a YouTube Link:
```bash
python3 karaoke_cli.py --source "https://www.youtube.com/watch?v=cDNDVtoJhik"
```

#### From a Local Audio File:
```bash
python3 karaoke_cli.py --source "path/to/song.mp3" --lang fa
```

#### With Pre-Supplied Lyrics:
```bash
python3 karaoke_cli.py --source "song.mp3" --lyrics "lyrics.txt"
```

#### Non-Interactive Mode (Automated):
```bash
python3 karaoke_cli.py --source "https://..." --non-interactive
```

#### Inspect Song Logs, Detect Crashes/Freezes & View ML Telemetry:
```bash
python3 karaoke_cli.py --check-logs
```

---

## 📦 Generated Outputs

Each run creates an isolated directory in `./out/<song_name>/`:
- `instrumental.wav`: Clean vocal-free backing track
- `vocals.wav`: Isolated vocal stem (used for precision lyric timing)
- `lyrics.json`: Full word-level timestamped dataset with tri-tier lyrics
- `lyrics.lrc`: Standard synchronized lyric file
- `karaoke.ass`: Styled SubStation Alpha subtitle file with `\k` pause holds & `\K` karaoke sweeps
- `<Artist>_-_<Title>_(Karaoke).mp4` & `karaoke.mp4`: Complete 1080p video ready to play or upload
- `pipeline.log`: Millisecond-timestamped execution log, Python tracebacks, freeze stack dumps & `faulthandler` crash traces
- `telemetry.jsonl`: Structured step-by-step latency, RSS memory, and watchdog events
- `run_status.json`: Live heartbeat & crash/freeze status file

Plus a global cross-song dataset in `./logs/runs_history.jsonl` for ML benchmarking and error analytics.

---

## ☁️ Deploying to Modal (Serverless Worker)

Deploy the GPU/CPU worker to your Modal account:
```bash
modal deploy Serverless_implementation/modal/modal_worker.py
```
This deploys:
- `process_song_web`: Web trigger endpoint
- `process_song_gpu`: A10G GPU stem separation & alignment worker
- `render_mp4_web`: Web trigger for video rendering
- `render_mp4_cpu`: CPU video burning worker
