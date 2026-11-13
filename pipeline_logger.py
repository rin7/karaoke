"""
pipeline_logger.py - Per-Song Execution Logging, Watchdog Freeze/Crash Detection & ML Telemetry
Provides:
1. Per-song logs inside each song's output folder (`out/<song_id>/`):
   - `pipeline.log`: Timestamped human-readable log + C/Python crash tracebacks (flushed live)
   - `telemetry.jsonl`: Structured event stream (`step_start`, `step_end`, `stall_warning`, `error`, `run_summary`)
   - `run_status.json`: Atomic live state + heartbeat file for crash & freeze detection
2. Global ML & Operations Dataset (`logs/runs_history.jsonl`):
   - Cross-song structured dataset of stage latencies, memory usage, vocal coverage,
     alignment quality metrics, phonetic compliance, and failure diagnostics for ML optimization.
3. Crash & Freeze Detection:
   - Background watchdog thread updating heartbeats and capturing live main-thread stack traces
     (`sys._current_frames()`) whenever a stage exceeds its stall threshold.
   - Automatic stale-PID crash detection when a previous process was killed (`SIGKILL` / OOM / crash).
"""

import os
import re
import sys
import json
import time
import uuid
import signal
import socket
import platform
import resource
import threading
import traceback
import faulthandler
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Dict, Any, List, Optional


DEFAULT_GLOBAL_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def iso_now() -> str:
    """UTC ISO-8601 timestamp with millisecond precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def get_peak_rss_mb() -> float:
    """Return peak resident memory usage (RSS) of the current process in MB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(usage / (1024.0 * 1024.0), 2)
        return round(usage / 1024.0, 2)
    except Exception:
        return 0.0


def is_pid_alive(pid: int) -> bool:
    """Check whether an OS process with the given PID is still running."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """Write JSON atomically via a temporary file so crashes never leave corrupt JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp_path, path)


def _append_jsonl(path: str, record: Dict[str, Any], fsync: bool = False) -> None:
    """Append a single JSON object line to a JSONL file and flush immediately."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        if fsync:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


def detect_and_recover_stale_run(
    output_dir: str,
    global_log_dir: Optional[str] = None,
    freeze_threshold_sec: float = 45.0
) -> Optional[Dict[str, Any]]:
    """
    Inspect `run_status.json` in `output_dir` to detect if a previous run hard-crashed
    (dead PID while status == 'running') or is currently frozen (alive PID with stale heartbeat).
    Automatically updates crashed runs so post-mortem and ML logs stay accurate.
    """
    status_path = os.path.join(output_dir, "run_status.json")
    if not os.path.exists(status_path):
        return None

    try:
        with open(status_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return None

    if state.get("status") != "running":
        return None

    old_pid = int(state.get("pid", 0))
    now_ep = time.time()
    last_hb = float(state.get("last_heartbeat_epoch", state.get("started_epoch", now_ep)))
    hb_age = round(max(0.0, now_ep - last_hb), 2)

    if old_pid != os.getpid() and not is_pid_alive(old_pid):
        # Previous process died abruptly (SIGKILL / OOM / segfault / power loss) without cleanup!
        crashed_at = iso_now()
        state["status"] = "crashed"
        state["crash_detected_at"] = crashed_at
        state["heartbeat_age_at_detection_sec"] = hb_age
        state["error"] = {
            "type": "UnexpectedProcessTermination",
            "message": (
                f"Process PID {old_pid} terminated unexpectedly during step "
                f"'{state.get('current_step', 'unknown')}' (last heartbeat {hb_age}s ago)."
            ),
            "failed_step": state.get("current_step", "unknown")
        }
        _atomic_write_json(status_path, state)

        telemetry_path = os.path.join(output_dir, "telemetry.jsonl")
        _append_jsonl(
            telemetry_path,
            {
                "timestamp": crashed_at,
                "event": "crash_recovered",
                "run_id": state.get("run_id"),
                "crashed_pid": old_pid,
                "failed_step": state.get("current_step"),
                "last_heartbeat_at": state.get("last_heartbeat_at"),
                "heartbeat_age_sec": hb_age,
            },
            fsync=True,
        )

        g_dir = global_log_dir or os.environ.get("KARAOKE_LOG_DIR") or DEFAULT_GLOBAL_LOG_DIR
        history_path = os.path.join(g_dir, "runs_history.jsonl")
        _append_jsonl(
            history_path,
            {
                "timestamp": crashed_at,
                "run_id": state.get("run_id"),
                "song_folder": os.path.basename(os.path.abspath(output_dir)),
                "source": state.get("source"),
                "status": "crashed",
                "failed_step": state.get("current_step"),
                "error_type": "UnexpectedProcessTermination",
                "error_message": state["error"]["message"],
                "stage_durations_sec": state.get("stage_durations_sec", {}),
            },
            fsync=True,
        )
        return {"diagnosis": "crashed", "state": state}

    if old_pid != os.getpid() and is_pid_alive(old_pid) and hb_age >= freeze_threshold_sec:
        return {"diagnosis": "frozen", "heartbeat_age_sec": hb_age, "state": state}

    if old_pid != os.getpid() and is_pid_alive(old_pid):
        return {"diagnosis": "running", "heartbeat_age_sec": hb_age, "state": state}

    return None


def compute_ml_quality_features(
    phonetic_lines: List[Dict[str, Any]],
    duration_sec: float,
    align_meta: Optional[Dict[str, Any]] = None,
    stem_cache_hit: bool = False,
    online_lyrics_found: bool = False
) -> Dict[str, Any]:
    """
    Compute structured numerical features per song for ML training, anomaly detection,
    and pipeline quality benchmarking.
    """
    n_lines = len(phonetic_lines)
    if n_lines == 0:
        return {
            "lines_count": 0,
            "words_count": 0,
            "vocal_coverage_ratio": 0.0,
            "overlap_violations_count": 0,
            "ascii_phonetic_compliance_rate": 0.0,
            "stem_cache_hit": stem_cache_hit,
            "online_lyrics_found": online_lyrics_found,
        }

    total_words = 0
    line_durations: List[float] = []
    overlap_violations = 0
    out_of_bounds = 0
    ascii_compliant_lines = 0
    translated_lines = 0

    for idx, ln in enumerate(phonetic_lines):
        s = float(ln.get("start", 0.0))
        e = float(ln.get("end", s))
        dur = max(0.0, e - s)
        line_durations.append(dur)
        total_words += len(ln.get("words", []))

        if idx + 1 < n_lines:
            next_s = float(phonetic_lines[idx + 1].get("start", e))
            if e > next_s + 0.01:
                overlap_violations += 1

        if duration_sec > 0 and e > duration_sec + 1.0:
            out_of_bounds += 1

        p_txt = str(ln.get("phonetic", ""))
        if p_txt and all(ord(c) < 128 for c in p_txt) and not re.search(r"\d", p_txt):
            ascii_compliant_lines += 1

        t_txt = str(ln.get("translation", "")).strip()
        if t_txt and t_txt != "(Sing-along phonetic track)":
            translated_lines += 1

    total_sung_sec = sum(line_durations)
    span_sec = max(0.0, float(phonetic_lines[-1].get("end", 0.0)) - float(phonetic_lines[0].get("start", 0.0)))

    features: Dict[str, Any] = {
        "lines_count": n_lines,
        "words_count": total_words,
        "avg_words_per_line": round(total_words / n_lines, 2),
        "avg_line_duration_sec": round(total_sung_sec / n_lines, 3),
        "min_line_duration_sec": round(min(line_durations), 3),
        "max_line_duration_sec": round(max(line_durations), 3),
        "lyric_timeline_span_sec": round(span_sec, 2),
        "vocal_coverage_ratio": round(total_sung_sec / max(1.0, duration_sec), 4),
        "overlap_violations_count": overlap_violations,
        "out_of_bounds_count": out_of_bounds,
        "ascii_phonetic_compliance_rate": round(ascii_compliant_lines / n_lines, 4),
        "translation_coverage_rate": round(translated_lines / n_lines, 4),
        "stem_cache_hit": stem_cache_hit,
        "online_lyrics_found": online_lyrics_found,
    }
    if align_meta:
        features["alignment_method"] = align_meta.get("method", "unknown")
        features["chunks_processed"] = align_meta.get("chunks_processed", 0)
    return features


class _TeeStream:
    """Mirrors stdout/stderr to the terminal while writing timestamped lines into pipeline.log."""

    def __init__(self, orig_stream, log_file, lock: threading.Lock, prefix: str = ""):
        self._orig = orig_stream
        self._log = log_file
        self._lock = lock
        self._prefix = prefix
        self._at_line_start = True

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._orig.write(data)
        self._orig.flush()

        # Skip pure carriage-return progress bar redraws in log file to keep logs clean & readable
        if "\r" in data and "\n" not in data:
            return len(data)

        clean_data = data.replace("\r", "")
        with self._lock:
            if not self._log.closed:
                for chunk in clean_data.splitlines(keepends=True):
                    if self._at_line_start and chunk.strip():
                        self._log.write(f"[{iso_now()}] {self._prefix}{chunk}")
                    else:
                        self._log.write(chunk)
                    self._at_line_start = chunk.endswith("\n")
                self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._orig.flush()
        with self._lock:
            if not self._log.closed:
                self._log.flush()

    def fileno(self) -> int:
        return self._orig.fileno()

    def isatty(self) -> bool:
        return getattr(self._orig, "isatty", lambda: False)()


class SongRunLogger:
    """
    Context-managed per-song logger, watchdog freeze detector, and ML telemetry recorder.
    """

    def __init__(
        self,
        source: str,
        output_dir: str,
        global_log_dir: Optional[str] = None,
        heartbeat_interval_sec: float = 5.0,
        default_stall_sec: float = 300.0,
        capture_stdio: bool = True
    ):
        self.run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.source = source
        self.output_dir = os.path.abspath(output_dir)
        self.global_log_dir = os.path.abspath(
            global_log_dir or os.environ.get("KARAOKE_LOG_DIR") or DEFAULT_GLOBAL_LOG_DIR
        )
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.default_stall_sec = default_stall_sec
        self.capture_stdio = capture_stdio

        self.log_path = os.path.join(self.output_dir, "pipeline.log")
        self.telemetry_path = os.path.join(self.output_dir, "telemetry.jsonl")
        self.status_path = os.path.join(self.output_dir, "run_status.json")
        self.history_path = os.path.join(self.global_log_dir, "runs_history.jsonl")

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._main_thread_id = threading.get_ident()

        self._log_fh = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._orig_sigint = None
        self._orig_sigterm = None

        self.started_epoch = time.time()
        self.started_at = iso_now()
        self.current_step = "initializing"
        self.step_started_epoch = self.started_epoch
        self.current_step_stall_sec = default_stall_sec
        self._stall_alerted_for_step = False
        self._paused_for_user_input = False

        self.stage_durations_sec: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}
        self.ml_features: Dict[str, Any] = {}
        self.artifacts: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self.stall_events: List[Dict[str, Any]] = []
        self.previous_crash_recovery: Optional[Dict[str, Any]] = None

    def __enter__(self) -> "SongRunLogger":
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.global_log_dir, exist_ok=True)

        # 1. Check if a previous run in this folder crashed or froze
        self.previous_crash_recovery = detect_and_recover_stale_run(
            self.output_dir, global_log_dir=self.global_log_dir
        )

        # 2. Open per-song pipeline.log
        self._log_fh = open(self.log_path, "a", encoding="utf-8")
        self._write_raw_log(
            f"\n{'='*72}\n"
            f" RUN START: {self.run_id} | PID={os.getpid()} | Host={socket.gethostname()}\n"
            f" Source:    {self.source}\n"
            f" Platform:  {platform.platform()} | Python {platform.python_version()}\n"
            f"{'='*72}\n"
        )

        if self.previous_crash_recovery and self.previous_crash_recovery.get("diagnosis") == "crashed":
            prev_st = self.previous_crash_recovery["state"]
            self._write_raw_log(
                f"[RECOVERY] Detected previous crashed run '{prev_st.get('run_id')}' "
                f"(died during step '{prev_st.get('current_step')}'). Logged to crash history.\n"
            )

        # 3. Enable C-level faulthandler into pipeline.log for segfault/OOM diagnostics
        try:
            faulthandler.enable(file=self._log_fh, all_threads=True)
        except Exception:
            pass

        # 4. Tee stdout & stderr into pipeline.log
        if self.capture_stdio:
            self._orig_stdout = sys.stdout
            self._orig_stderr = sys.stderr
            sys.stdout = _TeeStream(self._orig_stdout, self._log_fh, self._lock, prefix="")
            sys.stderr = _TeeStream(self._orig_stderr, self._log_fh, self._lock, prefix="[STDERR] ")

        # 5. Register signal handlers (if running on main thread) so SIGINT/SIGTERM log cleanly
        if threading.current_thread() is threading.main_thread():
            self._orig_sigint = signal.getsignal(signal.SIGINT)
            self._orig_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, self._handle_termination_signal)
            signal.signal(signal.SIGTERM, self._handle_termination_signal)

        # 6. Write initial run_status.json & telemetry event
        self._flush_status(status="running")
        self.log_event("run_start", {
            "source": self.source,
            "pid": os.getpid(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }, fsync=True)

        # 7. Launch background heartbeat & freeze-detection watchdog
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name=f"KaraokeWatchdog-{self.run_id}",
            daemon=True,
        )
        self._watchdog_thread.start()
        return self

    def _handle_termination_signal(self, signum: int, _frame) -> None:
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        raise KeyboardInterrupt(f"Pipeline interrupted by signal {sig_name} ({signum})")

    def _write_raw_log(self, msg: str) -> None:
        with self._lock:
            if self._log_fh and not self._log_fh.closed:
                self._log_fh.write(f"[{iso_now()}] {msg}")
                self._log_fh.flush()

    def log_event(self, event_name: str, payload: Optional[Dict[str, Any]] = None, fsync: bool = False) -> None:
        rec = {
            "timestamp": iso_now(),
            "run_id": self.run_id,
            "step": self.current_step,
            "event": event_name,
            "rss_mb": get_peak_rss_mb(),
        }
        if payload:
            rec.update(payload)
        _append_jsonl(self.telemetry_path, rec, fsync=fsync)

    def log_warning(self, message: str) -> None:
        self.warnings.append(message)
        self._write_raw_log(f"[WARN] {message}\n")
        self.log_event("warning", {"message": message})

    def _flush_status(self, status: str, error_info: Optional[Dict[str, Any]] = None) -> None:
        now_ep = time.time()
        state: Dict[str, Any] = {
            "run_id": self.run_id,
            "status": status,
            "pid": os.getpid(),
            "source": self.source,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "started_epoch": round(self.started_epoch, 3),
            "last_heartbeat_at": iso_now(),
            "last_heartbeat_epoch": round(now_ep, 3),
            "elapsed_sec": round(now_ep - self.started_epoch, 2),
            "current_step": self.current_step,
            "current_step_elapsed_sec": round(now_ep - self.step_started_epoch, 2),
            "peak_rss_mb": get_peak_rss_mb(),
            "stall_detected": len(self.stall_events) > 0,
            "stage_durations_sec": self.stage_durations_sec,
            "metadata": {
                k: v for k, v in self.metadata.items()
                if k in ("title", "artist", "language", "duration", "source_type")
            },
            "ml_features": self.ml_features,
            "error": error_info,
        }
        _atomic_write_json(self.status_path, state)

    def _capture_main_thread_stack(self) -> str:
        """Capture the exact live Python stack trace of the main pipeline thread on freeze/stall."""
        try:
            frames = sys._current_frames()
            frame = frames.get(self._main_thread_id)
            if frame:
                return "".join(traceback.format_stack(frame))
        except Exception as e:
            return f"<stack capture unavailable: {e}>"
        return "<no frame found>"

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_sec):
            try:
                now_ep = time.time()
                self._flush_status(status="running")

                if self._paused_for_user_input:
                    continue

                step_elapsed = now_ep - self.step_started_epoch
                if step_elapsed >= self.current_step_stall_sec and not self._stall_alerted_for_step:
                    self._stall_alerted_for_step = True
                    live_stack = self._capture_main_thread_stack()
                    stall_rec = {
                        "step": self.current_step,
                        "step_elapsed_sec": round(step_elapsed, 2),
                        "threshold_sec": self.current_step_stall_sec,
                        "main_thread_stack": live_stack,
                    }
                    self.stall_events.append(stall_rec)
                    self._write_raw_log(
                        f"[WATCHDOG FREEZE ALERT] Step '{self.current_step}' has been running for "
                        f"{step_elapsed:.1f}s (threshold={self.current_step_stall_sec:.1f}s).\n"
                        f"Live Main-Thread Call Stack:\n{live_stack}\n"
                    )
                    self.log_event("stall_warning", stall_rec, fsync=True)
                    self._flush_status(status="running")
            except Exception:
                pass

    @contextmanager
    def step(self, step_name: str, stall_timeout_sec: Optional[float] = None):
        """Track start, end, latency, memory, and stall timeout for a pipeline step."""
        self.current_step = step_name
        self.step_started_epoch = time.time()
        self.current_step_stall_sec = stall_timeout_sec if stall_timeout_sec is not None else self.default_stall_sec
        self._stall_alerted_for_step = False

        self._flush_status(status="running")
        self.log_event("step_start", {"step": step_name, "stall_timeout_sec": self.current_step_stall_sec}, fsync=True)
        t0 = time.time()
        try:
            yield
        except BaseException as exc:
            dur = round(time.time() - t0, 3)
            self.stage_durations_sec[step_name] = dur
            self.log_event(
                "step_error",
                {
                    "step": step_name,
                    "duration_sec": dur,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                fsync=True,
            )
            raise
        else:
            dur = round(time.time() - t0, 3)
            self.stage_durations_sec[step_name] = dur
            self.log_event("step_end", {"step": step_name, "duration_sec": dur}, fsync=True)
            self._flush_status(status="running")

    @contextmanager
    def pause_watchdog_for_input(self):
        """Pause freeze detection while waiting on interactive human CLI input."""
        self._paused_for_user_input = True
        try:
            yield
        finally:
            self._paused_for_user_input = False
            self.step_started_epoch = time.time()

    def record_metadata(self, meta: Dict[str, Any]) -> None:
        self.metadata = dict(meta)
        self._flush_status(status="running")

    def record_ml_features(self, features: Dict[str, Any]) -> None:
        self.ml_features = dict(features)
        self._flush_status(status="running")

    def record_artifacts(self, artifacts: Dict[str, Any]) -> None:
        sizes: Dict[str, int] = {}
        for key, path in artifacts.items():
            if isinstance(path, str) and os.path.exists(path):
                try:
                    sizes[f"{key}_bytes"] = os.path.getsize(path)
                except OSError:
                    pass
        self.artifacts = {"paths": artifacts, "sizes": sizes}

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # 1. Stop watchdog thread
        self._stop_event.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)

        total_elapsed = round(time.time() - self.started_epoch, 3)
        song_dur = float(self.metadata.get("duration", 0.0) or 0.0)
        rtf = round(total_elapsed / song_dur, 4) if song_dur > 0 else None

        error_info: Optional[Dict[str, Any]] = None
        if exc_type is not None:
            final_status = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
            tb_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            error_info = {
                "type": exc_type.__name__,
                "message": str(exc_val),
                "failed_step": self.current_step,
                "traceback": tb_str,
            }
            self._write_raw_log(
                f"\n[PIPELINE {final_status.upper()}] Step='{self.current_step}' | "
                f"{exc_type.__name__}: {exc_val}\n{tb_str}\n"
            )
        else:
            final_status = "completed"
            self.current_step = "completed"

        # 2. Finalize per-song status & telemetry
        self._flush_status(status=final_status, error_info=error_info)

        summary_record: Dict[str, Any] = {
            "timestamp": iso_now(),
            "run_id": self.run_id,
            "song_folder": os.path.basename(self.output_dir),
            "source": self.source,
            "status": final_status,
            "title": self.metadata.get("title"),
            "artist": self.metadata.get("artist"),
            "language": self.metadata.get("language"),
            "song_duration_sec": round(song_dur, 2),
            "total_elapsed_sec": total_elapsed,
            "realtime_factor": rtf,
            "peak_rss_mb": get_peak_rss_mb(),
            "stage_durations_sec": self.stage_durations_sec,
            "ml_features": self.ml_features,
            "artifact_sizes": self.artifacts.get("sizes", {}),
            "warnings_count": len(self.warnings),
            "stall_alerts_count": len(self.stall_events),
            "failed_step": error_info["failed_step"] if error_info else None,
            "error_type": error_info["type"] if error_info else None,
            "error_message": error_info["message"] if error_info else None,
        }

        self.log_event("run_summary", summary_record, fsync=True)
        _append_jsonl(self.history_path, summary_record, fsync=True)

        self._write_raw_log(
            f" RUN END: {self.run_id} | status={final_status} | elapsed={total_elapsed:.2f}s | "
            f"peak_rss={get_peak_rss_mb():.1f}MB\n{'='*72}\n"
        )

        # 3. Restore signal handlers, stdio streams, and faulthandler cleanly before closing file
        if threading.current_thread() is threading.main_thread():
            if self._orig_sigint is not None:
                signal.signal(signal.SIGINT, self._orig_sigint)
            if self._orig_sigterm is not None:
                signal.signal(signal.SIGTERM, self._orig_sigterm)

        if self.capture_stdio and self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr

        try:
            faulthandler.disable()
        except Exception:
            pass

        with self._lock:
            if self._log_fh and not self._log_fh.closed:
                self._log_fh.close()

        # Never suppress exceptions so CLI/caller still handles errors normally
        return False


def scan_all_song_runs(
    out_root_dir: str,
    global_log_dir: Optional[str] = None,
    freeze_threshold_sec: float = 45.0
) -> Dict[str, Any]:
    """
    Scan all song folders in `out/` and the global `runs_history.jsonl` log to detect
    any active, completed, failed, crashed, or frozen runs and compute aggregate ML stats.
    """
    g_dir = global_log_dir or os.environ.get("KARAOKE_LOG_DIR") or DEFAULT_GLOBAL_LOG_DIR
    songs_report: List[Dict[str, Any]] = []

    if os.path.isdir(out_root_dir):
        for entry in sorted(os.listdir(out_root_dir)):
            song_dir = os.path.join(out_root_dir, entry)
            if not os.path.isdir(song_dir):
                continue

            recovery = detect_and_recover_stale_run(
                song_dir, global_log_dir=g_dir, freeze_threshold_sec=freeze_threshold_sec
            )
            status_path = os.path.join(song_dir, "run_status.json")
            if os.path.exists(status_path):
                try:
                    with open(status_path, "r", encoding="utf-8") as f:
                        st = json.load(f)
                    diag = recovery["diagnosis"] if recovery else st.get("status", "unknown")
                    songs_report.append({
                        "song_folder": entry,
                        "run_id": st.get("run_id"),
                        "diagnosis": diag,
                        "current_step": st.get("current_step"),
                        "elapsed_sec": st.get("elapsed_sec"),
                        "last_heartbeat_at": st.get("last_heartbeat_at"),
                        "title": st.get("metadata", {}).get("title"),
                        "language": st.get("metadata", {}).get("language"),
                        "lines_count": st.get("ml_features", {}).get("lines_count"),
                        "error": st.get("error"),
                    })
                except Exception:
                    pass
            elif os.path.exists(os.path.join(song_dir, "lyrics.json")):
                try:
                    with open(os.path.join(song_dir, "lyrics.json"), "r", encoding="utf-8") as f:
                        lj = json.load(f)
                    meta = lj.get("metadata", {})
                    lines = lj.get("lines", [])
                    has_mp4 = any(fn.endswith(".mp4") for fn in os.listdir(song_dir))
                    songs_report.append({
                        "song_folder": entry,
                        "run_id": "pre_logger_run",
                        "diagnosis": "completed" if has_mp4 else "partial_no_video",
                        "current_step": "completed" if has_mp4 else "step5_phonetics_and_subtitles",
                        "elapsed_sec": None,
                        "last_heartbeat_at": None,
                        "title": meta.get("title"),
                        "language": meta.get("language"),
                        "lines_count": len(lines),
                        "error": None,
                    })
                except Exception:
                    pass

    history_path = os.path.join(g_dir, "runs_history.jsonl")
    history_records: List[Dict[str, Any]] = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history_records.append(json.loads(line))
                    except Exception:
                        pass

    completed = [r for r in history_records if r.get("status") == "completed"]
    avg_rtf = (
        round(sum(r["realtime_factor"] for r in completed if r.get("realtime_factor")) / len(completed), 3)
        if completed else None
    )

    return {
        "total_logged_runs": len(history_records),
        "completed_runs": len(completed),
        "failed_runs": sum(1 for r in history_records if r.get("status") == "failed"),
        "crashed_runs": sum(1 for r in history_records if r.get("status") == "crashed"),
        "avg_realtime_factor": avg_rtf,
        "songs": songs_report,
    }


if __name__ == "__main__":
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    report = scan_all_song_runs(default_out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
