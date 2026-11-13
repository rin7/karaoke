/* Karaoke Agent Studio v0.1 — Client Controller */
(() => {
  "use strict";

  const STEP_ORDER = [
    "step1_ingest",
    "step2_stem_separation",
    "step3_lyrics_search",
    "step4_vocal_timing",
    "step5_phonetics_and_subtitles",
    "step6_video_render",
  ];

  const LANG_LABELS = {
    hy: "Armenian (Հայերեն)",
    fa: "Persian (فارسی)",
    ar: "Arabic (العربية)",
    auto: "Auto-Detected",
  };

  // Fallback for :user-invalid in older browsers + ARIA invalid sync
  const UserInvalidFallback = (() => {
    const dirtyState = new WeakMap();
    const updateState = (input) => {
      if (!input.checkValidity) return;
      const isValid = input.checkValidity();
      input.classList.toggle("user-invalid-fallback", !isValid);
      input.setAttribute("aria-invalid", !isValid ? "true" : "false");
    };
    const handleEvent = (event) => {
      const input = event.target;
      if (event.type === "reset") {
        const controls = input.elements || [];
        for (const c of controls) {
          dirtyState.delete(c);
          c.classList.remove("user-invalid-fallback");
          c.removeAttribute("aria-invalid");
        }
        return;
      }
      if (!input || !input.checkValidity) return;
      const st = dirtyState.get(input) || { hasInteracted: false, hasBlurred: false };
      if (event.type === "input" || event.type === "change") {
        st.hasInteracted = true;
        dirtyState.set(input, st);
        if (st.hasBlurred) updateState(input);
      } else if (event.type === "blur") {
        st.hasBlurred = true;
        dirtyState.set(input, st);
        if (st.hasInteracted) updateState(input);
      }
    };
    const init = (root = document) => {
      if (CSS.supports("selector(:user-invalid)")) {
        const syncAria = (el) => {
          if (el && typeof el.matches === "function" && el.checkValidity) {
            el.setAttribute("aria-invalid", el.matches(":user-invalid") ? "true" : "false");
          }
        };
        root.addEventListener("blur", (e) => syncAria(e.target), true);
        root.addEventListener("input", (e) => {
          if (e.target && e.target.hasAttribute && e.target.hasAttribute("aria-invalid")) {
            syncAria(e.target);
          }
        });
        return;
      }
      root.addEventListener("blur", handleEvent, true);
      root.addEventListener("input", handleEvent);
      root.addEventListener("change", handleEvent);
      root.addEventListener("reset", handleEvent, true);
    };
    return { init };
  })();

  const form = document.getElementById("convert-form");
  const urlGroup = document.getElementById("url-group");
  const fileGroup = document.getElementById("file-group");
  const sourceUrlInput = document.getElementById("source-url");
  const audioFileInput = document.getElementById("audio-file");
  const manualLyricsGroup = document.getElementById("manual-lyrics-group");
  const lyricsAutoRadio = document.getElementById("lyrics-auto");
  const lyricsManualRadio = document.getElementById("lyrics-manual");
  const submitBtn = document.getElementById("submit-btn");
  const formFeedback = document.getElementById("form-feedback");

  const jobPanel = document.getElementById("job-panel");
  const jobStepLabel = document.getElementById("job-step-label");
  const jobElapsed = document.getElementById("job-elapsed");
  const watchdogBadge = document.getElementById("watchdog-badge");
  const jobProgress = document.getElementById("job-progress");
  const stepList = document.getElementById("step-list");
  const jobLogTail = document.getElementById("job-log-tail");

  const playerSubtitle = document.getElementById("player-subtitle");
  const playerLangBadge = document.getElementById("player-lang-badge");
  const videoEl = document.getElementById("karaoke-video");
  const emptyPlayer = document.getElementById("empty-player");
  const downloadBar = document.getElementById("download-bar");
  const lyricsList = document.getElementById("lyrics-list");
  const songsGrid = document.getElementById("songs-grid");
  const refreshLibraryBtn = document.getElementById("refresh-library-btn");
  const healthText = document.getElementById("health-text");

  let activePollTimer = null;
  let currentSongLines = [];
  let lyricButtons = [];

  UserInvalidFallback.init(form);

  function formatTimestamp(sec) {
    const s = Math.max(0, Math.floor(Number(sec) || 0));
    const mm = String(Math.floor(s / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  }

  function updateSourceTypeUI() {
    const selected = form.elements.source_type.value;
    const isUrl = selected === "url";
    urlGroup.classList.toggle("hidden", !isUrl);
    fileGroup.classList.toggle("hidden", isUrl);
    sourceUrlInput.required = isUrl;
    audioFileInput.required = !isUrl;
  }

  function updateLyricsModeUI() {
    const isManual = form.elements.lyrics_mode.value === "manual";
    manualLyricsGroup.classList.toggle("hidden", !isManual);
    lyricsManualRadio.setAttribute("aria-expanded", String(isManual));
    lyricsAutoRadio.setAttribute("aria-expanded", String(!isManual));
  }

  form.addEventListener("change", (e) => {
    if (e.target.name === "source_type") updateSourceTypeUI();
    if (e.target.name === "lyrics_mode") updateLyricsModeUI();
  });

  sourceUrlInput.addEventListener("input", () => {
    sourceUrlInput.setCustomValidity("");
    formFeedback.textContent = "";
  });

  function setDownloadLink(id, url) {
    const el = document.getElementById(id);
    if (!el) return;
    if (url) {
      el.href = url;
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
    }
  }

  function renderLyricsViewer(lines) {
    currentSongLines = Array.isArray(lines) ? lines : [];
    lyricButtons = [];
    lyricsList.replaceChildren();

    if (!currentSongLines.length) {
      const p = document.createElement("p");
      p.className = "muted-msg";
      p.textContent = "No timed lyric lines found for this track.";
      lyricsList.appendChild(p);
      return;
    }

    currentSongLines.forEach((ln, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "lyric-item";
      btn.dataset.index = String(idx);

      const top = document.createElement("div");
      top.className = "lyric-top-row";
      const nativeSpan = document.createElement("span");
      nativeSpan.className = "lyric-native";
      nativeSpan.dir = "auto";
      nativeSpan.textContent = ln.native || "";
      const timeSpan = document.createElement("span");
      timeSpan.className = "lyric-time";
      timeSpan.textContent = `${formatTimestamp(ln.start)} – ${formatTimestamp(ln.end)}`;
      top.append(nativeSpan, timeSpan);

      const phon = document.createElement("div");
      phon.className = "lyric-phonetic";
      phon.textContent = ln.phonetic || "";

      const trans = document.createElement("div");
      trans.className = "lyric-translation";
      trans.textContent = ln.translation || "";

      btn.append(top, phon, trans);
      btn.addEventListener("click", () => {
        if (!videoEl.classList.contains("hidden")) {
          videoEl.currentTime = Number(ln.start) || 0;
          videoEl.play().catch(() => {});
        }
      });

      lyricsList.appendChild(btn);
      lyricButtons.push(btn);
    });
  }

  videoEl.addEventListener("timeupdate", () => {
    const t = videoEl.currentTime;
    let activeIdx = -1;
    for (let i = 0; i < currentSongLines.length; i++) {
      const ln = currentSongLines[i];
      if (t >= ln.start && t <= ln.end + 0.15) {
        activeIdx = i;
        break;
      }
    }
    lyricButtons.forEach((btn, idx) => {
      const isAct = idx === activeIdx;
      if (isAct && !btn.classList.contains("active-lyric")) {
        btn.classList.add("active-lyric");
        btn.scrollIntoView({ block: "nearest", behavior: "smooth" });
      } else if (!isAct) {
        btn.classList.remove("active-lyric");
      }
    });
  });

  async function loadSongIntoPlayer(songFolder) {
    try {
      const resp = await fetch(`/api/songs/${encodeURIComponent(songFolder)}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const song = await resp.json();

      playerSubtitle.textContent = song.title || songFolder;
      const langCode = song.language || "auto";
      playerLangBadge.textContent = LANG_LABELS[langCode] || langCode;
      playerLangBadge.classList.remove("hidden");

      const urls = song.urls || {};
      if (urls.video_mp4) {
        videoEl.src = `${urls.video_mp4}?t=${song.updated_epoch || Date.now()}`;
        videoEl.classList.remove("hidden");
        emptyPlayer.classList.add("hidden");
      } else {
        videoEl.removeAttribute("src");
        videoEl.classList.add("hidden");
        emptyPlayer.classList.remove("hidden");
      }

      downloadBar.classList.remove("hidden");
      setDownloadLink("dl-video", urls.video_mp4);
      setDownloadLink("dl-inst", urls.instrumental_wav);
      setDownloadLink("dl-voc", urls.vocals_wav);
      setDownloadLink("dl-ass", urls.karaoke_ass);
      setDownloadLink("dl-lrc", urls.lyrics_lrc);
      setDownloadLink("dl-json", urls.lyrics_json);
      setDownloadLink("dl-log", urls.pipeline_log);

      renderLyricsViewer(song.lines || []);
    } catch (err) {
      formFeedback.textContent = `Could not load song '${songFolder}': ${err.message}`;
    }
  }

  async function refreshLibrary(autoSelectFirst = false) {
    try {
      const [songsResp, healthResp] = await Promise.all([
        fetch("/api/songs"),
        fetch("/api/health"),
      ]);

      if (healthResp.ok) {
        const h = await healthResp.json();
        const t = h.telemetry || {};
        healthText.textContent = `${(t.songs || []).length} songs in library · ${t.crashed_runs || 0} crashes`;
      }

      if (!songsResp.ok) return;
      const data = await songsResp.json();
      const songs = data.songs || [];
      songsGrid.replaceChildren();

      if (!songs.length) {
        const msg = document.createElement("p");
        msg.className = "muted-msg";
        msg.textContent = "No converted songs yet. Paste a YouTube link above to create your first karaoke track!";
        songsGrid.appendChild(msg);
        return;
      }

      songs.forEach((s) => {
        const card = document.createElement("article");
        card.className = "song-card";

        if (s.urls && s.urls.thumbnail) {
          const img = document.createElement("img");
          img.className = "song-thumb";
          img.src = s.urls.thumbnail;
          img.alt = "";
          img.loading = "lazy";
          card.appendChild(img);
        }

        const body = document.createElement("div");
        body.className = "song-body";

        const titleEl = document.createElement("div");
        titleEl.className = "song-title";
        titleEl.textContent = s.title || s.song_folder;

        const metaRow = document.createElement("div");
        metaRow.className = "song-meta-row";
        const langSpan = document.createElement("span");
        langSpan.textContent = LANG_LABELS[s.language] || s.language || "auto";
        const linesSpan = document.createElement("span");
        linesSpan.textContent = `· ${s.lines_count || 0} lines`;
        const durSpan = document.createElement("span");
        durSpan.textContent = s.duration_seconds ? `· ${formatTimestamp(s.duration_seconds)}` : "";
        metaRow.append(langSpan, linesSpan, durSpan);

        const actions = document.createElement("div");
        actions.className = "song-actions";
        const openBtn = document.createElement("button");
        openBtn.type = "button";
        openBtn.className = "secondary-btn";
        openBtn.textContent = "Open in Studio Player";
        openBtn.addEventListener("click", () => {
          loadSongIntoPlayer(s.song_folder);
          window.scrollTo({ top: 0, behavior: "smooth" });
        });
        actions.appendChild(openBtn);

        body.append(titleEl, metaRow, actions);
        card.appendChild(body);
        songsGrid.appendChild(card);
      });

      if (autoSelectFirst && songs[0] && videoEl.classList.contains("hidden")) {
        loadSongIntoPlayer(songs[0].song_folder);
      }
    } catch (_err) {
      // Ignore transient network errors on initial load
    }
  }

  function updateStepListUI(currentStep, status) {
    const curIdx = STEP_ORDER.indexOf(currentStep);
    Array.from(stepList.children).forEach((li, idx) => {
      li.classList.remove("step-active", "step-done");
      if (status === "completed" || (curIdx > idx)) {
        li.classList.add("step-done");
      } else if (curIdx === idx && status === "running") {
        li.classList.add("step-active");
      }
    });
  }

  async function pollJob(jobId) {
    try {
      const resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (!resp.ok) return;
      const job = await resp.json();

      jobStepLabel.textContent = job.progress_label || job.status;
      jobElapsed.textContent = `${job.elapsed_seconds || 0}s`;
      jobProgress.value = Number(job.progress_percent) || 5;
      jobLogTail.textContent = job.log_tail || "Starting pipeline...";
      jobLogTail.scrollTop = jobLogTail.scrollHeight;

      updateStepListUI(job.current_step, job.status);

      watchdogBadge.className = "badge";
      if (job.status === "failed" || job.watchdog === "crashed") {
        watchdogBadge.classList.add("badge-err");
        watchdogBadge.textContent = "Error Detected";
      } else if (job.watchdog === "frozen" || job.watchdog === "stall_warning") {
        watchdogBadge.classList.add("badge-warn");
        watchdogBadge.textContent = "Watchdog: Stall Alert";
      } else if (job.status === "completed") {
        watchdogBadge.classList.add("badge-ok");
        watchdogBadge.textContent = "Completed";
      } else {
        watchdogBadge.classList.add("badge-ok");
        watchdogBadge.textContent = "Watchdog: Healthy";
      }

      if (job.status === "completed") {
        clearInterval(activePollTimer);
        activePollTimer = null;
        submitBtn.disabled = false;
        submitBtn.textContent = "Create 1080p Karaoke Video";
        await refreshLibrary(false);
        await loadSongIntoPlayer(job.song_folder);
      } else if (job.status === "failed") {
        clearInterval(activePollTimer);
        activePollTimer = null;
        submitBtn.disabled = false;
        submitBtn.textContent = "Create 1080p Karaoke Video";
        formFeedback.textContent = job.error || "Conversion failed. Check the live execution log below.";
        await refreshLibrary(false);
      }
    } catch (_err) {
      // Keep polling across momentary hiccups
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    formFeedback.textContent = "";

    if (!form.checkValidity()) {
      form.reportValidity();
      return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Converting Song...";
    jobPanel.classList.remove("hidden");
    jobProgress.value = 4;
    jobStepLabel.textContent = "Submitting conversion job...";

    try {
      const formData = new FormData(form);
      const resp = await fetch("/api/jobs", {
        method: "POST",
        body: formData,
      });
      const payload = await resp.json();
      if (!resp.ok) {
        throw new Error(payload.detail || `Server returned ${resp.status}`);
      }

      if (activePollTimer) clearInterval(activePollTimer);
      await pollJob(payload.job_id);
      activePollTimer = setInterval(() => pollJob(payload.job_id), 1500);
    } catch (err) {
      submitBtn.disabled = false;
      submitBtn.textContent = "Create 1080p Karaoke Video";
      formFeedback.textContent = err.message;
    }
  });

  refreshLibraryBtn.addEventListener("click", () => refreshLibrary(false));

  updateSourceTypeUI();
  updateLyricsModeUI();
  refreshLibrary(true);
})();
