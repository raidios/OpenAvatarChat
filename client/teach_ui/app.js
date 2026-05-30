/* eslint-disable */
// 单页示教/调试 UI。无构建步骤，直接由 FastAPI StaticFiles 暴露。
//
// 顶层逻辑：
// - 一条 /ws/state WebSocket 持续接收 5Hz 状态快照，刷新所有面板的 readonly 显示。
// - REST 接口处理"动作类"请求（录制、保存、播放、标定、绑定、debug say）。
// - 摄像头：用 <img src> + cache busting，省得自己写解码逻辑。
// - 麦克风监听：单独一条 /ws/mic WebSocket，二进制是 16k mono int16 PCM；用
//   AudioContext + AudioBuffer 直接以 16k 采样率塞进 destination；多数现代
//   浏览器会自动重采样到设备速率。

(() => {
  const $ = (id) => document.getElementById(id);
  const fmt = (v, digits = 2) => (typeof v === "number" ? v.toFixed(digits) : "—");

  // ---------------- /ws/state ----------------
  let stateWs = null;
  let lastState = null;
  function ensureStateWs() {
    if (stateWs && stateWs.readyState <= 1) return;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    stateWs = new WebSocket(`${proto}//${location.host}/ws/state`);
    stateWs.onopen = () => setIndicator("ws connected", "on");
    stateWs.onclose = () => {
      setIndicator("ws closed", "off");
      setTimeout(ensureStateWs, 1500);
    };
    stateWs.onerror = () => setIndicator("ws error", "off");
    stateWs.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "state") onState(msg);
      } catch (e) { /* ignore */ }
    };
  }
  function setIndicator(text, cls) {
    const el = $("conn-indicator");
    el.textContent = text;
    el.className = `indicator ${cls}`;
  }

  function onState(s) {
    lastState = s;
    const t = s.telemetry || {};
    if (t.mcu) {
      $("st-bat").textContent = `${fmt(t.mcu.bat_voltage, 2)} V`;
      $("st-yaw").textContent = `${fmt(t.mcu.yaw_deg, 1)}°`;
      $("st-vel").textContent = `${t.mcu.vel[0]}/${t.mcu.vel[1]}/${t.mcu.vel[2]} mm·s/mrad·s`;
    }
    if (t.ps2) {
      $("st-ljoy").textContent = `${t.ps2.ljoy_lr.toString(16)}/${t.ps2.ljoy_ud.toString(16)}`;
      $("st-rjoy").textContent = `${t.ps2.rjoy_lr.toString(16)}/${t.ps2.rjoy_ud.toString(16)}`;
      $("st-ps2btn").textContent = `b1=${t.ps2.btn1.toString(16)} b2=${t.ps2.btn2.toString(16)}`;
    }
    if (t.ctrl) {
      $("st-ctrl").textContent = `${t.ctrl.source_name} [${t.ctrl.flags_str}]`;
    }
    if (s.chat) {
      $("st-chat").textContent =
        `${s.chat.connected ? "connected" : "off"}  wake=${s.chat.wake_session ? "Y" : "N"}  vad=${s.chat.vad_enabled ? "Y" : "N"}`;
      $("st-expr").textContent = s.chat.override_expression || "(none)";
    }
    if (s.player) {
      $("st-motion").textContent =
        s.player.current_clip_id
          ? `playing ${s.player.current_clip_id} (${s.player.current_progress_ms}/${s.player.current_total_ms} ms)  +${s.player.queued} queued`
          : (s.player.queued > 0 ? `${s.player.queued} queued` : "idle");
    }
    if (s.recorder) {
      const r = s.recorder;
      $("rec-status").textContent =
        `${r.state}  samples=${r.sample_count}  dur=${r.duration_ms} ms`;
      if (r.state === "recording") {
        $("rec-start").disabled = true;
        $("rec-stop").disabled = false;
      } else {
        $("rec-start").disabled = false;
        $("rec-stop").disabled = true;
      }
    }
    if (s.calib) {
      const c = s.calib;
      $("calib-progress").textContent =
        `${c.state}  captured=${c.captured}/${c.min_frames}  corners=${c.last_corner_count}  cooldown=${fmt(c.auto_cooldown_left, 1)}s${c.error_msg ? "  ERR: " + c.error_msg : ""}`;
      const running = c.state === "running";
      $("ca-capture").disabled = !running;
      $("ca-finish").disabled = !running;
      $("ca-abort").disabled = !running;
      $("ca-start").disabled = running;
    } else {
      $("calib-progress").textContent = "未启动";
    }
  }

  // ---------------- camera ----------------
  let camRunning = false;
  function refreshCam() {
    const overlay = $("cam-overlay").checked ? 1 : 0;
    const fps = Math.max(1, Math.min(15, parseInt($("cam-fps").value, 10) || 10));
    if (camRunning) {
      $("cam-img").src = `/api/camera/stream?overlay=${overlay}&fps=${fps}&_=${Date.now()}`;
    }
  }
  $("cam-start").onclick = () => { camRunning = true; refreshCam(); };
  $("cam-stop").onclick = () => {
    camRunning = false;
    // 触发浏览器关闭对应 MJPEG 连接；空 src 在 Chrome 会显示破图，所以塞 1x1。
    $("cam-img").src = "data:image/gif;base64,R0lGODlhAQABAAAAACw=";
  };
  $("cam-overlay").onchange = refreshCam;
  $("cam-fps").onchange = refreshCam;
  $("cam-snap").onclick = async () => {
    const overlay = $("cam-overlay").checked ? 1 : 0;
    const url = `/api/camera/snapshot?overlay=${overlay}&_=${Date.now()}`;
    window.open(url, "_blank");
  };

  // ---------------- calibration ----------------
  async function loadCalibCurrent() {
    try {
      const r = await fetch("/api/calib/current");
      if (!r.ok) return;
      const d = await r.json();
      if (!d.exists) {
        $("calib-current").textContent = `当前 calib：(不存在，path=${d.path})`;
        return;
      }
      const c = d.data || {};
      $("calib-current").textContent =
        `当前 calib：fx=${fmt(c.fx)} fy=${fmt(c.fy)} cx=${fmt(c.cx)} cy=${fmt(c.cy)}  RMS=${fmt(c.rms_error, 4)}  res=${(c.resolution || []).join("x")}`;
    } catch (e) {}
  }
  loadCalibCurrent();

  $("ca-start").onclick = async () => {
    const body = {
      rows: +$("ca-rows").value,
      cols: +$("ca-cols").value,
      square_mm: +$("ca-sq").value,
      min_frames: +$("ca-min").value,
      auto_interval_s: +$("ca-int").value,
    };
    const r = await fetch("/api/calib/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) alert("start failed: " + await r.text());
  };
  $("ca-capture").onclick = async () => {
    const r = await fetch("/api/calib/capture", { method: "POST" });
    if (!r.ok) alert("capture failed: " + await r.text());
  };
  $("ca-finish").onclick = async () => {
    const r = await fetch("/api/calib/finish", { method: "POST" });
    const d = await r.json().catch(() => ({}));
    if (d.result) {
      $("calib-result").textContent = JSON.stringify(d.result, null, 2);
      loadCalibCurrent();
    } else {
      $("calib-result").textContent = JSON.stringify(d, null, 2);
    }
  };
  $("ca-abort").onclick = async () => {
    await fetch("/api/calib/abort", { method: "POST" });
  };

  // ---------------- mic monitor ----------------
  let micWs = null;
  let micAudioCtx = null;
  let micNextPlay = 0;
  function micLog(msg) { $("mic-info").textContent = msg; }
  $("mic-toggle").onclick = async () => {
    if (micWs && micWs.readyState <= 1) {
      micWs.close();
      micWs = null;
      $("mic-toggle").textContent = "开启监听";
      micLog("已停止");
      if (micAudioCtx) { await micAudioCtx.close(); micAudioCtx = null; }
      return;
    }
    try {
      micAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      // 0 表示初始化为"立即开始播"，每收到一个 100ms block 就 schedule 到
      // micNextPlay；如果落后则跳到 audioCtx.currentTime 重新对齐。
      micNextPlay = micAudioCtx.currentTime;
    } catch (e) {
      alert("AudioContext init failed: " + e);
      return;
    }
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    micWs = new WebSocket(`${proto}//${location.host}/ws/mic`);
    micWs.binaryType = "arraybuffer";
    $("mic-toggle").textContent = "关闭监听";
    micLog("connecting…");
    micWs.onopen = () => micLog("已连接，等待 PCM…");
    micWs.onclose = (ev) => {
      micLog(`已断开 (${ev.code}${ev.reason ? " " + ev.reason : ""})`);
      $("mic-toggle").textContent = "开启监听";
    };
    micWs.onerror = () => micLog("error");
    micWs.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "mic_info") {
            micLog(`PCM ${msg.sample_rate} Hz, ${msg.channels}ch, ${msg.format}`);
          } else if (msg.type === "mic_stats") {
            const peak = msg.peak_dbfs == null ? "-inf" : msg.peak_dbfs.toFixed(1);
            $("mic-meter-text").textContent = `rms=${msg.rms}  peak=${peak} dB ${msg.muted ? "🔇" : ""}`;
            const level = msg.peak_dbfs == null ? 0 : Math.min(100, Math.max(0, (msg.peak_dbfs + 60) * 100 / 60));
            $("mic-meter-bar").style.width = `${level}%`;
          }
        } catch (e) {}
        return;
      }
      // binary: int16 PCM @ 16k mono
      const i16 = new Int16Array(ev.data);
      const f32 = new Float32Array(i16.length);
      for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768.0;
      const buf = micAudioCtx.createBuffer(1, f32.length, 16000);
      buf.copyToChannel(f32, 0);
      const src = micAudioCtx.createBufferSource();
      src.buffer = buf;
      src.connect(micAudioCtx.destination);
      const now = micAudioCtx.currentTime;
      if (micNextPlay < now + 0.02) micNextPlay = now + 0.02;
      src.start(micNextPlay);
      micNextPlay += buf.duration;
    };
  };

  // ---------------- recording ----------------
  $("rec-start").onclick = () => fetch("/api/clips/record/start", { method: "POST" });
  $("rec-stop").onclick = async () => {
    await fetch("/api/clips/record/stop", { method: "POST" });
    refreshPreview();
  };
  $("rec-discard").onclick = async () => {
    await fetch("/api/clips/record/discard", { method: "POST" });
    refreshPreview();
  };

  let previewSamples = [];
  async function refreshPreview() {
    try {
      const r = await fetch("/api/clips/record/preview");
      const d = await r.json();
      previewSamples = d.samples || [];
      $("crop-start").value = 0;
      $("crop-end").value = d.duration_ms || 0;
      drawPreview();
    } catch (e) {}
  }
  function drawPreview() {
    const c = $("rec-canvas");
    const ctx = c.getContext("2d");
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "#444"; ctx.beginPath();
    ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
    if (!previewSamples.length) {
      ctx.fillStyle = "#888";
      ctx.fillText("(无数据)", 10, 20);
      return;
    }
    const maxT = previewSamples[previewSamples.length - 1].t_ms || 1;
    const maxV = 1500;
    const drawCh = (key, color) => {
      ctx.strokeStyle = color; ctx.beginPath();
      previewSamples.forEach((s, i) => {
        const x = (s.t_ms / maxT) * W;
        const y = H / 2 - (s[key] / maxV) * (H / 2);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };
    drawCh("vx", "#5dadff");
    drawCh("vy", "#5ad88a");
    drawCh("vw", "#ef9a5a");
    // crop markers
    const cs = +$("crop-start").value;
    const ce = +$("crop-end").value;
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.fillRect((cs / maxT) * W, 0, ((ce - cs) / maxT) * W, H);
  }
  $("crop-start").addEventListener("input", drawPreview);
  $("crop-end").addEventListener("input", drawPreview);

  $("rec-save").onclick = async () => {
    const body = {
      name: $("rec-name").value.trim() || `clip_${Date.now()}`,
      crop_start_ms: +$("crop-start").value || 0,
      crop_end_ms: +$("crop-end").value || null,
    };
    const r = await fetch("/api/clips/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      alert("save failed: " + await r.text());
      return;
    }
    refreshClips();
  };

  // ---------------- clip library ----------------
  async function refreshClips() {
    const list = $("clip-list");
    const r = await fetch("/api/clips");
    const clips = await r.json();
    list.innerHTML = "";
    if (!clips.length) {
      list.innerHTML = "<p class='hint'>(尚无 clip。先录制一段再保存)</p>";
    } else {
      for (const c of clips) {
        const el = document.createElement("div");
        el.className = "clip";
        el.innerHTML = `
          <span class="clip-name">${c.name}</span>
          <span class="clip-meta">${c.duration_ms}ms · ${c.sample_count} samples</span>
          <button data-act="play">▶ play</button>
          <button data-act="rename">rename</button>
          <button data-act="delete" class="danger">delete</button>
        `;
        el.querySelector('[data-act="play"]').onclick = () => fetch(`/api/clips/${c.id}/play`, { method: "POST" });
        el.querySelector('[data-act="rename"]').onclick = async () => {
          const nm = prompt("rename to", c.name);
          if (!nm) return;
          await fetch(`/api/clips/${c.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: nm }) });
          refreshClips();
        };
        el.querySelector('[data-act="delete"]').onclick = async () => {
          if (!confirm("delete?")) return;
          await fetch(`/api/clips/${c.id}`, { method: "DELETE" });
          refreshClips();
        };
        list.appendChild(el);
      }
    }
    // refresh bindings too — they need the clip list for the <select>
    refreshBindings(clips);
  }

  // ---------------- bindings ----------------
  const EMOTIONS = ["happy", "shy", "apologize", "scared"];
  const EXPRESSIONS = ["happy", "joy", "dead", "frown", "listen", "tag", "blink"];

  async function refreshBindings(clipsHint) {
    const r = await fetch("/api/bindings");
    const data = await r.json();
    const clips = clipsHint || (await (await fetch("/api/clips")).json());
    const root = $("binding-cards");
    root.innerHTML = "";
    for (const key of EMOTIONS) {
      const b = data[key] || { clip_id: null, expression_id: "" };
      const card = document.createElement("div");
      card.className = "binding-card";
      const clipOpts = `<option value="">(none, 占位 500ms)</option>` +
        clips.map(c => `<option value="${c.id}"${c.id === b.clip_id ? " selected" : ""}>${c.name}</option>`).join("");
      const exprOpts = EXPRESSIONS.map(e =>
        `<option value="${e}"${e === b.expression_id ? " selected" : ""}>${e}</option>`).join("");
      card.innerHTML = `
        <h3>${key}</h3>
        <label>clip</label>
        <select data-field="clip">${clipOpts}</select>
        <label>expression</label>
        <select data-field="expr">${exprOpts}</select>
        <div class="row">
          <button data-act="save" class="primary">保存</button>
          <button data-act="trigger">试触发</button>
        </div>
      `;
      card.querySelector('[data-act="save"]').onclick = async () => {
        const clipSel = card.querySelector('[data-field="clip"]').value || "";
        const exprSel = card.querySelector('[data-field="expr"]').value;
        await fetch(`/api/bindings/${key}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ clip_id: clipSel, expression_id: exprSel }),
        });
      };
      card.querySelector('[data-act="trigger"]').onclick =
        () => fetch(`/api/bindings/${key}/trigger`, { method: "POST" });
      root.appendChild(card);
    }
  }

  $("say-btn").onclick = async () => {
    const text = $("say-text").value.trim();
    if (!text) return;
    await fetch("/api/say", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  };

  // ---------------- init ----------------
  ensureStateWs();
  refreshClips();
  setInterval(refreshPreview, 1000);
})();
