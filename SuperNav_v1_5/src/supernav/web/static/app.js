"use strict";

const $ = (id) => document.getElementById(id);
let selected = "", stream = null, latest = null, displayed = null;
let paused = false, replayIndex = 0, replayTimer = null, mapZoom = 1;
let sessionSignature = "", eventSignature = "", imageURL = "", imageGeneration = 0;
let imageLoading = false, pendingImageURL = "";
let connected = false;
let selectedOverlay = null, selectedCapture = null, perceptionSignature = "";
let latestTrace = null, traceSignature = "", traceBusy = false;
const number = (value, digits = 0) => Number.isFinite(value) ? value.toFixed(digits) : "—";
const title = (s) => String(s || "").replace(/^hab_/, "").replaceAll("_", " ");
const eventTime = (stamp) => {
  const date = new Date(typeof stamp === "number" ? stamp > 1e12 ? stamp : stamp * 1000 : stamp);
  return Number.isNaN(date.getTime()) ? String(stamp) : date.toLocaleString();
};
const elapsed = (seconds) => {
  seconds = Math.max(0, Math.floor(seconds || 0));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
};

function connection(ok, message) {
  connected = ok;
  $("connection").textContent = message;
  $("connection-dot").className = `status-dot ${ok ? "connected" : "error"}`;
}

function notice(message) {
  $("notice").hidden = !message;
  $("notice").textContent = message;
}

async function sessions() {
  try {
    const response = await fetch("/api/sessions", {signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error("Session list unavailable");
    const {sessions: rows} = await response.json();
    const signature = JSON.stringify(rows.map((r) => [r.id, r.scene, r.status, r.source]));
    if (signature !== sessionSignature) {
      sessionSignature = signature;
      $("session").replaceChildren();
      if (!rows.length) $("session").add(new Option("Waiting for a session", ""));
      for (const row of rows) {
        const label = row.source === "demo" ? "Demo · Illustrated apartment" : `${row.source === "recording" ? "Recording" : row.status === "running" ? "Live" : title(row.status)} · ${row.scene || row.session_id.slice(0, 8)}`;
        $("session").add(new Option(label, row.id));
      }
      const requested = new URLSearchParams(location.hash.slice(1)).get("session");
      const chosen = rows.find((r) => r.id === selected) || rows.find((r) => r.id === requested) || rows[0];
      if (chosen) {
        $("session").value = chosen.id;
        if (selected !== chosen.id) selectSession(chosen.id);
      } else if (selected) {
        selectSession("");
      }
    }
    if (!rows.length) connection(true, "Ready to connect");
    else if (!stream) selectSession($("session").value);
  } catch (error) {
    connection(false, "Server disconnected");
    notice("The viewer server is unreachable. Keeping the last view while reconnecting…");
  }
}

function selectSession(key) {
  stream?.close();
  stream = null;
  stopReplay();
  selected = key;
  paused = false;
  latest = displayed = null;
  mapZoom = 1;
  eventSignature = "";
  imageURL = "";
  imageGeneration++;
  imageLoading = false;
  pendingImageURL = "";
  selectedOverlay = selectedCapture = null;
  perceptionSignature = traceSignature = "";
  latestTrace = null;
  renderPerception({});
  renderTrace({events: [], status: "waiting"});
  $("camera").hidden = true;
  $("demo-canvas").hidden = true;
  $("empty-camera").hidden = false;
  $("camera-overlay").hidden = true;
  $("paused-label").hidden = true;
  $("replay-controls").hidden = true;
  $("pause").disabled = true;
  $("follow").disabled = true;
  notice("");
  drawMap(null);
  for (const id of ["scene", "agent", "steps", "distance", "elapsed", "claim", "pose-x", "pose-y", "pose-z", "pose-heading"]) $(id).textContent = "—";
  $("instruction").textContent = "Your next exploration starts here.";
  $("action").textContent = "Waiting for agent activity";
  $("session-state").textContent = "Waiting";
  $("state-note").textContent = "No session selected";
  $("source-badge").textContent = "NO SESSION";
  $("source-badge").className = "badge";
  $("activity").replaceChildren();
  $("event-count").textContent = "0 EVENTS";
  $("frame-info").textContent = "Waiting for observations";
  if (!key) return;
  history.replaceState(null, "", `#session=${encodeURIComponent(key)}`);
  connection(false, "Connecting to session…");
  stream = new EventSource(`/api/sessions/${encodeURIComponent(key)}/events`);
  stream.onopen = () => { if (key === selected) connection(true, "Stream connected"); };
  stream.addEventListener("snapshot", (event) => {
    if (key !== selected) return;
    try {
      const next = JSON.parse(event.data);
      const first = !latest;
      latest = next;
      connection(true, next.source === "recording" ? "Recording connected" : "Stream connected");
      if (first && next.source === "recording") replayIndex = Math.max(0, next.frames.length - 1);
      if (!paused) render(next);
    } catch (error) {
      notice("This session update could not be displayed. Waiting for the next observation…");
    }
  });
  stream.addEventListener("unavailable", () => {
    stream?.close(); stream = null;
    connection(false, "Session unavailable");
    notice("This session is no longer available. Choose another session to continue.");
  });
  stream.onerror = () => {
    if (key !== selected) return;
    connection(false, "Reconnecting…");
    notice("Live connection interrupted. The last observation is preserved; reconnecting automatically…");
  };
}

function render(data) {
  displayed = data;
  const recording = data.source === "recording", demo = data.source === "demo";
  $("scene").textContent = data.scene || "Unknown scene";
  $("agent").textContent = data.agent || "Connected agent";
  $("source-badge").textContent = demo ? "DEMONSTRATION" : recording ? "RECORDING" : data.status === "running" ? "LIVE SESSION" : title(data.status).toUpperCase();
  $("source-badge").className = `badge ${demo ? "demo" : !recording && data.status === "running" ? "live" : ""}`;
  $("steps").textContent = number(data.step);
  $("distance").replaceChildren(document.createTextNode(`${number(data.path_length_m, 2)} `));
  const unit = document.createElement("small"); unit.textContent = "m"; $("distance").append(unit);
  $("session-state").textContent = demo ? "Demo" : title(data.status).replace(/^./, (s) => s.toUpperCase());
  $("state-note").textContent = recording ? "Recorded session · final statistics" : data.status === "running" ? "Receiving agent observations" : data.status === "interrupted" ? "Producer stopped before session close" : "Session has ended";
  $("instruction").textContent = data.instruction || `Exploring ${data.scene || "the scene"}.`;
  const claim = typeof data.agent_claim === "object" && data.agent_claim !== null ? data.agent_claim.task_outcome || data.agent_claim.status || data.agent_claim.outcome : data.agent_claim;
  $("claim").textContent = claim ? title(claim) : "No completion claim";
  $("action").textContent = title(data.action) || "Waiting for an action";
  $("map-mode").textContent = recording ? "Complete recorded path" : data.trail_truncated ? "Recent trail" : "Live trail";
  $("view-label").textContent = demo ? "ILLUSTRATED DEMO · NOT SIMULATOR OUTPUT" : recording ? "RECORDED OBSERVATION · RGB" : "FIRST PERSON · RGB";
  $("replay-controls").hidden = !recording;
  $("pause").disabled = recording ? !data.frames.length : false;
  $("follow").disabled = recording;
  $("follow").classList.toggle("selected", !recording && !paused);
  $("follow").querySelector("span").textContent = recording ? "Recorded" : paused ? "Follow live" : "Following live";
  updatePause();
  const position = data.pose?.position || [];
  ["x", "y", "z"].forEach((axis, i) => { $("pose-" + axis).textContent = number(position[i], 2); });
  $("pose-heading").textContent = Number.isFinite(data.pose?.heading_deg) ? `${number(data.pose.heading_deg, 0)}°` : "—";
  if (recording) {
    $("replay").max = Math.max(0, data.frames.length - 1);
    replayIndex = Math.min(replayIndex, data.frames.length - 1);
    renderReplay();
  } else if (demo) {
    $("camera").hidden = true;
    $("demo-canvas").hidden = false;
    $("empty-camera").hidden = true;
    $("camera-overlay").hidden = false;
    drawDemo(data);
    $("frame-info").textContent = `SAMPLE STEP ${String(data.step).padStart(3, "0")}`;
  } else {
    $("demo-canvas").hidden = true;
    loadFrame(data.image_url);
    $("frame-info").textContent = `STEP ${number(data.step)} · RGB ${number(data.frame_revision)}`;
  }
  if (!recording) renderPerception(data);
  const warnings = [];
  if (demo) warnings.push("Demo mode · A procedural illustration of the interface. No simulator or model is running.");
  if (data.capture_error) warnings.push("Camera capture failed; showing the last available image. Agent execution continues.");
  if (data.status === "interrupted") warnings.push("The capture process stopped without closing this session. These are its last recorded observations.");
  notice(warnings.join(" "));
  drawMap(data);
  renderEvents(data);
  updateTime();
}

function loadFrame(url) {
  pendingImageURL = url;
  if (!url) {
    $("camera").hidden = true;
    $("empty-camera").hidden = false;
    $("camera-overlay").hidden = true;
    return;
  }
  // One request at a time: a slow connection must still display frames instead
  // of continuously discarding every response when newer snapshots arrive.
  if (imageLoading || url === imageURL) return;
  imageLoading = true;
  const generation = imageGeneration;
  const preload = new Image();
  preload.onload = () => {
    if (generation !== imageGeneration) return;
    imageLoading = false;
    if (!paused) {
      imageURL = url;
      preload.id = "camera";
      preload.alt = "First-person observation from the navigation agent";
      $("camera").replaceWith(preload);
      $("empty-camera").hidden = true;
      $("camera-overlay").hidden = false;
      if (pendingImageURL !== url) loadFrame(pendingImageURL);
    }
  };
  preload.onerror = () => {
    if (generation !== imageGeneration) return;
    imageLoading = false;
    imageURL = "";
    notice("This image is unavailable. Waiting for another frame…");
    if (pendingImageURL !== url && !paused) loadFrame(pendingImageURL);
  };
  preload.src = url;
}

function renderReplay() {
  if (!displayed?.frames) return;
  const frame = displayed.frames[replayIndex];
  $("replay").value = Math.max(0, replayIndex);
  $("replay-count").textContent = `${frame ? replayIndex + 1 : 0} / ${displayed.frames.length}`;
  $("frame-info").textContent = "Recorded frames · 8 fps playback";
  loadFrame(frame?.url);
  $("frame-age").textContent = frame ? new Date(frame.timestamp * 1000).toLocaleTimeString() : "No recorded images";
  renderPerception(displayed, frame?.timestamp);
}

function updatePause() {
  const recording = displayed?.source === "recording";
  $("pause-icon").textContent = recording ? (replayTimer ? "Ⅱ" : "▷") : (paused ? "▷" : "Ⅱ");
  $("pause-text").textContent = recording ? (replayTimer ? "Pause replay" : "Play recording") : (paused ? "Resume view" : "Pause view");
  $("paused-label").hidden = !paused;
}

function stopReplay() {
  clearInterval(replayTimer);
  replayTimer = null;
  updatePause();
}

$("pause").onclick = () => {
  if (displayed?.source === "recording") {
    if (replayTimer) return stopReplay();
    if (replayIndex >= displayed.frames.length - 1) replayIndex = -1;
    replayTimer = setInterval(() => {
      replayIndex++;
      renderReplay();
      if (replayIndex >= displayed.frames.length - 1) stopReplay();
    }, 125);
    updatePause();
  } else {
    paused = !paused;
    if (!paused && latest) { imageURL = ""; render(latest); if (latestTrace) renderTrace(latestTrace); }
    updatePause();
    $("follow").classList.toggle("selected", !paused);
    $("follow").querySelector("span").textContent = paused ? "Follow live" : "Following live";
  }
};
$("follow").onclick = () => { paused = false; imageURL = ""; if (latest) render(latest); if (latestTrace) renderTrace(latestTrace); };
$("replay").oninput = () => { stopReplay(); replayIndex = Number($("replay").value); renderReplay(); };
$("session").onchange = () => selectSession($("session").value);
$("fullscreen").onclick = async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $("camera-stage").requestFullscreen(); }
  catch { notice("Full-screen viewing is unavailable in this browser."); }
};
for (const id of ["help-button", "empty-guide"]) $(id).onclick = () => $("help").showModal();
$("help").onclick = (event) => { if (event.target === $("help")) { const r = $("help").getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) $("help").close(); } };

function renderEvents(data) {
  const events = data.events || [];
  const signature = JSON.stringify(events);
  if (signature === eventSignature) return;
  eventSignature = signature;
  $("activity").replaceChildren();
  $("event-count").textContent = `${events.length} RECENT EVENTS`;
  if (!events.length) {
    const li = document.createElement("li"); li.className = "activity-empty";
    li.textContent = "Observations and actions will appear as the agent explores.";
    $("activity").append(li);
  }
  for (const event of [...events].reverse()) {
    const li = document.createElement("li"), clock = document.createElement("time"), dot = document.createElement("i");
    clock.textContent = elapsed(event.timestamp - data.started_at);
    dot.className = `event-dot ${event.ok === false ? "failed" : ""}`;
    const main = document.createElement("div"), heading = document.createElement("strong"), sub = document.createElement("small"), step = document.createElement("span");
    main.className = "event-main";
    heading.textContent = title(event.action);
    sub.textContent = event.ok === false ? "Returned an error" : event.phase === "started" ? "Action started" : event.phase === "finished" ? "Action returned" : title(event.phase);
    main.append(heading, sub);
    step.className = "event-step"; step.textContent = Number.isFinite(event.step) ? `STEP ${event.step}` : "";
    li.append(clock, dot, main, step); $("activity").append(li);
  }
}

function updateTime() {
  if (!displayed) return;
  const now = displayed.ended_at || (paused || !connected || displayed.status !== "running" ? displayed.updated_at : Date.now() / 1000);
  $("elapsed").textContent = elapsed(now - displayed.started_at);
  if (displayed.source === "demo") $("frame-age").textContent = "DEMO";
  else if (displayed.source !== "recording") {
    const age = displayed.frame_at ? Math.max(0, Date.now() / 1000 - displayed.frame_at) : null;
    $("frame-age").textContent = age === null ? "Awaiting frame" : age < 2 ? "JUST NOW" : `${Math.floor(age)}s SINCE FRAME`;
  }
}

function expandImage(url, label) {
  if (!url) return;
  $("expanded-image").src = url;
  $("image-dialog-label").textContent = label;
  $("image-dialog").showModal();
}

function renderPerception(data, frameTime) {
  const groups = data.panoramas || [], overlays = data.overlays || [];
  const eligible = frameTime ? overlays.filter((o) => o.timestamp <= frameTime) : overlays;
  const overlay = selectedOverlay !== null ? overlays.find((o) => o.id === selectedOverlay) : eligible.at(-1);
  const availableGroups = frameTime ? groups.filter((g) => g.timestamp <= frameTime) : groups;
  const group = selectedCapture !== null ? groups.find((g) => g.capture_seq === selectedCapture) : availableGroups.at(-1);
  const signature = JSON.stringify([group, overlay, overlays.map((o) => o.id)]);
  if (signature === perceptionSignature) return;
  perceptionSignature = signature;
  if (latestTrace) renderTrace(latestTrace);
  $("surround").replaceChildren();
  for (const [direction, arrow] of [["front", "↑"], ["right", "→"], ["back", "↓"], ["left", "←"]]) {
    const button = document.createElement("button"), label = document.createElement("span"), missing = document.createElement("span");
    const view = group?.views?.[direction];
    button.className = `surround-view ${overlay?.capture_seq === group?.capture_seq && overlay?.direction === direction ? "selected" : ""}`;
    button.setAttribute("aria-label", `Expand ${direction} observation`);
    label.className = "view-direction"; label.textContent = `${arrow} ${direction.toUpperCase()}`;
    missing.className = "view-missing"; missing.textContent = view ? "Loading…" : group ? "View not captured" : "Awaiting observation";
    button.append(missing);
    if (view?.image_url) {
      const img = new Image(); img.alt = `${direction} observation, capture ${group.capture_seq}`;
      img.onload = () => {missing.hidden = true;};
      img.onerror = () => {img.hidden = true;missing.hidden = false;missing.textContent = "Image unavailable";};
      img.src = view.image_url;button.append(img);
      button.onclick = () => expandImage(view.image_url, `${direction.toUpperCase()} · CAPTURE ${group.capture_seq} · ${view.image_ref || ""}`);
    } else button.disabled = true;
    button.append(label); $("surround").append(button);
  }
  $("surround-meta").textContent = group ? `Capture ${group.capture_seq} · ${new Date(group.timestamp * 1000).toLocaleTimeString()}${selectedCapture !== null ? " · decision view" : " · latest observation"}` : selectedCapture !== null ? `Capture ${selectedCapture} is no longer in the live history.` : "Waiting for the agent’s panorama";
  $("overlay-select").replaceChildren(new Option("Latest available point", ""));
  for (const entry of [...overlays].reverse()) $("overlay-select").add(new Option(`#${entry.id} · ${String(entry.direction || "view").toUpperCase()} · capture ${entry.capture_seq ?? "—"}`, String(entry.id)));
  $("overlay-select").value = selectedOverlay === null ? "" : String(selectedOverlay);
  $("point-overlay").hidden = !overlay;
  $("overlay-empty").hidden = !!overlay;
  if (overlay) {
    $("point-overlay").onerror = () => {$("point-overlay").hidden = true;$("overlay-empty").hidden = false;$("overlay-empty").textContent = "This overlay is no longer available.";};
    $("point-overlay").src = overlay.image_url;
    $("overlay-meta").textContent = `${String(overlay.direction || "view").toUpperCase()} · capture ${overlay.capture_seq ?? "—"}${overlay.point ? ` · point [${overlay.point.join(", ")}]` : ""} · ${new Date(overlay.timestamp * 1000).toLocaleTimeString()}`;
    $("overlay-image-button").onclick = () => expandImage(overlay.image_url, `${overlay.tool || "POINT OVERLAY"} · ${overlay.image_ref || ""}`);
  } else {
    $("overlay-empty").textContent = "The agent’s marked target will appear here.";
    $("overlay-meta").textContent = "Overlay and source view stay linked to the same capture.";
    $("overlay-image-button").onclick = null;
  }
}

function chooseOverlay(id) {
  selectedOverlay = id;
  selectedCapture = displayed?.overlays?.find((o) => o.id === id)?.capture_seq ?? null;
  perceptionSignature = "";
  renderPerception(displayed || {});
}
$("overlay-select").onchange = () => chooseOverlay($("overlay-select").value ? Number($("overlay-select").value) : null);
$("latest-observation").onclick = () => {selectedOverlay = selectedCapture = null;perceptionSignature = "";renderPerception(displayed || {});};

async function refreshTrace() {
  if (!selected || traceBusy) return;
  const key = selected;
  traceBusy = true;
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(key)}/trace`, {signal: AbortSignal.timeout(5000)});
    if (!response.ok) return;
    const trace = await response.json();
    if (key !== selected) return;
    latestTrace = trace;
    if (!paused) renderTrace(trace);
  } catch { /* The main connection indicator already reports network failures. */ }
  finally {traceBusy = false;}
}

function renderTrace(trace) {
  const filter = $("trace-filter").value;
  const signature = JSON.stringify([trace, filter, displayed?.overlays?.map((o) => o.id)]);
  if (signature === traceSignature) return;
  traceSignature = signature;
  const thought = (trace.events || []).findLast((event) => event.kind === "thought");
  $("latest-thought").hidden = !thought;
  $("latest-thought-text").textContent = thought?.text || "";
  $("latest-thought-label").textContent = thought?.channel === "agent message" ? "LATEST AGENT NOTE" : "LATEST RECORDED REASONING";
  const open = new Set([...$("trace").querySelectorAll("details[open]")].map((el) => el.dataset.key));
  const scroll = $("trace").scrollTop;
  const follow = $("trace").scrollHeight - $("trace").clientHeight - scroll < 40;
  const events = (trace.events || []).filter((event) => filter === "all" || event.kind === filter);
  $("trace-status").textContent = trace.status === "available" ? `${trace.events.length} recent events · recorded agent output` : "Waiting for the linked agent’s output";
  $("trace").replaceChildren();
  if (!events.length) {
    const p = document.createElement("p"); p.className = "trace-empty";
    p.textContent = filter === "tool" ? "Tool calls will appear here with their arguments and results." : "Agent messages and recorded reasoning appear when the client emits them. No reasoning is inferred.";
    $("trace").append(p);
  }
  for (const event of events) {
    const article = document.createElement("article"); article.className = "trace-entry";
    if (event.kind === "thought") {
      const label = document.createElement("div"), text = document.createElement("p");
      label.className = "trace-label";label.textContent = String(event.channel || "Agent message").toUpperCase();
      text.className = "trace-message";text.textContent = event.text;
      article.append(label, text);
    } else {
      const details = document.createElement("details"), summary = document.createElement("summary"), name = document.createElement("span"), state = document.createElement("span");
      details.className = "trace-tool";details.dataset.key = event.id;details.open = open.has(event.id);
      name.textContent = event.name;state.className = `trace-state ${event.status === "running" ? "running" : event.status === "error" ? "error" : ""}`;
      state.textContent = event.status === "completed" ? "RETURNED" : String(event.status).toUpperCase();summary.append(name,state);details.append(summary);
      for (const [heading, value] of [["ARGUMENTS",event.arguments], ["RESULT",event.result]]) {
        const h = document.createElement("h3"), pre = document.createElement("pre");h.textContent = heading;
        pre.textContent = value == null ? "Awaiting result…" : typeof value === "string" ? value : JSON.stringify(value,null,2);
        details.append(h,pre);
      }
      const point = event.arguments?.point;
      const candidates = (displayed?.overlays || []).filter((o) => o.tool === event.name && (
        event.image_ref || event.arguments?.image_ref ? o.image_ref === (event.image_ref || event.arguments.image_ref) :
        displayed.source === "recording" && event.audit_tool_seq != null ? o.id === event.audit_tool_seq :
        point && JSON.stringify(o.point) === JSON.stringify(point) && (!event.arguments.view || o.direction === event.arguments.view)));
      const audited = displayed?.source === "recording" && event.audit_tool_seq != null
        ? displayed.overlays?.find((o) => o.tool === event.name && o.id === event.audit_tool_seq) : null;
      const overlay = audited || (candidates.length === 1 ? candidates[0] : null);
      if (overlay) {
        const button = document.createElement("button");button.className = "trace-overlay-link";button.textContent = `View point overlay · capture ${overlay.capture_seq}`;
        button.onclick = () => {chooseOverlay(overlay.id);$("overlay-heading").scrollIntoView({behavior:"smooth",block:"center"});};details.append(button);
      }
      article.append(details);
    }
    const source = document.createElement("p");source.className = "trace-origin";
    source.textContent = `${event.source} : ${event.line}${event.result_line ? ` → ${event.result_line}` : ""}${event.timestamp ? ` · ${event.timestamp_kind || "emitted"} ${eventTime(event.timestamp)}` : " · native event order"}`;
    if (event.timing_source) source.title = `Timestamp from ${event.timing_source}`;
    article.append(source);$("trace").append(article);
  }
  $("trace").scrollTop = follow ? $("trace").scrollHeight : scroll;
}
$("trace-filter").onchange = () => {traceSignature = "";if(latestTrace)renderTrace(latestTrace);};

function canvasContext(canvas) {
  const rect = canvas.getBoundingClientRect(), ratio = Math.min(devicePixelRatio || 1, 2);
  if (!rect.width || !rect.height) return null;
  canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  return {ctx, w: rect.width, h: rect.height};
}

function drawMap(data) {
  const size = canvasContext($("map")); if (!size) return;
  const {ctx, w, h} = size, trail = (data?.trajectory || []).filter((p) => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite));
  $("map-empty").hidden = trail.length > 0;
  let minX = -3, maxX = 3, minZ = -3, maxZ = 3;
  if (trail.length) {
    const xs = trail.map((p) => p[0]), zs = trail.map((p) => p[2]);
    minX = Math.min(...xs); maxX = Math.max(...xs); minZ = Math.min(...zs); maxZ = Math.max(...zs);
  }
  const scale = Math.min((w - 90) / Math.max(3, maxX - minX), (h - 80) / Math.max(3, maxZ - minZ)) * mapZoom;
  const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
  const project = (p) => [w / 2 + (p[0] - cx) * scale, h / 2 + (p[2] - cz) * scale];
  ctx.fillStyle = "#efefeb"; ctx.fillRect(0, 0, w, h);
  const grid = Math.pow(10, Math.floor(Math.log10(55 / scale)));
  const spacing = grid * scale;
  const origin = project([0, 0, 0]);
  ctx.strokeStyle = "#deded7"; ctx.lineWidth = 0.6;
  for (let x = ((origin[0] % spacing) + spacing) % spacing; x < w; x += spacing) {ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();}
  for (let y = ((origin[1] % spacing) + spacing) % spacing; y < h; y += spacing) {ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();}
  ctx.fillStyle = "#77776f"; ctx.font = "9px monospace"; ctx.fillText("+X →", 14, 21); ctx.fillText("+Z ↓", 14, 36);
  $("map-scale").textContent = `${number(grid, grid < 1 ? Math.ceil(-Math.log10(grid)) : 0)} m GRID · WORLD X/Z`;
  if (!trail.length) return;
  ctx.beginPath(); trail.forEach((p, i) => {const [x, y] = project(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);});
  ctx.strokeStyle = "#ff7f00"; ctx.lineWidth = 2.5; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.stroke();
  const start = project(trail[0]); ctx.beginPath(); ctx.arc(...start, 4.5, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = "#111"; ctx.lineWidth = 1.5; ctx.stroke();
  const p = data.pose?.position || trail.at(-1), [x, y] = project(p), heading = (data.pose?.heading_deg || 0) * Math.PI / 180;
  ctx.save(); ctx.translate(x, y); ctx.rotate(heading);
  ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, 35, -.5, .5); ctx.closePath(); ctx.fillStyle = "#ff7f0029"; ctx.fill();
  ctx.beginPath(); ctx.arc(0, 0, 11, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
  ctx.beginPath(); ctx.moveTo(9, 0); ctx.lineTo(-6, -5); ctx.lineTo(-3, 0); ctx.lineTo(-6, 5); ctx.closePath(); ctx.fillStyle = "#111"; ctx.fill(); ctx.restore();
}

// A deliberately illustrated, local demo. It never pretends to be Habitat RGB.
function drawDemo(data) {
  const size = canvasContext($("demo-canvas")); if (!size) return;
  const {ctx, w, h} = size;
  const t = data.step / 80, eye = [Math.sin(t) * 1.4, 1.65, 6.5 + Math.cos(t) * .8], yaw = -Math.sin(t) * .13;
  const polys = [];
  function face(vertices, color) {
    const points = vertices.map(([x,y,z]) => { x -= eye[0]; y -= eye[1]; z = eye[2] - z; return [x*Math.cos(yaw)+z*Math.sin(yaw), y, z*Math.cos(yaw)-x*Math.sin(yaw)]; });
    if (points.some((p) => p[2] < .1)) return;
    polys.push({depth: points.reduce((n,p)=>n+p[2],0)/points.length, color, points: points.map(([x,y,z])=>[w/2 + x*w*.8/z, h*.48 - y*w*.8/z])});
  }
  function box(x,y,z,dx,dy,dz,colors) {
    face([[x,y,z+dz],[x+dx,y,z+dz],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],colors[0]);
    face([[x,y+dy,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],colors[1]);
    face([[x,y,z],[x,y,z+dz],[x,y+dy,z+dz],[x,y+dy,z]],colors[2]);
    face([[x+dx,y,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x+dx,y,z+dz]],colors[2]);
  }
  ctx.fillStyle = "#dfd9c9"; ctx.fillRect(0,0,w,h);
  face([[-5,0,-3],[5,0,-3],[5,0,8],[-5,0,8]],"#bdb199");
  face([[-5,0,-3],[5,0,-3],[5,4,-3],[-5,4,-3]],"#e4e1d5");
  face([[-5,0,8],[-5,0,-3],[-5,4,-3],[-5,4,8]],"#c9cabc");
  face([[5,0,-3],[5,0,8],[5,4,8],[5,4,-3]],"#d2cfbf");
  for (let z=-3;z<7;z+=.65) face([[-5,.006,z],[5,.006,z],[5,.006,z+.013],[-5,.006,z+.013]],"#a99b82");
  face([[-2.5,.01,-1],[2.8,.01,-1],[2.8,.01,3],[-2.5,.01,3]],"#e4dfd0");
  box(-3.7,.65,-2.97,2.2,2,.035,["#9caa9c","#d7d8c9","#8b9688"]);
  for (let x=-3.7;x<=-1.5;x+=1.1) box(x,.65,-2.9,.035,2,.025,["#f7f2e5","#f7f2e5","#f7f2e5"]);
  box(-3.7,1.63,-2.9,2.2,.035,.025,["#f7f2e5","#f7f2e5","#f7f2e5"]);
  box(.35,0,-2.95,1.28,2.7,.02,["#827a68","#998d76","#998d76"]);
  box(.45,0,-2.88,1.07,2.58,.02,["#b1a58f","#b1a58f","#b1a58f"]);
  box(-.6,.5,1,.95,.2,1,["#d87839","#ea924d","#a85325"]);
  box(-.6,.7,.95,.95,.9,.18,["#d87839","#ea924d","#a85325"]);
  for (const x of [-.54,.2]) for (const z of [1.1,1.82]) box(x,0,z,.07,.51,.07,["#665744","#665744","#554837"]);
  box(2.4,0,-1.7,1.2,.85,.7,["#8c7255","#b19a76","#70583e"]);
  box(2.58,.86,-1.45,.32,.34,.3,["#d0c1a1","#e1d5bf","#bba781"]);
  box(2.55,1.2,-1.46,.38,.35,.34,["#61765b","#819376","#435b3e"]);
  box(-3.5,0,.3,.5,.6,.5,["#c7b293","#dfcbae","#a58f72"]);
  box(-3.62,.6,.22,.7,.8,.65,["#788368","#95a083","#5d6c50"]);
  polys.sort((a,b)=>b.depth-a.depth);
  for (const poly of polys) {ctx.beginPath();poly.points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fillStyle=poly.color;ctx.fill();}
  const shade = ctx.createLinearGradient(0,0,w,h);shade.addColorStop(0,"#fff1ca18");shade.addColorStop(1,"#2b271726");ctx.fillStyle=shade;ctx.fillRect(0,0,w,h);
}

$("zoom-in").onclick = () => {mapZoom = Math.min(8, mapZoom*1.3);drawMap(displayed);};
$("zoom-out").onclick = () => {mapZoom = Math.max(.25, mapZoom/1.3);drawMap(displayed);};
$("fit-map").onclick = () => {mapZoom = 1;drawMap(displayed);};
new ResizeObserver(() => {drawMap(displayed);if (displayed?.source === "demo") drawDemo(displayed);}).observe($("map"));
new ResizeObserver(() => {if (displayed?.source === "demo") drawDemo(displayed);}).observe($("camera-stage"));
document.addEventListener("visibilitychange", () => { if (!document.hidden) sessions(); });
window.addEventListener("pagehide", () => {stream?.close();stopReplay();});
setInterval(sessions, 3000);
setInterval(updateTime, 1000);
setInterval(refreshTrace, 1000);
renderPerception({});
sessions();
