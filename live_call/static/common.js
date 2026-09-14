// Shared helpers for the demo pages: verdict labels, the score timeline, the warning tone.

const STATE_TEXT = {
  listening: "Listening",
  genuine: "Genuine voice",
  suspicious: "Suspicious: may be synthetic",
  likely_fake: "Likely cloned voice",
};

function stateText(state) {
  return STATE_TEXT[state] || state;
}

function pct(x, digits = 2) {
  return x === null || x === undefined ? "–" : (100 * x).toFixed(digits) + "%";
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

// Draw P(genuine) over time. points: [{t, score, smoothed}], threshold in [0, 1].
function drawTimeline(canvas, points, threshold) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const padL = 44, padR = 10, padT = 10, padB = 22;
  const tMax = Math.max(20, ...points.map((p) => p.t));
  const x = (t) => padL + ((w - padL - padR) * t) / tMax;
  const y = (v) => padT + (h - padT - padB) * (1 - v);

  ctx.font = "11px Segoe UI, sans-serif";
  ctx.fillStyle = "#5d6b7a";
  ctx.strokeStyle = "#e3e8ef";
  for (const v of [0, 0.5, 1]) {
    ctx.beginPath(); ctx.moveTo(padL, y(v)); ctx.lineTo(w - padR, y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(1), 8, y(v) + 4);
  }
  ctx.fillText("P(genuine)", padL + 4, padT + 10);
  ctx.fillText(Math.round(tMax) + " s", w - padR - 30, h - 6);

  if (threshold !== null && threshold !== undefined) {
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = "#c62f2f";
    ctx.beginPath(); ctx.moveTo(padL, y(threshold)); ctx.lineTo(w - padR, y(threshold)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#c62f2f";
    ctx.fillText("alert threshold " + threshold.toFixed(3), w - padR - 140, y(threshold) - 4);
  }

  const scored = points.filter((p) => p.score !== null && p.score !== undefined);
  ctx.fillStyle = "rgba(36, 88, 214, 0.35)";
  for (const p of scored) { ctx.beginPath(); ctx.arc(x(p.t), y(p.score), 3, 0, 7); ctx.fill(); }
  const smooth = points.filter((p) => p.smoothed !== null && p.smoothed !== undefined);
  if (smooth.length) {
    ctx.strokeStyle = "#2458d6"; ctx.lineWidth = 2;
    ctx.beginPath();
    smooth.forEach((p, i) => (i ? ctx.lineTo(x(p.t), y(p.smoothed)) : ctx.moveTo(x(p.t), y(p.smoothed))));
    ctx.stroke(); ctx.lineWidth = 1;
  }
}

// A short two-tone warning, synthesised in the browser (no audio files shipped).
let toneCtx = null;
function warningTone(level = "beep") {
  try {
    toneCtx = toneCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (toneCtx.state === "suspended") toneCtx.resume();
    if (level === "unlock") return; // first click: allow sound later, play nothing now
    const now = toneCtx.currentTime;
    const beeps = level === "sms" ? 3 : 2;
    for (let i = 0; i < beeps; i++) {
      const osc = toneCtx.createOscillator(), gain = toneCtx.createGain();
      osc.type = "sine";
      osc.frequency.value = i % 2 ? 660 : 880;
      gain.gain.setValueAtTime(0.0001, now + i * 0.28);
      gain.gain.exponentialRampToValueAtTime(0.25, now + i * 0.28 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.28 + 0.24);
      osc.connect(gain).connect(toneCtx.destination);
      osc.start(now + i * 0.28); osc.stop(now + i * 0.28 + 0.26);
    }
  } catch (e) { /* audio not allowed yet: the banner still shows */ }
}

async function getJSON(url, options) {
  const r = await fetch(url, options);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

async function renderHealth(container) {
  try {
    const h = await getJSON("/api/health");
    const d = h.detector;
    container.replaceChildren(
      el("span", { class: "pill " + (d.checkpoint_found ? "ok" : "bad"), text: "model: " + (d.checkpoint || "not set") }),
      el("span", { class: "pill " + (d.threshold !== null ? "ok" : "bad"), text: "threshold: " + (d.threshold !== null ? d.threshold.toFixed(3) : "missing") }),
      el("span", { class: "pill " + (h.twilio.configured ? "ok" : "warn"), text: "Twilio: " + (h.twilio.configured ? "live" : "dry run") }),
      el("span", { class: "pill", text: h.live_sessions + " live call(s)" }),
    );
  } catch (e) {
    container.replaceChildren(el("span", { class: "pill bad", text: "server unreachable" }));
  }
}
