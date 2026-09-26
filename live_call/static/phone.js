// Shared handset logic for /phone/caller and /phone/receiver.
//
// Each handset holds two connections to the server:
//   1. a WebSocket to /ws/phone  - call control (dial, ring, accept, hang up)
//   2. a WebRTC peer connection  - the audio itself
// The caller's audio is the monitored one; the receiver's is carried but not scored
// (monitor:false), so the receiver's own voice can never raise a warning.

const ICE = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };

// Opus defaults to a thrifty, narrow, discontinuous stream. The detector is trained
// on telephone audio and reads fine waveform detail, so a low-bitrate encode of a
// REAL voice can score like a synthetic one. Ask for full-band, constant, no DTX.
function highQualityOpus(sdp) {
  const m = sdp.match(/a=rtpmap:(\d+) opus\/48000/i);
  if (!m) return sdp;
  const pt = m[1];
  const wanted = "maxaveragebitrate=128000;maxplaybackrate=48000;sprop-maxcapturerate=48000;" +
                 "stereo=0;cbr=1;useinbandfec=1;usedtx=0";
  if (new RegExp("a=fmtp:" + pt + " ").test(sdp)) {
    return sdp.replace(new RegExp("(a=fmtp:" + pt + " )(.*)"), (_x, head, rest) => {
      const kept = rest.split(";").filter((kv) => !/^(maxaveragebitrate|stereo|cbr|usedtx|useinbandfec|maxplaybackrate|sprop-maxcapturerate)=/.test(kv.trim()));
      return head + [...kept, wanted].filter(Boolean).join(";");
    });
  }
  const line = new RegExp("(a=rtpmap:" + pt + " opus/48000[^\r\n]*)");
  return sdp.replace(line, "$1\r\na=fmtp:" + pt + " " + wanted);
}

function qs(name, fallback) {
  const v = new URLSearchParams(location.search).get(name);
  return v === null || v === "" ? fallback : v;
}

function mmss(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
}

function initials(name) {
  return (name || "?")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0].toUpperCase())
    .join("");
}

// ----------------------------------------------------------------- call control
class Line {
  constructor({ room, role, name, number }) {
    Object.assign(this, { room, role, name, number });
    this.handlers = {};
    this.ws = null;
    this.state = "idle";
  }

  on(type, fn) {
    this.handlers[type] = fn;
    return this;
  }

  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const q = new URLSearchParams({
      room: this.room, role: this.role, name: this.name, number: this.number,
    });
    this.ws = new WebSocket(`${proto}://${location.host}/ws/phone?${q}`);
    this.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.state) this.state = msg.state;
      (this.handlers[msg.type] || (() => {}))(msg);
      (this.handlers["*"] || (() => {}))(msg);
    };
    this.ws.onclose = () => (this.handlers["closed"] || (() => {}))();
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve(this);
      this.ws.onerror = reject;
    });
  }

  send(action, extra = {}) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action, ...extra }));
    }
  }
}

// ---------------------------------------------------------------------- audio
class Media {
  constructor() {
    this.ctx = null;
    this.stream = null;
    this.pc = null;
    this.pcId = null;
    this.clipEl = null;
    this.rafId = null;
  }

  // mode: "mic" | "clip" (clipUrl required) | "file" (file required)
  async open(mode, { clipUrl, file, meterBar } = {}) {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    if (mode === "mic") {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
      });
      this.meter(this.ctx.createMediaStreamSource(this.stream), meterBar);
      return this.stream;
    }
    const url = mode === "file" ? URL.createObjectURL(file) : clipUrl;
    this.clipEl = new Audio(url);
    this.clipEl.crossOrigin = "anonymous";
    this.clipEl.loop = true; // a call should not fall silent mid-demo
    const src = this.ctx.createMediaElementSource(this.clipEl);
    const dest = this.ctx.createMediaStreamDestination();
    src.connect(dest);
    this.meter(src, meterBar);
    this.stream = dest.stream;
    return this.stream;
  }

  playClip() {
    if (this.clipEl) this.clipEl.play().catch(() => {});
  }

  meter(node, bar) {
    if (!bar) return;
    const analyser = this.ctx.createAnalyser();
    analyser.fftSize = 512;
    node.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (const v of data) { const x = (v - 128) / 128; sum += x * x; }
      bar.style.width = Math.min(100, Math.sqrt(sum / data.length) * 320).toFixed(0) + "%";
      this.rafId = requestAnimationFrame(tick);
    };
    tick();
  }

  // Connect to the server and start sending. onRemote gets the far end's audio.
  async connect({ room, role, name, monitor }, onRemote) {
    this.pc = new RTCPeerConnection(ICE);
    this.stream.getTracks().forEach((t) => this.pc.addTrack(t, this.stream));
    this.pc.ontrack = (ev) => onRemote(ev.streams[0]);
    const answer = await this.negotiate({ room, role, name, monitor });
    this.pcId = answer.pc_id;
    return this.pcId;
  }

  // Also used for renegotiation: the server adds the other party's track to our
  // existing connection and asks us to re-offer, which is what makes audio two-way.
  async negotiate(params) {
    const offer = await this.pc.createOffer({ offerToReceiveAudio: true });
    offer.sdp = highQualityOpus(offer.sdp);
    await this.pc.setLocalDescription(offer);
    this.raiseBitrate();
    await new Promise((resolve) => {
      if (this.pc.iceGatheringState === "complete") return resolve();
      const check = () => this.pc.iceGatheringState === "complete" && resolve();
      this.pc.addEventListener("icegatheringstatechange", check);
      setTimeout(resolve, 2000);
    });
    const body = {
      sdp: this.pc.localDescription.sdp,
      type: this.pc.localDescription.type,
      pc_id: this.pcId || undefined,
      ...params,
    };
    const answer = await getJSON("/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await this.pc.setRemoteDescription(answer);
    return answer;
  }

  // Ask the encoder for a high, steady bitrate. A default WebRTC call spends as
  // few bits as it can, and the detector reads fine waveform detail, so a thrifty
  // encoder can make a real voice look synthetic.
  raiseBitrate(bps = 128000) {
    for (const sender of this.pc.getSenders()) {
      if (!sender.track || sender.track.kind !== "audio") continue;
      const params = sender.getParameters();
      params.encodings = params.encodings && params.encodings.length ? params.encodings : [{}];
      params.encodings[0].maxBitrate = bps;
      params.encodings[0].networkPriority = "high";
      params.encodings[0].priority = "high";
      sender.setParameters(params).catch(() => {});
    }
  }

  close() {
    if (this.rafId) cancelAnimationFrame(this.rafId);
    if (this.clipEl) { this.clipEl.pause(); this.clipEl = null; }
    if (this.pc) { this.pc.close(); this.pc = null; }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) this.ctx.close().catch(() => {});
    this.ctx = this.stream = null;
    this.pcId = null;
  }
}

// ------------------------------------------------------------------- ringtone
let ringCtx = null;
let ringTimer = null;

function startRing(kind = "ring") {
  stopRing();
  try {
    ringCtx = ringCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (ringCtx.state === "suspended") ringCtx.resume();
  } catch (e) { return; }
  const burst = () => {
    const now = ringCtx.currentTime;
    // Two short tones, the usual double-ring cadence.
    for (const offset of kind === "ring" ? [0, 0.4] : [0]) {
      const osc = ringCtx.createOscillator();
      const gain = ringCtx.createGain();
      osc.type = "sine";
      osc.frequency.value = kind === "ring" ? 440 : 400;
      gain.gain.setValueAtTime(0.0001, now + offset);
      gain.gain.exponentialRampToValueAtTime(kind === "ring" ? 0.22 : 0.1, now + offset + 0.03);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.33);
      osc.connect(gain).connect(ringCtx.destination);
      osc.start(now + offset);
      osc.stop(now + offset + 0.35);
    }
  };
  burst();
  ringTimer = setInterval(burst, 2000);
}

function stopRing() {
  if (ringTimer) { clearInterval(ringTimer); ringTimer = null; }
}

// Unlock audio on the first user gesture, so the ringtone and warning can play.
function unlockAudio() {
  try {
    ringCtx = ringCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (ringCtx.state === "suspended") ringCtx.resume();
  } catch (e) { /* the banner still shows without sound */ }
}

// --------------------------------------------------------------- call timer
function startTimer(el, since) {
  const t0 = since ? since * 1000 : Date.now();
  const tick = () => (el.textContent = mmss((Date.now() - t0) / 1000));
  tick();
  return setInterval(tick, 1000);
}
