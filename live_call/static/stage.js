// The product's shared shell: the navigation rail, the status light, the verdict key, the
// presenter's notes and keyboard, and source tooltips. Every page calls Stage.mount("<key>").
//
// No number lives in this file. Where the presenter's line carries one, it is filled
// from /api/headline or /api/real_calls, which read the committed result files.

const Stage = (() => {
  const BRAND = { name: "Asli", tagline: "Cloned-voice call protection" };

  // What the product does, in the order a user meets it. Keys 1-6 jump here.
  const NAV = [
    { key: "home", title: "Overview", href: "/demo", icon: "home", what: "status, recent calls and what it protects" },
    { key: "live", title: "Live call", href: "/live", icon: "phone", what: "call a protected phone; only the receiver is warned" },
    { key: "try", title: "Check audio", href: "/try", icon: "wave", what: "upload, record or pick a recording to check" },
    { key: "history", title: "Call log", href: "/history", icon: "list", what: "every call it has judged, with its verdict" },
    { key: "lanes", title: "Compare", href: "/lanes", icon: "columns", what: "one call through four detectors side by side" },
    { key: "limits", title: "Performance", href: "/limits", icon: "chart", what: "measured results and known limits" },
  ];
  // The evidence pages sit inside Performance, reached from its tabs.
  const EVIDENCE = [
    { key: "limits", title: "Results and limits", href: "/limits" },
    { key: "threat", title: "Why it is needed", href: "/threat" },
    { key: "network", title: "Real phone network", href: "/network" },
    { key: "trust", title: "Trust check", href: "/trust" },
  ];
  // The review's eight-minute walkthrough: the arrow keys follow it.
  const FLOW = [
    { key: "home", title: "Overview", minutes: "0:45" },
    { key: "lanes", title: "Compare detectors", minutes: "2:00" },
    { key: "live", title: "Live call", minutes: "3:00" },
    { key: "limits", title: "Performance", minutes: "2:00" },
  ];
  // The same four names on every page, whatever a lane is called in DFD_LANES.
  const NAMES = { s1: "English-only", deployed: "Deployed adapter", final: "Final recipe", ssl: "Larger encoder" };
  const ABOUT = {
    s1: "Trained on English speech only",
    deployed: "Our first adapter: Hindi-English clones over a phone line",
    final: "Far more speakers and a real mobile codec",
    ssl: "A bigger model on the same data: the next candidate",
  };

  // What the person at the keyboard says, one breath each.
  const SAY = {
    home: (d) => [
      `A familiar voice calls in Hindi-English and asks for an OTP. The usual detector, trained on English, calls ${d.s1Fake} of genuine Hindi-English speech fake: it cannot protect this call.`,
      "Asli listens to the call as it happens and warns only the person being called.",
    ],
    lanes: () => [
      "One call, four detectors. Play the genuine caller: the English-only detector rejects a real person; ours let them through.",
      "Now the Tortoise clone, a tool none of them was trained on: the final recipe catches it. The larger encoder is one checkpoint, our next candidate.",
    ],
    live: () => [
      "Please hold this phone. A clone is calling you. [amber] A caution. [red, spoken] Likely cloned.",
      "The caller sees nothing, because a caller who knows they were caught learns to evade.",
    ],
    limits: () => [
      "Where it stands, with numbers: what it catches on real calls, and what it cannot do yet.",
      "The larger encoder does better on spontaneous speech and Chatterbox, but it is one checkpoint: the next candidate, not a replacement.",
    ],
    try: () => ["Any recording, judged exactly like a live call: upload a file, record a line, or pick a prepared sample."],
    history: () => ["Every call it has judged since it started, with the strongest warning each one got."],
    threat: (d) => [`The usual detector calls ${d.s1Fake} of genuine Hindi-English speech fake.`],
    network: (d) => [
      `Thirty real mobile calls, scores only: the final recipe holds where the codec breaks the old model, and catches ${d.xttsCaught} of ${d.xttsCalls} seen-tool clones.`,
    ],
    trust: () => ["Scramble the words: same verdict. It reads how the voice was produced, not what is said."],
  };

  const ICONS = {
    home: '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M10 21v-6h4v6"/>',
    phone: '<path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1.9.4 1.8.7 2.7a2 2 0 0 1-.5 2.1L8 9.8a16 16 0 0 0 6 6l1.3-1.3a2 2 0 0 1 2.1-.4c.9.3 1.8.6 2.7.7a2 2 0 0 1 1.7 2z"/>',
    wave: '<path d="M3 12h1M7 8v8M11 4v16M15 8v8M19 10v4M21 12h0"/>',
    list: '<path d="M9 6h12M9 12h12M9 18h12"/><path d="M4 6h.01M4 12h.01M4 18h.01"/>',
    columns: '<rect x="3" y="4" width="7.5" height="16" rx="2"/><rect x="13.5" y="4" width="7.5" height="16" rx="2"/>',
    chart: '<path d="M3 21h18"/><path d="M6 17v-6M11 17V6M16 17v-9M20 17v-4"/>',
  };
  const svg = (paths, cls = "") =>
    `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${paths}</svg>`;
  const MARK = `<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.8 19.5 6v5.6c0 4.6-3.2 8.3-7.5 9.6-4.3-1.3-7.5-5-7.5-9.6V6z"/><path d="M8 12h1.6l1.3-2.6 2.2 5.2 1.3-2.6H16"/></svg>`;

  let data = null;

  const $ = (sel, root = document) => root.querySelector(sel);

  function make(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const child of [].concat(children)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  async function json(url, options) {
    const r = await fetch(url, options);
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (e) { /* not json */ }
      throw new Error(detail);
    }
    return r.json();
  }

  // A number from a results file, with the file and field as its tooltip.
  function cite(entry, format = (v) => v, cls = "") {
    if (!entry || entry.value === null || entry.value === undefined) return make("span", { class: "faint", text: "–" });
    return make("span", { class: ("cite " + cls).trim(), title: "read from " + entry.source, text: format(entry.value) });
  }
  const pct = (digits = 1) => (v) => Number(v).toFixed(digits) + "%";
  const nameOf = (role, fallback = "") => NAMES[role] || fallback;

  // A "Details" toggle: the button shows or hides every [data-details] block in `root`.
  function details(button, root = document, onChange = () => {}) {
    const set = (open) => {
      root.querySelectorAll("[data-details]").forEach((el) => (el.hidden = !open));
      button.setAttribute("aria-expanded", open ? "true" : "false");
      button.textContent = open ? "Hide details" : "Details";
      document.body.classList.toggle("details-open", open);
      onChange(open);
    };
    button.addEventListener("click", () => set(button.getAttribute("aria-expanded") !== "true"));
    set(false);
    return set;
  }

  // One request shared by every caller (the notes and a page's cards ask at once).
  function sayData() {
    if (data) return data;
    data = (async () => {
      const out = { s1Fake: "most", xttsCaught: "most", xttsCalls: "the" };
      try {
        const h = await json("/api/headline");
        out.s1Fake = h.s1.genuine_called_fake_pct.value.toFixed(1) + "%";
      } catch (e) { /* notes still read without the number */ }
      try {
        const r = await json("/api/real_calls");
        const row = r.table.find((x) => x.kind === "clone_replay" && x.tool === "xtts_v2");
        out.xttsCaught = row.final.likely.value;
        out.xttsCalls = row.final.calls.value;
      } catch (e) { /* idem */ }
      return out;
    })();
    return data;
  }

  const sectionOf = (key) => (EVIDENCE.some((e) => e.key === key) ? "limits" : key);

  function rail(key) {
    const section = sectionOf(key);
    return make("nav", { class: "rail", "aria-label": BRAND.name }, [
      make("a", { class: "brand", href: "/demo", title: `${BRAND.name}: ${BRAND.tagline}` }, [
        make("span", { class: "mark", html: MARK }),
        make("b", { text: BRAND.name }),
      ]),
      make("div", { class: "nav" }, NAV.map((n, i) => make("a", {
        href: n.href,
        class: n.key === section ? "on" : "",
        "aria-current": n.key === section ? "page" : null,
        title: `${n.title}: ${n.what} (${i + 1})`,
      }, [make("span", { class: "ic", html: svg(ICONS[n.icon]) }), make("span", { class: "t", text: n.title })]))),
      make("div", { class: "status", id: "stagePills" }, [make("span", { class: "pill", text: "connecting…" })]),
    ]);
  }

  // The evidence pages share one tab strip under Performance.
  function tabs(key) {
    if (!EVIDENCE.some((e) => e.key === key)) return null;
    return make("nav", { class: "tabs", "aria-label": "Performance" }, EVIDENCE.map((e) => make("a", {
      href: e.href, class: e.key === key ? "on" : "", "aria-current": e.key === key ? "page" : null, text: e.title,
    })));
  }

  function shortModel(name) {
    if (!name) return "no model";
    if (name.startsWith("final_")) return NAMES.final;
    if (name.startsWith("lora_norm_rvc")) return NAMES.deployed;
    if (name.includes("ssl")) return NAMES.ssl;
    if (name.startsWith("best")) return NAMES.s1; // the baseline folder holds the English-only model
    return "another model";
  }

  // One light: is protection on? The details for the mouse.
  async function pills() {
    const box = $("#stagePills");
    if (!box) return;
    try {
      const h = await json("/api/health");
      const d = h.detector;
      const ready = d.model_loaded && d.threshold !== null && h.demo_clips;
      const detail = `${shortModel(d.checkpoint)}${d.model_loaded ? "" : " (loading)"} · threshold from ${d.threshold_source || "missing"} · ${h.demo_clips} prepared calls`;
      box.replaceChildren(make("span", {
        class: "pill " + (ready ? "ok" : d.checkpoint_found ? "warn" : "bad"),
        title: detail,
        text: ready ? "Ready" : d.checkpoint_found ? "Loading…" : "Model missing",
      }));
    } catch (e) {
      box.replaceChildren(make("span", { class: "pill bad", text: "Offline" }));
    }
  }

  // The same three verdict words and colours on every page, in one line.
  function keyLine() {
    return make("div", { class: "vkey", "aria-label": "What the verdicts mean" }, [
      make("span", { class: "k s-genuine" }, [make("span", { class: "light" }), "Genuine"]),
      make("span", { class: "k s-suspicious" }, [make("span", { class: "light" }), "Caution"]),
      make("span", { class: "k s-likely_fake" }, [make("span", { class: "light still" }), "Likely cloned"]),
      make("span", { class: "means", text: "Caution = a few suspicious seconds in a row. Likely cloned = more of them." }),
    ]);
  }

  function flowStep(key) {
    const i = FLOW.findIndex((f) => f.key === key);
    return { i, prev: i > 0 ? FLOW[i - 1] : null, next: i >= 0 ? FLOW[i + 1] || null : FLOW[0] };
  }
  const hrefOf = (key) => (NAV.find((n) => n.key === key) || EVIDENCE.find((e) => e.key === key) || NAV[0]).href;

  async function presenter(key) {
    const { i, prev, next } = flowStep(key);
    const lines = (SAY[key] || (() => []))(await sayData());
    const step = i >= 0 ? FLOW[i] : null;
    const drawer = make("aside", { class: "presenter", id: "presenter", "aria-label": "Presenter notes" }, [
      make("div", { class: "handle" }, [
        keyLine(),
        make("span", { class: "spacer" }),
        make("button", { class: "notes-btn", type: "button", onclick: () => toggleNotes(), html: "notes <kbd>N</kbd>" }),
      ]),
      make("div", { class: "body" }, [
        make("div", { class: "say" }, [
          make("div", { class: "meta", text: step ? `Walkthrough ${i + 1} of ${FLOW.length} · ${step.title} · ${step.minutes}` : "For questions" }),
          ...lines.map((t) => make("p", { text: t })),
        ]),
        make("div", { class: "navs" }, [
          prev ? make("a", { class: "btn ghost", href: hrefOf(prev.key), text: "← " + prev.title }) : null,
          next ? make("a", { class: "btn", href: hrefOf(next.key), text: next.title + " →" }) : null,
        ]),
      ]),
    ]);
    document.body.append(drawer);
  }

  // Only the presenter's own toggle is remembered, so notes never open by themselves.
  function toggleNotes(force, remember = true) {
    const p = $("#presenter");
    if (!p) return;
    const open = force === undefined ? !p.classList.contains("open") : force;
    p.classList.toggle("open", open);
    if (!remember) return;
    try { localStorage.setItem("stage.notes", open ? "1" : "0"); } catch (e) { /* private window */ }
  }

  function keys(key) {
    document.addEventListener("keydown", (ev) => {
      if (ev.target.closest("input, select, textarea, [contenteditable]") || ev.ctrlKey || ev.metaKey || ev.altKey) return;
      const { prev, next } = flowStep(key);
      if (ev.key === "ArrowRight") {
        if (next) location.href = hrefOf(next.key);
      } else if (ev.key === "ArrowLeft") {
        if (prev) location.href = hrefOf(prev.key);
        else if (key !== "home") location.href = NAV[0].href;
      } else if (ev.key === "n" || ev.key === "N") {
        toggleNotes();
      } else if (ev.key === "f" || ev.key === "F") {
        if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => {});
      } else if (ev.key >= "1" && ev.key <= String(NAV.length)) {
        location.href = NAV[Number(ev.key) - 1].href;
      }
    });
  }

  function toast(message, ms = 6000) {
    let t = $("#stageToast");
    if (!t) { t = make("div", { class: "toast", id: "stageToast", role: "alert" }); document.body.append(t); }
    t.textContent = message;
    t.classList.add("show");
    clearTimeout(t._timer);
    t._timer = setTimeout(() => t.classList.remove("show"), ms);
  }

  function favicon() {
    if (document.querySelector("link[rel=icon]")) return;
    const icon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#2a63e8"/><path d="M16 5l9 4v6.5c0 5.4-3.8 9.7-9 11.2-5.2-1.5-9-5.8-9-11.2V9z" fill="none" stroke="#fff" stroke-width="2.4" stroke-linejoin="round"/></svg>`;
    document.head.append(make("link", { rel: "icon", href: "data:image/svg+xml," + encodeURIComponent(icon) }));
  }

  async function mount(key) {
    favicon();
    const page = NAV.find((n) => n.key === key) || EVIDENCE.find((e) => e.key === key);
    if (page) document.title = `${page.title} · ${BRAND.name}`;
    document.body.classList.add("stage");
    document.body.prepend(rail(key));
    const strip = tabs(key);
    const main = $("main");
    if (strip && main) main.prepend(strip);
    keys(key);
    pills();
    setInterval(pills, 10000);
    await presenter(key);
    let open = false;
    try { open = localStorage.getItem("stage.notes") === "1"; } catch (e) { /* private window */ }
    toggleNotes(open, false);
  }

  // The dashboard socket, reconnecting: every lane window and call verdict arrives here.
  function socket(onMessage) {
    let ws = null;
    let alive = true;
    const open = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws/dashboard`);
      ws.onmessage = (ev) => { try { onMessage(JSON.parse(ev.data)); } catch (e) { /* ignore */ } };
      ws.onclose = () => { if (alive) setTimeout(open, 1500); };
    };
    open();
    return { close() { alive = false; if (ws) ws.close(); } };
  }

  const WORDS = { listening: "Listening", genuine: "Genuine", suspicious: "Caution", likely_fake: "Likely cloned" };
  const word = (state) => WORDS[state] || state || "Listening";

  // The presenter's lines for one page, numbers filled from the results API.
  async function lines(key) {
    return (SAY[key] || (() => []))(await sayData());
  }

  // A small verdict chip: light and word, in the verdict's colour.
  function chip(state, text) {
    return make("span", { class: "vchip s-" + (state || "listening") }, [make("span", { class: "light still" }), text || word(state)]);
  }

  // Prepared calls are named for what they are, never for how they turned out.
  const CALL_NAMES = {
    tortoise_caught: "Tortoise cloned voice, call 1 (unseen tool)",
    tortoise_clone: "Tortoise cloned voice, call 2 (unseen tool)",
    hiacc_spontaneous: "Genuine caller speaking freely",
  };
  const callName = (id, title) => CALL_NAMES[id] || title || String(id).replace(/_/g, " ");

  // A judged call, described for people: what it was and where it came from.
  function callInfo(s) {
    const meta = s.meta || {};
    const truth = meta.truth === "spoof" ? "known clone" : meta.truth === "bonafide" ? "known genuine" : null;
    if (s.kind === "replay") return { title: callName(meta.clip, s.label), source: "Prepared call", truth };
    if (s.kind === "webrtc") return { title: "Call to the protected phone", source: "Browser phone", truth };
    if (s.kind === "twilio") return { title: s.label, source: "Phone network", truth };
    return { title: s.label, source: s.kind, truth };
  }
  const clock = (unix) => new Date(unix * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const secs = (s) => {
    if (s === null || s === undefined) return "–";
    const r = Math.round(s);
    return `${Math.floor(r / 60)}:${String(r % 60).padStart(2, "0")}`;
  };

  return { BRAND, NAV, EVIDENCE, FLOW, NAMES, ABOUT, ICONS, svg, mount, make, json, cite, pct, toast, socket, word, chip, callName, callInfo, clock, secs, shortModel, nameOf, details, lines };
})();
