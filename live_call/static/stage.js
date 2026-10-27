// The final-review demo's shared shell: scene stepper, status pills, presenter notes,
// keyboard navigation and source tooltips. Every scene page calls Stage.mount("<key>").
//
// No number lives in this file. Where the presenter's line carries one, it is filled
// from /api/headline or /api/real_calls, which read the committed result files.

const Stage = (() => {
  const SCENES = [
    { key: "threat", n: 1, title: "The threat", href: "/threat", minutes: "0:30" },
    { key: "lanes", n: 2, title: "Three detectors", href: "/lanes", minutes: "2:00" },
    { key: "live", n: 3, title: "Live call", href: "/live", minutes: "3:00" },
    { key: "network", n: 4, title: "Real network", href: "/network", minutes: "2:00" },
    { key: "trust", n: 5, title: "Why trust it", href: "/trust", minutes: "1:00" },
    { key: "limits", n: 6, title: "Limits + try it", href: "/limits", minutes: "1:30" },
  ];

  // What the person at the keyboard says, one breath each (plan section 6).
  const SAY = {
    threat: (d) => [
      `A familiar voice calls in Hindi-English and asks for an OTP. An English-trained detector calls ${d.s1Fake} of real Hindi-English speakers fake, so it cannot protect this call.`,
    ],
    lanes: () => [
      "Same call, three detectors. The English one rejects a real person. Our adapter fixes that.",
      "The final recipe, trained on far more voices and a real mobile codec, also catches the tool none of them ever saw.",
    ],
    live: () => [
      "Please hold this phone. A clone is calling you. [amber] A caution. [red, spoken] Likely cloned.",
      "The caller sees nothing, because a caller who knows they were caught learns to evade.",
    ],
    network: (d) => [
      "We called each other over a real mobile network. Hear what the codec does.",
      `For two of us the old model collapses at the codec; the final recipe holds for all three, and catches ${d.xttsCaught} of ${d.xttsCalls} seen-tool clones. Scores only; we never play our own voices.`,
    ],
    trust: () => [
      "Scramble the words: same verdict. It reads how the voice was produced, not what is said.",
      "Every number is tested against a cheap-statistics floor.",
    ],
    limits: () => [
      "What it cannot do yet, with numbers, and what we did about each overnight.",
      "Now read this line into the microphone.",
    ],
    hub: () => [
      "Six scenes, about ten minutes. Arrow keys move between scenes; N opens these notes; F goes full screen.",
    ],
  };

  const ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h2l2-6 4 12 3-9 2 3h5"/></svg>`;
  let current = null;
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

  // One request shared by every caller (the drawer and the overview's cards ask at once).
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

  function topbar(key) {
    const steps = make("nav", { class: "steps", "aria-label": "Demo scenes" },
      SCENES.map((s) => make("a", {
        href: s.href,
        class: s.key === key ? "on" : (current && s.n < current.n ? "done" : ""),
        title: `Scene ${s.n}: ${s.title} (${s.minutes})`,
      }, [make("span", { class: "n", text: s.n }), make("span", { class: "t", text: s.title })])));
    return make("header", { class: "topbar" }, [
      make("a", { class: "brand", href: "/demo", title: "Demo overview" }, [
        make("span", { class: "mark", html: ICON }),
        make("span", {}, [make("b", { text: "Cloned-voice call detection" }), make("small", { text: "Hindi-English · final review" })]),
      ]),
      steps,
      make("div", { class: "pills", id: "stagePills" }, [make("span", { class: "pill", text: "connecting…" })]),
    ]);
  }

  function shortModel(name) {
    if (!name) return "no model";
    if (name.startsWith("final_")) return "final recipe";
    if (name.startsWith("lora_norm_rvc")) return "deployed adapter";
    if (name === "best.pt") return "English-only S1";
    return name.replace(/\.pt$/, "");
  }

  async function pills() {
    const box = $("#stagePills");
    if (!box) return;
    try {
      const h = await json("/api/health");
      const d = h.detector;
      const items = [
        make("span", { class: "pill " + (d.model_loaded ? "ok" : d.checkpoint_found ? "warn" : "bad"),
          title: (d.checkpoint || "no checkpoint") + (d.model_loaded ? " (loaded)" : " (loading)"),
          text: shortModel(d.checkpoint) + (d.model_loaded ? "" : " · loading") }),
        make("span", { class: "pill " + (d.threshold !== null ? "ok" : "bad"), title: d.threshold_source || "",
          text: "threshold " + (d.threshold !== null ? d.threshold.toFixed(3) : "missing") }),
      ];
      items.push(make("span", { class: "pill " + (h.demo_clips ? "ok" : "bad"),
        title: (h.demo_clips_dir || "") + (h.lanes !== undefined ? ` · ${h.lanes} detector lane(s)` : ""),
        text: h.demo_clips + " calls" }));
      box.replaceChildren(...items);
    } catch (e) {
      box.replaceChildren(make("span", { class: "pill bad", text: "server unreachable" }));
    }
  }

  async function presenter(key) {
    const scene = SCENES.find((s) => s.key === key);
    const lines = (SAY[key] || (() => []))(await sayData());
    const prev = scene ? SCENES[scene.n - 2] : null;
    const next = scene ? SCENES[scene.n] : SCENES[0];
    const drawer = make("aside", { class: "presenter", id: "presenter", "aria-label": "Presenter notes" }, [
      make("div", { class: "handle", onclick: () => toggleNotes() }, [
        make("b", { text: scene ? `Scene ${scene.n} · ${scene.title}` : "Demo overview" }),
        make("span", { text: scene ? `${scene.minutes} on the clock` : "about ten minutes" }),
        make("span", { class: "spacer" }),
        make("span", { html: "presenter notes <kbd>N</kbd> · scenes <kbd>←</kbd> <kbd>→</kbd> · full screen <kbd>F</kbd>" }),
      ]),
      make("div", { class: "body" }, [
        make("div", { class: "say" }, lines.map((t) => make("p", { text: t }))),
        make("div", {}, [
          make("div", { class: "meta", text: "What to say, one breath each" }),
          make("div", { class: "navs" }, [
            prev ? make("a", { class: "btn ghost", href: prev.href, text: "← " + prev.title }) : null,
            next ? make("a", { class: "btn", href: next.href, text: next.title + " →" }) : make("a", { class: "btn ghost", href: "/demo", text: "Overview" }),
          ]),
        ]),
      ]),
    ]);
    document.body.append(drawer);
  }

  // Only the presenter's own toggle is remembered: the overview opening its notes by
  // default must not leave them open over every scene on the projector.
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
      const scene = SCENES.find((s) => s.key === key);
      if (ev.key === "ArrowRight") {
        const next = scene ? SCENES[scene.n] : SCENES[0];
        if (next) location.href = next.href;
      } else if (ev.key === "ArrowLeft") {
        const prev = scene ? SCENES[scene.n - 2] : null;
        location.href = prev ? prev.href : "/demo";
      } else if (ev.key === "n" || ev.key === "N") {
        toggleNotes();
      } else if (ev.key === "f" || ev.key === "F") {
        if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => {});
      } else if (/^[1-6]$/.test(ev.key)) {
        location.href = SCENES[Number(ev.key) - 1].href;
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
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#2a63e8"/><path d="M5 16h3l3-8 5 16 4-12 2 4h5" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    document.head.append(make("link", { rel: "icon", href: "data:image/svg+xml," + encodeURIComponent(svg) }));
  }

  async function mount(key) {
    current = SCENES.find((s) => s.key === key) || null;
    favicon();
    document.body.classList.add("stage");
    document.body.prepend(topbar(key));
    keys(key);
    pills();
    setInterval(pills, 10000);
    if (key === "home") return; // the project overview is not a scene: no presenter notes
    await presenter(key);
    let open = false;
    try { open = localStorage.getItem("stage.notes") === "1"; } catch (e) { /* private window */ }
    if (key === "hub" && window.innerHeight > 900) open = true; // the overview opens them on a tall screen
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

  // The presenter's lines for one scene, numbers filled from the results API.
  async function lines(key) {
    return (SAY[key] || (() => []))(await sayData());
  }

  return { SCENES, mount, make, json, cite, pct, toast, socket, word, shortModel, lines };
})();
