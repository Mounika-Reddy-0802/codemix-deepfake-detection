// The final-review demo's shared shell: scene stepper, readiness light, presenter notes,
// the verdict key, keyboard navigation and source tooltips. Every page calls
// Stage.mount("<key>").
//
// No number lives in this file. Where the presenter's line carries one, it is filled
// from /api/headline or /api/real_calls, which read the committed result files.

const Stage = (() => {
  // Four scenes, about eight minutes: one message each.
  const SCENES = [
    { key: "threat", n: 1, title: "The threat", href: "/threat", minutes: "0:45" },
    { key: "lanes", n: 2, title: "Four detectors", href: "/lanes", minutes: "2:00" },
    { key: "live", n: 3, title: "Live call", href: "/live", minutes: "3:00" },
    { key: "limits", n: 4, title: "Limits", href: "/limits", minutes: "2:00" },
  ];
  // For questions: kept out of the flow, reached from scene 4's "More" menu.
  const MORE = [
    { key: "network", title: "Real network", href: "/network", what: "the 30 real mobile calls and what the codec does" },
    { key: "trust", title: "Why trust it", href: "/trust", what: "shuffle a call into 100 ms pieces: same verdict" },
    { key: "try", title: "Try it", href: "/try", what: "read a line into the microphone, judged as a call" },
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
    threat: (d) => [
      `A familiar voice calls in Hindi-English and asks for an OTP. The usual detector cannot help: it calls ${d.s1Fake} of genuine Hindi-English speech fake.`,
    ],
    lanes: () => [
      "Same call, four detectors. Play the genuine caller: the English-only detector rejects a real person; ours let them through.",
      "Now the Tortoise clone, a tool none of them was trained on: the final recipe catches it. The larger encoder is one checkpoint, our next candidate.",
    ],
    live: () => [
      "Please hold this phone. A clone is calling you. [amber] A caution. [red, spoken] Likely cloned.",
      "The caller sees nothing, because a caller who knows they were caught learns to evade.",
    ],
    limits: () => [
      "What it cannot do yet, with numbers: spontaneous speech, unseen tools, English over the phone line.",
      "The larger encoder does better on the first two, but it is one checkpoint: the next candidate, not a replacement.",
    ],
    network: (d) => [
      `Thirty real mobile calls, scores only: the final recipe holds where the codec breaks the old model, and catches ${d.xttsCaught} of ${d.xttsCalls} seen-tool clones.`,
    ],
    trust: () => ["Scramble the words: same verdict. It reads how the voice was produced, not what is said."],
    try: () => ["Read the line on the screen; it is judged exactly as a live call."],
    hub: () => ["Four scenes, about eight minutes. Keys 1 to 4 jump to a scene; N opens these notes; F goes full screen."],
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
        make("span", {}, [make("b", { text: "Cloned-voice call detection" }), make("small", { text: "Hindi-English phone calls" })]),
      ]),
      steps,
      make("div", { class: "pills", id: "stagePills" }, [make("span", { class: "pill", text: "connecting…" })]),
    ]);
  }

  function shortModel(name) {
    if (!name) return "no model";
    if (name.startsWith("final_")) return NAMES.final;
    if (name.startsWith("lora_norm_rvc")) return NAMES.deployed;
    if (name.includes("ssl")) return NAMES.ssl;
    if (name.startsWith("best")) return NAMES.s1; // the baseline folder holds the English-only model
    return "another model";
  }

  // One light for the room, the details for the presenter's mouse.
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
      box.replaceChildren(make("span", { class: "pill bad", text: "Server unreachable" }));
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

  async function presenter(key) {
    const scene = SCENES.find((s) => s.key === key);
    const more = MORE.find((m) => m.key === key);
    const lines = (SAY[key] || (() => []))(await sayData());
    const prev = scene ? SCENES[scene.n - 2] : more ? SCENES.at(-1) : null;
    const next = scene ? SCENES[scene.n] : more ? null : SCENES[0];
    const drawer = make("aside", { class: "presenter", id: "presenter", "aria-label": "Presenter notes" }, [
      make("div", { class: "handle" }, [
        keyLine(),
        make("span", { class: "spacer" }),
        make("button", { class: "notes-btn", type: "button", onclick: () => toggleNotes(), html: "notes <kbd>N</kbd>" }),
      ]),
      make("div", { class: "body" }, [
        make("div", { class: "say" }, [
          make("div", { class: "meta", text: scene ? `Scene ${scene.n} of ${SCENES.length} · ${scene.title} · ${scene.minutes}` : more ? `For questions · ${more.title}` : "Demo overview · about eight minutes" }),
          ...lines.map((t) => make("p", { text: t })),
        ]),
        make("div", { class: "navs" }, [
          prev ? make("a", { class: "btn ghost", href: prev.href, text: "← " + prev.title }) : null,
          next ? make("a", { class: "btn", href: next.href, text: next.title + " →" }) : make("a", { class: "btn ghost", href: "/demo", text: "Overview" }),
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
      const more = MORE.some((m) => m.key === key);
      if (ev.key === "ArrowRight") {
        const next = scene ? SCENES[scene.n] : more ? null : SCENES[0];
        if (next) location.href = next.href;
      } else if (ev.key === "ArrowLeft") {
        const prev = scene ? SCENES[scene.n - 2] : more ? SCENES.at(-1) : null;
        location.href = prev ? prev.href : "/demo";
      } else if (ev.key === "n" || ev.key === "N") {
        toggleNotes();
      } else if (ev.key === "f" || ev.key === "F") {
        if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => {});
      } else if (ev.key >= "1" && ev.key <= String(SCENES.length)) {
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

  // A scene renamed by its page (scene 2 counts the detectors it was given): stepper and notes.
  function retitle(key, title) {
    const s = SCENES.find((x) => x.key === key);
    if (!s) return;
    s.title = title;
    const step = document.querySelector(`.steps a[href="${s.href}"]`);
    if (step) {
      step.querySelector(".t").textContent = title;
      step.title = `Scene ${s.n}: ${title} (${s.minutes})`;
    }
    const meta = document.querySelector("#presenter .say .meta");
    if (meta && current && current.key === key) meta.textContent = `Scene ${s.n} of ${SCENES.length} · ${title} · ${s.minutes}`;
  }

  return { SCENES, MORE, NAMES, ABOUT, mount, make, json, cite, pct, toast, socket, word, shortModel, nameOf, details, lines, retitle };
})();
