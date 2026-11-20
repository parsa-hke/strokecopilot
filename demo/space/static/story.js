/* Stroke Decision Copilot — the story page (DEC-32 §3.1), with three reader challenges. Every value comes from the Space's own API and is set with
   textContent. The patient is fixture fx01 (scripted). The answers and flags are the models' real runs on that fixture. */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const h = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = String(text); return n; };
  const NS = "http://www.w3.org/2000/svg";
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const narrow = () => matchMedia("(max-width: 900px)").matches;
  const getJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u); return r.json(); });
  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const later = (fn, ms) => setTimeout(fn, reduce ? 0 : ms);
  const hm = (m) => { const t = Math.round(m); return `${Math.floor(t / 60)} h ${String(t % 60).padStart(2, "0")}`; };
  const CASE = "fx01_thrombolysis_clear", MODEL = "claude-sonnet-5-5";
  const LABEL = { "claude-sonnet-5-5": "Sonnet 5.5", "qwen3.5-27b": "Qwen3.5-27B", "gemma-3-27b": "Gemma 3 27B", "medgemma-27b-text": "MedGemma 27B",
    "gemma-3-4b": "Gemma 3 4B", "gemma-3-4b-tuned": "Gemma 3 4B tuned" };
  const S = {}, scenes = {};
  const score = { spot: undefined, guess: undefined, verify: undefined };   // the reader's three challenges
  let reached = 0;   // furthest chapter seen, for the case file

  /* ------------------------------------------------ the case file (HUD) and the scorecard */
  function pip(n) { $$("#hud-pips i")[n].classList.add("on"); renderScore(); }
  function renderHud() {
    if (!S.file) return;
    const v = S.file.arms.rulebook.vision, R = S.file.rules, rec = S.ans.recommendation, c = S.ans.counts;
    const note = S.file.arms.rulebook.note.split("\n").slice(1, 5).join(" ");
    const rows = [[1, "Patient", note],
      [2, "Scan numbers", `Core ${v.core_ml} mL · at risk ${v.penumbra_ml} mL · clot in ${v.occlusion_site} · ASPECTS ${v.aspects} · ${v.haemorrhage.present ? "bleed" : "no bleed"}`],
      [3, "Rulebook", `Drug: ${R.thrombolysis.value}. Removal: ${R.thrombectomy.value}.`],
      [4, "Sonnet 5.5", `${rec.thrombolysis.recommend ? "Give" : "Do not give"} the drug; ${rec.thrombectomy.recommend ? "attempt" : "do not attempt"} removal.`],
      [5, "Verifier", `${c.claims - c.flagged} of ${c.claims} claims hold; ${c.critical} marked critical.`]];
    $("#hud-list").replaceChildren(...rows.map(([n, k, text]) => { const d = h("div", reached >= n ? "" : "locked"); d.append(h("dt", null, k), h("dd", null, reached >= n ? text : "Keep scrolling")); return d; }));
  }
  function renderScore() {
    const spot = { core: ["In the core", "Your tap landed in the tissue the model marked as lost."], risk: ["In the tissue at risk", "Inside the area at risk, outside the core."],
      out: ["Missed it", "Early strokes barely show on a plain CT."] }[score.spot] || ["Skipped", "You did not tap the scan."];
    const g = score.guess, v = score.verify;
    const tiles = [["Spot the stroke", ...spot], ["Call the model", g && g.made ? `${g.hits} of ${g.made}` : "Skipped", g && g.made ? "Guesses that matched Sonnet 5.5." : "You did not guess."],
      ["Be the verifier", v ? `Caught ${v.caught} of ${v.bad}` : "Skipped", v ? (v.extra ? `You also doubted ${v.extra} that hold${v.extra === 1 ? "s" : ""}.` : v.picked ? "No false alarms." : "You trusted every claim.") : "You did not reach the marking."]];
    $("#score").replaceChildren(...tiles.map(([k, val, sub]) => { const d = h("div", "sc"); d.append(h("div", "k", k), h("div", "v", val), h("div", "s", sub)); return d; }));
  }

  /* ------------------------------------------------ 1 · the clock: scrolling moves time from onset to the scan */
  scenes.ch1 = {
    step(i) { if (i === 0) scramble($(".scramble")); },
    progress(p) {
      const mins = 13 * 60 + 30 + 150 * p, hh = Math.floor(mins / 60), mm = Math.floor(mins % 60), gone = Math.round(150 * p);
      const t = `${hh}:${String(mm).padStart(2, "0")}`;
      $("#clk-time").textContent = t; $("#hud-clock").textContent = t;
      $("#clk-since").textContent = gone < 1 ? "onset" : `${Math.floor(gone / 60)} h ${String(gone % 60).padStart(2, "0")} min since onset`;
      $("#d-fill").style.strokeDasharray = `${(gone / 270) * 553} 553`;
    },
  };
  function scramble(el) {   // her words, briefly lost
    if (!el || reduce || el.dataset.busy) return;
    const text = el.dataset.text, abc = "abcdefghijklmnopqrstuvwxyz"; let n = 0; el.dataset.busy = "1";
    const id = setInterval(() => {
      n++; el.textContent = [...text].map((c, k) => (c === " " || k < (n - 10) * 1.2 ? c : abc[Math.floor(Math.random() * 26)])).join("");
      if (n > 24) { clearInterval(id); el.textContent = text; delete el.dataset.busy; }
    }, 70);
  }

  /* ------------------------------------------------ 2 · the scan: challenge 1, then the model's outline */
  scenes.ch2 = (() => {
    const viewer = $("#viewer"), base = $("#scan-base"), top = $("#scan-top"), handle = $("#wipe-handle"), pinEl = $("#pin");
    const SER = [["story_ncct.png", "Plain CT"], ["story_cbf.png", "Blood flow"], ["story_tmax.png", "Tmax"], ["model", "Model view"]];
    let revealed = false, pos = 100, mask = null, tap = null, series = "story_ncct.png";
    function setPos(p, animate) {
      pos = clamp(p, 0, 100);
      const t = animate && !reduce ? "clip-path .9s cubic-bezier(.2,.7,.2,1), left .9s cubic-bezier(.2,.7,.2,1)" : "none";
      top.style.transition = t; handle.style.transition = t; top.style.clipPath = `inset(0 0 0 ${pos}%)`; handle.style.left = `${pos}%`;
    }
    function setSeries(k, animate) {
      series = k;
      base.src = `/api/bank/img/${k === "model" ? "story_ncct.png" : k}`;
      setPos(k === "model" ? 3 : 100, animate); handle.hidden = k !== "model" || !revealed;
      $$("#series button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.k === k)));
      const name = SER.find(([f]) => f === k)[1];
      $("#scan-label").textContent = `${name} · ISLES'24 case ${S.story ? S.story.alias : "D-10"}`;
    }
    async function buildMask() {   // read the model's red and amber marks back out of the overlay image
      try {
        await Promise.all([base.decode(), top.decode()]);
        const n = 180, c = document.createElement("canvas"); c.width = c.height = n; const g = c.getContext("2d", { willReadFrequently: true });
        g.drawImage(top, 0, 0, n, n); const a = g.getImageData(0, 0, n, n).data;
        mask = new Uint8Array(n * n);
        for (let i = 0; i < n * n; i++) { const r = a[i * 4], gg = a[i * 4 + 1], b = a[i * 4 + 2];
          if (r > 150 && r > gg * 1.6 && r > b * 1.6) mask[i] = 2; else if (r > 120 && gg > 80 && r > b * 1.5 && gg > b * 1.25) mask[i] = 1; }
      } catch (e) { mask = null; }
    }
    function classify(x, y) {
      if (!mask) return "out";
      const n = 180, cx = Math.round(x * n), cy = Math.round(y * n); let best = 0;
      for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) { const X = cx + dx, Y = cy + dy; if (X >= 0 && Y >= 0 && X < n && Y < n) best = Math.max(best, mask[Y * n + X]); }
      return best === 2 ? "core" : best === 1 ? "risk" : "out";
    }
    function reveal() {
      if (revealed) return; revealed = true;
      $("#scan-prompt").style.opacity = 0; viewer.classList.add("wiping"); $("#series").hidden = false;
      $("#scan-badge").textContent = "Challenge 1 done"; $("#scan-badge").classList.add("done");
      score.spot = tap ? classify(tap.x, tap.y) : null;
      const say = { core: ["Your tap landed in the ", "predicted core", "."], risk: ["Your tap landed in the ", "tissue at risk", ", outside the core."],
        out: ["Your tap landed ", "outside", " everything the model marked."] }[score.spot] || ["You did not tap. Here is what the model marked.", "", ""];
      $("#scan-result").replaceChildren(say[0], h("b", null, say[1]), say[2]);
      setSeries("model", true); pip(0);
    }
    viewer.addEventListener("pointerdown", (e) => {
      const r = viewer.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      if (!revealed) { tap = { x, y }; pinEl.hidden = false; pinEl.style.left = `${x * 100}%`; pinEl.style.top = `${y * 100}%`; later(reveal, 650); return; }
      if (series !== "model") return;
      viewer.setPointerCapture(e.pointerId); viewer.dataset.drag = "1"; setPos(x * 100);
    });
    viewer.addEventListener("pointermove", (e) => { if (!viewer.dataset.drag) return; const r = viewer.getBoundingClientRect(); setPos(((e.clientX - r.left) / r.width) * 100); });
    const stop = () => { delete viewer.dataset.drag; }; viewer.addEventListener("pointerup", stop); viewer.addEventListener("pointercancel", stop);
    viewer.addEventListener("keydown", (e) => {
      if (!revealed && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); reveal(); return; }
      const d = { ArrowLeft: -5, ArrowRight: 5 }[e.key]; if (d && revealed && series === "model") { e.preventDefault(); setPos(pos + d); }
    });
    return {
      build() {
        const box = $("#series"); box.hidden = true;
        SER.forEach(([k, label]) => { const b = h("button", null, label); b.type = "button"; b.dataset.k = k; b.setAttribute("aria-pressed", "false"); b.addEventListener("click", () => setSeries(k, true)); box.append(b); });
        setPos(100); buildMask();
      },
      step(i) {
        if (i >= 1) reveal();
        if (revealed) setSeries(i === 2 ? "story_ncct.png" : "model", true);
        $("#scanline").classList.toggle("on", i === 2 && !reduce);
        if (i === 2) $("#scan-label").textContent = "Bleed check · plain CT";
        const box = $("#scan-chips");
        if (i === 3 && !box.childElementCount) {
          const v = S.file.arms.rulebook.vision;
          [`core ${v.core_ml} mL`, `at risk ${v.penumbra_ml} mL`, `clot: ${v.occlusion_site}`, `ASPECTS ${v.aspects}`, `bleed: ${v.haemorrhage.present ? "yes" : "no"}`]
            .forEach((t, k) => { const c = h("span", "chip", t); box.append(c); later(() => c.classList.add("in"), 120 * k + 60); });
        } else if (i !== 3) box.replaceChildren();
      },
    };
  })();

  /* ------------------------------------------------ 3 · the rulebook: read a passage, then run the rule yourself */
  scenes.ch3 = (() => {
    const tiles = {}, fed = { thrombolysis: new Set(), thrombectomy: new Set() }; let showing = null;
    const NAME = { thrombolysis: "the clot-busting drug", thrombectomy: "clot removal" }, WORD = { yes: "Yes", no: "No", undetermined: "Undetermined" };
    function factText(k) {
      const f = S.row.features;
      return { onset_to_imaging: `${hm(f.minutes)} from onset`, lkw_to_imaging: `${hm(f.minutes)} since last well`, nihss: `NIHSS ${f.nihss}`, "haemorrhage.present": f.haemorrhage ? "Bleed" : "No bleed",
        occlusion_site: `Clot in ${f.site}`, premorbid_mrs: `mRS ${f.mrs}`, aspects: `ASPECTS ${f.aspects}`, m2_dominance: "M2 dominant?" }[k] || k.replace(/[._]/g, " ");
    }
    function read(id) {
      const p = S.passages[id]; if (!p) return;
      $$(".tile.sel").forEach((t) => t.classList.remove("sel")); tiles[id].classList.add("sel");
      $("#reader .rid").textContent = `${id} · ${p.pointer}`; $("#reader .rtext").textContent = p.text;
    }
    function light(ids) { Object.entries(tiles).forEach(([id, t]) => t.classList.toggle("lit", ids.includes(id))); if (ids[0]) read(ids[0]); }
    function settle(k) {
      const r = S.file.rules[k], done = fed[k].size >= r.deciding.length, v = $("#r-verdict");
      v.textContent = done ? WORD[r.value] || r.value : "?"; v.classList.toggle("in", done);
      $("#r-why").textContent = done ? r.reason : `${r.deciding.length - fed[k].size} to go`;
      light(done ? r.rule_ids.filter((id) => tiles[id]) : []);
      if (done) renderHud();
    }
    function show(k) {
      showing = k; const r = S.file.rules[k], box = $("#r-facts");
      $("#runner").hidden = false; $("#r-title").textContent = `Rule for ${NAME[k]} · tap each fact`;
      box.replaceChildren(...r.deciding.map((key) => {
        const b = h("button", `fact${fed[k].has(key) ? " fed" : ""}`, factText(key)); b.type = "button";
        b.addEventListener("click", () => { fed[k].add(key); b.classList.add("fed"); settle(k); }); return b; }),
      ...(r.unknown || []).map((key) => {
        const b = h("button", "fact unknown", factText(key)); b.type = "button"; b.title = "The system cannot see this";
        b.addEventListener("click", () => { b.classList.remove("shake"); void b.offsetWidth; b.classList.add("shake"); $("#r-why").textContent = "The system cannot see this, so the rule cannot finish."; }); return b; }));
      settle(k);
    }
    return {
      build() {
        const box = $("#tiles");
        Object.entries(S.passages).forEach(([id, p]) => {
          const b = h("button", "tile"); b.type = "button"; b.title = id; b.dataset.t = p.treatment; b.setAttribute("role", "listitem"); b.setAttribute("aria-label", `Passage ${id}`);
          b.addEventListener("click", () => read(id)); box.append(b); tiles[id] = b;
        });
        const F = [["all", "All"], ["thrombolysis", "Drug"], ["thrombectomy", "Removal"]];
        $("#filters").replaceChildren(...F.map(([k, l], n) => { const b = h("button", null, l); b.type = "button"; b.setAttribute("aria-pressed", String(n === 0));
          b.addEventListener("click", () => { $$("#filters button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
            Object.values(tiles).forEach((t) => t.classList.toggle("off", k !== "all" && t.dataset.t !== k && t.dataset.t !== "both")); }); return b; }));
      },
      finish() { for (const k of ["thrombolysis", "thrombectomy"]) S.file.rules[k].deciding.forEach((x) => fed[k].add(x)); if (showing) show(showing); },
      step(i) {
        if (i === 0) { $("#runner").hidden = true; light([]); showing = null; return; }
        if (i >= 2) S.file.rules.thrombolysis.deciding.forEach((x) => fed.thrombolysis.add(x));   // scrolled past: the rule finishes itself
        show(i === 1 ? "thrombolysis" : "thrombectomy");
      },
    };
  })();

  /* ------------------------------------------------ 4 · your call: challenge 2, then all six models */
  scenes.ch4 = (() => {
    const guess = {}; let revealed = false;
    const ROWS = [["thrombolysis", "Clot-busting drug", ["Give it", "Don't"]], ["thrombectomy", "Clot removal", ["Attempt it", "Don't"]]];
    const said = (k, v) => (v == null ? "–" : v ? ROWS.find((r) => r[0] === k)[2][0] : "Don't");
    function reveal() {
      if (revealed) return; revealed = true;
      const rec = S.ans.recommendation, R = S.file.rules;
      const t = h("table"), head = h("tr"); ["", "You", "Rulebook", "Sonnet 5.5"].forEach((x) => head.append(h("th", null, x))); t.append(head);
      let hits = 0, made = 0;
      ROWS.forEach(([k, label]) => {
        const tr = h("tr"), mine = guess[k], model = rec[k].recommend;
        if (mine != null) { made++; if (mine === model) hits++; }
        const cells = [label, said(k, mine), R[k].value === "undetermined" ? "Undetermined" : R[k].value === "yes" ? "Yes" : "No", said(k, model)].map((x) => h("td", null, x));
        if (mine != null) cells[1].className = mine === model ? "hit" : "miss";
        tr.append(...cells); t.append(tr);
      });
      score.guess = { hits, made };
      const out = $("#reveal"); out.replaceChildren(t, h("p", "say", made ? `You matched Sonnet on ${hits} of ${made}.` : "You did not guess, so here is its answer."));
      out.hidden = false; $("#lock").classList.add("locked"); $("#ask").classList.add("locked"); $(".q").textContent = "Sonnet 5.5 recommends";
      const bd = $("#viz-call .badge"); bd.textContent = "Challenge 2 done"; bd.classList.add("done");
      pip(1);
    }
    function others() {
      const box = $("#others"); if (box.childElementCount) { box.hidden = false; return; }
      const ans = S.file.arms.rulebook.answers, ref = S.ans.recommendation;
      box.append(h("p", "oh", "All six models, same file"));
      Object.keys(LABEL).filter((m) => ans[m] && ans[m].available).forEach((m, n) => {
        const r = ans[m].recommendation, c = ans[m].counts, row = h("div", "om");
        if (r.thrombolysis.recommend !== ref.thrombolysis.recommend || r.thrombectomy.recommend !== ref.thrombectomy.recommend) row.classList.add("odd");
        const name = h("span", null, LABEL[m]); name.append(h("small", null, ` · ${c.flagged} of ${c.claims} claims flagged`));
        row.append(name, h("span", `a${r.thrombolysis.recommend ? " y" : ""}`, r.thrombolysis.recommend ? "drug: give" : "drug: don't"),
          h("span", `a${r.thrombectomy.recommend ? " y" : ""}`, r.thrombectomy.recommend ? "removal: attempt" : "removal: don't"));
        box.append(row); later(() => row.classList.add("in"), 90 * n + 60);
      });
      box.hidden = false;
    }
    return {
      build() {
        const box = $("#ask");
        ROWS.forEach(([k, label, opts]) => {
          const row = h("div", "row"), lbl = h("div", "lbl", label); lbl.append(h("small", null, k));
          const seg = h("div", "seg"); seg.setAttribute("role", "group"); seg.setAttribute("aria-label", label);
          opts.forEach((o, n) => { const b = h("button", null, o); b.type = "button"; b.setAttribute("aria-pressed", "false");
            b.addEventListener("click", () => { guess[k] = n === 0; $$("button", seg).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
              $("#lock").disabled = !(guess.thrombolysis != null && guess.thrombectomy != null); });
            seg.append(b); });
          row.append(lbl, seg); box.append(row);
        });
        $("#lock").addEventListener("click", reveal);
      },
      step(i) { if (i >= 1) reveal(); if (i >= 2) { others(); $("#viz-call").dataset.others = "1"; } else { $("#others").hidden = true; delete $("#viz-call").dataset.others; } },
    };
  })();

  /* ------------------------------------------------ 5 · the marking: challenge 3, then the real verifier */
  scenes.ch5 = (() => {
    let items = [], closed = false, fake = null, near = null, looked = false;
    function close() {
      if (closed) return; closed = true;
      const list = $("#claims"); list.classList.remove("open"); list.classList.add("closed");
      $("#mk-prompt").hidden = true; $("#mk-badge").textContent = "Challenge 3 done"; $("#mk-badge").classList.add("done");
      const picked = items.filter((li) => li.classList.contains("sus")), bad = items.filter((li) => li.classList.contains("bad"));
      score.verify = { picked: picked.length, bad: bad.length, caught: picked.filter((li) => li.classList.contains("bad")).length, extra: picked.filter((li) => !li.classList.contains("bad")).length };
      pip(2);
    }
    function lookup() {   // the citation check, drawn: the cited name is looked for among the 47 and is not there
      if (looked || !fake) return; looked = true;
      const cells = $$("#lk-strip i"), ids = Object.keys(S.passages), hit = ids.indexOf(near), say = $("#lk-say");
      say.replaceChildren("Looking for ", h("b", null, fake), " among the 47…");
      cells.forEach((c, k) => { later(() => c.classList.add("scan"), 16 * k); later(() => c.classList.remove("scan"), 16 * k + 90); });
      later(() => { if (cells[hit]) cells[hit].classList.add("near");
        say.replaceChildren("Sonnet cited ", h("b", null, fake), ". No passage has that name. The closest is ", h("em", null, near), "."); }, 16 * cells.length + 150);
    }
    return {
      build() {
        const claims = S.ans.claims, list = $("#claims"); list.classList.add("open");
        $("#mk-count").textContent = `${claims.length} claims · Sonnet 5.5`;
        items = claims.map((c) => {
          const li = h("li", c.flagged ? "bad" : ""), svg = document.createElementNS(NS, "svg"), p = document.createElementNS(NS, "path");
          svg.setAttribute("class", "m"); svg.setAttribute("viewBox", "0 0 30 26"); svg.setAttribute("aria-hidden", "true");
          p.setAttribute("d", c.flagged ? "M6 4 L24 22 M24 4 L6 22" : "M4 14 L11 21 L27 3"); svg.append(p);
          const body = h("div"); body.append(c.text, h("span", "tag", "you doubted"));
          if (c.flagged) body.append(h("span", "why", `${c.reason}${c.critical ? " Marked critical: it would change a decision." : ""}`));
          li.append(svg, body); li.tabIndex = 0; li.setAttribute("role", "button"); li.setAttribute("aria-pressed", "false");
          const toggle = () => { if (closed) return; li.classList.toggle("sus"); li.setAttribute("aria-pressed", String(li.classList.contains("sus"))); };
          li.addEventListener("click", toggle); li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
          list.append(li); return li;
        });
        const cited = [...S.ans.recommendation.thrombolysis.rule_ids, ...S.ans.recommendation.thrombectomy.rule_ids];
        fake = cited.find((id) => !S.passages[id]) || null;
        if (fake) { const tok = (s) => new Set(s.split("-")), ft = tok(fake);
          near = Object.keys(S.passages).map((id) => [[...tok(id)].filter((t) => ft.has(t)).length, id]).sort((a, b) => b[0] - a[0])[0][1]; }
        $("#lk-strip").replaceChildren(...Object.keys(S.passages).map(() => h("i")));
      },
      step(i) {
        $("#viz-mark").dataset.step = i;
        if (i >= 1) close();
        items.forEach((li, k) => {
          const bad = li.classList.contains("bad");
          later(() => li.classList.add("in"), i === 0 ? 70 * k : 0);
          later(() => li.classList.toggle("checked", bad ? i >= 2 : i >= 1), i === 1 ? 110 * k : 0);
          li.classList.toggle("fade", i === 2 && !bad);
        });
        $("#lookup").hidden = i < 2 || !fake; if (i >= 2) lookup();
        const v = score.verify, out = $("#mk-result");
        if (i >= 2 && v) out.replaceChildren(v.picked ? "You doubted " : "You trusted every claim. ", ...(v.picked ? [h("b", null, `${v.picked}`), ` and caught ${v.caught} of the ${v.bad} that fail.`,
          v.extra ? ` ${v.extra} of your doubts ${v.extra > 1 ? "were" : "was"} a false alarm.` : ""] : [`${v.bad} of them fail.`]));
        else out.replaceChildren();
      },
    };
  })();

  /* ------------------------------------------------ 6 · the bigger picture */
  scenes.ch6 = (() => {
    const METRICS = [["hallucination_rate", "All flags", "Share of claims the verifier flagged"], ["content_hallucination_rate", "False content", "Share of claims flagged for false content"],
      ["critical_hallucination_rate", "Critical", "Share of claims flagged that would change a decision"]];
    let metric = METRICS[0][0];
    function draw() {
      const rows = S.charts.c9.rows, max = Math.max(...rows.map((r) => r[metric].value));
      $$("#bars .bar").forEach((b) => { const r = rows.find((x) => x.model === b.dataset.m), v = r[metric].value;
        b.querySelector(".track i").style.width = `${(v / max) * 100}%`; b.querySelector(".n").textContent = `${(v * 100).toFixed(1)}%`; });
      $("#bars-cap").textContent = `${METRICS.find((m) => m[0] === metric)[2]}, by model.`;
    }
    return {
      build() {
        const cases = S.index.cases, isReal = (c) => c.kind === "development" || c.kind === "heldout", tip = $("#tip");
        $("#dots").replaceChildren(...cases.map((c) => {
          const d = h("i", c.id === CASE ? "hero" : isReal(c) ? "real" : "");
          d.addEventListener("pointerenter", (e) => { const r = c.rules || {};
            tip.replaceChildren(h("b", null, c.id === CASE ? `${c.title} (her case)` : c.title), ...(r.thrombolysis ? [h("div", null, `Rulebook, drug: ${r.thrombolysis.value}`), h("div", null, `Rulebook, removal: ${r.thrombectomy.value}`)] : []));
            tip.hidden = false; tip.style.left = `${Math.min(e.clientX + 14, innerWidth - 290)}px`; tip.style.top = `${e.clientY + 14}px`; });
          d.addEventListener("pointerleave", () => { tip.hidden = true; }); return d; }));
        const n = (k) => cases.filter((c) => c.kind === k).length;
        $("#dots-cap").textContent = `${n("scenario")} scenarios, ${n("fixture")} fixtures and ${cases.filter(isReal).length} real patients (ISLES'24, brighter). The red one is hers.`;
        $("#n-cases").textContent = cases.length; $("#n-cases2").textContent = cases.length;
        const rows = [...S.charts.c9.rows].sort((a, b) => a.hallucination_rate.value - b.hallucination_rate.value);
        $("#bars").replaceChildren(...rows.map((r) => { const b = h("div", `bar${r.model === MODEL ? " hero" : ""}`), tr = h("div", "track"); tr.append(h("i")); b.dataset.m = r.model;
          b.append(h("span", null, LABEL[r.model] || r.model), tr, h("span", "n", "")); return b; }));
        $("#metric").replaceChildren(...METRICS.map(([k, l], i) => { const b = h("button", null, l); b.type = "button"; b.setAttribute("aria-pressed", String(i === 0));
          b.addEventListener("click", () => { metric = k; $$("#metric button").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); draw(); }); return b; }));
        const [yes, all] = S.charts.human.flags_confirmed;
        $("#flags").replaceChildren(...Array.from({ length: all }, (_, k) => { const s = h("i"); s.dataset.yes = k < yes ? "1" : ""; return s; }));
        $("#flags-cap").textContent = `Each square is one verifier flag. People confirmed ${yes} of ${all}.`;
        $("#n-conf").textContent = `${yes} of its ${all}`;
      },
      step(i) {
        $("#pane-cases").hidden = i !== 0; $("#pane-bars").hidden = i !== 1; $("#pane-human").hidden = i !== 2; $("#metric").hidden = i !== 1;
        $("#big-label").textContent = ["All cases", "Six models", "The check on the checker"][i] || "";
        if (i === 0) $$("#dots i").forEach((d, k) => later(() => d.classList.add("in"), 4 * k));
        if (i === 1) later(draw, 60);
        if (i === 2) $$("#flags i").forEach((f, k) => later(() => f.classList.toggle("yes", !!f.dataset.yes), 30 * k));
      },
    };
  })();

  /* ------------------------------------------------ the scroll engine */
  const current = {};
  function setStep(ch, i) {
    if (current[ch] === i) return; current[ch] = i;
    $$(`#${ch} .step`).forEach((s, k) => s.classList.toggle("on", k === i));
    const n = +ch.slice(2); if (n > reached) { if (reached < 3 && n > 3) scenes.ch3.finish(); reached = n; renderHud(); }
    scenes[ch].step(i);
  }
  function watch() {
    const line = narrow() ? "-74% 0px -26% 0px" : "-50% 0px -50% 0px";
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { const sec = e.target.closest(".scrolly"); setStep(sec.id, $$(".step", sec).indexOf(e.target)); } }), { rootMargin: line });
    $$(".scrolly .step").forEach((s) => io.observe(s));
    const tick = () => {
      const steps = $$("#ch1 .step"), mid = innerHeight * (narrow() ? 0.74 : 0.5), c = (el) => { const r = el.getBoundingClientRect(); return r.top + r.height / 2; };
      scenes.ch1.progress(clamp((mid - c(steps[0])) / (c(steps[steps.length - 1]) - c(steps[0]))));
      $("#progress").style.width = `${clamp(scrollY / (document.documentElement.scrollHeight - innerHeight)) * 100}%`;
      $("#hud").classList.toggle("live", scrollY > innerHeight * 0.6 && $("#ch7").getBoundingClientRect().top > innerHeight * 0.5);
    };
    let queued = false;
    addEventListener("scroll", () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; tick(); }); }, { passive: true });
    addEventListener("resize", tick); tick();
    // the room's light follows the chapter; title cards rise in once
    const tones = $$("[data-tone]").filter((el) => el !== document.body);
    const toneIO = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) document.body.dataset.tone = e.target.dataset.tone; }), { rootMargin: "-50% 0px -50% 0px" });
    tones.forEach((el) => toneIO.observe(el));
    const seenIO = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) e.target.classList.add("seen"); }), { threshold: 0.35 });
    $$(".chap").forEach((el) => seenIO.observe(el));
    // the chapter rail
    const secs = $$("main > section"), rail = $("#rail");
    const links = secs.map((s) => { const a = h("a"); a.href = `#${s.id}`; a.append(h("span", null, s.dataset.title), h("i")); a.setAttribute("aria-label", s.dataset.title); rail.append(a); return a; });
    const railIO = new IntersectionObserver((es) => es.forEach((e) => { if (!e.isIntersecting) return; const k = secs.indexOf(e.target);
      links.forEach((a, j) => { a.classList.toggle("on", j === k); a.classList.toggle("done", j < k); }); }), { rootMargin: "-50% 0px -50% 0px" });
    secs.forEach((s) => railIO.observe(s));
  }

  /* ------------------------------------------------ small touches */
  $$(".frame").forEach((f) => f.addEventListener("pointermove", (e) => { const r = f.getBoundingClientRect(); f.style.setProperty("--mx", `${e.clientX - r.left}px`); f.style.setProperty("--my", `${e.clientY - r.top}px`); }));
  $(".scramble").addEventListener("pointerenter", (e) => scramble(e.target));
  $("#hud-pill").addEventListener("click", () => { const p = $("#hud-panel"); p.hidden = !p.hidden; $("#hud-pill").setAttribute("aria-expanded", String(!p.hidden)); });
  addEventListener("keydown", (e) => { if (e.key === "Escape") { $("#hud-panel").hidden = true; $("#hud-pill").setAttribute("aria-expanded", "false"); } });
  (function countUp() {   // the cover's clock runs up to her two and a half hours
    const el = $("#cover-t"); if (reduce) { el.textContent = "2 h 30"; return; }
    const t0 = performance.now(), run = (now) => { const p = clamp((now - t0) / 1700), e = 1 - Math.pow(1 - p, 3); el.textContent = hm(150 * e); if (p < 1) requestAnimationFrame(run); };
    requestAnimationFrame(run);
  })();

  /* ------------------------------------------------ start */
  Promise.all([getJSON(`/api/bank/case/${CASE}`), getJSON("/api/bank/passages"), getJSON("/api/bank/charts"), getJSON("/api/bank/index"), getJSON("/api/bank/img/story.json").catch(() => null)])
    .then(([file, passages, charts, index, story]) => {
      Object.assign(S, { file, passages, charts, index, story, ans: file.arms.rulebook.answers[MODEL], row: index.cases.find((c) => c.id === CASE) });
      const va = charts.vision_a, vb = charts.vision_b;
      $("#va-fact").textContent = `On ${va.cases} development cases its outline overlaps the truth with a Dice of ${va.dice.mean.toFixed(2)}, and its volume is off by ${va.abs_volume_error_ml.median} mL (median).`;
      $("#vb-fact").textContent = `On ${vb.scans.toLocaleString("en")} scans it had never seen, it caught ${(vb.sensitivity * 100).toFixed(1)}% of bleeds. On this stroke dataset it also raised ${vb.isles_flags} false alarms in ${vb.isles_scans} scans.`;
      ["ch2", "ch3", "ch4", "ch5", "ch6"].forEach((k) => scenes[k].build());
      renderHud(); renderScore(); watch();
    })
    .catch(() => { $(".cover .fine").textContent = "The story could not load its data. Please reload."; });
})();
