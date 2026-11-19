/* Stroke Decision Copilot — the public Space's page (DEC-32). Plain JS, no third-party code. Every value from the
   server is set with textContent, never as HTML. */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  function h(tag, attrs = {}, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid instanceof Node ? kid : String(kid));
    return n;
  }
  const getJSON = async (url) => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); };
  const pct = (v, d = 1) => (v == null ? "n/a" : `${(100 * v).toFixed(d)}%`);
  const hm = (m) => `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`;

  const MODEL_LABEL = { "claude-sonnet-5-5": "Sonnet 5.5", "qwen3.5-27b": "Qwen3.5-27B", "gemma-3-27b": "Gemma 3 27B",
    "medgemma-27b-text": "MedGemma 27B", "gemma-3-4b": "Gemma 3 4B", "gemma-3-4b-tuned": "Gemma 3 4B tuned" };
  const MODELS = Object.keys(MODEL_LABEL);
  const SLOT = Object.fromEntries(MODELS.map((m, i) => [m, `var(--s${i + 1})`]));
  const FIELD_GROUPS = [["rationale", "Rationale"], ["thrombolysis.rule_ids", "Thrombolysis: rules cited"],
    ["thrombolysis.blocking_factors", "Thrombolysis: reasons against"], ["thrombectomy.rule_ids", "Thrombectomy: rules cited"],
    ["thrombectomy.blocking_factors", "Thrombectomy: reasons against"], ["uncertainties", "Uncertainties"]];
  const SUBTYPE = { wrong_number: "Wrong number", invented_finding: "Invented finding", wrong_rule: "Wrong rule",
    misapplied_rule: "Misapplied rule", faulty_derivation: "Faulty derivation", uncited_rule: "Uncited rule",
    misattributed_source: "Mis-attributed source" };
  const CONTRACT_ROWS = [["haemorrhage.present", "Bleed on plain CT"], ["haemorrhage.confidence", "Bleed probability"],
    ["core_ml", "Core (mL)"], ["penumbra_ml", "Tmax > 6 s, core included (mL)"], ["mismatch_ratio", "Mismatch ratio"],
    ["mismatch_volume_ml", "Mismatch volume (mL)"], ["occlusion_site", "Occlusion site"], ["aspects", "ASPECTS"],
    ["confidence.core", "Confidence, core"], ["confidence.penumbra", "Confidence, penumbra"], ["warnings", "Warnings"]];

  const state = { meta: null, index: [], charts: null, passages: {}, cases: {} };

  /* ---------------------------------------------------------------- theme */
  function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem("theme"); } catch (e) { /* storage blocked */ }
    if (saved) document.documentElement.dataset.theme = saved;
    $(".theme").addEventListener("click", () => {
      const dark = getComputedStyle(document.documentElement).colorScheme.includes("dark");
      const next = dark ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
    });
  }

  /* ---------------------------------------------------------------- answer view (shared) */
  function get(obj, path) { return path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj); }
  function fmt(v) {
    if (v === null || v === undefined) return "null (unknown)";
    if (Array.isArray(v)) return v.length ? v.join("; ") : "none";
    if (typeof v === "boolean") return v ? "yes" : "no";
    return String(v);
  }
  function decisionWord(rec) { return rec === true ? "Recommended" : rec === false ? "Not recommended" : "No answer"; }
  function ruleWord(r) { return r === true || r === "yes" ? "yes" : r === false || r === "no" ? "no" : "undetermined"; }

  function noteWithMarks(note, quotes) {
    const box = h("p", { class: "note" });
    const spans = [];
    quotes.filter(Boolean).forEach((q) => {
      const i = note.indexOf(q);
      if (i >= 0) spans.push([i, i + q.length]);
    });
    spans.sort((a, b) => a[0] - b[0]);
    let pos = 0;
    for (const [a, b] of spans) {
      if (a < pos) continue;
      box.append(note.slice(pos, a), h("mark", { text: note.slice(a, b) }));
      pos = b;
    }
    box.append(note.slice(pos));
    return box;
  }

  function renderAnswer(root, view, { title } = {}) {
    root.replaceChildren();
    const rec = view.recommendation || {};
    const dec = view.decisions || {};
    root.append(h("div", { class: "decisions" }, ["thrombolysis", "thrombectomy"].map((d) => {
      const x = dec[d] || {};
      const agrees = x.agrees === true ? h("span", { class: "agree", text: "✓ agrees with the rulebook" })
        : x.agrees === false ? h("span", { class: "disagree", text: "✕ differs from the rulebook" })
        : h("span", { class: "muted", text: "rulebook undetermined: either answer accepted" });
      return h("div", { class: "dec" }, h("div", { class: "k", text: d === "thrombolysis" ? "Thrombolysis (clot-busting drug)" : "Thrombectomy (clot removal)" }),
        h("div", { class: "v", text: decisionWord(rec[d] && rec[d].recommend) }),
        h("div", { class: "vs" }, `Rulebook: ${ruleWord(x.rulebook)} · `, agrees));
    })));
    const c = view.counts || {};
    root.append(h("div", { class: "counts" }, h("span", { text: `${c.claims ?? 0} claims checked` }),
      c.flagged ? h("span", { class: "flag-chip", text: `✕ ${c.flagged} flagged` }) : h("span", { class: "agree", text: "✓ none flagged" }), h("span", { text: `${c.critical ?? 0} would change a decision` })));

    const left = h("div", { class: "pane" });
    const right = h("div", { class: "pane" });
    root.append(h("div", { class: "panes" }, left, right));
    const passageBox = h("div");

    function drawLeft(sel) {
      const fields = new Set(), quotes = [];
      if (sel) {
        [...sel.sources, ...sel.used].forEach((s) => {
          if (s.type === "vision" && (s.field || s.key)) fields.add(s.field || s.key);
          if (s.type === "note") quotes.push(s.quote || s.span);
        });
      }
      const table = h("table", { class: "contract" }, CONTRACT_ROWS.map(([path, label]) => {
        const top = path.split(".")[0];
        return h("tr", { class: fields.has(path) || fields.has(top) ? "hl" : null }, h("td", { text: path }),
          h("td", {}, h("div", { class: "muted small", text: label }), fmt(get(view.vision, path))));
      }));
      left.replaceChildren(h("h4", { text: "Triage note" }), noteWithMarks(view.note || "", quotes),
        h("h4", { text: "Imaging contract (what vision reported)", style: "margin-top:14px" }), table);
      const first = left.querySelector("mark, tr.hl");
      if (first && sel) {
        if (getComputedStyle(left).position === "sticky") left.scrollTop = Math.max(0, first.offsetTop - left.offsetTop - 60);
        else first.scrollIntoView({ block: "nearest" });
      }
    }

    function drawPassages(sel) {
      passageBox.replaceChildren();
      if (!sel) return;
      const ids = [...new Set([...sel.sources.filter((s) => s.type === "passage").map((s) => s.id), ...(sel.supported_passages || [])])];
      ids.forEach((id) => {
        const p = state.passages[id];
        passageBox.append(h("div", { class: "passage" }, h("div", { class: "pid", text: `${id}${p ? ` · ${p.pointer}` : " · not a rulebook passage"}` }),
          p ? h("div", { text: p.text }) : null));
      });
    }

    let selected = null;
    const list = h("div");
    FIELD_GROUPS.forEach(([field, label]) => {
      const items = (view.claims || []).filter((x) => x.field === field);
      if (!items.length) return;
      list.append(h("div", { class: "group-title", text: label }));
      const ul = h("ul", { class: "claims" });
      items.forEach((cl) => {
        const li = h("li", { class: `claim ${cl.flagged ? "bad" : "ok"}`, role: "button", tabindex: "0", "aria-pressed": "false" },
          h("span", { class: "ic", "aria-hidden": "true", text: cl.flagged ? "✕" : "✓" }),
          h("div", {}, h("div", { text: cl.text || "(claim)" }),
            h("div", { class: "meta", text: `${cl.kind || "claim"} · ${cl.sources.length ? cl.sources.map((s) => s.field || s.id || "note").join(", ") : "no source cited"}` }),
            cl.flagged ? h("div", { class: "why" }, h("strong", { text: `${SUBTYPE[cl.subtype] || cl.subtype}${cl.critical ? " (would change a decision)" : ""}: ` }), cl.reason || "") : null));
        const pick = () => {
          list.querySelectorAll(".claim").forEach((n) => n.setAttribute("aria-pressed", "false"));
          if (selected === cl) { selected = null; drawLeft(null); drawPassages(null); return; }
          selected = cl; li.setAttribute("aria-pressed", "true"); drawLeft(cl); drawPassages(cl);
        };
        li.addEventListener("click", pick);
        li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
        ul.append(li);
      });
      list.append(ul);
    });
    right.append(h("h4", { text: title || "The answer, claim by claim" }), h("p", { class: "muted small", text: "Click a claim to see what it rests on." }), list, passageBox);
    drawLeft(null);
  }

  /* ---------------------------------------------------------------- hero and story */
  function renderHero() {
    const c = state.charts;
    const sonnet = c.c9.rows.find((r) => r.model === "claude-sonnet-5-5");
    const tiles = [[`${state.index.length}`, "cases answered by up to six models"],
      [pct(sonnet.hallucination_rate.value), "of Sonnet 5.5's claims flagged by the verifier"],
      [pct(c.vision_b.sensitivity, 0), "of bleeds caught on unseen RSNA scans"],
      [`${c.human.flags_confirmed[0]}/${c.human.flags_confirmed[1]}`, "verifier flags that humans confirmed"]];
    $("#hero-stats").replaceChildren(...tiles.map(([v, l]) => h("div", { class: "stat" }, h("div", { class: "v", text: v }), h("div", { class: "l", text: l }))));
  }

  function storySteps() {
    const c = state.charts, va = c.vision_a, vb = c.vision_b;
    const sample = state.cases.story;
    return [
      { t: "Perfusion CT", p: ["A CT perfusion scan shows how blood reaches each part of the brain. Where it arrives late (Tmax), tissue is at risk; where flow has collapsed, it is already lost.",
        "We use ISLES'24: 149 real stroke patients, 129 for development and 20 held back until the final test."],
        stage: () => h("div", {}, h("h4", { text: "Tissue at risk, schematic" }), brainSchematic(),
          h("p", { class: "muted small", text: "Schematic. Core: tissue already lost. Penumbra: at risk, still salvageable." })) },
      { t: "Core and penumbra", p: ["An nnU-Net model, trained five times on different folds, outlines the core. The penumbra is the guideline's own rule: Tmax over 6 seconds.",
        `On development cases its Dice is ${va.dice.mean.toFixed(2)} and its volume is off by ${va.abs_volume_error_ml.median} mL (median). Volumes matter more than outlines: the rules cut at 21, 31, 51 and 70 mL.`],
        stage: () => h("div", {}, h("h4", { text: "How often the core lands on the wrong side of a cut-off" }), cutoffBars(va),
          h("p", { class: "muted small", text: `${va.cases} development cases, each scored by the fold model that never trained on it.` })) },
      { t: "Bleed check", p: ["A second model reads the plain CT for any bleed, which rules out clot-busting drugs. It was trained on 21,744 RSNA scans and tuned to miss as few bleeds as possible.",
        `On RSNA scans it never saw, it caught ${pct(vb.sensitivity, 1)} of bleeds. But on ISLES'24's presumed bleed-free scans it raised ${vb.isles_flags} false alarms in ${vb.isles_scans}: the new hospitals' scans look different.`],
        stage: () => h("div", { class: "tiles" }, tile(vb.scan_auc.toFixed(3), "AUC, RSNA test"), tile(pct(vb.sensitivity, 1), "bleeds caught"),
          tile(pct(vb.specificity, 1), "non-bleeds cleared"), tile(`${vb.isles_flags}/${vb.isles_scans}`, "ISLES'24 scans flagged")) },
      { t: "The rulebook", p: ["The guideline cannot be copied, so the team rewrote every criterion in its own words: 47 passages, each pointing to its source section.",
        "The language model gets the whole rulebook in its prompt, so it can never miss a rule because a search failed."],
        stage: () => h("div", {}, h("h4", { text: "Two of the 47 passages" }),
          ...["AHA2026-4.6.1-within-4.5h", "DEFUSE3-inclusion-imaging"].filter((id) => state.passages[id]).map((id) =>
            h("div", { class: "passage" }, h("div", { class: "pid", text: `${id} · ${state.passages[id].pointer}` }), h("div", { text: state.passages[id].text })))) },
      { t: "The reasoner", p: ["The language model reads the imaging numbers, the triage note and the rulebook, and decides on each treatment separately.",
        "Every sentence of its reasoning must cite what it rests on: a number, a line of the note, or a passage."],
        stage: () => sample ? h("div", {}, h("h4", { text: "Sonnet 5.5 on a late-window case" }), miniAnswer(sample)) : h("p", { text: "Loading…" }) },
      { t: "The verifier", p: ["A second pass splits the answer into single claims and checks each one: numbers against the contract, patient facts against the note, rules against the passages, with a smaller model as judge.",
        "A claim that does not trace is a hallucination. If believing it would change a decision, it is critical."],
        stage: () => h("div", {}, h("h4", { text: "What gets checked" }), h("ul", {},
          h("li", { text: "Numbers: exact after rounding, and comparisons recomputed in code" }),
          h("li", { text: "Patient facts: against the note's own fact sheet" }),
          h("li", { text: "Rules: retrieved passages, then a judge on scope and exceptions" }),
          h("li", { text: "Citations: does the declared source back the claim?" }),
          h("li", { text: "Omissions: a fact that should have changed the decision but was ignored" }))) },
    ];
  }
  function tile(v, l) { return h("div", { class: "stat" }, h("div", { class: "v", text: v }), h("div", { class: "l", text: l })); }
  function cutoffBars(va) {
    const box = h("div");
    window.Charts.bars(box, Object.entries(va.wrong_side).map(([c, n]) => ({ label: `${c} mL`, value: n / va.cases, n,
      tip: [`${c} mL cut-off`, `${n} of ${va.cases} cases on the wrong side`] })));
    return box;
  }
  function brainSchematic() {
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 300 220"); svg.setAttribute("class", "chart"); svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Schematic brain slice with a small core inside a larger penumbra on one side");
    const add = (n, a) => { const e = document.createElementNS(NS, n); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); svg.appendChild(e); return e; };
    add("ellipse", { cx: 150, cy: 110, rx: 105, ry: 92, fill: "var(--surface-3)", stroke: "var(--line-2)", "stroke-width": 2 });
    add("line", { x1: 150, y1: 20, x2: 150, y2: 200, stroke: "var(--line-2)" });
    add("ellipse", { cx: 200, cy: 100, rx: 48, ry: 40, fill: "var(--s4)", opacity: 0.45 });
    add("ellipse", { cx: 208, cy: 104, rx: 18, ry: 15, fill: "var(--critical)", opacity: 0.85 });
    const t = (x, y, s) => { const e = add("text", { x, y }); e.textContent = s; };
    t(222, 160, "penumbra"); t(186, 72, "core");
    return svg;
  }
  function miniAnswer(file) {
    const a = file.arms.rulebook.answers["claude-sonnet-5-5"];
    const rec = a.recommendation;
    return h("div", {}, h("div", { class: "decisions" }, ["thrombolysis", "thrombectomy"].map((d) =>
      h("div", { class: "dec" }, h("div", { class: "k", text: d }), h("div", { class: "v", text: decisionWord(rec[d].recommend) })))),
      h("ul", { class: "claims" }, rec.rationale.slice(0, 4).map((r) => h("li", { class: "claim ok" }, h("span", { class: "ic", text: "→" }), h("div", { text: r.claim })))));
  }

  function initStory() {
    const steps = storySteps(), list = $("#steps"), stage = $("#stage");
    let current = -1;
    const show = (i) => {
      if (i === current) return;
      current = i;
      list.querySelectorAll(".step").forEach((s, j) => s.classList.toggle("on", j === i));
      stage.replaceChildren(steps[i].stage());
    };
    steps.forEach((s, i) => list.append(h("li", { class: "step", onclick: () => show(i) },
      h("div", { class: "n", text: `STEP ${i + 1} OF ${steps.length}` }), h("h3", { text: s.t }), ...s.p.map((p) => h("p", { text: p })))));
    show(0);
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => entries.forEach((e) => {
        if (e.isIntersecting) show([...list.children].indexOf(e.target));
      }), { rootMargin: "-45% 0px -45% 0px" });
      list.querySelectorAll(".step").forEach((s) => io.observe(s));
    }
  }

  /* ---------------------------------------------------------------- explorer */
  const KINDS = [["fixture", "Fixtures"], ["scenario", "Scenarios"], ["development", "Real cases"]];
  let currentKind = "fixture";
  function routeText(row) {
    const names = { thrombolysis: "Thrombolysis", thrombectomy: "Thrombectomy" };
    return ["thrombolysis", "thrombectomy"].map((d) => row.rules[d] ? `${names[d]} ${row.rules[d].value}` : "").filter(Boolean).join(" · ");
  }
  function renderCaseList() {
    const q = ($("#case-filter").value || "").toLowerCase();
    const rows = state.index.filter((r) => r.kind === currentKind && (!q || JSON.stringify(r.rules).toLowerCase().includes(q) || r.title.toLowerCase().includes(q)));
    $("#case-items").replaceChildren(...rows.map((r) => h("li", {}, h("button", { type: "button", "data-id": r.id, onclick: () => openCase(r.id) },
      h("div", { class: "t", text: r.title }), h("div", { class: "r", text: routeText(r) })))));
  }
  function initExplorer() {
    const tabs = $("#case-tabs");
    KINDS.forEach(([k, label]) => tabs.append(h("button", { type: "button", role: "tab", "aria-selected": String(k === currentKind),
      onclick: (e) => { currentKind = k; tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b === e.currentTarget))); renderCaseList(); } },
      `${label} (${state.index.filter((r) => r.kind === k).length})`)));
    $("#case-filter").addEventListener("input", renderCaseList);
    renderCaseList();
    openCase("fx02_thrombectomy_large_mismatch");
  }
  async function loadCase(id) {
    if (!state.cases[id]) state.cases[id] = await getJSON(`/api/bank/case/${encodeURIComponent(id)}`);
    return state.cases[id];
  }
  async function openCase(id) {
    document.querySelectorAll("#case-items button").forEach((b) => b.setAttribute("aria-current", String(b.dataset.id === id)));
    const view = $("#case-view");
    view.replaceChildren(h("p", { class: "muted", text: "Loading…" }));
    const file = await loadCase(id);
    const row = state.index.find((r) => r.id === id);
    const arms = Object.keys(file.arms);
    let arm = arms.includes("predicted") ? "predicted" : arms[0];
    let model = "claude-sonnet-5-5";
    const body = h("div");
    const head = h("div", { class: "case-head" });
    function draw() {
      const a = file.arms[arm];
      const ans = a.answers[model];
      head.replaceChildren(...[h("h3", { text: row ? row.title : id }),
        h("div", { class: "chips", role: "group", "aria-label": "Model" }, MODELS.filter((m) => a.answers[m]).map((m) =>
          h("button", { type: "button", class: "chip", "aria-pressed": String(m === model), disabled: !a.answers[m].available,
            title: a.answers[m].available ? "" : a.answers[m].reason, onclick: () => { model = m; draw(); } },
            h("span", { class: "sw", style: `background:${SLOT[m]}` }), MODEL_LABEL[m]))),
        arms.length > 1 ? h("div", { class: "chips", role: "group", "aria-label": "Vision arm" }, arms.map((x) =>
          h("button", { type: "button", class: "chip", "aria-pressed": String(x === arm), onclick: () => { arm = x; draw(); } },
            x === "predicted" ? "Predicted vision" : "Ground-truth vision"))) : null].filter(Boolean));
      if (!ans || !ans.available) {
        body.replaceChildren(h("div", { class: "notice" }, h("strong", { text: `${MODEL_LABEL[model]} has no verified answer here` }), ans ? ans.reason : ""));
        return;
      }
      renderAnswer(body, { vision: a.vision, note: a.note, ...ans }, { title: `${MODEL_LABEL[model]}'s answer, claim by claim` });
      if (file.kind === "development") body.prepend(h("p", { class: "muted small", text: arm === "predicted"
        ? "Real development case: the contract is what our vision models predicted."
        : "The same patient with the reference final infarct as core and no bleed: what the answer would be if vision were right." }));
    }
    view.replaceChildren(head, body);
    draw();
  }

  /* ---------------------------------------------------------------- results */
  function card(title, sub, src) {
    const c = h("div", { class: "card" }, h("h3", { text: title }), sub ? h("p", { class: "sub", text: sub }) : null);
    return { root: c, body: c, src: () => c.append(h("div", { class: "src", text: src })) };
  }
  function renderResults() {
    const c = state.charts, grid = $("#result-grid"), C = window.Charts;
    // C9
    {
      const k = card("Hallucination rate by model", "Share of claims the verifier could not trace, with 95% case-bootstrap intervals. Same 175 cases for every model.", c.c9.source);
      const metrics = [["hallucination_rate", "All claims", 0.5], ["content_hallucination_rate", "False content only", 0.5], ["critical_hallucination_rate", "Would change a decision", 0.1]];
      const plot = h("div"), seg = h("div", { class: "seg", role: "group", "aria-label": "Measure" });
      const draw = (m) => {
        seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.m === m[0])));
        plot.replaceChildren();
        C.dotCI(plot, c.c9.rows.map((r) => ({ label: MODEL_LABEL[r.model], value: r[m[0]].value, ci: r[m[0]].ci95, color: SLOT[r.model],
          tip: [MODEL_LABEL[r.model], `${pct(r[m[0]].value)} (${r[m[0]].ci95.map((v) => pct(v)).join(" to ")})`, `${r[m[0]].numerator} of ${r[m[0]].denominator}`] })), { max: m[2] });
      };
      metrics.forEach((m) => seg.append(h("button", { type: "button", "data-m": m[0], onclick: () => draw(m) }, m[1])));
      k.body.append(seg, plot);
      draw(metrics[0]);
      C.table(k.body, ["Model", "All", "False content", "Critical"], c.c9.rows.map((r) => [MODEL_LABEL[r.model], pct(r.hallucination_rate.value), pct(r.content_hallucination_rate.value), pct(r.critical_hallucination_rate.value)]));
      k.src(); grid.append(k.root);
    }
    // Rulebook
    {
      const k = card("Giving the model the rulebook", "False-content rate without the rulebook in the prompt → with it, on the same 100 cases.", c.rulebook.source);
      C.dumbbell(k.body, c.rulebook.rows.map((r) => ({ label: MODEL_LABEL[r.model], from: r.without, to: r.with, color: SLOT[r.model],
        tip: [MODEL_LABEL[r.model], `${pct(r.without)} → ${pct(r.with)}`, `difference ${pct(r.difference)} (${r.ci95.map((v) => pct(v)).join(" to ")})`] })),
        { max: 0.4, fromLabel: "without the rulebook", toLabel: "with the rulebook" });
      C.table(k.body, ["Model", "Without", "With", "Difference"], c.rulebook.rows.map((r) => [MODEL_LABEL[r.model], pct(r.without), pct(r.with), pct(r.difference)]));
      k.src(); grid.append(k.root);
    }
    // Human check
    {
      const hc = c.human, k = card("Do people agree with the verifier?", `${hc.claims} claims, each labelled by two team members blind to the model.`, hc.source);
      k.body.append(h("div", { class: "tiles" }, tile(hc.kappa.toFixed(2), "agreement between people (κ)"), tile(pct(hc.agreement, 0), "verifier agrees with people"),
        tile(`${hc.flags_confirmed[0]}/${hc.flags_confirmed[1]}`, "verifier flags people confirmed"), tile(`${hc.human_errors_flagged[0]}/${hc.human_errors_flagged[1]}`, "human-found errors the verifier caught")));
      const names = { clean: "Claims it passed", content_error: "False-content flags", citation_error: "Citation flags" };
      C.bars(k.body, hc.by_verdict.map((v) => ({ label: names[v.verdict] || v.verdict, value: v.agreement, n: v.n, tip: [names[v.verdict], `people agree ${pct(v.agreement)}`, `${v.n} claims`] })));
      k.body.append(h("p", { class: "muted small", text: "It almost never passes a false claim, but it over-flags, especially citations. Its rates are upper bounds." }));
      k.src(); grid.append(k.root);
    }
    // Small model
    {
      const k = card("Fine-tuning a small model", "Gemma 3 4B, prompted and LoRA-tuned on synthetic gold answers. Fewer hallucinations, but it learned to say no.", c.small.source);
      const half = h("div"), half2 = h("div");
      half.append(h("div", { class: "muted small", text: "Hallucination rate on routes never seen in training" }));
      C.dotCI(half, c.small.rows.map((r) => ({ label: MODEL_LABEL[r.model], value: r.held_out_rate.value, ci: r.held_out_rate.ci95, color: SLOT[r.model] })), { max: 0.5 });
      half2.append(h("div", { class: "muted small", text: "Thrombectomy cases where the rulebook says yes, and the model said yes" }));
      C.bars(half2, c.small.rows.map((r) => ({ label: MODEL_LABEL[r.model], value: r.thrombectomy.yes_caught / r.thrombectomy.yes, n: r.thrombectomy.yes,
        tip: [MODEL_LABEL[r.model], `${r.thrombectomy.yes_caught} of ${r.thrombectomy.yes}`] })));
      k.body.append(half, half2);
      C.table(k.body, ["Model", "Rate (held-out routes)", "Thrombectomy yes caught"], c.small.rows.map((r) => [MODEL_LABEL[r.model], pct(r.held_out_rate.value), `${r.thrombectomy.yes_caught}/${r.thrombectomy.yes}`]));
      k.src(); grid.append(k.root);
    }
    // Vision B
    {
      const vb = c.vision_b, k = card("The bleed detector on unseen patients", `${vb.scans.toLocaleString()} RSNA test scans from ${vb.patients.toLocaleString()} patients, opened once for these figures.`, vb.source);
      k.body.append(h("div", { class: "tiles" }, tile(vb.scan_auc.toFixed(3), "AUC"), tile(pct(vb.sensitivity, 1), "sensitivity"), tile(pct(vb.specificity, 1), "specificity"), tile(vb.scan_ece.toFixed(3), "calibration error")));
      const rel = h("div", { style: "max-width:320px;margin-top:10px" });
      C.reliability(rel, vb.reliability);
      k.body.append(rel);
      C.table(k.body, ["Bleed subtype", "Caught", "Bleeds"], vb.subtypes.map((s) => [s.subtype, String(s.caught), String(s.bleeds)]));
      k.body.append(h("p", { class: "muted small", text: `Item 2 asked whether a rule on the slab scores could cut ISLES'24's ${c.item2.flags_none} false alarms. The best pre-registered rule cleared ${c.item2.flags_none - c.item2.flags_proposal}; the flagged slabs sit in brain tissue (${pct(c.item2.heat_in_brain, 0)} of the heat), so it is a domain shift no slab rule fixes.` }));
      k.src(); grid.append(k.root);
    }
    // Sensitivity
    {
      const s = c.scenario_sensitivity, k = card("Does a 20% core error change decisions?", "Core volume scaled by 0.8 and 1.2, decisions recomputed by the rules.", "EXP-63 (DEC-31 D3 and its supplement)");
      C.bars(k.body, [["Synthetic scenarios, core ×0.8", s["x0.8"]], ["Synthetic scenarios, core ×1.2", s["x1.2"]]].map(([l, r]) => ({ label: l, value: r.changed_share, n: r.decisions,
        tip: [l, `${r.changed} of ${r.decisions} decisions change`, `${r.yes_no_flips} yes/no flips`] })), { left: 220 });
      k.body.append(h("p", { class: "muted small", text: "On the 129 real development cases, none of 258 decisions changed: there the unknown occlusion site, ASPECTS and the bleed flag decide, not the core." }));
      k.src(); grid.append(k.root);
    }
  }

  /* ---------------------------------------------------------------- builder */
  const FIELDS = {
    timing: [["onset", "How onset was known", "select", { witnessed: "Witnessed onset", unwitnessed: "Not witnessed (last known well)", wake_up: "Woke with symptoms" }],
      ["minutes", "Onset or last known well to imaging", "range", hm], ["sleep_minutes", "Time asleep before waking", "range", hm, (v) => v.onset === "wake_up"]],
    patient: [["age", "Age (years)", "range"], ["sex", "Sex", "select", { female: "Female", male: "Male" }], ["nihss", "NIHSS on arrival", "range"],
      ["mrs", "Pre-stroke mRS", "range"], ["deficit", "Presenting deficit", "deficit"]],
    imaging: [["core_ml", "Core (mL)", "range"], ["penumbra_ml", "Tmax > 6 s, core included (mL)", "range"],
      ["site", "Occlusion site", "select", { ICA: "ICA", M1: "M1", M2: "M2", basilar: "Basilar", none: "None found", unknown: "Unknown (no CTA)" }],
      ["aspects", "ASPECTS", "aspects"], ["haemorrhage", "Bleed on plain CT", "check"]],
    history: [["contraindication", "Contraindication", "select", { none: "None", warfarin_inr: "Warfarin", apixaban: "Apixaban (DOAC)", head_trauma: "Head injury in the last 14 days", previous_ich: "Previous brain haemorrhage", recent_surgery: "Recent major surgery" }],
      ["inr", "INR", "range", null, (v) => v.contraindication === "warfarin_inr"], ["antiplatelet", "Takes an antiplatelet", "check", null, (v) => !["warfarin_inr", "apixaban"].includes(v.contraindication)]],
  };
  const values = {};
  const fieldNodes = {};
  function buildControls() {
    const m = state.meta;
    Object.assign(values, m.defaults);
    if (values.inr == null) values.inr = 1.0;
    for (const [group, defs] of Object.entries(FIELDS)) {
      const box = $(`#f-${group}`);
      defs.forEach(([name, label, kind, extra, visible]) => {
        let input, out;
        if (kind === "range") {
          const [lo, hi, step] = m.bounds[name];
          input = h("input", { type: "range", min: lo, max: hi, step, value: values[name] ?? lo, "aria-label": label });
          out = h("output");
          const show = () => { out.textContent = extra ? extra(+input.value) : input.value; };
          input.addEventListener("input", () => { values[name] = +input.value; show(); refresh(); });
          show();
          fieldNodes[name] = h("div", { class: "field" }, h("label", {}, label, h("div", { class: "row" }, input, out)));
          fieldNodes[name].sync = () => { input.value = values[name]; show(); };
        } else if (kind === "check") {
          input = h("input", { type: "checkbox" });
          input.checked = !!values[name];
          input.addEventListener("change", () => { values[name] = input.checked; refresh(); });
          fieldNodes[name] = h("div", { class: "field check" }, h("label", {}, input, label));
          fieldNodes[name].sync = () => { input.checked = !!values[name]; };
        } else {
          let opts;
          if (kind === "select") opts = Object.entries(extra);
          if (kind === "deficit") opts = m.deficits.map((d, i) => [String(i), d]);
          if (kind === "aspects") opts = [["", "Unknown (not available)"], ...Array.from({ length: 11 }, (_, i) => [String(10 - i), String(10 - i)])];
          input = h("select", { "aria-label": label }, opts.map(([v, t]) => h("option", { value: v, text: t })));
          const read = () => {
            const v = input.value;
            values[name] = kind === "deficit" ? +v : kind === "aspects" ? (v === "" ? null : +v) : v;
          };
          input.value = values[name] == null ? "" : String(values[name]);
          input.addEventListener("change", () => { read(); refresh(); });
          fieldNodes[name] = h("div", { class: "field" }, h("label", {}, label, input));
          fieldNodes[name].sync = () => { input.value = values[name] == null ? "" : String(values[name]); };
        }
        fieldNodes[name].visible = visible;
        box.append(fieldNodes[name]);
      });
    }
    refresh();
    const presets = $("#preset");
    state.index.filter((r) => r.inputs).forEach((r) => presets.append(h("option", { value: r.id, text: `${r.title} — ${routeText(r)}` })));
    presets.addEventListener("change", () => {
      const r = state.index.find((x) => x.id === presets.value);
      Object.assign(values, m.defaults, { inr: 1.0 }, r ? r.inputs : {});
      if (values.inr == null) values.inr = 1.0;
      if (values.sleep_minutes == null) values.sleep_minutes = m.defaults.sleep_minutes;
      Object.values(fieldNodes).forEach((f) => f.sync && f.sync());
      refresh();
    });
    $("#live-status").textContent = m.live_enabled
      ? `Live runs are on: up to ${m.limits.per_hour} an hour per visitor. A run takes about one to two minutes.`
      : "Live runs are off right now: you will get the closest precomputed case.";
    $("#builder").addEventListener("submit", (e) => { e.preventDefault(); run(); });
  }
  function refresh() {
    for (const f of Object.values(fieldNodes)) f.hidden = f.visible ? !f.visible(values) : false;
    const aspects = fieldNodes.aspects && fieldNodes.aspects.querySelector("select");
    if (aspects) { aspects.disabled = values.site === "basilar"; }
  }
  function payload() {
    const out = { ...values };
    if (out.onset !== "wake_up") delete out.sleep_minutes;
    if (out.contraindication !== "warfarin_inr") delete out.inr;
    if (out.site === "basilar") out.aspects = null;
    return out;
  }

  const STEPS = [["note", "Writing the triage note"], ["rules", "Applying the rulebook in code"], ["reasoner", "Sonnet 5.5 is deciding"],
    ["extractor", "Splitting the answer into claims"], ["judge", "Checking each claim"], ["score", "Scoring"]];
  async function run() {
    const btn = $("#run"), err = $("#builder-error"), list = $("#progress"), out = $("#run-result");
    err.hidden = true; btn.disabled = true; out.replaceChildren();
    const started = Date.now(), items = {};
    list.replaceChildren(...STEPS.map(([k, l]) => (items[k] = h("li", {}, h("span", { class: "dot" }), h("span", { text: l }), h("span", { class: "time" })))));
    const tick = setInterval(() => { const li = list.querySelector("li.running .time"); if (li) li.textContent = `${Math.round((Date.now() - started) / 1000)} s`; }, 500);
    const finish = () => { clearInterval(tick); btn.disabled = false; };
    let res;
    try {
      res = await fetch("/api/run", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ inputs: payload() }) });
    } catch (e) { finish(); err.textContent = "Could not reach the server. Try again in a moment."; err.hidden = false; return; }
    if (!res.ok) {
      finish();
      const body = await res.json().catch(() => ({}));
      err.textContent = body.error ? `Please check: ${body.error}` : `The server answered ${res.status}.`; err.hidden = false; return;
    }
    const { run_id } = await res.json();
    const es = new EventSource(`/api/run/${run_id}/events`);
    let expected = null;
    es.addEventListener("progress", (e) => {
      const ev = JSON.parse(e.data);
      const li = items[ev.step];
      if (ev.step === "rules" && ev.detail) expected = ev.detail.expected;
      if (!li) return;
      li.className = ev.status === "done" ? "done" : "running";
      if (ev.status === "done") li.querySelector(".time").textContent = `${ev.t.toFixed(0)} s`;
      if (ev.step === "judge" && ev.status === "running" && ev.detail) li.querySelector(".time").textContent = `${ev.detail.calls} checks`;
      const idx = STEPS.findIndex(([k]) => k === ev.step);
      if (ev.status === "done" && STEPS[idx + 1]) { const nx = items[STEPS[idx + 1][0]]; if (nx.className !== "done") nx.className = "running"; }
    });
    es.addEventListener("result", (e) => {
      es.close(); finish();
      const r = JSON.parse(e.data);
      Object.values(items).forEach((li) => { if (r.source !== "live" && li.className === "running") li.className = ""; });
      const label = r.source === "live" ? "Answered live" : r.source === "cache" ? "From the cache" : "Precomputed fallback";
      out.append(h("div", { class: `notice ${r.source === "precomputed" ? "fallback" : ""}` }, h("strong", { text: label }), r.message,
        r.source === "precomputed" && r.differs && r.differs.length ? h("div", { class: "muted small", text: `Differs from your inputs in: ${r.differs.join(", ")}.` }) : null,
        r.source === "live" && r.seconds ? h("div", { class: "muted small", text: `${r.seconds.total} s in all.` }) : null));
      const ans = h("div");
      renderAnswer(ans, r.view, { title: r.source === "precomputed" ? `The closest precomputed case (${r.preset_title})` : "Sonnet 5.5's answer, claim by claim" });
      out.append(ans);
    });
    es.onerror = () => { if (es.readyState === EventSource.CLOSED) return; };
  }

  /* ---------------------------------------------------------------- start */
  async function start() {
    initTheme();
    try {
      const [meta, index, charts, passages] = await Promise.all([getJSON("/api/meta"), getJSON("/api/bank/index"), getJSON("/api/bank/charts"), getJSON("/api/bank/passages")]);
      state.meta = meta; state.index = index.cases; state.charts = charts; state.passages = passages;
      state.cases.story = await loadCase("fx02_thrombectomy_large_mismatch");
    } catch (e) {
      $("#hero-stats").replaceChildren(h("p", { class: "error", text: "The page could not load its data. Please reload." }));
      return;
    }
    renderHero(); initStory(); initExplorer(); renderResults(); buildControls();
  }
  document.addEventListener("DOMContentLoaded", start);
})();
