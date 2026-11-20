/* Stroke Decision Copilot — the live run, drawn step by step (DEC-32 Space). Plain JS, no third-party code.
   It shows only what the server reports: the note and imaging numbers, the rulebook's verdicts, the decisions,
   the claims and their checks. Every server value is set with textContent. */
(function () {
  "use strict";
  function h(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = String(text); return n; }
  const reduce = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wait = (ms) => new Promise((r) => setTimeout(r, reduce() ? 0 : ms));
  const FACT = { occlusion_site: "occlusion site", nihss: "NIHSS", premorbid_mrs: "pre-stroke mRS", onset_to_imaging: "time to imaging",
    lkw_to_imaging: "time to imaging", aspects: "ASPECTS", core_ml: "core volume", mismatch_ratio: "mismatch ratio",
    mismatch_volume_ml: "mismatch volume", age: "age", "haemorrhage.present": "bleed", head_trauma: "head injury",
    previous_ich: "previous bleed", anticoagulant: "anticoagulant", inr: "INR", penumbra_ml: "Tmax volume", recent_surgery: "recent surgery" };
  const STEPS = [["note", "Note"], ["rules", "Rulebook"], ["reasoner", "Deciding"], ["extractor", "Claims"], ["judge", "Checking"], ["score", "Score"]];
  const DECISIONS = ["thrombolysis", "thrombectomy"];
  const decWord = (r) => (r === true ? "Recommended" : r === false ? "Not recommended" : "No answer");

  function mount(root, { passageIds = [] } = {}) {
    const rail = h("ol", "rv-rail"), items = {};
    STEPS.forEach(([k, l], i) => {
      const li = h("li"); li.append(h("span", "rv-node", i + 1), h("span", "rv-lbl", l), h("span", "rv-t", ""));
      rail.append(li); items[k] = li;
    });
    const clock = h("span", "rv-clock", ""), bar = h("div", "rv-bar"), fill = h("i"); bar.append(fill);
    const P = {};
    const panel = (k, title) => { const s = h("section", "rv-p"); s.append(h("h5", null, title)); P[k] = s; return s; };
    const PANEL = { note: "note", rules: "rules", reasoner: "reasoner", extractor: "claims", judge: "claims", score: "score" };
    const focus = (k) => Object.entries(P).forEach(([name, p]) => p.classList.toggle("cur", name === k));   // one step in view, where the page asks for it
    const note = h("pre", "rv-note"), vchips = h("div", "rv-vchips");
    const rulesBox = h("div", "rv-rules"), pass = h("div", "rv-pass"), decs = h("div", "rv-decs");
    const claims = h("div", "rv-claims"), claimText = h("p", "rv-claimtext"), score = h("div", "rv-score");
    panel("note", "1 · Triage note").append(note, vchips, h("p", "rv-foot", "Dashed: the imaging numbers the reasoner is given. No scan is read during a live run."));
    panel("rules", "2 · The rulebook, applied in code").append(rulesBox);
    panel("reasoner", `3 · Sonnet 5.5 reads the note, the numbers and all ${passageIds.length} passages`).append(pass, decs);
    panel("claims", "4–5 · Split into claims, then each one checked").append(claims, claimText);
    panel("score", "6 · Score").append(score);
    const head = h("div", "rv-head"); head.append(rail, clock);
    const grid = h("div", "rv-grid"); grid.append(P.note, P.rules, P.reasoner, P.claims, P.score);
    root.replaceChildren(head, bar, grid);
    const tiles = passageIds.map((id) => { const t = h("i"); t.title = id; pass.append(t); return [id, t]; });

    let started = 0, tick = 0, scan = 0, pulse = 0, slots = [];
    const stopLoops = () => { clearInterval(scan); clearInterval(pulse); scan = pulse = 0; };
    const api = {
      reset() {
        stopLoops(); clearInterval(tick);
        Object.values(items).forEach((li) => { li.className = ""; li.lastChild.textContent = ""; });
        Object.values(P).forEach((p) => p.classList.remove("on"));
        [note, vchips, rulesBox, decs, claims, score].forEach((n) => n.replaceChildren());
        tiles.forEach(([, t]) => t.className = "");
        claimText.textContent = ""; fill.style.width = "0"; clock.textContent = ""; slots = [];
        root.querySelectorAll(".rv-tag").forEach((t) => t.remove());
        started = performance.now();
        tick = setInterval(() => { clock.textContent = `${Math.round((performance.now() - started) / 1000)} s`; }, 500);
      },
      step(key, status, t) {
        const li = items[key]; if (!li) return;
        li.className = status;
        if (status === "running" && PANEL[key]) focus(PANEL[key]);
        if (status === "done" && t != null) li.lastChild.textContent = `${Math.round(t)} s`;
        if (status === "pre") li.lastChild.textContent = "–";
        const idx = STEPS.findIndex(([k]) => k === key);
        if (status === "done" || status === "pre") fill.style.width = `${((idx + 1) / STEPS.length) * 100}%`;
        const next = STEPS[idx + 1];
        if (status === "done" && next && !items[next[0]].className) items[next[0]].className = "running";
      },
      async note(text, vision) {
        P.note.classList.add("on");
        const caret = h("span", "rv-caret"), node = document.createTextNode("");
        note.replaceChildren(node, caret);
        const s = String(text || "");
        for (let i = 0; i < s.length; i += 5) { node.textContent = s.slice(0, i); await wait(10); }
        node.textContent = s; caret.remove();
        const v = vision || {}, f = (x) => (x == null ? "n/a" : x);
        const chips = [`core ${f(v.core_ml)} mL`, `Tmax>6 s ${f(v.penumbra_ml)} mL`, `mismatch ${f(v.mismatch_ratio)}`, `site ${f(v.occlusion_site)}`,
          `ASPECTS ${f(v.aspects)}`, `bleed ${v.haemorrhage && v.haemorrhage.present ? "yes" : "no"}`];
        for (const c of chips) { const n = h("span", "rv-vchip", c); vchips.append(n); await wait(50); n.classList.add("in"); }
      },
      async rules(expected) {
        P.rules.classList.add("on");
        for (const d of DECISIONS) {
          const r = (expected || {})[d]; if (!r) continue;
          const row = h("div", "rv-rule"), verdict = h("span", "rv-verdict", "…"), facts = h("div", "rv-facts");
          row.append(h("span", null, d[0].toUpperCase() + d.slice(1)), verdict); rulesBox.append(row, facts);
          await wait(250);
          verdict.textContent = r.value; verdict.classList.add(String(r.value));
          for (const label of new Set((r.deciding || []).map((k) => FACT[k] || k.replace(/[._]/g, " ")))) {
            const n = h("span", "rv-fact", label); facts.append(n); await wait(70); n.classList.add("in");
          }
        }
      },
      reasoning(on) {
        P.reasoner.classList.add("on"); clearInterval(scan); scan = 0;
        if (!on || reduce() || !tiles.length) { tiles.forEach(([, t]) => t.classList.remove("scan")); return; }
        let i = 0;
        scan = setInterval(() => { tiles.forEach(([, t], j) => t.classList.toggle("scan", Math.abs(j - i) < 3)); i = (i + 1) % (tiles.length + 6); }, 60);
      },
      async decisions(rec) {
        api.reasoning(false);
        if (decs.childElementCount || !rec) return;
        for (const d of DECISIONS) {
          const card = h("div", "rv-dec"); card.append(h("div", "k", d), h("div", "v", decWord(rec[d] && rec[d].recommend)));
          decs.append(card); await wait(200); card.classList.add("in");
        }
      },
      async claimSlots(n) {
        P.claims.classList.add("on");
        if (slots.length) return;
        for (let i = 0; i < n; i++) { const s = h("span", "rv-claim"); s.tabIndex = 0; claims.append(s); slots.push(s); await wait(30); s.classList.add("in"); }
      },
      say(text) { P.claims.classList.add("on"); claimText.textContent = text; },
      judging(calls) {
        P.claims.classList.add("on"); claimText.textContent = calls ? `${calls} checks so far…` : "Checking…";
        if (pulse || reduce() || !slots.length) return;
        let i = 0; pulse = setInterval(() => { slots.forEach((s, j) => s.classList.toggle("checking", j === i % slots.length)); i++; }, 140);
      },
      async finish(view, { label, tag } = {}) {
        stopLoops(); clearInterval(tick);
        if (!view) return;
        if (tag) [P.reasoner, P.claims, P.score].forEach((p) => p.firstChild.append(h("span", "rv-tag", tag)));
        const cited = new Set((view.claims || []).flatMap((c) => (c.sources || []).filter((s) => s.type === "passage").map((s) => s.id)));
        P.reasoner.classList.add("on");
        tiles.forEach(([id, t]) => { t.classList.remove("scan"); t.classList.toggle("cited", cited.has(id)); });
        if (!decs.childElementCount) { focus("reasoner"); await wait(300); }
        await api.decisions(view.recommendation);
        if (label) await wait(700);
        focus("claims");
        const list = view.claims || [];
        if (slots.length !== list.length) { claims.replaceChildren(); slots = []; await api.claimSlots(list.length); }
        for (let i = 0; i < list.length; i++) {
          const c = list[i], s = slots[i];
          s.classList.remove("checking"); s.classList.add("checking"); claimText.textContent = c.text; await wait(label ? 60 : 120);
          s.classList.remove("checking"); s.classList.add(c.flagged ? "flag" : "ok"); s.textContent = c.flagged ? "✕" : "✓";
          s.title = c.text; s.setAttribute("aria-label", `${c.flagged ? "Flagged" : "Supported"}: ${c.text}`);
          const show = () => { claimText.textContent = c.flagged && c.reason ? `${c.text} — flagged: ${c.reason}` : c.text; };
          s.addEventListener("pointerenter", show); s.addEventListener("focus", show);
        }
        claimText.textContent = "Hover or tab to any square to read its claim.";
        await wait(500); focus("score");
        P.score.classList.add("on");
        const k = view.counts || {}, dec = DECISIONS.map((d) => (view.decisions || {})[d] || {});
        const agree = dec.filter((x) => x.agrees === true).length, differ = dec.filter((x) => x.agrees === false).length, open = dec.length - agree - differ;
        const cells = [[k.claims, "claims"], [k.flagged, "flagged"], [k.critical, "would change a decision"], [agree, "agree with the rulebook"]];
        if (differ) cells.push([differ, "differ from the rulebook"]);
        if (open) cells.push([open, "rulebook undetermined"]);
        score.replaceChildren(...cells.map(([n, l]) => { const d = h("div"); d.append(h("b", null, n == null ? "–" : n), l); return d; }));
        fill.style.width = "100%";
        if (label) { clock.textContent = label; }
      },
    };
    return api;
  }
  window.RunViz = { mount, STEPS };
})();
