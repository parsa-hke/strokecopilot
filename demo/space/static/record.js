/* Stroke Decision Copilot — the record page: methods, every result chart, every documented experiment and the case
   explorer (DEC-32). Plain JS. Every server value is set with textContent. */
(function () {
  "use strict";
  const { $, h, getJSON, pct, hm, MODEL_LABEL, MODELS, SLOT, state, decisionWord, ruleWord, renderAnswer, routeText, loadCase, tile } = window.SDC;

  function cutoffBars(va) {
    const box = h("div");
    window.Charts.bars(box, Object.entries(va.wrong_side).map(([c, n]) => ({ label: `${c} mL`, value: n / va.cases, n,
      tip: [`${c} mL cut-off`, `${n} of ${va.cases} cases on the wrong side`] })));
    return box;
  }

  /* ---------------------------------------------------------------- explorer */
  const KINDS = [["fixture", "Fixtures"], ["scenario", "Scenarios"], ["development", "Real cases"], ["heldout", "Held-out cases"]];
  let currentKind = "fixture";
  function renderCaseList() {
    const q = ($("#case-filter").value || "").toLowerCase();
    const rows = state.index.filter((r) => r.kind === currentKind && (!q || JSON.stringify(r.rules).toLowerCase().includes(q) || r.title.toLowerCase().includes(q)));
    $("#case-items").replaceChildren(...rows.map((r) => h("li", {}, h("button", { type: "button", "data-id": r.id, onclick: () => openCase(r.id) },
      h("div", { class: "t", text: r.title }), h("div", { class: "r", text: routeText(r) })))));
  }
  function initExplorer() {
    const tabs = $("#case-tabs");
    KINDS.filter(([k]) => state.index.some((r) => r.kind === k)).forEach(([k, label]) => tabs.append(h("button", { type: "button", role: "tab", "data-kind": k, "aria-selected": String(k === currentKind),
      onclick: () => selectKind(k) },
      `${label} (${state.index.filter((r) => r.kind === k).length})`)));
    $("#case-filter").addEventListener("input", renderCaseList);
    renderCaseList();
    openCase("fx01_thrombolysis_clear");   // the story's patient
  }
  function selectKind(k) {
    currentKind = k;
    $("#case-tabs").querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.kind === k)));
    renderCaseList();
  }
  function showCase(id) {   // from a results card: open the case in the explorer and bring it into view
    const row = state.index.find((r) => r.id === id);
    if (!row) return;
    selectKind(row.kind);
    openCase(id);
    $("#case-view").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
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
      if (file.kind === "development" || file.kind === "heldout") body.prepend(h("p", { class: "muted small", text: arm === "predicted"
        ? (file.kind === "heldout" ? "Held-out case: one of the 20 patients no model or threshold saw until everything was frozen. The contract is what our vision models predicted."
          : "Real development case: the contract is what our vision models predicted.")
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
      const ho = c.heldout ? ` On the ${c.heldout.cases} held-out patients, ${c.heldout.rule_sensitivity["x0.8"].changed + c.heldout.rule_sensitivity["x1.2"].changed} of ${c.heldout.rule_sensitivity["x0.8"].decisions + c.heldout.rule_sensitivity["x1.2"].decisions} rule decisions changed` +
        (c.heldout.sonnet_sensitivity ? `, and ${c.heldout.sonnet_sensitivity["x0.8"].changed + c.heldout.sonnet_sensitivity["x1.2"].changed} of ${c.heldout.sonnet_sensitivity["x0.8"].decisions_compared + c.heldout.sonnet_sensitivity["x1.2"].decisions_compared} of Sonnet 5.5's own answers.` : ".") : "";
      k.body.append(h("p", { class: "muted small", text: "On the 129 real development cases, none of 258 decisions changed: there the unknown occlusion site, ASPECTS and the bleed flag decide, not the core." + ho }));
      k.src(); grid.append(k.root);
    }
    if (c.heldout) renderHeldout(c.heldout, grid, C);
  }

  function renderHeldout(ho, grid, C) {
    const word = (t) => Object.entries(t).map(([k, n]) => `${k} ${n}`).join(", ");
    // Correctness
    {
      const k = card(`The final test: ${ho.cases} held-out patients`,
        "Decisions that match the rules applied to the reference contract (true infarct, no bleed), with the vision models' own contract as input. Undetermined truths are not scored.", ho.source);
      k.body.append(h("div", { class: "notice" }, h("strong", { text: "Read with care. " }),
        `On these patients the right thrombectomy answer is always no (the occlusion site is unknown), and thrombolysis is undetermined for ${ho.truth.thrombolysis.undetermined || 0} of ${ho.cases}. A model that leans to "no" scores well here without being better.`));
      k.body.append(h("div", { class: "tiles" }, tile(`${ho.rules_on_predicted.correct}/${ho.rules_on_predicted.scored}`, "the rules themselves, on the predicted contract"),
        tile(`${ho.vision.dice.mean.toFixed(2)}`, "core Dice, held-out"), tile(`${ho.vision.abs_volume_error_ml.median} mL`, "median core volume error"),
        tile(`${ho.vision.bleed_flags.flagged}/${ho.vision.bleed_flags.scans}`, "scans flagged for a bleed")));
      C.dotCI(k.body, ho.rows.map((r) => ({ label: MODEL_LABEL[r.model], value: r.predicted.rate, ci: r.predicted.ci95, color: SLOT[r.model],
        tip: [MODEL_LABEL[r.model], `${r.predicted.correct} of ${r.predicted.scored} decisions right`, r.predicted.no_answer ? `${r.predicted.no_answer} cases without a valid answer` : "every case answered"] })), { max: 1 });
      C.table(k.body, ["Model", "Predicted vision", "Ground-truth vision", "No valid answer"], ho.rows.map((r) => [MODEL_LABEL[r.model],
        `${r.predicted.correct}/${r.predicted.scored}`, r.groundtruth ? `${r.groundtruth.correct}/${r.groundtruth.scored}` : "n/a", String(r.predicted.no_answer)]));
      k.body.append(h("p", { class: "muted small", text: `Truth by the rules: thrombolysis ${word(ho.truth.thrombolysis)}; thrombectomy ${word(ho.truth.thrombectomy)}.` }));
      k.src(); grid.append(k.root);
    }
    // Error propagation
    {
      const k = card("Where wrong decisions came from", "Each wrong decision with predicted vision: did the vision contract mislead the model, or did the model reason wrongly from a correct one? And did the verifier flag it?", ho.source);
      const cell = (t) => t.wrong ? `${t.wrong} (${t.caught} caught)` : "0";
      C.table(k.body, ["Model", "From vision", "From reasoning"], ho.rows.map((r) => [MODEL_LABEL[r.model], cell(r.errors.vision), cell(r.errors.reasoning)]), { open: true });
      const ex = ho.rows.flatMap((r) => r.examples.map((e) => ({ ...e, model: r.model }))).filter((e) => e.cause === "vision");
      if (ex.length) k.body.append(h("p", { class: "muted small", text: "Vision errors, open the case:" }), h("div", { class: "chips" }, ex.map((e) =>
        h("button", { type: "button", class: "chip", onclick: () => showCase(e.alias) }, `${e.alias} · ${MODEL_LABEL[e.model]} · ${e.caught ? "caught" : "missed"}`))));
      k.body.append(h("p", { class: "muted small", text: "The verifier checks claims against the contract it was given. When the contract itself is wrong (a false bleed flag) and the answer restates it faithfully, there is nothing for it to catch." }));
      k.src(); grid.append(k.root);
    }
    // Realism check
    {
      const rows = ho.rows.filter((r) => r.rate_development);
      const k = card("Did the hallucination rates hold on new patients?", "Share of claims flagged by the verifier: development real cases → held-out patients, predicted vision. Unpaired: different patients.", ho.source);
      C.dumbbell(k.body, rows.map((r) => ({ label: MODEL_LABEL[r.model], from: r.rate_development.value, to: r.rate_heldout.value, color: SLOT[r.model],
        tip: [MODEL_LABEL[r.model], `${pct(r.rate_development.value)} → ${pct(r.rate_heldout.value)}`, `held-out ${r.rate_heldout.ci95.map((v) => pct(v)).join(" to ")}, ${r.rate_heldout.denominator} claims`] })),
        { max: 0.5, fromLabel: "development", toLabel: "held-out" });
      C.table(k.body, ["Model", "Development", "Held-out", "Format failures"], rows.map((r) => [MODEL_LABEL[r.model], pct(r.rate_development.value), pct(r.rate_heldout.value), `${r.format_failures}/${ho.cases}`]));
      k.src(); grid.append(k.root);
    }
  }

  /* ---------------------------------------------------------------- overview and the vision models */
  function renderOverview() {
    const c = state.charts, sonnet = c.c9.rows.find((r) => r.model === "claude-sonnet-5-5");
    $("#overview-tiles").replaceChildren(tile(`${state.index.length}`, "cases answered by up to six models"), tile(`${Object.keys(state.passages).length}`, "rulebook passages"),
      tile(sonnet.hallucination_rate.denominator.toLocaleString("en"), "claims checked for Sonnet 5.5 alone"), tile(pct(sonnet.hallucination_rate.value), "of those claims flagged by the verifier"),
      tile(`${c.human.flags_confirmed[0]}/${c.human.flags_confirmed[1]}`, "verifier flags that humans confirmed"));
    const va = c.vision_a, vb = c.vision_b;
    $("#va-tiles").replaceChildren(tile(va.dice.mean.toFixed(2), "Dice, mean (overlap with the true outline, 0 to 1)"), tile(`${va.abs_volume_error_ml.median} mL`, "volume error, median"),
      tile(`${va.cases}`, "development cases, each scored by the fold model that never saw it"),
      ...(c.heldout ? [tile(c.heldout.vision.dice.mean.toFixed(2), `Dice on the ${c.heldout.cases} held-out patients (five-fold ensemble)`), tile(`${c.heldout.vision.abs_volume_error_ml.median} mL`, "held-out volume error, median")] : []));
    $("#va-note").textContent = "The model was trained on the final infarct, not the acute core, so a low overlap early on is expected; what the rules use is the volume.";
    $("#va-cutoffs").replaceChildren(cutoffBars(va));
    $("#vb-tiles").replaceChildren(tile(vb.scan_auc.toFixed(3), `AUC on ${vb.scans.toLocaleString("en")} unseen RSNA scans`), tile(pct(vb.sensitivity, 1), "of bleeds caught"),
      tile(pct(vb.specificity, 1), "of non-bleeds cleared"), tile(`${vb.isles_flags}/${vb.isles_scans}`, "ISLES'24 scans flagged, all presumed bleed-free"));
    $("#vb-note").textContent = "Tuned to miss as few bleeds as possible. On the stroke dataset's scans, from other hospitals, it raises many false alarms: a shift in how the scans look, reported as a limit.";
  }

  /* ---------------------------------------------------------------- every experiment (record.json, compiled from the project's documents) */
  function renderRecord(rec) {
    const box = $("#groups"), groups = rec.groups || [];
    if (rec.note) $("#record-note").textContent = `${$("#record-note").textContent} ${rec.note}`;
    const WORD = { done: "done" };
    box.replaceChildren(...groups.map((g) => {
      const d = h("details", { class: "group", id: `g-${g.id}` });
      d.append(h("summary", {}, h("h3", { text: g.title }), h("span", { class: "n", text: `${g.experiments.length} experiment${g.experiments.length === 1 ? "" : "s"}` }), g.summary ? h("p", { text: g.summary }) : null));
      d.append(h("div", { class: "exps" }, g.experiments.map((e) => {
        const row = h("article", { class: "exp" }, h("div", { class: "id" }, e.id, h("span", { class: "st", text: WORD[e.status] || e.status || "" })),
          h("div", {}, h("h4", { text: e.title || e.id }),
            e.asked ? h("p", {}, h("b", { text: "Asked" }), e.asked) : null,
            e.method ? h("p", {}, h("b", { text: "How" }), e.method) : null,
            e.result ? h("p", { class: "res" }, h("b", { text: "Found" }), e.result) : h("p", {}, h("b", { text: "Found" }), "No result recorded."),
            e.numbers && e.numbers.length ? h("div", { class: "nums" }, e.numbers.map((n) => h("span", {}, n.label, h("b", { text: n.value })))) : null,
            e.note ? h("p", {}, h("b", { text: "Note" }), e.note) : null,
            h("div", { class: "src", text: [e.decisions && e.decisions.length ? e.decisions.join(", ") : null, e.source].filter(Boolean).join(" · ") })));
        row.dataset.text = JSON.stringify(e).toLowerCase();
        return row;
      })));
      return d;
    }));
    const total = groups.reduce((n, g) => n + g.experiments.length, 0), count = $("#rec-count"), toggle = $("#rec-toggle");
    const filter = () => {
      const q = $("#rec-search").value.trim().toLowerCase(); let shown = 0;
      box.querySelectorAll(".group").forEach((g) => {
        let any = 0;
        g.querySelectorAll(".exp").forEach((x) => { const ok = !q || x.dataset.text.includes(q); x.hidden = !ok; if (ok) any++; });
        g.hidden = !any; if (q) g.open = any > 0; shown += any;
      });
      count.textContent = q ? `${shown} of ${total} experiments match` : `${total} experiments in ${groups.length} groups`;
    };
    $("#rec-search").addEventListener("input", filter);
    toggle.addEventListener("click", () => { const open = toggle.textContent === "Open all"; box.querySelectorAll(".group").forEach((g) => { g.open = open; }); toggle.textContent = open ? "Close all" : "Open all"; });
    filter();
    if (groups[0]) box.querySelector(".group").open = true;
    $("#decs-list").replaceChildren(...(rec.decisions || []).map((d) => h("div", {}, h("b", {}, h("span", { text: d.id }), d.title || ""), d.text || "")));
    if (rec.limits && rec.limits.length) $("#limits-list").replaceChildren(...rec.limits.map((l) => h("li", { text: l })));
    if (location.hash.startsWith("#g-")) { const g = document.getElementById(location.hash.slice(1)); if (g) { g.open = true; g.scrollIntoView(); } }
  }

  /* ---------------------------------------------------------------- start */
  function initToc() {
    const links = [...document.querySelectorAll("#toc a")], secs = links.map((a) => document.querySelector(a.getAttribute("href")));
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) links.forEach((a, i) => a.classList.toggle("on", secs[i] === e.target)); }), { rootMargin: "-30% 0px -65% 0px" });
    secs.forEach((s) => s && io.observe(s));
  }
  async function start() {
    initToc();
    try {
      const [index, charts, passages] = await Promise.all([getJSON("/api/bank/index"), getJSON("/api/bank/charts"), getJSON("/api/bank/passages")]);
      state.index = index.cases; state.charts = charts; state.passages = passages;
    } catch (e) { $("#overview-tiles").replaceChildren(h("p", { class: "error", text: "The page could not load its data. Please reload." })); return; }
    renderOverview(); renderResults(); initExplorer();
    try { renderRecord(await getJSON("record.json")); }
    catch (e) { $("#groups").replaceChildren(h("p", { class: "muted", text: "The experiment record could not be loaded." })); }
    if (location.hash) { const t = document.getElementById(location.hash.slice(1)); if (t) t.scrollIntoView(); }
  }
  document.addEventListener("DOMContentLoaded", start);
})();
