/* Stroke Decision Copilot — the live run page (DEC-32 §4), as a short game: pick a patient, bet on the model, watch the
   run, see the score. The builder's values, the server calls and the fallback are as before. Plain JS; every server
   value is set with textContent. Points and badges live in this browser only (localStorage, optional). */
(function () {
  "use strict";
  const { $, h, getJSON, pct, hm, MODEL_LABEL, MODELS, SLOT, state, decisionWord, ruleWord, renderAnswer, routeText, loadCase, tile } = window.SDC;

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
    $("#live-badge").classList.toggle("on", !!m.live_enabled);
    $("#builder").addEventListener("submit", (e) => e.preventDefault());
  }
  /* the patient, summed up as you set the values; the ring is the time used of the 4.5-hour window */
  function summary() {
    const m = state.meta, v = values, mins = v.minutes || 0, frac = Math.min(1, mins / 270), C = 2 * Math.PI * 42;
    $("#p-ring").style.strokeDasharray = `${frac * C} ${C}`; $("#p-ring").style.stroke = mins > 270 ? "var(--ink3)" : "var(--red)";
    $("#p-time").textContent = `${Math.floor(mins / 60)} h ${String(mins % 60).padStart(2, "0")}`;
    const since = v.onset === "witnessed" ? "from onset" : v.onset === "wake_up" ? "since last known well (woke with symptoms)" : "since last known well";
    $("#p-who").textContent = `${v.age}-year-old ${v.sex === "female" ? "woman" : "man"} with ${m.deficits[v.deficit] || "a deficit"}, imaged ${hm(mins)} ${since}.`;
    const facts = [[`NIHSS ${v.nihss}`], [`mRS ${v.mrs}`], [`core ${v.core_ml} mL`], [`Tmax>6 s ${v.penumbra_ml} mL`], [`site ${v.site}`],
      [`ASPECTS ${v.site === "basilar" || v.aspects == null ? "n/a" : v.aspects}`], [v.haemorrhage ? "bleed on CT" : "no bleed", v.haemorrhage],
      ...(v.contraindication && v.contraindication !== "none" ? [[v.contraindication === "warfarin_inr" ? `warfarin, INR ${v.inr}` : v.contraindication.replace(/_/g, " "), true]] : [])];
    $("#p-facts").replaceChildren(...facts.map(([t, warn]) => h("span", { class: warn ? "warn" : null, text: t })));
  }
  function refresh() {
    summary();
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

  const STEP_TEXT = { note: "Writing the triage note", rules: "Applying the rulebook in code", reasoner: "Sonnet 5.5 is deciding",
    extractor: "Splitting the answer into claims", judge: "Checking each claim", score: "Scoring" };
  const sleep = (ms) => new Promise((r) => setTimeout(r, matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : ms));
  const announce = (k, status) => { $("#rv-status").textContent = `${STEP_TEXT[k] || k}: ${status === "done" ? "done" : "in progress"}`; };

  /* ---------------------------------------------------------------- the game: screens, points, badges */
  const SCREENS = ["pick", "bet", "run", "result"];
  function go(name) {
    SCREENS.forEach((k, i) => { $(`#s-${k}`).hidden = k !== name; $("#gsteps").children[i].className = k === name ? "on" : SCREENS.indexOf(name) > i ? "done" : ""; });
    $("#board").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }
  const BADGES = [["first", "First shift", "Finish one run."], ["perfect", "Perfect read", "Call all three bets right."],
    ["caught", "Caught one", "The verifier flags a claim the model made."], ["clean", "Clean sheet", "Every claim holds."],
    ["undecided", "Beyond the book", "The rulebook cannot decide, and says so."], ["split", "Split decision", "The model goes against the rulebook."],
    ["both", "Both treatments", "The model recommends the drug and the removal."], ["neither", "Hands off", "The model recommends neither."]];
  const KEY = "sdc-shift-v1";
  let save = { points: 0, runs: 0, badges: [] };
  try { const raw = JSON.parse(localStorage.getItem(KEY) || "null"); if (raw && Array.isArray(raw.badges)) save = { points: +raw.points || 0, runs: +raw.runs || 0, badges: raw.badges.filter((b) => BADGES.some(([k]) => k === b)) }; } catch (e) { /* storage blocked: the game still works, it just forgets */ }
  const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(save)); } catch (e) { /* ignore */ } };
  function drawTally(fresh = []) {
    $("#tally").replaceChildren(h("span", {}, h("b", { text: save.points.toLocaleString("en") }), " points"), h("span", {}, h("b", { text: save.runs }), ` run${save.runs === 1 ? "" : "s"}`),
      h("span", {}, h("b", { text: `${save.badges.length}/${BADGES.length}` }), " badges"));
    $("#shelf").replaceChildren(...BADGES.map(([k, name, how]) => h("div", { class: `gbadge${save.badges.includes(k) ? " got" : ""}${fresh.includes(k) ? " fresh" : ""}` },
      h("i", { "aria-hidden": "true", text: save.badges.includes(k) ? "★" : "?" }), h("b", { text: name }), h("span", { text: how }))));
  }

  /* ---- 1 · the six patients */
  const CARDS = [["fx01_thrombolysis_clear", "The early arrival"], ["fx02_thrombectomy_large_mismatch", "The late find"], ["fx03_haemorrhage_present", "The dark patch"],
    ["fx04_large_core_no_mismatch", "The heavy scan"], ["fx05_borderline_onset", "The close call"]];
  let picked = null;
  function pick(id, card) {
    const sel = $("#preset");
    sel.value = id; sel.dispatchEvent(new Event("change"));
    picked = id;
    document.querySelectorAll("#cards .pcard").forEach((c) => c.setAttribute("aria-pressed", String(c === card)));
    $("#picked").hidden = false;
    if (card) $("#picked").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
  }
  function buildCards() {
    const box = $("#cards"), C = 2 * Math.PI * 42;
    const make = (id, name, row) => {
      const i = row && row.inputs, mins = i ? i.minutes : 0;
      const NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg"), ring = h("span", { class: "mini" });
      svg.setAttribute("viewBox", "0 0 100 100"); svg.setAttribute("aria-hidden", "true");
      [["bg", null], ["fg", `${Math.min(1, mins / 270) * C} ${C}`]].forEach(([cls, dash]) => { const c = document.createElementNS(NS, "circle");
        c.setAttribute("class", cls); c.setAttribute("cx", 50); c.setAttribute("cy", 50); c.setAttribute("r", 42); if (dash) c.setAttribute("stroke-dasharray", dash); svg.append(c); });
      ring.append(svg);
      ring.append(h("em", { text: i ? `${Math.floor(mins / 60)}h${String(mins % 60).padStart(2, "0")}` : "?" }));
      const card = h("button", { type: "button", class: `pcard${i ? "" : " mystery"}`, "aria-pressed": "false" }, ring,
        h("span", { class: "pc-body" }, h("b", { text: name }),
          h("span", { text: i ? `${i.age}-year-old ${i.sex === "female" ? "woman" : "man"}, ${Math.floor(mins / 60)} h ${String(mins % 60).padStart(2, "0")} since ${i.onset === "witnessed" ? "onset" : "last seen well"}` : "A patient drawn at random from the 150 scenarios" }),
          i ? h("span", { class: "pc-chips" }, h("i", { text: `NIHSS ${i.nihss}` }), h("i", { text: `core ${i.core_ml} mL` }), h("i", { text: `site ${i.site}` })) : null));
      card.addEventListener("click", () => {
        if (i) { pick(id, card); return; }
        const pool = state.index.filter((r) => r.kind === "scenario" && r.inputs); pick(pool[Math.floor(Math.random() * pool.length)].id, card);
      });
      return card;
    };
    box.replaceChildren(...CARDS.map(([id, name]) => make(id, name, state.index.find((r) => r.id === id))).filter(Boolean), make(null, "Surprise me", null));
  }

  /* ---- 2 · the three bets */
  const BETS = [["drug", "The clot-busting drug", "Will Sonnet recommend it?", ["Give it", "Don't"]], ["removal", "Clot removal", "Will Sonnet recommend it?", ["Attempt it", "Don't"]],
    ["flag", "The verifier", "Will it flag any of Sonnet's claims?", ["Yes, at least one", "No, all clean"]]];
  let bets = {};
  function buildBets() {
    bets = {}; $("#run").disabled = true;
    $("#bets").replaceChildren(...BETS.map(([k, name, ask, opts]) => {
      const row = h("div", { class: "bet", role: "group", "aria-label": name });
      const btns = opts.map((o, n) => h("button", { type: "button", "aria-pressed": "false", text: o, onclick: () => {
        bets[k] = n === 0; btns.forEach((b, j) => b.setAttribute("aria-pressed", String(j === n)));
        $("#run").disabled = BETS.some(([key]) => bets[key] == null);
      } }));
      row.append(h("div", { class: "bet-q" }, h("b", { text: name }), h("span", { text: ask })), h("div", { class: "bet-a" }, btns));
      return row;
    }));
    $("#bet-who").textContent = $("#p-who").textContent;
  }

  /* ---- 3 · the run */
  let viz = null;
  function startViz(title, sub) {
    if (!viz) viz = window.RunViz.mount($("#runviz"), { passageIds: Object.keys(state.passages) });
    viz.reset(); $("#run-title").textContent = title; $("#run-sub").textContent = sub;
    go("run"); return viz;
  }
  async function run() {
    const err = $("#builder-error");
    err.hidden = true; $("#run").disabled = true;
    let res;
    try {
      res = await fetch("/api/run", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ inputs: payload() }) });
    } catch (e) { $("#run").disabled = false; err.textContent = "Could not reach the server. Try again in a moment."; err.hidden = false; return; }
    if (!res.ok) {
      $("#run").disabled = false;
      const body = await res.json().catch(() => ({}));
      err.textContent = body.error ? `Please check: ${body.error}` : `The server answered ${res.status}.`; err.hidden = false; return;
    }
    const { run_id } = await res.json();
    const v = startViz("The run", "Each step is drawn as the server reports it."), done = new Set(), placed = { ...bets };
    let queue = Promise.resolve();
    const enqueue = (fn) => { queue = queue.then(fn).catch(() => { /* drawing must never stop the result */ }); };
    const es = new EventSource(`/api/run/${run_id}/events`);
    es.addEventListener("progress", (e) => enqueue(async () => {
      const ev = JSON.parse(e.data), d = ev.detail || {};
      announce(ev.step, ev.status); $("#run-sub").textContent = `${STEP_TEXT[ev.step] || ev.step}…`;
      if (ev.status === "running") {
        v.step(ev.step, "running");
        if (ev.step === "reasoner") v.reasoning(true);
        if (ev.step === "extractor") v.say("Splitting Sonnet's answer into single claims…");
        if (ev.step === "judge") v.judging(d.calls);
        return;
      }
      v.step(ev.step, "running");
      if (ev.step === "note") await v.note(d.note, d.vision);
      if (ev.step === "rules") { await v.rules(d.expected); await sleep(900); }
      if (ev.step === "reasoner") { await v.decisions(d.recommendation); await sleep(900); }
      if (ev.step === "extractor" && d.claims) await v.claimSlots(d.claims);
      v.step(ev.step, "done", ev.t); done.add(ev.step);
    }));
    es.addEventListener("result", (e) => { es.close(); enqueue(async () => {
      const r = JSON.parse(e.data);
      const label = r.source === "live" ? "Answered live" : r.source === "cache" ? "From the cache" : "Precomputed fallback";
      if (r.source !== "live") window.RunViz.STEPS.forEach(([k]) => { if (!done.has(k)) v.step(k, "pre"); });
      $("#run-sub").textContent = r.source === "live" ? "Marking the claims…" : `${label}: ${r.message}`;
      await v.finish(r.view, { label: r.source === "live" ? null : label.toLowerCase(),
        tag: r.source === "precomputed" ? `closest precomputed case: ${r.preset_title}` : r.source === "cache" ? "from the cache" : null });
      await sleep(1200);
      showResult(r, placed, label);
    }); });
    es.onerror = () => { if (es.readyState === EventSource.CLOSED) return; };
  }

  /* A recorded run, replayed through the same drawing: the story's patient, answered earlier by Sonnet 5.5. No bets. */
  async function replay() {
    const file = state.cases.story; if (!file) return;
    const v = startViz("A recorded run", "Replayed from the record. No model is called."), arm = file.arms.rulebook, ans = arm.answers["claude-sonnet-5-5"];
    const stepOn = (k) => { v.step(k, "running"); announce(k, "running"); $("#run-sub").textContent = `${STEP_TEXT[k]}… (recorded)`; };
    const fin = (k) => { v.step(k, "done"); announce(k, "done"); };
    stepOn("note"); await v.note(arm.note, arm.vision); fin("note");
    stepOn("rules"); await v.rules(file.rules); await sleep(900); fin("rules");
    stepOn("reasoner"); v.reasoning(true); await sleep(2400); await v.decisions(ans.recommendation); await sleep(900); fin("reasoner");
    stepOn("extractor"); v.say("Splitting Sonnet's answer into single claims…"); await sleep(700); await v.claimSlots(ans.claims.length); fin("extractor");
    stepOn("judge"); v.judging(0); await sleep(1400);
    const view = { vision: arm.vision, note: arm.note, expected: file.rules, ...ans };
    await v.finish(view, { label: "replay" }); fin("judge"); stepOn("score"); fin("score");
    await sleep(1200);
    const row = state.index.find((r) => r.id === file.id);
    showResult({ source: "replay", view, message: `The fixture “${row ? row.title : file.id}”, answered earlier by Sonnet 5.5 with the same prompt and verifier. Nothing was called.` }, null, "Recorded run");
  }

  /* ---- 4 · the result: bets scored against what the model and the verifier actually did */
  function showResult(r, placed, label) {
    const view = r.view, rec = view.recommendation || {}, c = view.counts || {}, dec = view.decisions || {};
    const out = { drug: rec.thrombolysis ? rec.thrombolysis.recommend : null, removal: rec.thrombectomy ? rec.thrombectomy.recommend : null, flag: (c.flagged || 0) > 0 };
    const said = { drug: (v) => (v == null ? "No answer" : v ? "Give it" : "Don't"), removal: (v) => (v == null ? "No answer" : v ? "Attempt it" : "Don't"), flag: (v) => (v ? `Yes: ${c.flagged} of ${c.claims}` : `No: all ${c.claims} hold`) };
    const mine = { drug: (v) => (v ? "Give it" : "Don't"), removal: (v) => (v ? "Attempt it" : "Don't"), flag: (v) => (v ? "Yes" : "No") };
    const rule = { drug: dec.thrombolysis, removal: dec.thrombectomy };
    let hits = 0;
    $("#res-bets").replaceChildren(...BETS.map(([k, name]) => {
      const ok = placed ? placed[k] === out[k] : null; if (ok) hits++;
      return h("div", { class: `rb${ok === true ? " hit" : ok === false ? " miss" : ""}` }, h("span", { class: "rb-i", "aria-hidden": "true", text: ok === true ? "✓" : ok === false ? "✕" : "·" }),
        h("div", {}, h("b", { text: name }), rule[k] ? h("small", { text: `Rulebook: ${ruleWord(rule[k].rulebook)}${rule[k].agrees === false ? " · the model differs" : rule[k].agrees == null ? " · either answer accepted" : ""}` }) : null),
        placed ? h("div", { class: "rb-you" }, h("small", { text: "You said" }), mine[k](placed[k])) : null,
        h("div", { class: "rb-out" }, h("small", { text: k === "flag" ? "The verifier" : "Sonnet 5.5" }), said[k](out[k])));
    }));
    const points = placed ? hits * 100 + (hits === 3 ? 50 : 0) : 0;
    $("#res-score").replaceChildren(h("b", { text: placed ? `+${points}` : "▶" }), h("span", { text: placed ? "points" : "replay" }));
    $("#res-title").textContent = !placed ? "A recorded run." : ["The model surprised you.", "One of three.", "Two of three.", "Perfect read."][hits];
    const sub = $("#res-sub"); sub.replaceChildren(h("strong", { text: `${label}. ` }),
      r.source === "live" && r.seconds ? `Sonnet 5.5 answered in ${r.seconds.total} s.` : r.message || "",
      r.source === "precomputed" && r.differs && r.differs.length ? ` It differs from your patient in: ${r.differs.join(", ")}. Your bets are scored against it.` : "");
    // every claim, as marked
    const text = $("#res-claimtext"); text.textContent = "Hover or tab to any square to read its claim.";
    $("#res-claims").replaceChildren(...(view.claims || []).map((cl) => {
      const s = h("span", { class: `rv-claim in ${cl.flagged ? "flag" : "ok"}`, tabindex: "0", text: cl.flagged ? "✕" : "✓", "aria-label": `${cl.flagged ? "Flagged" : "Supported"}: ${cl.text}` });
      const show = () => { text.textContent = cl.flagged && cl.reason ? `${cl.text} — flagged: ${cl.reason}` : cl.text; };
      s.addEventListener("pointerenter", show); s.addEventListener("focus", show); return s;
    }));
    $("#res-claims-k").textContent = `${c.claims || 0} claims, as the verifier marked them · ${c.flagged || 0} flagged · ${c.critical || 0} would change a decision`;
    // badges
    const earned = [];
    if (placed) {
      const got = (k, yes) => { if (yes && !save.badges.includes(k)) { save.badges.push(k); earned.push(k); } };
      got("first", true); got("perfect", hits === 3); got("caught", out.flag); got("clean", !out.flag && (c.claims || 0) > 0);
      got("undecided", Object.values(dec).some((d) => d && d.agrees == null)); got("split", Object.values(dec).some((d) => d && d.agrees === false));
      got("both", out.drug === true && out.removal === true); got("neither", out.drug === false && out.removal === false);
      save.points += points; save.runs += 1; persist();
    }
    $("#badges-new").replaceChildren(...(earned.length ? [h("p", { class: "kicker", text: earned.length > 1 ? "New badges" : "New badge" }),
      h("div", { class: "shelf" }, earned.map((k) => { const [, name, how] = BADGES.find(([b]) => b === k); return h("div", { class: "gbadge got fresh" }, h("i", { "aria-hidden": "true", text: "★" }), h("b", { text: name }), h("span", { text: how })); }))] : []));
    drawTally(earned);
    // the full answer, on demand
    const full = $("#run-result"); full.hidden = true; $("#full").setAttribute("aria-expanded", "false"); $("#full").textContent = "Read the full answer";
    renderAnswer(full, view, { title: r.source === "precomputed" ? `The closest precomputed case (${r.preset_title})` : "Sonnet 5.5's answer, claim by claim" });
    go("result");
  }

  /* ---------------------------------------------------------------- start */
  async function start() {
    drawTally();
    try {
      const [meta, index, passages] = await Promise.all([getJSON("/api/meta"), getJSON("/api/bank/index"), getJSON("/api/bank/passages")]);
      state.meta = meta; state.index = index.cases; state.passages = passages;
      state.cases.story = await loadCase("fx01_thrombolysis_clear");   // the story's patient, for the replay
    } catch (e) {
      $("#live-status").textContent = "The page could not load its data. Please reload."; return;
    }
    buildControls(); buildCards();
    $("#to-bets").addEventListener("click", () => { buildBets(); go("bet"); });
    $("#back-pick").addEventListener("click", () => go("pick"));
    $("#run").addEventListener("click", run);
    $("#replay").addEventListener("click", replay);
    $("#again").addEventListener("click", () => { go("pick"); });
    $("#full").addEventListener("click", () => { const f = $("#run-result"); f.hidden = !f.hidden; $("#full").setAttribute("aria-expanded", String(!f.hidden)); $("#full").textContent = f.hidden ? "Read the full answer" : "Hide the full answer"; });
    $("#reset").addEventListener("click", () => { save = { points: 0, runs: 0, badges: [] }; persist(); drawTally(); });
    const want = new URLSearchParams(location.search).get("preset");   // arriving from the story with her case
    if (want && [...$("#preset").options].some((o) => o.value === want)) {
      const k = CARDS.findIndex(([id]) => id === want); pick(want, k >= 0 ? $("#cards").children[k] : null);
    }
  }
  document.addEventListener("DOMContentLoaded", start);
})();
