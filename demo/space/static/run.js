/* Stroke Decision Copilot — the live run page: the scenario builder, the run drawn step by step, and a recorded
   replay (DEC-32 §4). Plain JS. Every server value is set with textContent. */
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
    $("#builder").addEventListener("submit", (e) => { e.preventDefault(); run(); });
    $("#replay").addEventListener("click", replay);
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
  let viz = null;
  function startViz() {
    if (!viz) viz = window.RunViz.mount($("#runviz"), { passageIds: Object.keys(state.passages) });
    $("#rv-idle").hidden = true; $("#runviz").hidden = false;
    viz.reset(); $("#run-result").replaceChildren();
    return viz;
  }
  const busy = (b) => { $("#run").disabled = b; $("#replay").disabled = b; };
  const announce = (k, status) => { $("#rv-status").textContent = `${STEP_TEXT[k] || k}: ${status === "done" ? "done" : "in progress"}`; };
  function resultNotice(label, message, extra) {
    return h("div", { class: `notice ${label === "Precomputed fallback" ? "fallback" : ""}` }, h("strong", { text: label }), message, extra);
  }

  async function run() {
    const err = $("#builder-error");
    err.hidden = true; busy(true);
    let res;
    try {
      res = await fetch("/api/run", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ inputs: payload() }) });
    } catch (e) { busy(false); err.textContent = "Could not reach the server. Try again in a moment."; err.hidden = false; return; }
    if (!res.ok) {
      busy(false);
      const body = await res.json().catch(() => ({}));
      err.textContent = body.error ? `Please check: ${body.error}` : `The server answered ${res.status}.`; err.hidden = false; return;
    }
    const { run_id } = await res.json();
    const v = startViz(), done = new Set();
    let queue = Promise.resolve();
    const enqueue = (fn) => { queue = queue.then(fn).catch(() => { /* drawing must never stop the result */ }); };
    const es = new EventSource(`/api/run/${run_id}/events`);
    es.addEventListener("progress", (e) => enqueue(async () => {
      const ev = JSON.parse(e.data), d = ev.detail || {};
      announce(ev.step, ev.status);
      if (ev.status === "running") {
        v.step(ev.step, "running");
        if (ev.step === "reasoner") v.reasoning(true);
        if (ev.step === "judge") v.judging(d.calls);
        return;
      }
      v.step(ev.step, "running");
      if (ev.step === "note") await v.note(d.note, d.vision);
      if (ev.step === "rules") await v.rules(d.expected);
      if (ev.step === "reasoner") await v.decisions(d.recommendation);
      if (ev.step === "extractor" && d.claims) await v.claimSlots(d.claims);
      v.step(ev.step, "done", ev.t); done.add(ev.step);
    }));
    es.addEventListener("result", (e) => { es.close(); enqueue(async () => {
      const r = JSON.parse(e.data);
      const label = r.source === "live" ? "Answered live" : r.source === "cache" ? "From the cache" : "Precomputed fallback";
      if (r.source !== "live") window.RunViz.STEPS.forEach(([k]) => { if (!done.has(k)) v.step(k, "pre"); });
      await v.finish(r.view, { label: r.source === "live" ? null : label.toLowerCase(),
        tag: r.source === "precomputed" ? `closest precomputed case: ${r.preset_title}` : r.source === "cache" ? "from the cache" : null });
      $("#run-result").append(resultNotice(label, r.message, [
        r.source === "precomputed" && r.differs && r.differs.length ? h("div", { class: "muted small", text: `Differs from your inputs in: ${r.differs.join(", ")}.` }) : null,
        r.source === "live" && r.seconds ? h("div", { class: "muted small", text: `${r.seconds.total} s in all.` }) : null]));
      const ans = h("div");
      renderAnswer(ans, r.view, { title: r.source === "precomputed" ? `The closest precomputed case (${r.preset_title})` : "Sonnet 5.5's answer, claim by claim" });
      $("#run-result").append(ans);
      $("#rv-status").textContent = `${label}. The answer is below the steps.`;
      busy(false);
    }); });
    es.onerror = () => { if (es.readyState === EventSource.CLOSED) return; };
  }

  /* A recorded run, replayed through the same drawing: the late-window fixture, answered earlier by Sonnet 5.5. */
  async function replay() {
    const file = state.cases.story; if (!file) return;
    busy(true);
    const v = startViz(), arm = file.arms.rulebook, ans = arm.answers["claude-sonnet-5-5"];
    const go = (k) => { v.step(k, "running"); announce(k, "running"); };
    const fin = (k) => { v.step(k, "done"); announce(k, "done"); };
    go("note"); await v.note(arm.note, arm.vision); fin("note");
    go("rules"); await v.rules(file.rules); fin("rules");
    go("reasoner"); v.reasoning(true); await sleep(2600); await v.decisions(ans.recommendation); fin("reasoner");
    go("extractor"); await sleep(500); await v.claimSlots(ans.claims.length); fin("extractor");
    go("judge"); v.judging(0); await sleep(1400);
    const view = { vision: arm.vision, note: arm.note, expected: file.rules, ...ans };
    await v.finish(view, { label: "replay" }); fin("judge");
    go("score"); fin("score");
    const row = state.index.find((r) => r.id === file.id);
    $("#run-result").append(resultNotice("Replay of a recorded run",
      `The fixture “${row ? row.title : file.id}”, answered earlier by Sonnet 5.5 with the same prompt and verifier. Nothing was called and nothing was spent.`));
    const out = h("div"); renderAnswer(out, view, { title: "Sonnet 5.5's answer, claim by claim" }); $("#run-result").append(out);
    $("#rv-status").textContent = "Replay finished. The answer is below the steps.";
    busy(false);
  }

  /* ---------------------------------------------------------------- start */
  async function start() {
    try {
      const [meta, index, passages] = await Promise.all([getJSON("/api/meta"), getJSON("/api/bank/index"), getJSON("/api/bank/passages")]);
      state.meta = meta; state.index = index.cases; state.passages = passages;
      state.cases.story = await loadCase("fx01_thrombolysis_clear");   // the story's patient, for the replay
    } catch (e) {
      $("#live-status").textContent = "The page could not load its data. Please reload."; return;
    }
    buildControls();
    const want = new URLSearchParams(location.search).get("preset"), sel = $("#preset");   // arriving from the story with her case
    if (want && [...sel.options].some((o) => o.value === want)) { sel.value = want; sel.dispatchEvent(new Event("change")); }
  }
  document.addEventListener("DOMContentLoaded", start);
})();
