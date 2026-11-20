/* Stroke Decision Copilot — shared helpers for every page of the public Space (DEC-32). Plain JS, no third-party code.
   Every value from the server is set with textContent, never as HTML. */
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

  function routeText(row) {
    const names = { thrombolysis: "Thrombolysis", thrombectomy: "Thrombectomy" };
    return ["thrombolysis", "thrombectomy"].map((d) => row.rules[d] ? `${names[d]} ${row.rules[d].value}` : "").filter(Boolean).join(" · ");
  }
  async function loadCase(id) {
    if (!state.cases[id]) state.cases[id] = await getJSON(`/api/bank/case/${encodeURIComponent(id)}`);
    return state.cases[id];
  }
  function tile(v, l) { return h("div", { class: "stat" }, h("div", { class: "v", text: v }), h("div", { class: "l", text: l })); }

  /* the top navigation marks the page you are on */
  function initNav() {
    const here = (location.pathname.split("/").pop() || "index.html");
    document.querySelectorAll(".nav a[data-page]").forEach((a) => { if (a.dataset.page === here) a.setAttribute("aria-current", "page"); });
  }
  document.addEventListener("DOMContentLoaded", initNav);

  window.SDC = { $, h, getJSON, pct, hm, MODEL_LABEL, MODELS, SLOT, FIELD_GROUPS, SUBTYPE, CONTRACT_ROWS, state, get, fmt, decisionWord, ruleWord,
    noteWithMarks, renderAnswer, routeText, loadCase, tile };
})();
