/* Small SVG charts for the Space (no library, no CDN). Each chart: thin marks, hairline grid, a hover tooltip and a
   table view beneath (tooltips enhance, never gate). Colours come from CSS tokens, by role. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const tip = () => document.getElementById("tooltip");

  function el(name, attrs = {}, parent) {
    const n = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  }
  function text(parent, x, y, s, attrs = {}) {
    const t = el("text", { x, y, ...attrs }, parent);
    t.textContent = s;
    return t;
  }
  function showTip(evt, lines) {
    const t = tip();
    t.replaceChildren(...lines.map((l, i) => { const d = document.createElement("div"); d.textContent = l; if (i === 0) d.style.fontWeight = "600"; return d; }));
    t.hidden = false;
    const x = Math.min(window.innerWidth - 300, evt.clientX + 14);
    t.style.left = `${x}px`;
    t.style.top = `${evt.clientY + 14}px`;
  }
  function hideTip() { tip().hidden = true; }
  function hover(node, lines) {
    node.setAttribute("tabindex", "0");
    node.addEventListener("mousemove", (e) => showTip(e, lines));
    node.addEventListener("mouseleave", hideTip);
    node.addEventListener("focus", () => { const r = node.getBoundingClientRect(); showTip({ clientX: r.right, clientY: r.top }, lines); });
    node.addEventListener("blur", hideTip);
  }
  function table(container, head, rows, { open = false } = {}) {
    const d = document.createElement("details");
    d.open = open;
    const s = document.createElement("summary"); s.textContent = "Show the numbers as a table"; d.appendChild(s);
    const t = document.createElement("table"); t.className = "tbl";
    const tr = document.createElement("tr");
    head.forEach((h, i) => { const th = document.createElement("th"); th.textContent = h; if (i) th.className = "num"; tr.appendChild(th); });
    t.appendChild(tr);
    rows.forEach((r) => { const row = document.createElement("tr"); r.forEach((c, i) => { const td = document.createElement("td"); td.textContent = c; if (i) td.className = "num"; row.appendChild(td); }); t.appendChild(row); });
    d.appendChild(t);
    container.appendChild(d);
  }
  const pct = (v, d = 1) => (v == null ? "n/a" : `${(100 * v).toFixed(d)}%`);
  function ticks(max, n = 5) { const step = max / n; return Array.from({ length: n + 1 }, (_, i) => +(i * step).toFixed(4)); }

  /* Dot + 95% interval per row. rows: [{label, value, ci:[lo,hi], color, tip:[...]}] */
  function dotCI(container, rows, { max = 0.5, format = pct, width = 560, rowH = 34, left = 140 } = {}) {
    const top = 8, bottom = 26, h = top + rows.length * rowH + bottom, right = 20;
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${width} ${h}`, role: "img" });
    const x = (v) => left + (Math.min(v, max) / max) * (width - left - right);
    ticks(max).forEach((t) => {
      el("line", { class: "grid", x1: x(t), x2: x(t), y1: top, y2: h - bottom }, svg);
      text(svg, x(t), h - 8, format(t, 0), { "text-anchor": "middle" });
    });
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2;
      text(svg, left - 10, y + 4, r.label, { "text-anchor": "end", style: "fill:var(--ink)" });
      if (r.ci) el("line", { class: "ci", x1: x(r.ci[0]), x2: x(r.ci[1]), y1: y, y2: y, stroke: r.color, opacity: 0.55 }, svg);
      el("circle", { cx: x(r.value), cy: y, r: 6, fill: r.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
      text(svg, x(r.ci ? r.ci[1] : r.value) + 8, y + 4, format(r.value), { style: "fill:var(--ink-2)" });
      const hit = el("rect", { class: "hit", x: 0, y: y - rowH / 2, width, height: rowH }, svg);
      hover(hit, r.tip || [r.label, format(r.value)]);
    });
    svg.setAttribute("aria-label", rows.map((r) => `${r.label} ${format(r.value)}`).join("; "));
    container.appendChild(svg);
    return svg;
  }

  /* Paired values per row: from (hollow) to (filled). rows: [{label, from, to, color, tip}] */
  function dumbbell(container, rows, { max = 0.3, format = pct, width = 560, rowH = 34, left = 140, fromLabel, toLabel } = {}) {
    const top = 8, bottom = 26, h = top + rows.length * rowH + bottom, right = 20;
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${width} ${h}`, role: "img" });
    const x = (v) => left + (Math.min(v, max) / max) * (width - left - right);
    ticks(max).forEach((t) => {
      el("line", { class: "grid", x1: x(t), x2: x(t), y1: top, y2: h - bottom }, svg);
      text(svg, x(t), h - 8, format(t, 0), { "text-anchor": "middle" });
    });
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2;
      text(svg, left - 10, y + 4, r.label, { "text-anchor": "end", style: "fill:var(--ink)" });
      el("line", { x1: x(r.from), x2: x(r.to), y1: y, y2: y, stroke: "var(--line-2)", "stroke-width": 2 }, svg);
      el("circle", { cx: x(r.from), cy: y, r: 5.5, fill: "var(--surface)", stroke: r.color, "stroke-width": 2 }, svg);
      el("circle", { cx: x(r.to), cy: y, r: 6, fill: r.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
      text(svg, x(Math.max(r.from, r.to)) + 9, y + 4, `${format(r.from)} → ${format(r.to)}`, { style: "fill:var(--ink-2)" });
      hover(el("rect", { class: "hit", x: 0, y: y - rowH / 2, width, height: rowH }, svg), r.tip || [r.label]);
    });
    container.appendChild(svg);
    const lg = document.createElement("div"); lg.className = "legend";
    lg.innerHTML = `<span><i style="border:2px solid var(--ink-2);background:var(--surface)"></i></span><span><i style="background:var(--ink-2)"></i></span>`;
    lg.children[0].append(fromLabel || "before"); lg.children[1].append(toLabel || "after");
    container.appendChild(lg);
  }

  /* Horizontal share bars, one hue. rows: [{label, value, n, tip}] */
  function bars(container, rows, { width = 560, rowH = 30, left = 140, color = "var(--s1)" } = {}) {
    const top = 6, bottom = 24, h = top + rows.length * rowH + bottom, right = 70;
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${width} ${h}`, role: "img" });
    const x = (v) => left + v * (width - left - right);
    ticks(1, 4).forEach((t) => {
      el("line", { class: "grid", x1: x(t), x2: x(t), y1: top, y2: h - bottom }, svg);
      text(svg, x(t), h - 6, pct(t, 0), { "text-anchor": "middle" });
    });
    rows.forEach((r, i) => {
      const y = top + i * rowH + 6;
      text(svg, left - 10, y + 13, r.label, { "text-anchor": "end", style: "fill:var(--ink)" });
      el("rect", { x: left, y, width: Math.max(2, x(r.value) - left), height: rowH - 12, rx: 4, fill: color }, svg);
      text(svg, x(r.value) + 6, y + 13, `${pct(r.value, 0)}${r.n != null ? ` (n ${r.n})` : ""}`);
      hover(el("rect", { class: "hit", x: 0, y: y - 6, width, height: rowH }, svg), r.tip || [r.label, pct(r.value)]);
    });
    container.appendChild(svg);
  }

  /* Reliability: predicted vs observed, the diagonal for reference. points: [{mean_prob, observed, count}] */
  function reliability(container, points, { width = 300 } = {}) {
    const pad = 34, h = width;
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${width} ${h}`, role: "img", "aria-label": "Reliability diagram" });
    const s = (v) => pad + v * (width - pad - 10), sy = (v) => h - pad - v * (h - pad - 10);
    [0, 0.5, 1].forEach((t) => {
      el("line", { class: "grid", x1: s(t), x2: s(t), y1: sy(0), y2: sy(1) }, svg);
      el("line", { class: "grid", x1: s(0), x2: s(1), y1: sy(t), y2: sy(t) }, svg);
      text(svg, s(t), h - 14, t.toFixed(1), { "text-anchor": "middle" });
      text(svg, pad - 6, sy(t) + 4, t.toFixed(1), { "text-anchor": "end" });
    });
    el("line", { x1: s(0), y1: sy(0), x2: s(1), y2: sy(1), stroke: "var(--ink-3)", "stroke-width": 1 }, svg);
    const d = points.map((p, i) => `${i ? "L" : "M"}${s(p.mean_prob)},${sy(p.observed)}`).join("");
    el("path", { d, fill: "none", stroke: "var(--s1)", "stroke-width": 2 }, svg);
    points.forEach((p) => {
      const c = el("circle", { cx: s(p.mean_prob), cy: sy(p.observed), r: 4.5, fill: "var(--s1)", stroke: "var(--surface)", "stroke-width": 2 }, svg);
      hover(c, [`Predicted ${pct(p.mean_prob)}`, `Observed ${pct(p.observed)}`, `${p.count} scans`]);
    });
    text(svg, width / 2, h - 1, "predicted probability", { "text-anchor": "middle" });
    container.appendChild(svg);
  }

  window.Charts = { dotCI, dumbbell, bars, reliability, table, pct, hover };
})();
