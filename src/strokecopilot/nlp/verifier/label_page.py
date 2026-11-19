"""The labelling webpage for human adjudication (plan §7 Phase 3 NLP B; docs/LABELLING_GUIDE.md).

One self-contained HTML file per labeller, opened from Drive in any browser, offline: no server, no account, no
external script. It holds the blinded sample (the sheet that adjudicate.blind_and_export writes) with what a labeller
needs to judge each claim: the case's imaging findings and triage note (by alias, never by case ID), the full answer
entry the claim came from, the sources the model declared, and the whole rulebook. It never holds the model, the run,
the case ID or the verifier's verdict: those stay in the key file.

Four labellers in two pairs (PAIRS; Parsa, 17 Nov 2026): the 100 claims are split into two halves, stratified by
model x verifier verdict (split_halves), and each pair labels one half, so every claim has two independent labels and
each person labels 50. Within a pair the first name fills label_A and the second label_B.

Labels are saved in the browser as the labeller works (localStorage, per labeller and sample) and downloaded as a CSV
in the sheet's own columns, so adjudicate.read_labels reads them after merge_sheets (per pair) and combine_sheets (the
two halves). The page can load a downloaded CSV back, to move to another computer; rows outside its half are ignored.
"""

import csv
import hashlib
import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from strokecopilot.nlp.reasoner.prompt import VISION_HIDDEN
from strokecopilot.nlp.verifier import adjudicate as adj

# Mixed pairs, one from the vision side and one from the NLP side, so shared background does not inflate kappa.
PAIRS = {1: ("Lorna", "Benny"), 2: ("Mohamed", "Hana")}
SPLIT_SEED = 20261117


def role(name):
    """(half, letter, partner) for a labeller: the pair's first name is A, the second B."""
    for half, (a, b) in PAIRS.items():
        if name == a:
            return half, "A", b
        if name == b:
            return half, "B", a
    raise ValueError(f"{name!r} is not a labeller ({', '.join(n for pair in PAIRS.values() for n in pair)}).")


def split_halves(key_rows, seed=SPLIT_SEED):
    """{sample_id: 1 or 2}: each model x verdict stratum split as evenly as it divides, an odd claim going to the half
    that has fewer so far, so both halves see every model and every kind of flag and hold 50 each."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in key_rows:
        strata[(r["model"], r["verdict"])].append(r["sample_id"])
    halves, count = {}, {1: 0, 2: 0}
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        first = 1 if count[1] <= count[2] else 2
        for i, sid in enumerate(ids):
            half = first if i % 2 == 0 else 3 - first
            halves[sid] = half
            count[half] += 1
    return halves
_ENTRY = re.compile(r"^(?P<field>[a-z_.]+)\[(?P<index>\d+)\]")


def answer_entry(rec, claim_id):
    """The output entry an atomic claim was taken from, as shown for context: a rationale entry's claim text, a
    blocking factor, an uncertainty, or a decision's cited passage ID. None if the ID does not point into `rec`."""
    m = _ENTRY.match(claim_id)
    if not m:
        return None
    field, i = m.group("field"), int(m.group("index"))
    if field == "rationale":
        entries = rec.get("rationale") or []
        return entries[i]["claim"] if i < len(entries) else None
    if field == "uncertainties":
        entries = rec.get("uncertainties") or []
        return entries[i] if i < len(entries) else None
    decision, _, part = field.partition(".")
    entries = (rec.get(decision) or {}).get(part) or []
    return entries[i] if i < len(entries) else None


def sample_items(sheet_rows, key_rows, load_inputs):
    """The page's items and cases from the sheet, the key and `load_inputs(run_folder, case) -> {vision, note}`.

    Each item: {sample_id, case_alias, field, claim, context, sources}; sources are the full declared sources
    ({type, field} | {type, quote} | {type, id}), read from the run's traces. Cases: {alias: {vision, note}}, with the
    vision JSON's case_id and model_version removed. The key is only read here; nothing from it beyond the claim's
    location reaches the page."""
    key = {r["sample_id"]: r for r in key_rows}
    items, cases = [], {}
    traces, outputs = {}, {}
    for row in sheet_rows:
        k = key[row["sample_id"]]
        folder, case = Path(k["run_folder"]), k["case"]
        if (folder, case) not in traces:
            traces[(folder, case)] = {c["id"]: c for c in json.loads((folder / "traces" / f"{case}.json").read_text())["claims"]}
            out = folder / "outputs" / f"{case}.json"
            outputs[(folder, case)] = json.loads(out.read_text()) if out.exists() else {}
        claim = traces[(folder, case)].get(k["claim_id"], {})
        context = answer_entry(outputs[(folder, case)], k["claim_id"])
        items.append({"sample_id": row["sample_id"], "case_alias": row["case_alias"], "field": row["field"],
                      "claim": row["claim"], "context": context if context != row["claim"] else None,
                      "sources": claim.get("declared_sources") or []})
        if row["case_alias"] not in cases:
            inputs = load_inputs(str(folder), case)
            vision = {f: v for f, v in inputs["vision"].items() if f not in VISION_HIDDEN}  # as the prompt shows it
            cases[row["case_alias"]] = {"vision": vision, "note": inputs["note"]}
    return items, cases


def sample_hash(items):
    blob = json.dumps([[i["sample_id"], i["claim"]] for i in items], ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def build_page(items, cases, passages, name, title="Verifier labelling"):
    """The HTML page for one labeller (a name in PAIRS), holding only the items and cases given: their half.
    passages: [{id, text}] in rulebook order."""
    half, letter, partner = role(name)
    used = {i["case_alias"] for i in items}
    data = {"labeller": letter, "name": name, "partner": partner, "half": half, "sample": sample_hash(items),
            "items": items, "cases": {a: c for a, c in cases.items() if a in used}, "passages": passages,
            "fields": adj.SHEET_FIELDS}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (_PAGE.replace("__TITLE__", html.escape(f"{title} — {name}"))
                 .replace("__DATA__", payload))


def build_consensus_page(items, cases, passages, prior, name="Parsa", title="Settling disagreements"):
    """The page for a third rater who settles a pair's disagreements (LABELLING_GUIDE "Consensus"): each item shows both
    labels and notes, and the page fills `consensus` and `note_consensus`. prior: {sample_id: [{name, letter, label,
    note}, ...]}. Blind like the labellers' pages: no model, run or verifier verdict."""
    used = {i["case_alias"] for i in items}
    shown = [{**i, "prior": prior[i["sample_id"]]} for i in items]
    data = {"labeller": "consensus", "name": name, "partner": "", "half": "both", "sample": sample_hash(items),
            "items": shown, "cases": {a: c for a, c in cases.items() if a in used}, "passages": passages,
            "fields": adj.SHEET_FIELDS}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (_PAGE.replace("__TITLE__", html.escape(f"{title} — {name}"))
                 .replace("__DATA__", payload))


def merge_consensus(combined_sheet, consensus_sheet, out_path):
    """The combined sheet with `consensus` and `note_consensus` from the third rater's download. Refuses a consensus
    row whose sample is not in the combined sheet, or one given where the pair already agreed."""
    def rows(path):
        with Path(path).open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    base, extra = rows(combined_sheet), {r["sample_id"]: r for r in rows(consensus_sheet) if r.get("consensus")}
    ids = {r["sample_id"] for r in base}
    if set(extra) - ids:
        raise ValueError(f"Consensus for samples not in the sheet: {sorted(set(extra) - ids)[:3]}.")
    for r in base:
        if r["sample_id"] in extra:
            if r["label_A"] == r["label_B"]:
                raise ValueError(f"{r['sample_id']}: the pair agreed; it needs no consensus.")
            r["consensus"], r["note_consensus"] = extra[r["sample_id"]]["consensus"], extra[r["sample_id"]].get("note_consensus", "")
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=adj.SHEET_FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in adj.SHEET_FIELDS} for r in base)
    return Path(out_path)


def merge_sheets(sheet_a, sheet_b, out_path):
    """One sheet with label_A/note_A from labeller A's download and label_B/note_B from B's; refuses two sheets of
    different samples. Returns out_path, ready for adjudicate.read_labels."""
    def rows(path):
        with Path(path).open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    a, b = rows(sheet_a), rows(sheet_b)
    if [r["sample_id"] for r in a] != [r["sample_id"] for r in b] or [r["claim"] for r in a] != [r["claim"] for r in b]:
        raise ValueError("The two sheets are not the same sample.")
    merged = []
    for ra, rb in zip(a, b):
        merged.append({**ra, "label_B": rb.get("label_B", ""), "note_B": rb.get("note_B", ""),
                       "label_A": ra.get("label_A", ""), "note_A": ra.get("note_A", "")})
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=adj.SHEET_FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in adj.SHEET_FIELDS} for r in merged)
    return Path(out_path)


def combine_sheets(pair_sheets, out_path):
    """The merged sheets of the two pairs as one sheet for adjudicate.read_labels, in sample order; refuses a sample
    ID that appears twice."""
    rows = []
    for path in pair_sheets:
        with Path(path).open(encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("A sample ID appears in both pairs' sheets.")
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=adj.SHEET_FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in adj.SHEET_FIELDS} for r in sorted(rows, key=lambda r: r["sample_id"]))
    return Path(out_path)


def kappa_by_pair(rows, halves):
    """{pair: {"labellers", "n", "kappa"}} over the rows both of the pair labelled (rows from adjudicate.read_labels)."""
    out = {}
    for half, names in PAIRS.items():
        both = [r for r in rows if halves.get(r["sample_id"]) == half and r["label_A"] and r["label_B"]]
        out[half] = {"labellers": list(names), "n": len(both),
                     "kappa": adj.cohen_kappa([r["label_A"] for r in both], [r["label_B"] for r in both]) if both else None}
    return out


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1d232b; --muted: #5b6573; --line: #dfe3e8; --accent: #2f5fd0;
  --err: #b42318; --err-bg: #fdecea; --ok: #1f7a4a; --ok-bg: #e7f5ec; --mark: #fff1b8; --chip: #eef1f5;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171c; --panel: #1c2027; --ink: #e6e9ee; --muted: #9aa4b2; --line: #2d333d; --accent: #7aa2ff;
          --err: #ff8a80; --err-bg: #3a1f1d; --ok: #7ad49c; --ok-bg: #1a3326; --mark: #5a4a12; --chip: #262b34; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--panel); border-bottom: 1px solid var(--line);
         display: flex; align-items: center; gap: 10px; padding: 8px 16px; flex-wrap: wrap; }
header button { padding: 5px 10px; font-size: 14px; }
header h1 { font-size: 16px; margin: 0; font-weight: 600; }
.who { color: var(--muted); font-size: 14px; }
.bar { flex: 1; min-width: 160px; height: 8px; background: var(--chip); border-radius: 4px; overflow: hidden; }
.bar > div { height: 100%; background: var(--accent); width: 0; transition: width .2s; }
.count { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 14px; }
button { font: inherit; border: 1px solid var(--line); background: var(--panel); color: var(--ink); border-radius: 8px;
         padding: 7px 12px; cursor: pointer; }
button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
main { display: grid; grid-template-columns: 220px 1fr; gap: 0; min-height: calc(100vh - 56px); }
nav { border-right: 1px solid var(--line); background: var(--panel); overflow-y: auto; max-height: calc(100vh - 56px);
      position: sticky; top: 56px; padding: 8px; }
nav .filter { display: flex; gap: 6px; margin: 4px 4px 10px; font-size: 13px; }
nav .filter button { padding: 3px 8px; font-size: 13px; }
nav .filter button.on { background: var(--chip); border-color: var(--accent); }
nav a { display: flex; align-items: center; gap: 8px; padding: 5px 8px; border-radius: 6px; color: var(--ink);
        text-decoration: none; font-size: 14px; font-variant-numeric: tabular-nums; }
nav a.cur { background: var(--chip); font-weight: 600; }
.dot { width: 10px; height: 10px; border-radius: 50%; border: 1.5px solid var(--muted); flex: none; }
.dot.error { background: var(--err); border-color: var(--err); }
.dot.no_error { background: var(--ok); border-color: var(--ok); }
section.work { padding: 20px 28px 60px; max-width: 1180px; }
.top { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 6px; }
.sid { font-weight: 700; font-size: 18px; }
.chip { background: var(--chip); border-radius: 999px; padding: 2px 10px; font-size: 13px; color: var(--muted); }
.claim { font-size: 21px; line-height: 1.4; margin: 10px 0 6px; font-weight: 500; }
.context { color: var(--muted); font-size: 14px; margin-bottom: 14px; }
.cols { display: grid; grid-template-columns: 1.1fr 1fr; gap: 18px; }
@media (max-width: 1000px) { .cols { grid-template-columns: 1fr; } main { grid-template-columns: 1fr; } nav { display: none; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }
.card h3 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); font-weight: 600; }
.src { border-left: 3px solid var(--line); padding: 4px 10px; margin: 8px 0; }
.src.passage { border-color: var(--accent); }
.src .kind { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.src .pid { font-family: ui-monospace, Menlo, monospace; font-size: 13px; }
.missing { color: var(--err); }
table.vis { border-collapse: collapse; width: 100%; font-size: 14px; }
table.vis td { padding: 3px 6px; border-bottom: 1px solid var(--line); vertical-align: top; }
table.vis td:first-child { font-family: ui-monospace, Menlo, monospace; color: var(--muted); width: 45%; }
table.vis tr.hit td { background: var(--mark); }
.note { white-space: pre-wrap; font-size: 14px; }
mark { background: var(--mark); color: inherit; padding: 0 1px; border-radius: 3px; }
.decide { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; margin: 0 0 16px; }
.choices { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
.choice { flex: 1; min-width: 200px; padding: 12px; border-radius: 10px; font-size: 16px; font-weight: 600; }
.choice.err.on { background: var(--err-bg); border-color: var(--err); color: var(--err); }
.choice.ok.on { background: var(--ok-bg); border-color: var(--ok); color: var(--ok); }
.choice small { display: block; font-weight: 400; color: var(--muted); font-size: 13px; }
textarea { width: 100%; min-height: 38px; font: inherit; font-size: 14px; color: var(--ink); background: var(--bg);
           border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
.navbtns { display: flex; justify-content: space-between; margin-top: 10px; gap: 8px; flex-wrap: wrap; }
.hint { color: var(--muted); font-size: 13px; }
.banner.warn { display: block; border-color: var(--err); background: var(--err-bg); }
.banner { background: var(--chip); border: 1px solid var(--accent); border-radius: 10px; padding: 10px 14px; margin-bottom: 14px; display: none; }
dialog { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); color: var(--ink); max-width: 860px; width: 92vw; padding: 0; }
dialog .inner { padding: 18px 22px; max-height: 80vh; overflow-y: auto; }
dialog h2 { margin-top: 0; }
.pass { padding: 8px 0; border-bottom: 1px solid var(--line); font-size: 14px; }
.pass b { font-family: ui-monospace, Menlo, monospace; font-size: 13px; }
input[type=search] { width: 100%; padding: 8px; font: inherit; border: 1px solid var(--line); border-radius: 8px; background: var(--bg); color: var(--ink); margin-bottom: 8px; }
kbd { border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px; padding: 0 5px; font-size: 12px; background: var(--chip); }
</style>
</head>
<body>
<header>
  <h1>Verifier labelling</h1>
  <span class="who" id="who"></span>
  <div class="bar"><div id="bar"></div></div>
  <span class="count" id="count"></span>
  <button id="btnRules">Rules</button>
  <button id="btnBook">Rulebook</button>
  <button id="btnLoad">Load labels…</button>
  <button class="primary" id="btnSave">Download my labels</button>
  <input type="file" id="file" accept=".csv" hidden>
</header>
<main>
  <nav>
    <div class="filter"><button data-f="all" class="on">All</button><button data-f="todo">To do</button><button data-f="done">Done</button></div>
    <div id="list"></div>
  </nav>
  <section class="work">
    <div class="banner warn" id="storeBanner">This browser is not saving your labels (private window, or storage switched off). Open the file in a normal Chrome, Edge, Firefox or Safari window, or press <b>Download my labels</b> before you close this tab.</div>
    <div class="banner" id="breakBanner">You have labelled 10 claims in a row. Take a short break before the next ones: the guide asks for one every 10 rows, because tiredness shows up as disagreement. <button id="btnBreakOk">OK</button></div>
    <div class="top"><span class="sid" id="sid"></span><span class="chip" id="field"></span><span class="chip" id="alias"></span></div>
    <div class="claim" id="claim"></div>
    <div class="context" id="context"></div>
    <div class="context" id="prior"></div>
    <div class="decide">
      <div class="choices">
        <button class="choice err" id="btnErr">Error <small>content false, or its citation wrong or missing · <kbd>E</kbd></small></button>
        <button class="choice ok" id="btnOk">No error <small>true and correctly cited, or only restates the decision · <kbd>N</kbd></small></button>
      </div>
      <textarea id="noteBox" placeholder="Note (optional; write why when you are unsure). Saved as you type."></textarea>
      <div class="navbtns">
        <button id="btnPrev">← Previous</button>
        <span class="hint"><kbd>E</kbd> error · <kbd>N</kbd> no error · <kbd>←</kbd> <kbd>→</kbd> move · <kbd>U</kbd> next to do · saved automatically in this browser</span>
        <button id="btnNext">Next →</button>
      </div>
    </div>
    <div class="cols">
      <div>
        <div class="card"><h3>Sources the model declared</h3><div id="sources"></div></div>
      </div>
      <div>
        <div class="card"><h3>Imaging findings (as the model received them)</h3><table class="vis" id="vision"></table></div>
        <div class="card"><h3>Triage note</h3><div class="note" id="note"></div></div>
      </div>
    </div>
  </section>
</main>

<dialog id="rules"><div class="inner">
  <h2>How to label (docs/LABELLING_GUIDE.md)</h2>
  <p>You are checking the verifier, not being tested. Give each claim your own reading; do not discuss it with your partner until you have both finished.</p>
  <p><b>Error</b>: the claim's content is false given the inputs and the rulebook (a wrong number, an invented finding, a rule stated wrongly or applied where its conditions do not hold), <i>or</i> the content is true but its citation is wrong or missing (a rule with no rulebook passage, or a declared source that does not support it).</p>
  <p><b>No error</b>: the claim is true and correctly cited, or it only restates the answer's own decision ("thrombolysis is not recommended").</p>
  <p>For each claim ask: (1) is the content true given the imaging findings, the note and the rulebook? (2) does the source the model cited support it? Error if either answer is no.</p>
  <ul>
    <li>Correct medicine is not enough: a true clinical rule that no rulebook passage states is an error.</li>
    <li><b>null means unknown.</b> Saying a null finding is unknown or not reported is fine; any value for it is an error.</li>
    <li><b>Numbers</b> are compared after rounding the true value to the claim's precision: "9 h" matches 9 h 0 min; "4 h" does not match 4 h 40 min.</li>
    <li><b>Uncertainties</b> have no sources in the answer's format: judge only whether their content is true.</li>
    <li><b>Blocking factors</b> and decision citations have no sources of their own; they may rest on the passages either decision cites.</li>
    <li>Whether an error would change a decision is measured separately; it does not decide the label.</li>
  </ul>
  <p>When unsure, choose the label these rules give and write why in the note. Take a break every 10 claims.</p>
  <button onclick="this.closest('dialog').close()">Close</button>
</div></dialog>
<dialog id="book"><div class="inner">
  <h2>Rulebook</h2>
  <input type="search" id="bookSearch" placeholder="Search passages (ID or words)">
  <div id="bookList"></div>
  <button onclick="this.closest('dialog').close()">Close</button>
</div></dialog>

<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const P = Object.fromEntries(D.passages.map(p => [p.id, p.text]));
  const col = D.labeller === 'consensus' ? ['consensus', 'note_consensus'] : D.labeller === 'A' ? ['label_A', 'note_A'] : ['label_B', 'note_B'];
  const storeKey = 'strokecopilot-labels-' + D.name + '-' + D.sample;
  let S = {}, canStore = true;
  try { localStorage.setItem(storeKey + '-probe', '1'); localStorage.removeItem(storeKey + '-probe');
        S = JSON.parse(localStorage.getItem(storeKey) || '{}'); } catch (e) { S = {}; canStore = false; }
  let cur = 0, filter = 'all', streak = 0;
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  function save() { try { localStorage.setItem(storeKey, JSON.stringify(S)); } catch (e) {} }
  function done() { return D.items.filter(i => S[i.sample_id] && S[i.sample_id].label).length; }

  $('who').textContent = D.labeller === 'consensus' ? D.name + ' · third rater · settling ' + D.items.length + ' disagreements'
    : D.name + ' · pair ' + D.half + ' with ' + D.partner + ' · labeller ' + D.labeller;
  if (canStore) $('storeBanner').remove();

  function renderList() {
    const list = $('list'); list.innerHTML = '';
    D.items.forEach((it, i) => {
      const lab = (S[it.sample_id] || {}).label || '';
      if (filter === 'todo' && lab) return;
      if (filter === 'done' && !lab) return;
      const a = document.createElement('a');
      a.href = '#'; a.className = i === cur ? 'cur' : '';
      a.innerHTML = '<span class="dot ' + lab + '"></span>' + esc(it.sample_id) + ' <span class="hint">' + esc(it.case_alias) + '</span>';
      a.onclick = e => { e.preventDefault(); go(i); };
      list.appendChild(a);
    });
    const n = done();
    $('count').textContent = n + ' / ' + D.items.length + ' labelled';
    $('bar').style.width = (100 * n / D.items.length) + '%';
  }

  function sourceHtml(s, it) {
    const c = D.cases[it.case_alias];
    if (s.type === 'vision') {
      const v = s.field.split('.').reduce((o, k) => (o == null ? undefined : o[k]), c.vision);
      return '<div class="src"><div class="kind">Imaging field</div><span class="pid">' + esc(s.field) + '</span> = <b>' + esc(JSON.stringify(v === undefined ? null : v)) + '</b></div>';
    }
    if (s.type === 'note') {
      const found = c.note.indexOf(s.quote) >= 0;
      return '<div class="src"><div class="kind">Note quote' + (found ? '' : ' — <span class="missing">not found word for word in the note</span>') + '</div>“' + esc(s.quote) + '”</div>';
    }
    if (s.type === 'passage') {
      const t = P[s.id];
      return '<div class="src passage"><div class="kind">Rulebook passage</div><span class="pid">' + esc(s.id) + '</span>' +
        (t ? '<div>' + esc(t) + '</div>' : '<div class="missing">No passage with this ID exists in the rulebook.</div>') + '</div>';
    }
    return '<div class="src">' + esc(JSON.stringify(s)) + '</div>';
  }

  function go(i) {
    cur = Math.max(0, Math.min(D.items.length - 1, i));
    const it = D.items[cur], c = D.cases[it.case_alias], st = S[it.sample_id] || {};
    $('sid').textContent = it.sample_id + ' · ' + (cur + 1) + ' of ' + D.items.length;
    $('field').textContent = it.field;
    $('alias').textContent = 'case ' + it.case_alias;
    $('claim').textContent = it.claim;
    $('context').innerHTML = it.context ? 'From the answer: <i>' + esc(it.context) + '</i>' : '';
    $('prior').innerHTML = it.prior ? '<b>The two labels:</b> ' + it.prior.map(p => esc(p.name) + ' (' + esc(p.letter) + '): <b>' +
      (p.label === 'error' ? 'Error' : 'No error') + '</b>' + (p.note ? ' — <i>' + esc(p.note) + '</i>' : '')).join('<br>') : '';
    let srcs = it.sources.map(s => sourceHtml(s, it)).join('');
    const cited = (it.claim.match(/[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9.]+)+/g) || []).filter(id => P[id] && !it.sources.some(s => s.id === id));
    if (cited.length) srcs += '<div class="hint" style="margin-top:8px">Passages named in the claim:</div>' + cited.map(id => sourceHtml({type: 'passage', id}, it)).join('');
    if (!it.sources.length) srcs = '<div class="hint">None declared. ' + (it.field === 'uncertainties' ? 'Uncertainties have no sources: judge only whether the content is true.' :
      'Blocking factors and decision citations have no sources of their own: they may rest on the passages either decision cites.') + '</div>' + srcs;
    $('sources').innerHTML = srcs;
    const hits = new Set(it.sources.filter(s => s.type === 'vision').map(s => s.field));
    $('vision').innerHTML = Object.entries(flatten(c.vision)).map(([k, v]) =>
      '<tr class="' + (hits.has(k) ? 'hit' : '') + '"><td>' + esc(k) + '</td><td>' + esc(JSON.stringify(v)) + '</td></tr>').join('');
    let note = esc(c.note);
    it.sources.filter(s => s.type === 'note' && c.note.indexOf(s.quote) >= 0).forEach(s => {
      const q = esc(s.quote); note = note.split(q).join('<mark>' + q + '</mark>');
    });
    $('note').innerHTML = note;
    $('btnErr').classList.toggle('on', st.label === 'error');
    $('btnOk').classList.toggle('on', st.label === 'no_error');
    $('noteBox').value = st.note || '';
    renderList();
    window.scrollTo(0, 0);
  }

  function flatten(o, pre) {
    const out = {};
    Object.entries(o).forEach(([k, v]) => {
      const key = pre ? pre + '.' + k : k;
      if (v && typeof v === 'object' && !Array.isArray(v)) Object.assign(out, flatten(v, key)); else out[key] = v;
    });
    return out;
  }

  function label(value) {
    const id = D.items[cur].sample_id;
    const had = (S[id] || {}).label;
    S[id] = Object.assign({}, S[id], {label: value, at: new Date().toISOString()});
    save();
    if (!had) { streak += 1; if (streak >= 10) { $('breakBanner').style.display = 'block'; streak = 0; } }
    go(cur);  // no automatic move: the labeller may still want to write a note; → or U moves on
  }
  function nextTodo() {
    for (let k = 1; k <= D.items.length; k++) {
      const i = (cur + k) % D.items.length;
      if (!(S[D.items[i].sample_id] || {}).label) { go(i); return; }
    }
  }

  $('btnErr').onclick = () => label('error');
  $('btnOk').onclick = () => label('no_error');
  $('btnPrev').onclick = () => go(cur - 1);
  $('btnNext').onclick = () => go(cur + 1);
  $('btnBreakOk').onclick = () => { $('breakBanner').style.display = 'none'; };
  $('noteBox').oninput = e => { const id = D.items[cur].sample_id; S[id] = Object.assign({}, S[id], {note: e.target.value}); save(); };
  document.querySelectorAll('nav .filter button').forEach(b => b.onclick = () => {
    filter = b.dataset.f; document.querySelectorAll('nav .filter button').forEach(x => x.classList.toggle('on', x === b)); renderList();
  });
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT' || document.querySelector('dialog[open]')) return;
    if (e.key === 'e' || e.key === 'E') label('error');
    else if (e.key === 'n' || e.key === 'N') label('no_error');
    else if (e.key === 'ArrowRight') go(cur + 1);
    else if (e.key === 'ArrowLeft') go(cur - 1);
    else if (e.key === 'u' || e.key === 'U') nextTodo();
  });

  function csvCell(v) { v = v == null ? '' : String(v); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
  $('btnSave').onclick = () => {
    const rows = [D.fields.join(',')];
    D.items.forEach(it => {
      const st = S[it.sample_id] || {};
      const r = {sample_id: it.sample_id, case_alias: it.case_alias, field: it.field, claim: it.claim,
                 declared_sources: it.sources.map(s => s.field || s.id || (s.quote || '').slice(0, 40)).join('; ') || '(none declared)'};
      r[col[0]] = st.label || ''; r[col[1]] = st.note || '';
      rows.push(D.fields.map(f => csvCell(r[f])).join(','));
    });
    const blob = new Blob([rows.join('\r\n') + '\r\n'], {type: 'text/csv'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'labels_' + D.name.toLowerCase() + '_' + new Date().toISOString().slice(0, 10) + '.csv';
    a.click();
  };

  function parseCsv(text) {
    const rows = []; let row = [], cell = '', q = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (q) { if (ch === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += ch; }
      else if (ch === '"') q = true;
      else if (ch === ',') { row.push(cell); cell = ''; }
      else if (ch === '\n' || ch === '\r') { if (ch === '\r' && text[i + 1] === '\n') i++; row.push(cell); rows.push(row); row = []; cell = ''; }
      else cell += ch;
    }
    if (cell || row.length) { row.push(cell); rows.push(row); }
    return rows;
  }
  $('btnLoad').onclick = () => $('file').click();
  $('file').onchange = e => {
    const f = e.target.files[0]; if (!f) return;
    f.text().then(text => {
      const rows = parseCsv(text), head = rows.shift();
      const ix = Object.fromEntries(head.map((h, i) => [h, i]));
      if (!(col[0] in ix)) { alert('This file has no ' + col[0] + ' column: it is not a labels file for labeller ' + D.labeller + '.'); return; }
      const ids = new Set(D.items.map(i => i.sample_id));
      let n = 0, other = 0;
      rows.forEach(r => {
        const id = r[ix.sample_id]; if (!id) return;
        if (!ids.has(id)) { other++; return; }  // a claim outside this half: ignored
        const lab = r[ix[col[0]]], note = r[ix[col[1]]] || '';
        if (lab === 'error' || lab === 'no_error' || note) { S[id] = {label: lab || undefined, note}; n++; }
      });
      if (!n) { alert('This file holds no labels for ' + D.name + '. Is it someone else\'s file?'); return; }
      save(); go(cur); alert('Loaded ' + n + ' labels.' + (other ? ' ' + other + ' rows for claims outside your half were ignored.' : ''));
    });
    e.target.value = '';
  };

  $('btnRules').onclick = () => $('rules').showModal();
  $('btnBook').onclick = () => { renderBook(''); $('book').showModal(); setTimeout(() => $('bookSearch').focus(), 50); };
  $('bookSearch').oninput = e => renderBook(e.target.value);
  function renderBook(q) {
    q = q.toLowerCase();
    $('bookList').innerHTML = D.passages.filter(p => !q || p.id.toLowerCase().includes(q) || p.text.toLowerCase().includes(q))
      .map(p => '<div class="pass"><b>' + esc(p.id) + '</b><div>' + esc(p.text) + '</div></div>').join('') || '<p class="hint">No passage matches.</p>';
  }
  window.addEventListener('beforeunload', save);
  const first = D.items.findIndex(i => !(S[i.sample_id] || {}).label);
  go(first < 0 ? 0 : first);
})();
</script>
</body>
</html>
"""
