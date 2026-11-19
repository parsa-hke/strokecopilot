"""DEC-31 D3: the Phase 4 analyses, defined before any held-out case is opened, and tested on development cases.

"Truth" is `rules.decide` on the ground-truth contract and the case's fact sheet (plan §9). An undetermined truth is
excluded and counted, never guessed. So is a missing answer (a format failure), as in C9.

- `correctness`: per model and vision arm, decisions right / decisions scored, per decision and pooled, with a case
  bootstrap interval. With 20 cases the intervals are wide.
- `c9_side_by_side`: C9's metrics (metrics.summarise, points 1-20) for one set of cells next to another, unpaired
  (held-out against EXP-37's 20 development real cases): §9's realism check.
- `error_propagation`: for each wrong decision in the predicted arm, with answer A and R(v) = rules.decide on contract v:
  a vision error if R(predicted) != truth; a reasoning error if R(predicted) = truth but A != truth. Separately, the
  verifier caught it if a claim of that decision is flagged. The schema ties `rule_ids` and `blocking_factors` to one
  decision, while rationale and uncertainty claims are shared by both. So "a claim of that decision" counts the
  decision's own fields and the shared ones (`caught`), and the decision's own fields alone are reported too
  (`caught_own_fields`). For each vision error, the predicted contract's fields that differ from ground truth.
- `rule_sensitivity`: core_ml x0.8 and x1.2 on each predicted contract (derived fields and the near-empty warning
  recomputed, schema-validated); the share of rule decisions that change. `perturbed_cases` builds the 40 cases for
  Sonnet's own answers, and `model_sensitivity` compares those answers with the unperturbed ones.
- `vision_summary`: ensemble Dice, absolute volume error, wrong side of 21/31/51/70 mL against the reference final
  infarct, and the bleed flags among presumed bleed-free scans.

Case IDs never reach a returned summary: examples carry the alias the caller gives (for example H-01), and `check_no_ids`
is run before anything is saved.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from strokecopilot.contract.validate import validate_vision_output
from strokecopilot.nlp.notes.facts import FactSheet
from strokecopilot.nlp.verifier import metrics as vm
from strokecopilot.nlp.verifier import rules
from strokecopilot.pipeline import compare
from strokecopilot.vision.haemorrhage.stats import wilson
from strokecopilot.vision.perfusion import confidence as conf
from strokecopilot.vision.perfusion import volumes
from strokecopilot.vision.perfusion.evaluate import DECISION_CUTOFFS_ML

DECISIONS = ("thrombolysis", "thrombectomy")
WORD = {rules.YES: True, rules.NO: False, rules.UNDETERMINED: None}
FACTORS = (0.8, 1.2)
N_BOOT = vm.N_BOOT
SEED = vm.SEED
SHARED_FIELDS = ("rationale", "uncertainties")


# ------------------------------------------------------------------------------------------- Inputs


def load_contracts(folder, ids):
    """{case: contract dict} for ids from a contract folder; refuses a file whose case_id differs from its name."""
    out = {}
    for case in ids:
        obj = json.loads((Path(folder) / f"{case}.json").read_text())
        if obj.get("case_id") != case:
            raise ValueError("A contract's case_id does not match its file name.")
        out[case] = obj
    return out


def load_facts(notes_dir, ids):
    """{case: FactSheet}, each checked against its note."""
    out = {}
    for case in ids:
        facts = FactSheet.from_json(Path(notes_dir) / f"{case}.facts.json")
        facts.check_against((Path(notes_dir) / f"{case}.txt").read_text())
        out[case] = facts
    return out


def decide(contract, facts):
    """{decision: True | False | None} from the rules in code."""
    return {d: WORD[v.value] for d, v in rules.decide(rules.case_from(contract, facts)).items()}


def load_cell(folder, metrics_name=None):
    """One run cell's answers and traces: {case: {"answer": {decision: bool} or None, "traces": [...] or None}}.
    The traces come from the cell's newest scoring (metrics_rescore.json and traces_rescore/ when present)."""
    folder = Path(folder)
    metrics_file = compare._metrics_file(folder, metrics_name)
    traces_dir = folder / compare._traces_dir_for(metrics_file.name)
    if not traces_dir.exists():
        traces_dir = folder / "traces"
    rows = json.loads(metrics_file.read_text()).get("per_case") or []
    out = {}
    for row in rows:
        case = row["case"]
        output = folder / "outputs" / f"{case}.json"
        answer = None
        if output.exists():
            rec = json.loads(output.read_text())
            answer = {d: rec[d]["recommend"] for d in DECISIONS}
        trace_file = traces_dir / f"{case}.json"
        out[case] = {"answer": answer, "traces": json.loads(trace_file.read_text())["traces"] if trace_file.exists() else None}
    return out


# ------------------------------------------------------------------------------------------- Correctness


def _bootstrap_ratio(num, den, n=N_BOOT, seed=SEED):
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) < vm.MIN_CASES_FOR_CI or den.sum() == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(num), size=(n, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    r = r[np.isfinite(r)]
    return [round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]


def correctness(cell, truth, n=N_BOOT, seed=SEED):
    """Decisions right against truth, per decision and pooled, with case-bootstrap intervals.
    cell: load_cell(); truth: {case: decide(ground truth)}. Undetermined truths and missing answers are counted."""
    cases = sorted(truth)
    out = {"cases": len(cases), "no_answer": sum(cell.get(c, {}).get("answer") is None for c in cases)}
    pooled_num, pooled_den = np.zeros(len(cases)), np.zeros(len(cases))
    for d in DECISIONS:
        num, den = np.zeros(len(cases)), np.zeros(len(cases))
        undetermined = 0
        for i, c in enumerate(cases):
            want, answer = truth[c][d], (cell.get(c) or {}).get("answer")
            if want is None:
                undetermined += 1
                continue
            if answer is None:
                continue
            den[i], num[i] = 1, float(answer[d] == want)
        pooled_num += num
        pooled_den += den
        out[d] = {"correct": int(num.sum()), "scored": int(den.sum()), "undetermined_truth": undetermined,
                  "rate": None if den.sum() == 0 else round(float(num.sum() / den.sum()), 4),
                  "ci95": _bootstrap_ratio(num, den, n, seed)}
    out["pooled"] = {"correct": int(pooled_num.sum()), "scored": int(pooled_den.sum()),
                     "rate": None if pooled_den.sum() == 0 else round(float(pooled_num.sum() / pooled_den.sum()), 4),
                     "ci95": _bootstrap_ratio(pooled_num, pooled_den, n, seed)}
    return out


# ------------------------------------------------------------------------------------------- C9 side by side


def c9_summary(folders, metrics_name=None):
    """C9's metrics for a pool of cells (compare.case_records, metrics.summarise): the headline rates with CIs."""
    records = [r for f in ([folders] if isinstance(folders, (str, Path)) else folders)
               for r in compare.case_records(f, metrics_name)]
    s = vm.summarise(records)
    keep = ("hallucination_rate", "content_hallucination_rate", "critical_hallucination_rate", "omission_rate",
            "source_attribution_accuracy", "recommendation_correctness", "format_failure_rate")
    return {"cases": s["cases"], "claims": s["claims"], **{k: {"value": s[k]["value"], "ci95": s[k]["ci95"],
                                                              "numerator": s[k]["numerator"], "denominator": s[k]["denominator"]}
                                                          for k in keep}}


def c9_side_by_side(heldout_folders, development_folders, metrics_name=None):
    """Held-out next to development, unpaired: different cases, so no difference with a paired interval is given."""
    return {"heldout": c9_summary(heldout_folders, metrics_name), "development": c9_summary(development_folders, metrics_name)}


# ------------------------------------------------------------------------------------------- Error propagation


def _crossed(a, b, cutoffs=DECISION_CUTOFFS_ML):
    """The cut-offs that lie between two volumes (None counts as unknown: no crossing is claimed)."""
    if a is None or b is None:
        return []
    return [c for c in cutoffs if (a < c) != (b < c)]


def contract_differences(predicted, truth):
    """What the predicted contract gets wrong against the ground-truth one, in the fields D3 names."""
    out = []
    crossed = _crossed(predicted.get("core_ml"), truth.get("core_ml"))
    if crossed:
        out.append("core across " + "/".join(f"{c:g}" for c in crossed) + " mL")
    if predicted["haemorrhage"]["present"] != truth["haemorrhage"]["present"]:
        out.append("bleed flag")
    if predicted.get("occlusion_site") != truth.get("occlusion_site"):
        out.append("occlusion site")
    return out


def _flagged(traces, decision):
    """(any flagged claim of the decision's own fields or the shared fields, any flagged claim of its own fields)."""
    own = {f"{decision}.rule_ids", f"{decision}.blocking_factors"}
    flagged = [t["field"] for t in (traces or []) if t.get("subtype") is not None]
    return any(f in own or f in SHARED_FIELDS for f in flagged), any(f in own for f in flagged)


def error_propagation(cell, predicted, truth_contracts, facts, aliases):
    """The D3 table for one model's predicted-arm cell. predicted / truth_contracts: {case: contract};
    facts: {case: FactSheet}; aliases: {case: alias}. Returns counts and examples by alias only."""
    rows, skipped = [], Counter()
    for case in sorted(truth_contracts):
        truth = decide(truth_contracts[case], facts[case])
        on_predicted = decide(predicted[case], facts[case])
        entry = cell.get(case) or {}
        answer = entry.get("answer")
        for d in DECISIONS:
            if truth[d] is None:
                skipped["undetermined truth"] += 1
                continue
            if answer is None:
                skipped["no answer"] += 1
                continue
            if answer[d] == truth[d]:
                continue
            caught, caught_own = _flagged(entry.get("traces"), d)
            vision = on_predicted[d] != truth[d]
            rows.append({"alias": aliases[case], "decision": d, "answer": answer[d], "truth": truth[d],
                         "rules_on_predicted": on_predicted[d], "cause": "vision" if vision else "reasoning",
                         "caught": caught, "caught_own_fields": caught_own,
                         "contract_differences": contract_differences(predicted[case], truth_contracts[case]) if vision else []})
    table = {cause: {"wrong": sum(r["cause"] == cause for r in rows),
                     "caught": sum(r["cause"] == cause and r["caught"] for r in rows),
                     "missed": sum(r["cause"] == cause and not r["caught"] for r in rows),
                     "caught_own_fields": sum(r["cause"] == cause and r["caught_own_fields"] for r in rows)}
             for cause in ("vision", "reasoning")}
    fields = Counter(f.split(" across")[0] if f.startswith("core") else f for r in rows for f in r["contract_differences"])
    return {"wrong_decisions": len(rows), "table": table, "vision_error_fields": dict(fields),
            "rules_undetermined_on_predicted": sum(r["rules_on_predicted"] is None for r in rows),
            "not_scored": dict(skipped), "examples": rows}


# ------------------------------------------------------------------------------------------- Sensitivity


def perturb(contract, factor):
    """A copy of a contract with core_ml scaled by factor (rounded to one decimal, DEC-14's mismatch rederived) and
    the near-empty warning recomputed (DEC-27). Penumbra, bleed, site and confidence are unchanged. Validated."""
    out = json.loads(json.dumps(contract))
    core = out.get("core_ml")
    if core is not None:
        out.update(volumes.contract_volumes(core * factor, out.get("penumbra_ml")))
        others = [w for w in out.get("warnings", []) if w != conf.NEAR_EMPTY_WARNING]
        out["warnings"] = others + conf.core_warnings(out["core_ml"])
    validate_vision_output(out)
    return out


def rule_sensitivity(predicted, facts, factors=FACTORS):
    """Per factor: how many rule decisions change when core_ml is scaled, among all decisions and among those
    determined before; yes<->no flips apart from moves to or from undetermined. Counts only."""
    out = {}
    for f in factors:
        changed = flips = to_undetermined = from_undetermined = determined = total = 0
        for case, contract in predicted.items():
            before, after = decide(contract, facts[case]), decide(perturb(contract, f), facts[case])
            for d in DECISIONS:
                total += 1
                determined += before[d] is not None
                if before[d] != after[d]:
                    changed += 1
                    if before[d] is not None and after[d] is not None:
                        flips += 1
                    elif after[d] is None:
                        to_undetermined += 1
                    else:
                        from_undetermined += 1
        out[f"x{f:g}"] = {"decisions": total, "changed": changed, "changed_share": round(changed / total, 4) if total else None,
                          "yes_no_flips": flips, "to_undetermined": to_undetermined, "from_undetermined": from_undetermined,
                          "determined_before": determined,
                          "flip_share_of_determined": round(flips / determined, 4) if determined else None}
    out["cases"] = len(predicted)
    return out


def perturbed_cases(cases, factors=FACTORS):
    """The extra requests for Sonnet (D3: 40 for 20 cases): each case dict (end_to_end.real_case shape) with its
    contract perturbed. The id gets a suffix; `expected` is rederived from the perturbed contract, as for any case."""
    out = []
    for case in cases:
        for f in factors:
            vision = perturb(case["vision"], f)
            out.append({**case, "id": f"{case['id']}__core_x{f:g}", "vision": vision, "base_id": case["id"], "factor": f,
                        "expected": decide(vision, case["facts"])})
    return out


def model_sensitivity(base_cell, perturbed_cell, factors=FACTORS):
    """How often the model's own answer changes when only core_ml is scaled. perturbed_cell keys are
    '<case>__core_x<factor>'. Counts only."""
    out = {}
    for f in factors:
        changed = compared = 0
        for case, entry in base_cell.items():
            other = perturbed_cell.get(f"{case}__core_x{f:g}")
            if not other or entry.get("answer") is None or other.get("answer") is None:
                continue
            for d in DECISIONS:
                compared += 1
                changed += entry["answer"][d] != other["answer"][d]
        out[f"x{f:g}"] = {"decisions_compared": compared, "changed": changed,
                          "changed_share": round(changed / compared, 4) if compared else None}
    return out


# ------------------------------------------------------------------------------------------- Vision on held-out


def vision_summary(rows, flags=None, scans=None):
    """rows: per-case score dicts (evaluate.case_metrics: dice, pred_ml, ref_ml, abs_error_ml, crosses_<c>).
    flags/scans: bleed flags among presumed bleed-free scans. Pooled over cases (n is small; no fold structure)."""
    dice = np.array([r["dice"] for r in rows], float)
    err = np.array([r["abs_error_ml"] for r in rows], float)
    signed = np.array([r["pred_ml"] - r["ref_ml"] for r in rows], float)
    out = {"cases": len(rows),
           "dice": {"mean": round(float(dice.mean()), 4), "sd": round(float(dice.std(ddof=1)), 4) if len(rows) > 1 else None,
                    "median": round(float(np.median(dice)), 4)},
           "abs_volume_error_ml": {"mean": round(float(err.mean()), 2), "median": round(float(np.median(err)), 2)},
           "signed_volume_error_ml_median": round(float(np.median(signed)), 2),
           "wrong_side": {f"{c:g}": int(sum(bool(r[f"crosses_{c:g}"]) for r in rows)) for c in DECISION_CUTOFFS_ML}}
    if flags is not None and scans:
        lo, hi = wilson(flags, scans)
        out["bleed_flags"] = {"flagged": int(flags), "scans": int(scans), "rate": round(flags / scans, 4),
                              "wilson_ci95": [round(lo, 4), round(hi, 4)]}
    return out


# ------------------------------------------------------------------------------------------- Guard


def check_no_ids(obj, ids):
    """Raise if any case ID appears anywhere in the JSON text of obj."""
    text = json.dumps(obj, default=str)
    found = sum(1 for i in ids if i and i in text)
    if found:
        raise ValueError(f"{found} case IDs reached a summary. Nothing was saved.")
    return text


def aliases_for(ids, prefix):
    """{case: '<prefix>-01', ...} in sorted order."""
    return {case: f"{prefix}-{i:02d}" for i, case in enumerate(sorted(ids), 1)}


# ------------------------------------------------------------------------------------------- The whole D3 report


def report(cells, predicted_dir, truth_dir, notes_dir, ids, alias_prefix, development_cells=None, vision_rows=None,
           flags=None, scans=None, rule_contracts=None, sonnet_perturbed=None):
    """Every D3 analysis for one case set. cells: {model: {"predicted": folder, "groundtruth": folder or None}};
    development_cells: {model: folder} for C9 side by side (held-out run only); vision_rows / flags / scans for
    vision_summary; rule_contracts: {case: contract} for rule_sensitivity (default: the predicted contracts of ids);
    sonnet_perturbed: (base cell folder, perturbed cell folder). Returns a dict with counts and aliases only."""
    ids = sorted(ids)
    predicted, truths = load_contracts(predicted_dir, ids), load_contracts(truth_dir, ids)
    facts = load_facts(notes_dir, ids)
    truth = {c: decide(truths[c], facts[c]) for c in ids}
    aliases = aliases_for(ids, alias_prefix)
    out = {"cases": len(ids),
           "truth": {d: dict(Counter({True: "yes", False: "no", None: "undetermined"}[truth[c][d]] for c in ids))
                     for d in DECISIONS},
           "rules_on_predicted_correct": correctness({c: {"answer": decide(predicted[c], facts[c])} for c in ids}, truth),
           "models": {}}
    for model, arms in cells.items():
        entry = {}
        for arm, folder in arms.items():
            if folder is None:
                continue
            cell = {c: v for c, v in load_cell(folder).items() if c in truth}
            if set(cell) != set(ids):
                raise ValueError(f"{model} {arm}: {len(set(ids) - set(cell))} cases missing from the cell.")
            entry[arm] = {"correctness": correctness(cell, truth)}
            if arm == "predicted":
                entry[arm]["error_propagation"] = error_propagation(cell, predicted, truths, facts, aliases)
        if development_cells and model in development_cells and arms.get("predicted"):
            entry["c9"] = c9_side_by_side(arms["predicted"], development_cells[model])
        out["models"][model] = entry
    rule_set = rule_contracts if rule_contracts is not None else predicted
    rule_facts = facts if rule_contracts is None else load_facts(notes_dir, sorted(rule_contracts))
    out["rule_sensitivity"] = rule_sensitivity(rule_set, rule_facts)
    if sonnet_perturbed:
        out["sonnet_sensitivity"] = model_sensitivity(load_cell(sonnet_perturbed[0]), load_cell(sonnet_perturbed[1]))
    if vision_rows is not None:
        out["vision"] = vision_summary(vision_rows, flags, scans)
    check_no_ids(out, set(ids) | set(rule_set))
    return out


def markdown(rep, title):
    """A readable summary of report() for runs/ and the CHANGELOG. Aliases and counts only."""
    def pct(r):
        if r is None or r.get("rate") is None:
            return "n/a"
        ci = r.get("ci95")
        return f"{r['correct']}/{r['scored']} ({100 * r['rate']:.0f}%" + (f", {100 * ci[0]:.0f}–{100 * ci[1]:.0f}%)" if ci else ")")

    L = [f"# {title}", "", f"Cases: {rep['cases']}. Truth (rules on the ground-truth contract): "
         + "; ".join(f"{d} {v}" for d, v in rep["truth"].items()) + ".", "",
         f"Rules applied to the predicted contracts get {pct(rep['rules_on_predicted_correct']['pooled'])} of decisions right.", "",
         "## Recommendation correctness (pooled over both decisions; case-bootstrap 95% interval)", "",
         "| Model | Predicted vision | Ground-truth vision | Thrombolysis (pred.) | Thrombectomy (pred.) |", "|---|---|---|---|---|"]
    for m, e in rep["models"].items():
        p, g = e.get("predicted", {}).get("correctness"), e.get("groundtruth", {}).get("correctness")
        L.append(f"| {m} | {pct(p and p['pooled'])} | {pct(g and g['pooled'])} | {pct(p and p['thrombolysis'])} | "
                 f"{pct(p and p['thrombectomy'])} |")
    L += ["", "## Error propagation (predicted arm)", "",
          "| Model | Wrong | Vision (caught / missed) | Reasoning (caught / missed) | Caught by own fields (vision, reasoning) | Vision-error fields |",
          "|---|---|---|---|---|---|"]
    for m, e in rep["models"].items():
        ep = e.get("predicted", {}).get("error_propagation")
        if ep:
            v, r = ep["table"]["vision"], ep["table"]["reasoning"]
            L.append(f"| {m} | {ep['wrong_decisions']} | {v['wrong']} ({v['caught']} / {v['missed']}) | {r['wrong']} "
                     f"({r['caught']} / {r['missed']}) | {v['caught_own_fields']}, {r['caught_own_fields']} | "
                     + (", ".join(f"{k} {n}" for k, n in ep["vision_error_fields"].items()) or "none") + " |")
    rs = rep["rule_sensitivity"]
    L += ["", f"## Rule-level sensitivity to core_ml ({rs['cases']} predicted contracts)", ""]
    for k in ("x0.8", "x1.2"):
        r = rs[k]
        L.append(f"- core {k}: {r['changed']}/{r['decisions']} decisions change ({100 * r['changed_share']:.1f}%): "
                 f"{r['yes_no_flips']} yes/no flips, {r['to_undetermined']} to undetermined, {r['from_undetermined']} from it.")
    if "sonnet_sensitivity" in rep:
        for k, r in rep["sonnet_sensitivity"].items():
            L.append(f"- Sonnet's own answers, core {k}: {r['changed']}/{r['decisions_compared']} change.")
    if "vision" in rep:
        v = rep["vision"]
        L += ["", f"## Vision ({v['cases']} cases)", "",
              f"- Dice mean {v['dice']['mean']} (sd {v['dice']['sd']}, median {v['dice']['median']}); absolute volume "
              f"error mean {v['abs_volume_error_ml']['mean']} mL (median {v['abs_volume_error_ml']['median']}); signed median "
              f"{v['signed_volume_error_ml_median']} mL.",
              "- Wrong side of " + ", ".join(f"{c} mL: {n}" for c, n in v["wrong_side"].items()) + "."]
        if "bleed_flags" in v:
            b = v["bleed_flags"]
            L.append(f"- Bleed flags on presumed bleed-free scans: {b['flagged']}/{b['scans']} ({100 * b['rate']:.1f}%, "
                     f"Wilson {100 * b['wilson_ci95'][0]:.1f}–{100 * b['wilson_ci95'][1]:.1f}%).")
    if any("c9" in e for e in rep["models"].values()):
        L += ["", "## C9 metrics: held-out next to development (unpaired)", "", "| Model | Held-out rate | Development rate |", "|---|---|---|"]
        for m, e in rep["models"].items():
            if "c9" in e:
                h, d = e["c9"]["heldout"]["hallucination_rate"], e["c9"]["development"]["hallucination_rate"]
                fmt = lambda x: "n/a" if x["value"] is None else f"{100 * x['value']:.1f}%" + (
                    f" [{100 * x['ci95'][0]:.1f}, {100 * x['ci95'][1]:.1f}]" if x["ci95"] else "")
                L.append(f"| {m} | {fmt(h)} | {fmt(d)} |")
    return "\n".join(L) + "\n"


def scenario_sensitivity(factors=FACTORS):
    """DEC-31's supplementary analysis (amended 19 Nov 2026): rule_sensitivity on the 150 synthetic scenarios
    (scenarios.generate), where the core can decide a case. Also lists which key routes the changed decisions came
    from. Free; no real case."""
    from strokecopilot.nlp.verifier import scenarios as sc

    items = sc.generate()
    contracts = {s.id: s.vision for s in items}
    facts = {s.id: s.facts for s in items}
    out = rule_sensitivity(contracts, facts, factors)
    for f in factors:
        kinds = Counter()
        for s in items:
            before, after = decide(s.vision, s.facts), decide(perturb(s.vision, f), s.facts)
            for d in DECISIONS:
                if before[d] != after[d]:
                    word = lambda v: {True: "yes", False: "no", None: "undetermined"}[v]
                    kinds[f"{d}: {word(before[d])} -> {word(after[d])}"] += 1
        out[f"x{f:g}"]["changes"] = dict(sorted(kinds.items()))
    return out
