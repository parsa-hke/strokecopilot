"""Read-outs of saved NLP results for the MAAI902 report's second feedback round (22 Nov 2026). Report only, free,
development data and synthetic scenarios; nothing is re-run and no held-out case is read (DEC-31, DEC-34).

- **EXP-88, the human check read again (EXP-41's labels):** the human-consensus error rate weighted back to all
  23,245 claims by the sampling strata (model x verifier verdict), pooled and per model, with two intervals; how many
  of the verifier's flags people confirmed, by flag type; and the labellers' kappa by stratum.
- **EXP-89, report tables:** per model, the share of ALL cases that got a valid answer with no false-content claim
  (so a model cannot gain by failing to answer); the largest subgroup difference on the scenarios, with its interval.
- **EXP-90, Qwen's format failures by cause,** from the recorded error of each failed answer.

These were asked for after every result had been seen, so they are post hoc descriptions, not tests: no p-value is
computed and DEC-36's list of t-tests is untouched. No case ID is written; IDs stay in memory.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 10_000
SEED = 20261122
MODELS = ("claude-sonnet-5-5", "qwen3.5-27b", "gemma-3-27b", "medgemma-27b-text", "gemma-3-4b", "gemma-3-4b-tuned")
CELLS = ("fixtures_rulebook", "scenarios_rulebook", "real_predicted")  # the 175 cases of the headline table
CITATION_WORDS = ("cit", "source")  # a labeller's note that names a missing or wrong citation


# ------------------------------------------------------------------------------------------- EXP-88: the human check


def strata_counts(rows):
    """{(model, verdict): {"n", "errors", "size"}} from sample rows {model, verdict, stratum_size, consensus}."""
    out = defaultdict(lambda: {"n": 0, "errors": 0, "size": 0})
    for r in rows:
        s = out[(r["model"], r["verdict"])]
        s["n"] += 1
        s["errors"] += r["consensus"] == "error"
        s["size"] = int(r["stratum_size"])
    return dict(out)


def _weighted(strata, draw):
    """The stratified rate over `strata` values, with each stratum's error share given by draw(stratum)."""
    total = sum(s["size"] for s in strata)
    return sum(s["size"] * draw(s) for s in strata) / total


def stratified_rate(strata, n_boot=N_BOOT, seed=SEED):
    """Human error rate weighted by stratum size, with two 95% intervals.

    - `bootstrap`: labels resampled within each stratum. A stratum with no error among its sampled claims contributes
      no spread, so this interval is too narrow whenever a large stratum shows 0 of n.
    - `jeffreys`: each stratum's rate drawn from Beta(errors + 0.5, n - errors + 0.5), which keeps the uncertainty of a
      0-of-n stratum. Reported beside the bootstrap for that reason.
    """
    strata = list(strata)
    rng = np.random.default_rng(seed)
    point = _weighted(strata, lambda s: s["errors"] / s["n"])
    boot = _weighted(strata, lambda s: rng.binomial(s["n"], s["errors"] / s["n"], n_boot) / s["n"])
    jeff = _weighted(strata, lambda s: rng.beta(s["errors"] + 0.5, s["n"] - s["errors"] + 0.5, n_boot))
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"value": float(point), "bootstrap_ci95": pct(boot), "jeffreys_ci95": pct(jeff),
            "claims": sum(s["size"] for s in strata), "sampled": sum(s["n"] for s in strata),
            "_boot": boot, "_jeff": jeff}


def rate_difference(a, b):
    """a minus b for two stratified_rate results (independent strata), with both intervals."""
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"value": a["value"] - b["value"], "bootstrap_ci95": pct(a["_boot"] - b["_boot"]),
            "jeffreys_ci95": pct(a["_jeff"] - b["_jeff"])}


def _public(result):
    return {k: v for k, v in result.items() if not k.startswith("_")}


def kappa(rows):
    from strokecopilot.nlp.verifier.adjudicate import cohen_kappa
    return cohen_kappa([r["label_a"] for r in rows], [r["label_b"] for r in rows]) if rows else None


def human_check(rows, n_boot=N_BOOT, seed=SEED):
    """Every EXP-88 number. rows: one per sampled claim, {sample_id, model, verdict (the sampling stratum),
    stratum_size, label_a, label_b, consensus, note_a, note_b, rescored (the verifier's verdict after the rescore)}."""
    strata = strata_counts(rows)
    models = sorted({m for m, _ in strata})
    per_model = {m: stratified_rate([s for (mm, _), s in strata.items() if mm == m], n_boot, seed) for m in models}
    pooled = stratified_rate(strata.values(), n_boot, seed)
    out = {"strata": {f"{m} / {v}": s for (m, v), s in sorted(strata.items())},
           "human_error_rate": {"pooled": _public(pooled), **{m: _public(per_model[m]) for m in models}}}
    if "claude-sonnet-5-5" in per_model:
        out["minus_sonnet"] = {m: rate_difference(per_model[m], per_model["claude-sonnet-5-5"])
                               for m in models if m != "claude-sonnet-5-5"}
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["rescored"]][0] += r["consensus"] == "error"
        by[r["rescored"]][1] += 1
    out["by_rescored_verdict"] = {v: {"human_errors": k, "claims": n} for v, (k, n) in sorted(by.items())}
    disagree = [r for r in rows if r["label_a"] != r["label_b"]]
    names_citation = [r for r in disagree
                      if any(w in (r.get("note_a", "") + " " + r.get("note_b", "")).lower() for w in CITATION_WORDS)]
    out["kappa"] = {"all": kappa(rows), "claims": len(rows),
                    "by_stratum": {v: {"kappa": kappa([r for r in rows if r["verdict"] == v]),
                                       "claims": sum(r["verdict"] == v for r in rows),
                                       "disagreements": sum(r["verdict"] == v for r in disagree)}
                                   for v in sorted({r["verdict"] for r in rows})}}
    out["disagreements"] = {"n": len(disagree), "note_names_a_citation": len(names_citation),
                            "settled_as_error": sum(r["consensus"] == "error" for r in disagree)}
    return out


def rescored_verdict(key_row, root="."):
    """The verifier's verdict on a sampled claim after the rescore (metric points 19 and 20), read from its run
    folder's `flagged_rescore.json`: clean when the claim is not in it."""
    from strokecopilot.nlp.verifier.adjudicate import verdict_of
    folder = Path(root) / key_row["run_folder"]
    flagged = json.loads((folder / "flagged_rescore.json").read_text())
    flagged = flagged.get("flagged", flagged) if isinstance(flagged, dict) else flagged
    for f in flagged:
        if f["case"] == key_row["case"] and f["claim_id"] == key_row["claim_id"]:
            return verdict_of(f["subtype"])
    return "clean"


def load_human_rows(folder="runs/EXP-41_labelling", root="."):
    """EXP-41's key and final labels joined, one row per sampled claim. The key never leaves the Mac."""
    folder = Path(folder)
    key = {r["sample_id"]: r for r in csv.DictReader(open(folder / "key.csv"))}
    rows = []
    for r in csv.DictReader(open(folder / "labels" / "final.csv")):
        k = key[r["sample_id"]]
        rows.append({"sample_id": r["sample_id"], "model": k["model"].split("/")[-1], "verdict": k["verdict"],
                     "stratum_size": k["stratum_size"], "label_a": r["label_A"], "label_b": r["label_B"],
                     "consensus": r["consensus"] or r["label_A"], "note_a": r["note_A"], "note_b": r["note_B"],
                     "rescored": rescored_verdict(k, root)})
    return rows


# ------------------------------------------------------------------------------------------- EXP-89: report tables


def _boot_share(flags, n_boot=N_BOOT, seed=SEED):
    x = np.asarray(flags, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def answered_without_false_content(records, n_boot=N_BOOT, seed=SEED):
    """Share of ALL cases with a valid answer in which no claim is flagged for false content (sub-types 1 to 5).
    A format failure counts against the model, so the share cannot rise by failing to answer."""
    ok = [int(not r["format_failure"] and not any(t.get("content_false") for t in r["traces"])) for r in records]
    return {"value": float(np.mean(ok)), "numerator": int(sum(ok)), "denominator": len(ok),
            "valid_answers": sum(not r["format_failure"] for r in records), "ci95": _boot_share(ok, n_boot, seed)}


def _claims(record):
    return sum(t.get("subtype") is not None for t in record["traces"]), len(record["traces"])


def _pooled_boot(records, rng, n_boot):
    h = np.array([_claims(r)[0] for r in records], dtype=float)
    n = np.array([_claims(r)[1] for r in records], dtype=float)
    idx = rng.integers(0, len(records), size=(n_boot, len(records)))
    return h.sum() / n.sum(), h[idx].sum(axis=1) / n[idx].sum(axis=1)


def subgroup_differences(records_by_model, groups, min_cases=10, n_boot=N_BOOT, seed=SEED):
    """Every pairwise difference in pooled hallucination rate between two groups of a grouping, per model, with an
    unpaired 95% case-bootstrap interval; and the largest in absolute size. Format failures hold no claims and are
    left out. A group under `min_cases` cases is skipped."""
    rows = []
    for model, records in records_by_model.items():
        valid = [r for r in records if not r["format_failure"] and r["traces"]]
        for grouping in sorted({g for r in valid for g in groups[r["case_id"]]}):
            by = defaultdict(list)
            for r in valid:
                label = groups[r["case_id"]].get(grouping)
                if label is not None:
                    by[label].append(r)
            labels = sorted(lab for lab, recs in by.items() if len(recs) >= min_cases)
            for i, a in enumerate(labels):
                for b in labels[i + 1:]:
                    rng = np.random.default_rng(seed)
                    pa, ba = _pooled_boot(by[a], rng, n_boot)
                    pb, bb = _pooled_boot(by[b], rng, n_boot)
                    d = ba - bb
                    rows.append({"model": model, "grouping": grouping, "a": a, "b": b, "cases_a": len(by[a]),
                                 "cases_b": len(by[b]), "rate_a": float(pa), "rate_b": float(pb),
                                 "difference": float(pa - pb),
                                 "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]})
    largest = max(rows, key=lambda r: abs(r["difference"])) if rows else None
    excludes = [r for r in rows if r["ci95"][0] > 0 or r["ci95"][1] < 0]
    return {"comparisons": len(rows), "interval_excludes_zero": len(excludes), "largest": largest, "rows": rows}


# ------------------------------------------------------------------------------------------- EXP-90: format failures


def failure_cause(error):
    """The cause of one failed answer, from its recorded error text."""
    e = error or ""
    if "JSON list, not an object" in e:
        return "a list, not an object"
    if "returned no content" in e or "no valid JSON" in e:
        return "no answer returned"
    if "Additional properties are not allowed" in e and any(k in e for k in ("'$defs'", "'title'", "'description'")):
        return "the output schema echoed back"
    if "hit the" in e and "token" in e or "finish_reason 'length'" in e or "finish_reason length" in e:
        return "cut off at the output limit"
    if "schema violation" in e:
        return "another schema violation"
    return "other"


def format_failure_causes(per_case_rows):
    """Counts of failure causes over metrics rows {case, error?}. Verifier failures are not format failures."""
    errors = [r["error"] for r in per_case_rows if r.get("error") and not r["error"].startswith("verifier failed")]
    return {"cases": len(per_case_rows), "format_failures": len(errors), "causes": dict(Counter(map(failure_cause, errors)))}


# ------------------------------------------------------------------------------------------- Command line


def _write(out, name, result, ids):
    from strokecopilot.vision.haemorrhage import isles_check as ic
    out = Path(out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    text = json.dumps(result, indent=1, default=float)
    if ic.ids_in(text, ids):
        raise ValueError("A case ID reached the summary. Nothing was saved.")
    (out / name).write_text(text)
    return text


def _root_of(model):
    return Path("runs/EXP-47_small_reasoner" if model.startswith("gemma-3-4b") else "runs/EXP-37_paid_runs")


def main(argv=None):
    import argparse

    from strokecopilot.data import splits
    from strokecopilot.nlp.verifier import scenarios as sc
    from strokecopilot.pipeline import compare as cp

    ap = argparse.ArgumentParser(description="Read-outs for the 902 report's second feedback round")
    ap.add_argument("what", choices=("human", "tables", "failures"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    folds = splits.load_folds()
    ids = list(splits.development_ids(folds)) + list(folds["heldout"])  # for the ID check only, never printed

    if args.what == "human":
        result = {"experiment": "EXP-88", **human_check(load_human_rows())}
    elif args.what == "tables":
        records = {m: [r for cell in CELLS for r in cp.case_records(_root_of(m) / m / cell)] for m in MODELS}
        from strokecopilot.analysis.subgroups import groups_from_facts
        groups = {s.id: groups_from_facts(s.facts) for s in sc.generate()}
        scen = {m: cp.case_records(_root_of(m) / m / "scenarios_rulebook") for m in MODELS}
        sub = subgroup_differences(scen, groups)
        result = {"experiment": "EXP-89",
                  "answered_without_false_content": {m: answered_without_false_content(records[m]) for m in MODELS},
                  "subgroup_differences_scenarios": sub}
    else:
        root = Path("runs/EXP-37_paid_runs/qwen3.5-27b")
        per_cell = {cell: format_failure_causes(json.loads((root / cell / "metrics_rescore.json").read_text())["per_case"])
                    for cell in CELLS}
        total = Counter()
        for c in per_cell.values():
            total.update(c["causes"])
        result = {"experiment": "EXP-90", "model": "qwen3.5-27b", "per_cell": per_cell,
                  "all_175": {"cases": sum(c["cases"] for c in per_cell.values()),
                              "format_failures": sum(c["format_failures"] for c in per_cell.values()),
                              "causes": dict(total)}}
    print(_write(args.out, "metrics.json", result, ids))


if __name__ == "__main__":
    main()
