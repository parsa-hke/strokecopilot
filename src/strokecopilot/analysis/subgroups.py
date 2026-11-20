"""Phase 5 task E (DEC-37): subgroup analysis from saved results, and SHAP rows 12 and 13 (surrogates).

Groupings (fixed before any count): age band (under 80, 80 and over), sex, centre, and onset-information pattern
(known: an onset-to-imaging time; wake-up; none: no onset time, last known well at most). A value the note gives as not
recorded is its own group. A group with fewer than MIN_CASES cases is reported with its counts only and marked too
small to conclude anything.

- **Vision, 129 development cases:** Vision A (EXP-27: Dice, absolute volume error, core on the wrong side of any of
  21/31/51/70 mL) and Vision B (EXP-33's contracts: the bleed flag at DEC-28, all presumed false alarms).
- **Reasoners, saved verified answers (rulebook arms, predicted vision; EXP-37, EXP-47):** hallucination rate and
  recommendation correctness per model, on the 150 synthetic scenarios (age band, sex, onset pattern) and on the 20
  development real cases (all four groupings; every group small).
- **Rows 12 and 13:** a gradient-boosted SURROGATE of "core on the wrong side of a cut-off" (12) and of "bleed flag" (13)
  from case features known at presentation (age, sex, centre, onset pattern, NIHSS, Tmax > 6 s volume, predicted core);
  out-of-fold AUC first, SHAP (`TreeExplainer`) only if its interval excludes 0.5.

Intervals: Wilson for counts, 95% case bootstrap (10,000, seed 20261104) for means and claim rates. No case ID is
returned; IDs stay in memory.
"""

import csv
import json
from pathlib import Path

import numpy as np

MIN_CASES = 10
SEED = 20261120
GROUPINGS = ("age band", "sex", "centre", "onset")
MODELS = ("claude-sonnet-5-5", "qwen3.5-27b", "gemma-3-27b", "medgemma-27b-text", "gemma-3-4b", "gemma-3-4b-tuned")


def _val(facts, key):
    f = facts.get(key)
    return None if f is None or f.type == "not_recorded" else f.value


def groups_from_facts(facts, centre=None):
    """{grouping: label} for one case's fact sheet (and centre, for real cases)."""
    age = _val(facts, "age")
    sex = _val(facts, "sex")
    if _val(facts, "wake_up_stroke"):
        onset = "wake-up"
    elif _val(facts, "onset_to_imaging") is not None:
        onset = "known"
    else:
        onset = "none"
    out = {"age band": "not recorded" if age is None else ("80 and over" if float(age) >= 80 else "under 80"),
           "sex": "not recorded" if sex is None else str(sex).lower(), "onset": onset}
    if centre is not None:
        out["centre"] = f"centre {centre}"
    return out


def wilson(k, n):
    from strokecopilot.vision.haemorrhage.stats import wilson as w
    return list(w(k, n)) if n else None


def boot_mean(x, n_boot=10000, seed=20261104):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return None
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def vision_table(rows, groups):
    """rows: [{case, dice, abs_error, wrong_side, flagged}]; groups: {case: {grouping: label}}."""
    out = {}
    for g in GROUPINGS:
        labels = sorted({groups[r["case"]].get(g) for r in rows if groups[r["case"]].get(g) is not None})
        if not labels:
            continue
        out[g] = {}
        for lab in labels:
            sel = [r for r in rows if groups[r["case"]].get(g) == lab]
            n = len(sel)
            small = n < MIN_CASES
            wrong, flags = sum(r["wrong_side"] for r in sel), sum(r["flagged"] for r in sel)
            out[g][lab] = {"cases": n, "too_small": small,
                           "dice_mean": float(np.mean([r["dice"] for r in sel])),
                           "dice_ci": None if small else boot_mean([r["dice"] for r in sel]),
                           "abs_error_ml_mean": float(np.mean([r["abs_error"] for r in sel])),
                           "abs_error_ml_median": float(np.median([r["abs_error"] for r in sel])),
                           "abs_error_ci": None if small else boot_mean([r["abs_error"] for r in sel]),
                           "wrong_side": wrong, "wrong_side_ci": None if small else wilson(wrong, n),
                           "bleed_flags": flags, "bleed_flags_ci": None if small else wilson(flags, n)}
    return out


def reasoner_table(records_by_model, groups):
    """records_by_model: {model: [case records]}; groups: {case: {grouping: label}}. Pooled rates per group."""
    from strokecopilot.nlp.verifier import metrics as vm
    out = {}
    for model, records in records_by_model.items():
        out[model] = {}
        for g in GROUPINGS:
            labels = sorted({groups[r["case_id"]].get(g) for r in records if groups[r["case_id"]].get(g) is not None})
            if not labels:
                continue
            out[model][g] = {}
            for lab in labels:
                sel = [r for r in records if groups[r["case_id"]].get(g) == lab]
                s = vm.summarise(sel)
                out[model][g][lab] = {"cases": len(sel), "too_small": len(sel) < MIN_CASES,
                                      **{k: {"value": s[k]["value"], "ci95": s[k]["ci95"]}
                                         for k in ("hallucination_rate", "recommendation_correctness")}}
    return out


def surrogate(X, y, names, n_boot=2000):
    """Out-of-fold AUC (5 stratified folds) with a case-bootstrap interval; SHAP only if the interval excludes 0.5."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    make = lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
                                              random_state=SEED)
    oof = np.zeros(len(y))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        oof[test] = make().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], oof[idx]))
    out = {"cases": int(len(y)), "positives": int(y.sum()), "auc": float(roc_auc_score(y, oof)),
           "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "explained": None}
    if out["ci95"][0] > 0.5:
        import shap
        sv = np.asarray(shap.TreeExplainer(make().fit(X, y)).shap_values(X))
        mag = np.abs(sv).mean(0)
        out["explained"] = {"share_of_attribution": dict(sorted(zip(names, (mag / mag.sum()).tolist()), key=lambda kv: -kv[1])),
                            "scale": "log-odds"}
    return out


def features(cases, groups, contracts, facts):
    """X for the surrogates: age, female, onset pattern, centre one-hot, NIHSS, Tmax > 6 s and predicted core volumes."""
    centres = sorted({groups[c].get("centre") for c in cases})
    names = ["age", "female", "onset known", "wake-up", "NIHSS", "Tmax > 6 s volume", "predicted core"] + centres
    X = []
    for c in cases:
        age = _val(facts[c], "age")
        nihss = _val(facts[c], "nihss")
        X.append([np.nan if age is None else float(age), float(groups[c]["sex"] == "female"),
                  float(groups[c]["onset"] == "known"), float(groups[c]["onset"] == "wake-up"),
                  np.nan if nihss is None else float(nihss), float(contracts[c]["penumbra_ml"] or 0),
                  float(contracts[c]["core_ml"] or 0)] + [float(groups[c].get("centre") == k) for k in centres])
    X = np.array(X)
    col_mean = np.nanmean(X, axis=0)
    return np.where(np.isnan(X), col_mean, X), names


def main(argv=None):
    import argparse

    from strokecopilot.config import local_drive_root
    from strokecopilot.data import isles24, splits
    from strokecopilot.nlp.verifier import scenarios as sc
    from strokecopilot.pipeline import compare as cp
    from strokecopilot.pipeline import heldout_analysis as ha
    from strokecopilot.vision.haemorrhage import isles_check as ic

    ap = argparse.ArgumentParser(description="Task E: subgroups, and SHAP rows 12-13")
    ap.add_argument("--out", default="runs/EXP-80_subgroups")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    drive = local_drive_root()
    folds = splits.load_folds()
    dev = list(splits.development_ids(folds))
    root = Path(__file__).resolve().parents[3] / "data" / "local" / isles24.SLIM_NAME
    facts = ha.load_facts(drive / "artifacts" / "isles24_notes_v1", dev)
    centre = {c: isles24.load_case(c, root, names=(), folds=folds).clinical.get("Center") for c in dev}
    groups = {c: groups_from_facts(facts[c], centre[c]) for c in dev}
    contracts = ha.load_contracts(drive / "artifacts" / "vision_outputs_v1", dev)
    per_case = {r["case"]: r for r in csv.DictReader(open(drive / "runs" / "EXP-27_visionA_error_followups" / "per_case.csv"))}
    crosses = lambda r: int(any(r[f"nnunet_crosses_{c}"] in ("True", "true", "1") for c in (21, 31, 51, 70)))
    rows = [{"case": c, "dice": float(per_case[c]["nnunet_dice"]), "abs_error": float(per_case[c]["nnunet_abs_error_ml"]),
             "wrong_side": crosses(per_case[c]), "flagged": int(contracts[c]["haemorrhage"]["present"])} for c in dev]
    result = {"experiment": "EXP-80", "min_cases": MIN_CASES, "vision": vision_table(rows, groups)}

    scen = sc.generate()
    s_groups = {s.id: groups_from_facts(s.facts) for s in scen}
    root_of = lambda m: Path("runs/EXP-47_small_reasoner" if m.startswith("gemma-3-4b") else "runs/EXP-37_paid_runs")
    scen_recs = {m: cp.case_records(root_of(m) / m / "scenarios_rulebook") for m in MODELS}
    real_recs = {m: cp.case_records(root_of(m) / m / "real_predicted") for m in MODELS}
    result["reasoners_scenarios"] = reasoner_table(scen_recs, s_groups)
    result["reasoners_real"] = reasoner_table(real_recs, {c: groups[c] for c in dev})

    X, names = features(dev, groups, contracts, facts)
    result["row12_surrogate_vision_a_wrong_side"] = surrogate(X, np.array([r["wrong_side"] for r in rows]), names)
    result["row13_surrogate_vision_b_flag"] = surrogate(X, np.array([r["flagged"] for r in rows]), names)
    text = json.dumps(result, indent=1, default=float)
    if ic.ids_in(text, dev + list(folds["heldout"])):
        raise ValueError("A case ID reached metrics.json. Nothing was saved.")
    (out / "metrics.json").write_text(text)
    print(json.dumps({k: result[k] for k in ("row12_surrogate_vision_a_wrong_side", "row13_surrogate_vision_b_flag")},
                     indent=1))


if __name__ == "__main__":
    main()
