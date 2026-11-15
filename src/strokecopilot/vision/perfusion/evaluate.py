"""Vision A metrics: Dice, volume error and decision-threshold crossings, per fold (plan §8.1, §9).

The same functions score the threshold baseline and nnU-Net. Reference volumes come from
native_volumes.csv, measured before resampling, while Dice is on the slim grid. Dice is reported on
the whole infarct and on the part with reliable Tmax, because a model cannot find infarct it has no
perfusion input for (docs/DATA.md). A case whose reference and prediction are both empty has no Dice.
"""

import statistics

import numpy as np

from strokecopilot.vision.perfusion.volumes import mask_ml

# Volume cut-offs the rules use: DAWN 21, 31 and 51 mL; DEFUSE-3 and the extended-window trials 70 mL.
DECISION_CUTOFFS_ML = (21.0, 31.0, 51.0, 70.0)


def dice(pred, truth):
    pred, truth = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    total = np.count_nonzero(pred) + np.count_nonzero(truth)
    if total == 0:
        return None
    return 2.0 * np.count_nonzero(pred & truth) / total


def crosses(pred_ml, ref_ml, cutoff):
    """True when prediction and reference fall on different sides of a cut-off (below vs not below)."""
    return (pred_ml < cutoff) != (ref_ml < cutoff)


def case_metrics(pred, infarct, valid, spacing_mm, ref_ml):
    """One case's scores for a predicted mask. ref_ml is the native infarct volume."""
    pred_ml = mask_ml(pred, spacing_mm)
    row = {
        "dice": dice(pred, infarct),
        "dice_reliable": dice(pred & valid, infarct & valid),
        "pred_ml": pred_ml,
        "ref_ml": ref_ml,
        "error_ml": pred_ml - ref_ml,
        "abs_error_ml": abs(pred_ml - ref_ml),
        "infarct_reliable_share": (float(np.count_nonzero(infarct & valid) / np.count_nonzero(infarct)) if np.any(infarct) else None),
    }
    for cutoff in DECISION_CUTOFFS_ML:
        row[f"crosses_{cutoff:g}"] = crosses(pred_ml, ref_ml, cutoff)
    return row


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def _percentiles(values, points=(5, 50, 95)):
    values = [v for v in values if v is not None]
    return {f"p{p}": float(np.percentile(values, p)) for p in points} if values else None


def summarise(rows, metrics=("dice", "dice_reliable", "abs_error_ml", "error_ml")):
    """Per-fold means, then mean ± std over folds, plus pooled percentiles and crossing counts.

    rows: dicts from case_metrics with a `fold` key added.
    """
    folds = sorted({row["fold"] for row in rows})
    per_fold = {}
    for fold in folds:
        in_fold = [row for row in rows if row["fold"] == fold]
        per_fold[str(fold)] = {m: _mean_std([row[m] for row in in_fold]) for m in metrics}
        per_fold[str(fold)]["cases"] = len(in_fold)
    across = {m: _mean_std([per_fold[str(f)][m]["mean"] for f in folds]) for m in metrics}
    crossings = {
        f"{cutoff:g}": {"cases": sum(row[f"crosses_{cutoff:g}"] for row in rows), "of": len(rows)}
        for cutoff in DECISION_CUTOFFS_ML
    }
    return {
        "cases": len(rows),
        "cases_without_dice": sum(row["dice"] is None for row in rows),
        "per_fold": per_fold,
        "mean_std_over_folds": across,
        "pooled_percentiles": {m: _percentiles([row[m] for row in rows]) for m in metrics},
        "threshold_crossings": crossings,
    }


def load_per_case(path, variant=None):
    """Rows of a per_case.csv (nnU-Net run or threshold baseline), with numbers and booleans parsed. `variant` keeps
    only that baseline variant (the baseline file holds one row per case per variant)."""
    import csv

    def parse(key, value):
        if value in ("", "None", "nan"):
            return None
        if key.startswith("crosses_"):
            return value in ("True", "true", "1", "1.0")
        try:
            return float(value)
        except ValueError:
            return value

    with open(path, newline="") as f:
        rows = [{k: parse(k, v) for k, v in row.items()} for row in csv.DictReader(f)]
    if variant is not None:
        rows = [row for row in rows if row.get("variant") == variant]
    for row in rows:
        row["fold"] = int(row["fold"])
    return rows


def paired_comparison(model_rows, baseline_rows, n_boot=5000, seed=0):
    """Model against baseline on the same cases (B5, plan §8.1). For Dice and absolute volume error: both means, the
    mean paired difference (model − baseline) with a bootstrap 95% CI over cases, how many cases each wins, and a
    Wilcoxon signed-rank p. For each decision cut-off: cases only the model puts on the wrong side, only the baseline,
    both, and an exact McNemar p on the discordant ones. Cases are matched by `case`; Dice skips cases where either is
    None. Returns no case IDs."""
    from scipy import stats

    base = {row["case"]: row for row in baseline_rows}
    pairs = [(row, base[row["case"]]) for row in model_rows if row["case"] in base]
    rng = np.random.default_rng(seed)
    out = {"cases": len(pairs), "metrics": {}, "crossings": {}}
    for metric, higher_better in (("dice", True), ("abs_error_ml", False)):
        m = np.array([a[metric] for a, b in pairs if a[metric] is not None and b[metric] is not None], dtype=float)
        b = np.array([b[metric] for a, b in pairs if a[metric] is not None and b[metric] is not None], dtype=float)
        diff = m - b
        boot = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
        nonzero = diff[diff != 0]
        out["metrics"][metric] = {
            "n": int(len(diff)), "model_mean": float(m.mean()), "baseline_mean": float(b.mean()),
            "model_median": float(np.median(m)), "baseline_median": float(np.median(b)),
            "mean_difference": float(diff.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "model_better": int(np.sum(diff > 0) if higher_better else np.sum(diff < 0)),
            "baseline_better": int(np.sum(diff < 0) if higher_better else np.sum(diff > 0)),
            "ties": int(np.sum(diff == 0)),
            "wilcoxon_p": float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) > 0 else None,
        }
    for cutoff in DECISION_CUTOFFS_ML:
        key = f"crosses_{cutoff:g}"
        only_model = sum(a[key] and not b[key] for a, b in pairs)
        only_base = sum(b[key] and not a[key] for a, b in pairs)
        discordant = only_model + only_base
        out["crossings"][f"{cutoff:g}"] = {
            "model": sum(a[key] for a, _ in pairs), "baseline": sum(b[key] for _, b in pairs),
            "only_model": only_model, "only_baseline": only_base, "both": sum(a[key] and b[key] for a, b in pairs),
            "mcnemar_p": float(stats.binomtest(only_model, discordant, 0.5).pvalue) if discordant else None,
        }
    return out
