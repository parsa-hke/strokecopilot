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
