"""Vision A confidence: `confidence.core` and `confidence.penumbra` for the contract (plan §6.1, §8.1; Task 2 of
docs/team/lorna_vision_a.md). Design approved by Parsa, 15 Nov 2026.

confidence.core uses only signals that exist when a prediction is made, never the true infarct:
1. the share of the at-risk area (Tmax > 6 s, plus a 10 mm margin) with no or unreliable Tmax data;
2. the size of nnU-Net's predicted core, as log(1 + mL);
3. how far apart nnU-Net's and the threshold's core volumes are, as |log(1 + a) - log(1 + b)|.
A small logistic regression turns them into the chance of a large error, and confidence.core is 1 minus that
chance. It is fitted cross-fitted by fold on the 129 development cases: fit on 4 folds, score the 5th, so no case
is scored by a mapping that saw it. A final fit on all 129 is saved for inference. Held-out cases are never used.

A large error is a case that lands on the wrong side of at least one decision cut-off (21, 31, 51 or 70 mL); a
volume error of 10 mL or more is reported too. The true infarct is used only to mark these, never as an input.

confidence.penumbra is 1 minus the share of the brain with no or unreliable Tmax data. ISLES'24 has no penumbra
label, so it is a data-quality score only and is not fitted.

No fold-ensemble disagreement: each development case was trained on by 4 of the 5 fold models.
Outputs that leave Drive carry counts and summaries only, never case IDs.
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize

from strokecopilot.config import ISLES24_SLIM
from strokecopilot.data.isles24_survey import PENUMBRA_TMAX_S, TISSUE_HU
from strokecopilot.vision.perfusion.error_analysis import _ids_in
from strokecopilot.vision.perfusion.evaluate import DECISION_CUTOFFS_ML
from strokecopilot.vision.perfusion.pod_job import FOLDS_SHA256, SLIM_SHA256
from strokecopilot.vision.perfusion.thresholds import reliable

# Margin around the Tmax > 6 s area in which missing Tmax data is counted (agreed with Lorna).
MARGIN_MM = 10.0
# The second, simpler measure of a large error: absolute volume error of at least this many mL. 10 mL is the
# smallest gap between two decision cut-offs (21 to 31 mL).
LARGE_ERROR_ML = 10.0
# The three signals, in the order the mapping uses them.
FEATURES = ("at_risk_no_data_share", "log_pred_ml", "log_disagreement")
# Ridge penalty on the standardised weights (not the intercept): keeps 4 numbers stable on about 100 cases.
L2 = 1.0
BANDS = ("low", "middle", "high")
CROSSINGS = tuple(f"nnunet_crosses_{c:g}" for c in DECISION_CUTOFFS_ML)
# Columns the scoring table needs (EXP-27's per_case.csv has them).
TABLE_COLUMNS = ("case", "fold", "ref_ml", "nnunet_pred_ml", "threshold_pred_ml", "nnunet_abs_error_ml", *CROSSINGS)


# ---------------------------------------------------------------------------- Signals from the scan


def brain_mask(ncct):
    """Tissue on the plain CT (TISSUE_HU, the range the threshold method uses)."""
    ncct = np.asarray(ncct, dtype=np.float32)
    return (ncct >= TISSUE_HU[0]) & (ncct <= TISSUE_HU[1])


def at_risk_no_data_share(tmax, cbf, spacing_mm, brain=None, margin_mm=MARGIN_MM):
    """Share of the at-risk zone with no or unreliable Tmax data.

    The zone is the Tmax > 6 s area grown by `margin_mm`, kept inside `brain` when given. With no Tmax > 6 s
    area, the zone is the whole brain. Returns None when the zone is empty.
    """
    tmax = np.asarray(tmax, dtype=np.float32)
    valid = reliable(tmax, cbf)
    at_risk = valid & (tmax > PENUMBRA_TMAX_S)
    brain = np.ones(tmax.shape, dtype=bool) if brain is None else np.asarray(brain, dtype=bool)
    if at_risk.any():
        sampling = np.broadcast_to(np.asarray(spacing_mm, dtype=float), (3,))
        zone = ndimage.distance_transform_edt(~at_risk, sampling=sampling) <= margin_mm
        zone &= brain | at_risk
    else:
        zone = brain
    n = int(np.count_nonzero(zone))
    return float(np.count_nonzero(zone & ~valid) / n) if n else None


def brain_no_data_share(tmax, cbf, brain):
    """Share of the brain with no or unreliable Tmax data, or None for an empty brain mask."""
    brain = np.asarray(brain, dtype=bool)
    n = int(np.count_nonzero(brain))
    return float(np.count_nonzero(brain & ~reliable(tmax, cbf)) / n) if n else None


def penumbra_confidence(brain_share):
    """confidence.penumbra: 1 minus the brain's no-data share. A data-quality score only (no penumbra label).
    With no measurable brain, 0."""
    if brain_share is None or (isinstance(brain_share, float) and np.isnan(brain_share)):
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - brain_share)))


def case_signals(case, margin_mm=MARGIN_MM):
    """The scan-based signals for one loaded isles24.Case (needs tmax, cbf and ncct arrays)."""
    arrays = case.arrays
    brain = brain_mask(arrays["ncct"])
    at_risk = at_risk_no_data_share(arrays["tmax"], arrays["cbf"], case.spacing_mm, brain, margin_mm)
    whole = brain_no_data_share(arrays["tmax"], arrays["cbf"], brain)
    return {"case": case.id, "at_risk_no_data_share": at_risk, "brain_no_data_share": whole,
            "confidence_penumbra": penumbra_confidence(whole)}


def add_volume_signals(table):
    """A copy of the table with the two volume signals, from nnU-Net's and the threshold's predicted mL."""
    table = table.copy()
    nn = np.log1p(table["nnunet_pred_ml"].astype(float).clip(lower=0))
    th = np.log1p(table["threshold_pred_ml"].astype(float).clip(lower=0))
    table["log_pred_ml"] = nn
    table["log_disagreement"] = (nn - th).abs()
    return table


def add_targets(table):
    """A copy of the table with the two error measures (from the true infarct; never used as inputs)."""
    table = table.copy()
    table["large_error"] = sum(table[c].astype(bool) for c in CROSSINGS) > 0
    table["error_10ml"] = table["nnunet_abs_error_ml"] >= LARGE_ERROR_ML
    return table


def check_table(table, expected_cases=None):
    """Raise, with counts only, if the scoring table lacks columns, repeats a case or has the wrong size."""
    missing = [c for c in TABLE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"The table lacks columns {missing}.")
    if table["case"].duplicated().any():
        raise ValueError(f"The table repeats {int(table['case'].duplicated().sum())} cases.")
    if expected_cases is not None and len(table) != expected_cases:
        raise ValueError(f"The table holds {len(table)} cases, not {expected_cases}.")


# ------------------------------------------------------------------------------- The mapping


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_mapping(X, y, l2=L2):
    """Logistic regression for P(large error) on standardised signals, with a ridge penalty on the weights.

    X: (n, 3) array of FEATURES, missing values allowed (filled with the training mean). y: booleans.
    Returns a JSON-safe dict: features, fill, mean, std, weights, intercept, n, positives.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("The training cases are all one class, so no mapping can be fitted.")
    fill = np.nanmean(X, axis=0)
    fill = np.where(np.isnan(fill), 0.0, fill)
    X = np.where(np.isnan(X), fill, X)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Z = (X - mean) / std

    def loss(params):
        b, w = params[0], params[1:]
        z = b + Z @ w
        # log(1 + exp(z)) - y z, written to stay finite for large |z|
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * np.sum(w ** 2))

    def grad(params):
        b, w = params[0], params[1:]
        r = _sigmoid(b + Z @ w) - y
        return np.concatenate([[r.sum()], Z.T @ r + l2 * w])

    start = np.zeros(Z.shape[1] + 1)
    result = optimize.minimize(loss, start, jac=grad, method="BFGS")
    return {"features": list(FEATURES), "fill": fill.tolist(), "mean": mean.tolist(), "std": std.tolist(),
            "weights": result.x[1:].tolist(), "intercept": float(result.x[0]), "n": int(len(y)),
            "positives": int(y.sum()), "l2": l2}


def predict_confidence(mapping, X):
    """confidence.core for rows of signals: 1 minus the mapping's chance of a large error, in 0 to 1."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    fill = np.asarray(mapping["fill"])
    X = np.where(np.isnan(X), fill, X)
    Z = (X - np.asarray(mapping["mean"])) / np.asarray(mapping["std"])
    p = _sigmoid(mapping["intercept"] + Z @ np.asarray(mapping["weights"]))
    return 1.0 - p


def cross_fitted_confidence(table, l2=L2):
    """Cross-fitted by fold: for each fold, fit on the other folds and score this one.

    table: with FEATURES, `fold` and `large_error`. Returns (confidence per row as an array in the table's order,
    {fold: mapping}). No case is scored by a mapping whose training cases include it.
    """
    X = table[list(FEATURES)].astype(float).to_numpy()
    y = table["large_error"].astype(bool).to_numpy()
    folds = table["fold"].astype(int).to_numpy()
    out = np.full(len(table), np.nan)
    mappings = {}
    for fold in sorted(set(folds.tolist())):
        train, test = folds != fold, folds == fold
        mapping = fit_mapping(X[train], y[train], l2)
        out[test] = predict_confidence(mapping, X[test])
        mappings[str(fold)] = mapping
    return out, mappings


def confidence_core(mapping, at_risk_share, nnunet_ml, threshold_ml):
    """confidence.core for one case at inference, from the saved mapping and that case's three inputs."""
    nn, th = np.log1p(max(0.0, nnunet_ml)), np.log1p(max(0.0, threshold_ml))
    share = np.nan if at_risk_share is None else at_risk_share
    return float(predict_confidence(mapping, [[share, nn, abs(nn - th)]])[0])


def save_mapping(path, mapping):
    Path(path).write_text(json.dumps(mapping, indent=1))
    return Path(path)


def load_mapping(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------- Bands and the check


def bands(confidence):
    """Three equal bands by confidence (ties broken by order): low, middle, high."""
    import pandas as pd

    ranks = pd.Series(confidence).rank(method="first")
    return pd.qcut(ranks, 3, labels=list(BANDS)).astype(str).to_numpy()


def _num(value, digits=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def band_table(table, confidence_col="confidence_core"):
    """Per confidence band and overall: cases, the confidence range, mean and median volume error, cases with
    10 mL or more, and cases on the wrong side of each cut-off and of any. Counts and summaries only."""
    table = table.copy()
    table["band"] = bands(table[confidence_col].to_numpy())

    def block(rows):
        out = {
            "cases": int(len(rows)),
            "confidence_min": _num(rows[confidence_col].min()),
            "confidence_max": _num(rows[confidence_col].max()),
            "abs_error_ml_mean": _num(rows["nnunet_abs_error_ml"].mean(), 1),
            "abs_error_ml_median": _num(rows["nnunet_abs_error_ml"].median(), 1),
            f"error_{LARGE_ERROR_ML:g}ml_or_more": int(rows["error_10ml"].sum()),
            "wrong_side_any": int(rows["large_error"].sum()),
        }
        for cutoff, column in zip(DECISION_CUTOFFS_ML, CROSSINGS):
            out[f"wrong_side_{cutoff:g}"] = int(rows[column].astype(bool).sum())
        return out

    result = {band: block(table[table["band"] == band]) for band in BANDS}
    result["all"] = block(table)
    return result


def done_check(band_rows, table, confidence_col="confidence_core"):
    """The task's "done when": low confidence goes with large errors. True when the low band has more cases on the
    wrong side of a cut-off than the high band, and more errors of 10 mL or more. Also the Spearman correlation
    between confidence and volume error (negative is the right direction). Counts only."""
    from scipy import stats

    low, high = band_rows["low"], band_rows["high"]
    key = f"error_{LARGE_ERROR_ML:g}ml_or_more"
    rho = stats.spearmanr(table[confidence_col], table["nnunet_abs_error_ml"]).correlation
    return {
        "low_more_wrong_side_than_high": low["wrong_side_any"] > high["wrong_side_any"],
        "low_more_10ml_errors_than_high": low[key] > high[key],
        "wrong_side_low_vs_high": [low["wrong_side_any"], high["wrong_side_any"]],
        "errors_10ml_low_vs_high": [low[key], high[key]],
        "spearman_confidence_vs_abs_error": _num(rho),
        "passes": bool(low["wrong_side_any"] > high["wrong_side_any"] and low[key] > high[key]),
    }


# ---------------------------------------------------------------------------- Run record


def write_config(run_dir, git_commit, experiment, inputs, notes=""):
    """config.json for the confidence run, written before it and never overwritten. A placeholder experiment ID
    is refused. Records the design, the signals and the measures. No case IDs."""
    if not re.fullmatch(r"EXP-\d+", str(experiment)):
        raise ValueError(f"Set the experiment ID from the Notion row (like EXP-30), not {experiment!r}.")
    if not git_commit:
        raise ValueError("git_commit is required: the commit the notebook installed.")
    run_dir = Path(run_dir)
    path = run_dir / "config.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists. config.json is written once, before the run.")
    config = {
        "experiment": experiment,
        "name": run_dir.name,
        "runner": "strokecopilot.vision.perfusion.confidence (notebooks/14_visionA_confidence.ipynb)",
        "git_commit": git_commit,
        "data_version": ISLES24_SLIM,
        "data_sha256": SLIM_SHA256,
        "folds_sha256": FOLDS_SHA256,
        "cases": "development only (folds 0-4); held-out cases are never used",
        "inputs": dict(inputs),
        "design": "approved by Parsa, 15 Nov 2026",
        "signals": {
            "at_risk_no_data_share": f"share of the Tmax > {PENUMBRA_TMAX_S:g} s area plus a {MARGIN_MM:g} mm margin "
                                     "(inside plain-CT tissue) with no or unreliable Tmax; the whole brain if "
                                     "there is no such area",
            "log_pred_ml": "log(1 + nnU-Net predicted mL)",
            "log_disagreement": "|log(1 + nnU-Net mL) - log(1 + threshold mL)|",
        },
        "mapping": f"logistic regression on standardised signals, ridge {L2:g} on the weights; "
                   "confidence.core = 1 - P(large error); cross-fitted by fold (fit on 4, score the 5th); "
                   "final fit on all development cases saved as mapping.json",
        "large_error": "nnU-Net volume on the wrong side of at least one of " + ", ".join(f"{c:g}" for c in DECISION_CUTOFFS_ML) + " mL",
        "second_measure": f"absolute volume error of {LARGE_ERROR_ML:g} mL or more",
        "bands": "three equal bands by confidence.core: low, middle, high",
        "penumbra": "confidence.penumbra = 1 - share of the brain (plain-CT tissue) with no or unreliable Tmax; "
                    "data-quality score only, ISLES'24 has no penumbra label",
        "not_used": "fold-ensemble disagreement (each development case was trained on by 4 of 5 models); "
                    "the true infarct as an input",
        "outputs_policy": "counts and summaries only, never case IDs",
        "notes": notes,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1))
    return config


def build_metrics(table, config, git_commit, band_rows, check, mappings, final_mapping):
    """metrics.json for the confidence run. Raises if any case ID appears anywhere in it."""
    if not git_commit:
        raise ValueError("git_commit is required: the commit this run installed.")
    penumbra = table["confidence_penumbra"]
    metrics = {
        "experiment": config.get("experiment"),
        "git_commit": git_commit,
        "config_git_commit": config.get("git_commit"),
        "data_version": config.get("data_version", ISLES24_SLIM),
        "cases": int(len(table)),
        "large_error_cases": int(table["large_error"].sum()),
        "error_10ml_cases": int(table["error_10ml"].sum()),
        "bands": band_rows,
        "done_check": check,
        "fold_mappings": mappings,
        "final_mapping": final_mapping,
        "penumbra": {"median": _num(penumbra.median()), "min": _num(penumbra.min()),
                     "under_0_9": int((penumbra < 0.9).sum())},
    }
    leaked = _ids_in(metrics, set(table["case"].astype(str)))
    if leaked:
        raise ValueError(f"{len(leaked)} case IDs reached metrics.json. Nothing was written.")
    return metrics


def write_metrics(run_dir, metrics):
    """Write metrics.json in the run folder, last, as it marks a finished run."""
    path = Path(run_dir) / "metrics.json"
    path.write_text(json.dumps(metrics, indent=1))
    return path
