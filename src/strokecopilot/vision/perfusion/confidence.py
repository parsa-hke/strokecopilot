"""Vision A confidence: `confidence.core` and `confidence.penumbra` for the contract (plan §6.1, §8.1; Task 2 of
docs/team/lorna_vision_a.md). Design approved by Parsa, 15 Nov 2026; Lorna decides the final version (16 Nov 2026).

confidence.core uses only signals that exist when a prediction is made, never the true infarct. Three versions
are fitted and reported side by side (VERSIONS); Lorna chose v2 after EXP-31 (CHOSEN_VERSION), which is the default
for fitting and for scoring one case (load_default_mapping, confidence_core):
- v1, the approved three: missing Tmax data around the at-risk area; nnU-Net's predicted core as log(1 + mL);
  the gap between nnU-Net's and the threshold's core volumes on that log scale;
- v2, v1 plus how close nnU-Net's volume is to the nearest decision cut-off (on the same log scale);
- v3, v2 plus the size of the Tmax > 6 s area as log(1 + mL), aimed at a tiny core inside a large at-risk area.
A small logistic regression turns the signals into the chance of a large error, and confidence.core is 1 minus that
chance. It is cross-fitted by fold on the 129 development cases (fit on 4 folds, score the 5th), so no case is
scored by a mapping that saw it. A final fit on all 129 is saved for inference. Held-out cases are never used.

Which values the contract uses (contract/build.py):
- development cases: the cross-fitted values in the confidence run's per_case.csv (EXP-31), read with
  development_confidence. The final mapping (mapping_v2.json) was fitted on all 129 development cases, so applying
  it to one of them is in-sample and flatters it (AUC 0.757 against 0.722 cross-fitted);
- held-out cases (Phase 4 only): mapping_v2.json from load_default_mapping, applied with case_confidence.

A large error is a case that lands on the wrong side of at least one decision cut-off (21, 31, 51 or 70 mL); a
volume error of 10 mL or more is reported too. The true infarct is used only to mark these, never as an input.

Where missing data is counted. The plain CT covers the whole head, but the perfusion scan covers only a slab of
slices, and its maps have no values outside the brain. So missing data is counted only in plain-CT tissue inside
the perfusion region: the slices from the first to the last one with perfusion values, and within each slice the
area the perfusion maps cover, with holes filled (a covered slice with no values at all takes the slab's outline).
Real gaps inside the brain still count; the scalp and the slices outside the slab do not.

confidence.penumbra is 1 minus the share of that region with no or unreliable Tmax data. ISLES'24 has no penumbra
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
from strokecopilot.vision.perfusion.volumes import mask_ml

# Margin around the Tmax > 6 s area in which missing Tmax data is counted (agreed with Lorna).
MARGIN_MM = 10.0
# The second, simpler measure of a large error: absolute volume error of at least this many mL. 10 mL is the
# smallest gap between two decision cut-offs (21 to 31 mL).
LARGE_ERROR_ML = 10.0
# Tmax background in the slim maps: no perfusion value at all (docs/DATA.md).
TMAX_BACKGROUND = -30.0
# The three versions of confidence.core, each a list of signals in the order the mapping uses them.
VERSIONS = {
    "v1": ("at_risk_no_data_share", "log_pred_ml", "log_disagreement"),
    "v2": ("at_risk_no_data_share", "log_pred_ml", "log_disagreement", "log_cutoff_distance"),
    "v3": ("at_risk_no_data_share", "log_pred_ml", "log_disagreement", "log_cutoff_distance", "log_tmax6_ml"),
}
# Lorna's choice after EXP-31 (16 Nov 2026): v2. It puts the most cut-off errors in the low band, and distance to a
# cut-off is the clearest clinical reason for doubt; v3 added nothing. The choice was made after seeing all three,
# and v1 and v2 differ by 2 cases (docs/CONFIDENCE_VISION_A.md).
CHOSEN_VERSION = "v2"
FEATURES = VERSIONS[CHOSEN_VERSION]
ALL_SIGNALS = VERSIONS["v3"]
# The chosen version's final mapping, as notebook 14 saves it in its run folder (EXP-31 on Drive).
DEFAULT_MAPPING_FILE = f"mapping_{CHOSEN_VERSION}.json"
# Ridge penalty on the standardised weights (not the intercept): keeps a few numbers stable on about 100 cases.
L2 = 1.0
BANDS = ("low", "middle", "high")
CROSSINGS = tuple(f"nnunet_crosses_{c:g}" for c in DECISION_CUTOFFS_ML)
# Columns the scoring table needs (EXP-27's per_case.csv has them).
TABLE_COLUMNS = ("case", "fold", "ref_ml", "nnunet_pred_ml", "threshold_pred_ml", "nnunet_abs_error_ml", *CROSSINGS)


# ---------------------------------------------------------------------------- Where missing data is counted


def brain_mask(ncct):
    """Tissue on the plain CT (TISSUE_HU, the range the threshold method uses)."""
    ncct = np.asarray(ncct, dtype=np.float32)
    return (ncct >= TISSUE_HU[0]) & (ncct <= TISSUE_HU[1])


def perfusion_region(tmax, cbf):
    """The part of the scan the perfusion study covers, with its holes filled (see the module text).

    Slices from the first to the last with any perfusion value; in each, the area with values, holes filled slice
    by slice. A slice inside that range with no values at all (a gap in coverage) takes the outline of all covered
    slices together, so its missing data still counts. Axial slices are along the last axis.
    """
    tmax = np.asarray(tmax, dtype=np.float32)
    has = (tmax != TMAX_BACKGROUND) | (np.asarray(cbf, dtype=np.float32) > 0)
    region = np.zeros(has.shape, dtype=bool)
    covered = np.flatnonzero(has.any(axis=(0, 1)))
    if covered.size == 0:
        return region
    for z in covered:
        region[:, :, z] = ndimage.binary_fill_holes(has[:, :, z])
    outline = ndimage.binary_fill_holes(region.any(axis=2))
    for z in range(covered[0], covered[-1] + 1):
        if not region[:, :, z].any():
            region[:, :, z] = outline
    return region


def measurement_area(ncct, tmax, cbf):
    """Plain-CT tissue inside the perfusion region: where missing Tmax data is counted."""
    return brain_mask(ncct) & perfusion_region(tmax, cbf)


# ---------------------------------------------------------------------------- Signals from the scan


def at_risk_no_data_share(tmax, cbf, spacing_mm, area=None, margin_mm=MARGIN_MM):
    """Share of the at-risk zone with no or unreliable Tmax data.

    The zone is the Tmax > 6 s area grown by `margin_mm`, kept inside `area` (measurement_area) when given. With
    no Tmax > 6 s area, the zone is the whole area. Returns None when the zone is empty.
    """
    tmax = np.asarray(tmax, dtype=np.float32)
    valid = reliable(tmax, cbf)
    at_risk = valid & (tmax > PENUMBRA_TMAX_S)
    area = np.ones(tmax.shape, dtype=bool) if area is None else np.asarray(area, dtype=bool)
    if at_risk.any():
        sampling = np.broadcast_to(np.asarray(spacing_mm, dtype=float), (3,))
        zone = ndimage.distance_transform_edt(~at_risk, sampling=sampling) <= margin_mm
        zone &= area | at_risk
    else:
        zone = area
    n = int(np.count_nonzero(zone))
    return float(np.count_nonzero(zone & ~valid) / n) if n else None


def no_data_share(tmax, cbf, area):
    """Share of `area` with no or unreliable Tmax data, or None for an empty area."""
    area = np.asarray(area, dtype=bool)
    n = int(np.count_nonzero(area))
    return float(np.count_nonzero(area & ~reliable(tmax, cbf)) / n) if n else None


# The earlier name, for the old whole-tissue measure kept as a before/after check.
brain_no_data_share = no_data_share


def tmax6_ml(tmax, cbf, spacing_mm):
    """Volume of the Tmax > 6 s area with reliable data, in mL (unrounded)."""
    return mask_ml(reliable(tmax, cbf) & (np.asarray(tmax, dtype=np.float32) > PENUMBRA_TMAX_S), spacing_mm)


def penumbra_confidence(share):
    """confidence.penumbra: 1 minus the no-data share of the measurement area. A data-quality score only (no
    penumbra label). With no measurable area, 0."""
    if share is None or (isinstance(share, float) and np.isnan(share)):
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - share)))


def case_signals(case, margin_mm=MARGIN_MM):
    """The scan-based signals for one loaded isles24.Case (needs tmax, cbf and ncct arrays).

    Also returns the old whole-tissue no-data share (`tissue_no_data_share`), the measurement area's size in
    slices, and the plain CT's tissue slices, for the before/after check of the slab and footprint fix.
    """
    arrays = case.arrays
    tissue = brain_mask(arrays["ncct"])
    area = tissue & perfusion_region(arrays["tmax"], arrays["cbf"])
    share = no_data_share(arrays["tmax"], arrays["cbf"], area)
    return {
        "case": case.id,
        "at_risk_no_data_share": at_risk_no_data_share(arrays["tmax"], arrays["cbf"], case.spacing_mm, area, margin_mm),
        "brain_no_data_share": share,
        "tissue_no_data_share": no_data_share(arrays["tmax"], arrays["cbf"], tissue),
        "tmax6_ml": tmax6_ml(arrays["tmax"], arrays["cbf"], case.spacing_mm),
        "area_slices": int(np.count_nonzero(area.any(axis=(0, 1)))),
        "tissue_slices": int(np.count_nonzero(tissue.any(axis=(0, 1)))),
        "confidence_penumbra": penumbra_confidence(share),
    }


def log_cutoff_distance(pred_ml):
    """How far a volume is from the nearest decision cut-off, as |log(1 + mL) - log(1 + cut-off)|."""
    pred = np.log1p(np.clip(np.asarray(pred_ml, dtype=float), 0, None))
    cuts = np.log1p(np.asarray(DECISION_CUTOFFS_ML, dtype=float))
    return np.min(np.abs(pred[..., None] - cuts), axis=-1)


def _volume(name, value):
    """A core volume as a float, or ValueError if it is missing or not finite (it must never become 0 mL)."""
    value = np.nan if value is None else float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} is {value}: a core volume is needed, and a missing one is not 0 mL.")
    return value


def signal_row(at_risk_share, nnunet_ml, threshold_ml, tmax6=None):
    """Every signal for one case from its inputs at inference (missing ones as NaN).

    Both core volumes are required: a missing or NaN volume raises ValueError rather than being read as 0 mL. A
    missing or NaN `tmax6` counts as missing, like a missing no-data share.
    """
    nnunet_ml, threshold_ml = _volume("nnunet_ml", nnunet_ml), _volume("threshold_ml", threshold_ml)
    nn, th = np.log1p(max(0.0, nnunet_ml)), np.log1p(max(0.0, threshold_ml))
    tmax6 = np.nan if tmax6 is None else float(tmax6)
    return {
        "at_risk_no_data_share": np.nan if at_risk_share is None else float(at_risk_share),
        "log_pred_ml": float(nn),
        "log_disagreement": float(abs(nn - th)),
        "log_cutoff_distance": float(log_cutoff_distance(nnunet_ml)),
        "log_tmax6_ml": np.nan if np.isnan(tmax6) else float(np.log1p(max(0.0, tmax6))),
    }


def add_volume_signals(table):
    """A copy of the table with the volume signals: log_pred_ml, log_disagreement, log_cutoff_distance, and
    log_tmax6_ml when the table has `tmax6_ml`."""
    table = table.copy()
    nn = np.log1p(table["nnunet_pred_ml"].astype(float).clip(lower=0))
    th = np.log1p(table["threshold_pred_ml"].astype(float).clip(lower=0))
    table["log_pred_ml"] = nn
    table["log_disagreement"] = (nn - th).abs()
    table["log_cutoff_distance"] = log_cutoff_distance(table["nnunet_pred_ml"].to_numpy())
    if "tmax6_ml" in table.columns:
        table["log_tmax6_ml"] = np.log1p(table["tmax6_ml"].astype(float).clip(lower=0))
    return table


def add_targets(table):
    """A copy of the table with the error measures (from the true infarct; never used as inputs): large_error,
    error_10ml, and undercall (predicted smaller than true and on the wrong side of a cut-off)."""
    table = table.copy()
    crosses = sum(table[c].astype(bool) for c in CROSSINGS) > 0
    table["large_error"] = crosses
    table["error_10ml"] = table["nnunet_abs_error_ml"] >= LARGE_ERROR_ML
    table["undercall"] = crosses & (table["nnunet_pred_ml"] < table["ref_ml"])
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


def fit_mapping(X, y, l2=L2, features=FEATURES):
    """Logistic regression for P(large error) on standardised signals, with a ridge penalty on the weights.

    X: (n, k) array of `features`, missing values allowed (filled with the training mean). y: booleans.
    Returns a JSON-safe dict: features, fill, mean, std, weights, intercept, n, positives.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(features):
        raise ValueError(f"X has {X.shape[-1] if X.ndim == 2 else '?'} columns for {len(features)} features.")
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

    result = optimize.minimize(loss, np.zeros(Z.shape[1] + 1), jac=grad, method="BFGS")
    return {"features": list(features), "fill": fill.tolist(), "mean": mean.tolist(), "std": std.tolist(),
            "weights": result.x[1:].tolist(), "intercept": float(result.x[0]), "n": int(len(y)),
            "positives": int(y.sum()), "l2": l2}


def predict_confidence(mapping, X):
    """confidence.core for rows of signals (in the mapping's feature order): 1 minus the chance of a large error."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    fill = np.asarray(mapping["fill"])
    X = np.where(np.isnan(X), fill, X)
    Z = (X - np.asarray(mapping["mean"])) / np.asarray(mapping["std"])
    return 1.0 - _sigmoid(mapping["intercept"] + Z @ np.asarray(mapping["weights"]))


def cross_fitted_confidence(table, features=FEATURES, l2=L2):
    """Cross-fitted by fold: for each fold, fit on the other folds and score this one.

    table: with `features`, `fold` and `large_error`. Returns (confidence per row, in the table's order,
    {fold: mapping}). No case is scored by a mapping whose training cases include it.
    """
    X = table[list(features)].astype(float).to_numpy()
    y = table["large_error"].astype(bool).to_numpy()
    folds = table["fold"].astype(int).to_numpy()
    out = np.full(len(table), np.nan)
    mappings = {}
    for fold in sorted(set(folds.tolist())):
        train, test = folds != fold, folds == fold
        mapping = fit_mapping(X[train], y[train], l2, features)
        out[test] = predict_confidence(mapping, X[test])
        mappings[str(fold)] = mapping
    return out, mappings


def confidence_core(mapping, at_risk_share, nnunet_ml, threshold_ml, tmax6=None):
    """confidence.core for one case at inference, from a saved mapping and the case's inputs. Use the chosen
    version's mapping (load_default_mapping, v2) unless comparing versions; v2 does not need `tmax6`.

    For held-out (Phase 4) cases only. A development case was among the final mapping's training cases, so use its
    cross-fitted value instead (development_confidence)."""
    row = signal_row(at_risk_share, nnunet_ml, threshold_ml, tmax6)
    return float(predict_confidence(mapping, [[row[f] for f in mapping["features"]]])[0])


def case_confidence(case, mapping, nnunet_ml, threshold_ml, margin_mm=MARGIN_MM):
    """The contract's confidence for one held-out (Phase 4) case: {"core": ..., "penumbra": ...}.

    case: a loaded isles24.Case (tmax, cbf and ncct arrays). mapping: load_default_mapping (mapping_v2.json).
    nnunet_ml, threshold_ml: the case's core volumes from nnU-Net and the threshold method. Development cases use
    development_confidence instead, because the mapping was fitted on them.
    """
    signals = case_signals(case, margin_mm)
    core = confidence_core(mapping, signals["at_risk_no_data_share"], nnunet_ml, threshold_ml, signals["tmax6_ml"])
    return {"core": core, "penumbra": signals["confidence_penumbra"]}


def development_confidence(per_case_csv, version=CHOSEN_VERSION, folds=None):
    """The cross-fitted confidence of every development case, from a confidence run's per_case.csv (EXP-31's):
    {case: {"core": confidence_core_<version>, "penumbra": confidence_penumbra}}.

    These are the honest values for development cases: each core value comes from a mapping that never saw the
    case. Raises ValueError, with counts only, for a missing column, a repeated case, or a value that is missing or
    outside 0 to 1, and HeldoutError for any held-out case before Phase 4 (splits.check_ids). `folds` is for tests.
    """
    import pandas as pd

    from strokecopilot.data.splits import check_ids

    if version not in VERSIONS:
        raise ValueError(f"Unknown version {version!r}; expected one of {sorted(VERSIONS)}.")
    core = f"confidence_core_{version}"
    table = pd.read_csv(per_case_csv)
    missing = [c for c in ("case", core, "confidence_penumbra") if c not in table.columns]
    if missing:
        raise ValueError(f"per_case.csv lacks columns {missing}.")
    if table["case"].duplicated().any():
        raise ValueError(f"per_case.csv repeats {int(table['case'].duplicated().sum())} cases.")
    values = table[[core, "confidence_penumbra"]].apply(pd.to_numeric, errors="coerce")
    bad = (values.isna() | (values < 0) | (values > 1)).any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} cases have a confidence that is missing, not a number or outside 0 to 1.")
    cases = table["case"].astype(str).tolist()
    check_ids(cases, folds)
    return {case: {"core": float(c), "penumbra": float(p)}
            for case, c, p in zip(cases, values[core], values["confidence_penumbra"])}


def save_mapping(path, mapping):
    Path(path).write_text(json.dumps(mapping, indent=1))
    return Path(path)


def load_mapping(path):
    return json.loads(Path(path).read_text())


def load_default_mapping(run_dir):
    """The chosen version's final mapping (`mapping_v2.json`) from a confidence run folder, such as EXP-31's.
    Raises if the file holds a different version's signals."""
    mapping = load_mapping(Path(run_dir) / DEFAULT_MAPPING_FILE)
    if mapping["features"] != list(FEATURES):
        raise ValueError(f"{DEFAULT_MAPPING_FILE} has signals {mapping['features']}, not {CHOSEN_VERSION}'s.")
    return mapping


# ---------------------------------------------------------------------------- Bands and the checks


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
    10 mL or more, cases on the wrong side of each cut-off and of any, and under-calls. Counts and summaries only."""
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
            "undercalls": int(rows["undercall"].sum()),
        }
        for cutoff, column in zip(DECISION_CUTOFFS_ML, CROSSINGS):
            out[f"wrong_side_{cutoff:g}"] = int(rows[column].astype(bool).sum())
        return out

    result = {band: block(table[table["band"] == band]) for band in BANDS}
    result["all"] = block(table)
    return result


def done_check(band_rows, table, confidence_col="confidence_core"):
    """The task's "done when": low confidence goes with large errors. Passes when the low band has more cases on the
    wrong side of a cut-off than the high band, and more errors of 10 mL or more. Also the Spearman correlation
    between confidence and volume error (negative is the right direction) and the under-calls in the high band."""
    from scipy import stats

    low, high = band_rows["low"], band_rows["high"]
    key = f"error_{LARGE_ERROR_ML:g}ml_or_more"
    rho = stats.spearmanr(table[confidence_col], table["nnunet_abs_error_ml"]).correlation
    return {
        "wrong_side_low_vs_high": [low["wrong_side_any"], high["wrong_side_any"]],
        "errors_10ml_low_vs_high": [low[key], high[key]],
        "undercalls_in_high_band": high["undercalls"],
        "undercalls_total": band_rows["all"]["undercalls"],
        "spearman_confidence_vs_abs_error": _num(rho),
        "passes": bool(low["wrong_side_any"] > high["wrong_side_any"] and low[key] > high[key]),
    }


def signals_alone(table, signals=ALL_SIGNALS):
    """Each signal on its own: its weight when fitted alone on all cases (positive means more risk, so lower
    confidence), and its Spearman correlation with the volume error. Counts and summaries only."""
    from scipy import stats

    out = {}
    y = table["large_error"].astype(bool).to_numpy()
    for signal in signals:
        x = table[[signal]].astype(float).to_numpy()
        mapping = fit_mapping(x, y, features=(signal,))
        ok = ~np.isnan(x[:, 0])
        rho = stats.spearmanr(x[ok, 0], table.loc[ok, "nnunet_abs_error_ml"]).correlation
        out[signal] = {"weight_alone": _num(mapping["weights"][0]),
                       "raises_confidence": bool(mapping["weights"][0] < 0),
                       "spearman_with_abs_error": _num(rho)}
    return out


def compare_versions(table, versions=VERSIONS):
    """Cross-fit every version on the same table and report each side by side: band table, done check, fold
    mappings and a final mapping fitted on all cases. Returns (results by version, confidences by version)."""
    results, confidences = {}, {}
    y = table["large_error"].astype(bool).to_numpy()
    for name, features in versions.items():
        scored = table.copy()
        scored["confidence_core"], mappings = cross_fitted_confidence(table, features)
        rows = band_table(scored)
        final = fit_mapping(table[list(features)].astype(float).to_numpy(), y, features=features)
        results[name] = {"features": list(features), "bands": rows, "done_check": done_check(rows, scored),
                         "fold_mappings": mappings, "final_mapping": final}
        confidences[name] = scored["confidence_core"].to_numpy()
    return results, confidences


def slab_check(table):
    """Before/after the slab and footprint fix: the no-data share over all plain-CT tissue against over the
    measurement area, and the slice counts. Medians and ranges only."""
    def summary(series):
        s = series.dropna()
        return {"median": _num(s.median()), "min": _num(s.min()), "max": _num(s.max())} if len(s) else None
    return {
        "tissue_no_data_share": summary(table["tissue_no_data_share"]),
        "area_no_data_share": summary(table["brain_no_data_share"]),
        "tissue_slices": summary(table["tissue_slices"].astype(float)),
        "area_slices": summary(table["area_slices"].astype(float)),
        "cases_area_share_over_0_1": int((table["brain_no_data_share"] > 0.1).sum()),
    }


# ---------------------------------------------------------------------------- Run record


def write_config(run_dir, git_commit, experiment, inputs, notes=""):
    """config.json for the confidence run, written before it and never overwritten. A placeholder experiment ID
    is refused. Records the design, the versions, the signals and the measures. No case IDs."""
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
        "design": "approved by Parsa, 15 Nov 2026; Lorna picks the version",
        "measurement_area": "plain-CT tissue inside the perfusion region: the slices from the first to the last with "
                            "perfusion values, and in each the perfusion maps' area with holes filled",
        "signals": {
            "at_risk_no_data_share": f"share of the Tmax > {PENUMBRA_TMAX_S:g} s area plus a {MARGIN_MM:g} mm margin, "
                                     "inside the measurement area, with no or unreliable Tmax; the whole area if "
                                     "there is no Tmax > 6 s area",
            "log_pred_ml": "log(1 + nnU-Net predicted mL)",
            "log_disagreement": "|log(1 + nnU-Net mL) - log(1 + threshold mL)|",
            "log_cutoff_distance": "|log(1 + nnU-Net mL) - log(1 + nearest cut-off)|",
            "log_tmax6_ml": "log(1 + mL of reliable Tmax > 6 s)",
        },
        "versions": {name: list(features) for name, features in VERSIONS.items()},
        "mapping": f"logistic regression on standardised signals, ridge {L2:g} on the weights; "
                   "confidence.core = 1 - P(large error); cross-fitted by fold (fit on 4, score the 5th); "
                   "final fit per version on all development cases saved as mapping_<version>.json",
        "large_error": "nnU-Net volume on the wrong side of at least one of "
                       + ", ".join(f"{c:g}" for c in DECISION_CUTOFFS_ML) + " mL",
        "second_measure": f"absolute volume error of {LARGE_ERROR_ML:g} mL or more",
        "undercall": "nnU-Net volume smaller than the true volume and on the wrong side of at least one cut-off",
        "bands": "three equal bands by confidence.core: low, middle, high",
        "penumbra": "confidence.penumbra = 1 - share of the measurement area with no or unreliable Tmax; "
                    "data-quality score only, ISLES'24 has no penumbra label",
        "not_used": "fold-ensemble disagreement (each development case was trained on by 4 of 5 models); "
                    "the true infarct as an input",
        "outputs_policy": "counts and summaries only, never case IDs",
        "notes": notes,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1))
    return config


def build_metrics(table, config, git_commit, versions, alone, slab):
    """metrics.json for the confidence run: every version's results, the signals-alone check, the slab check and
    confidence.penumbra. Raises if any case ID appears anywhere in it."""
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
        "undercall_cases": int(table["undercall"].sum()),
        "versions": versions,
        "signals_alone": alone,
        "slab_check": slab,
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
