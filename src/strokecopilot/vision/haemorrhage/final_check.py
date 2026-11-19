"""Vision B's final figures on the RSNA test split, opened once (DEC-22, DEC-31 §1 item 3; Phase 4).

The model is final: EXP-6's `best.pt`, its top-2 aggregation and Platt calibration, and DEC-28's threshold 0.175 on
the calibrated scale (`infer.contract_calibrator`). Nothing is chosen here. DEC-31 fixed what is reported:
- the scan AUC;
- sensitivity and specificity at 0.175, with patient-bootstrap intervals (whole patients resampled, as in
  operating_point.patient_bootstrap; a patient's scans are not independent);
- sensitivity per bleed subtype;
- the expected calibration error (ECE) of the calibrated scan probability.

Descriptive numbers that come for free are added: the slice AUC, Wilson intervals, counts, log loss and the reliability
bins (for the figure). They are labelled as extras.

`report` runs the same way on any split. The Kaggle notebook (25) runs it on validation first, where EXP-6 and DEC-28 are
known, and opens the test split only if that check passes. So the one opening is not spent on a broken pipeline.

Input: one row per slice with series_id, patient_id, any, the five subtype labels and prob (EXP-6's slice probability).
Everything returned is counts, rates and thresholds: no slice, scan or patient ID (docs/DATA_USE.md).
"""

import numpy as np

from strokecopilot.vision.haemorrhage import aggregate as agg
from strokecopilot.vision.haemorrhage import operating_point as op
from strokecopilot.vision.haemorrhage.stats import wilson

SUBTYPES = ("epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural")
BOOTSTRAP = op.BOOTSTRAP  # 2000, as EXP-17 and EXP-49
SEED = op.SEED
# EXP-6 and DEC-28 on validation, for the notebook's check before the test split is opened.
VAL_SCAN_AUC = 0.977
VAL_AUC_TOLERANCE = 0.005
VAL_SENSITIVITY_FLOOR = 0.955  # DEC-28's full-split threshold gives 96% sensitivity on validation by construction


def scan_frame(slices, calibrator):
    """One row per scan: series_id, patient_id, label, calibrated probability and 0/1 per subtype."""
    if (slices.groupby("series_id")["patient_id"].nunique() > 1).any():
        raise ValueError("A scan belongs to more than one patient.")
    scans = agg.scan_scores(slices, calibrator.method, calibrator.k)
    per_scan = slices.groupby("series_id").agg(patient_id=("patient_id", "first"), **{s: (s, "max") for s in SUBTYPES})
    scans = scans.merge(per_scan.reset_index(), on="series_id", how="left", validate="one_to_one")
    return scans.assign(prob=calibrator.apply(scans["score"].to_numpy()))


def _weighted_auc(labels, scores, weights):
    """AUC with a non-negative integer weight per scan (a bootstrap replicate). Ties count one half."""
    order = np.argsort(scores, kind="mergesort")
    s, y, w = scores[order], labels[order], weights[order]
    # group equal scores so ties share their pairs
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    pos = np.add.reduceat(w * y, starts)
    neg = np.add.reduceat(w * ~y, starts)
    neg_below = np.cumsum(neg) - neg
    total_pos, total_neg = pos.sum(), neg.sum()
    if total_pos == 0 or total_neg == 0:
        return np.nan
    return float((pos * (neg_below + 0.5 * neg)).sum() / (total_pos * total_neg))


def patient_bootstrap(scans, called, n=BOOTSTRAP, seed=SEED):
    """95% percentile intervals for scan AUC, sensitivity, specificity and each subtype's sensitivity, resampling
    whole patients. One set of replicates serves every quantity, so they are drawn from the same resamples."""
    patients, codes = np.unique(scans["patient_id"].to_numpy(), return_inverse=True)
    labels = scans["label"].to_numpy().astype(bool)
    scores = scans["prob"].to_numpy(dtype=float)
    called = np.asarray(called).astype(bool)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=n)
    out = {"auc": [], "sensitivity": [], "specificity": [], **{s: [] for s in SUBTYPES}}
    sub = {s: scans[s].to_numpy() == 1 for s in SUBTYPES}
    with np.errstate(invalid="ignore", divide="ignore"):
        for weights_by_patient in draws:
            w = weights_by_patient[codes].astype(float)
            out["auc"].append(_weighted_auc(labels, scores, w))
            out["sensitivity"].append((w * (called & labels)).sum() / (w * labels).sum())
            out["specificity"].append((w * (~called & ~labels)).sum() / (w * ~labels).sum())
            for s in SUBTYPES:
                out[s].append((w * (called & sub[s])).sum() / (w * sub[s]).sum() if sub[s].any() else np.nan)

    def ci(values):
        v = np.asarray(values, dtype=float)
        if np.isnan(v).all():
            return [None, None]
        return [op._r(np.nanpercentile(v, 2.5)), op._r(np.nanpercentile(v, 97.5))]

    return {"replicates": n, "seed": seed, "scan_auc_ci95": ci(out["auc"]), "sensitivity_ci95": ci(out["sensitivity"]),
            "specificity_ci95": ci(out["specificity"]), "subtype_sensitivity_ci95": {s: ci(out[s]) for s in SUBTYPES}}


def report(slices, calibrator, n=BOOTSTRAP, seed=SEED):
    """DEC-31's figures (and the labelled extras) for one split. `calibrator` carries DEC-28's threshold."""
    missing = {"series_id", "patient_id", "any", "prob", *SUBTYPES} - set(slices.columns)
    if missing:
        raise ValueError(f"Slice table lacks {sorted(missing)}.")
    scans = scan_frame(slices, calibrator)
    labels, probs = scans["label"].to_numpy().astype(bool), scans["prob"].to_numpy(dtype=float)
    called = probs >= calibrator.threshold
    c = op.rates(op.counts(called, labels))
    boot = patient_bootstrap(scans, called, n, seed)
    subtype = {}
    for s in SUBTYPES:
        mask = scans[s].to_numpy() == 1
        k, total = int(called[mask].sum()), int(mask.sum())
        subtype[s] = {"bleeds": total, "caught": k, "sensitivity": op._r(k / total) if total else None,
                      "wilson_ci95": [op._r(v) for v in wilson(k, total)],
                      "patient_bootstrap_ci95": boot["subtype_sensitivity_ci95"][s]}
    return {
        "threshold_calibrated": calibrator.threshold,
        "aggregation": {"method": calibrator.method, "k": calibrator.k, "calibration": calibrator.kind},
        "patients": int(scans["patient_id"].nunique()), "scans": int(len(scans)), "slices": int(len(slices)),
        "scan_bleed_rate": op._r(labels.mean()),
        "preregistered": {
            "scan_auc": op._r(agg.roc_auc(labels, probs)), "scan_auc_ci95": boot["scan_auc_ci95"],
            "sensitivity": c["sensitivity"], "sensitivity_ci95": boot["sensitivity_ci95"],
            "specificity": c["specificity"], "specificity_ci95": boot["specificity_ci95"],
            "subtype_sensitivity": subtype,
            "scan_ece": op._r(agg.ece(probs, labels)),
            "interval": f"patient bootstrap, {n} replicates, seed {seed}, 95% percentile",
        },
        "extras": {
            "counts": {k: c[k] for k in ("tp", "fn", "tn", "fp", "bleeds", "no_bleeds")},
            "sensitivity_wilson_ci95": c["sensitivity_ci95"], "specificity_wilson_ci95": c["specificity_ci95"],
            "missed_bleeds_per_100_bleeds": c["missed_bleeds_per_100_bleeds"],
            "false_alarms_per_100_no_bleed": c["false_alarms_per_100_no_bleed"],
            "slice_auc": op._r(agg.roc_auc(slices["any"], slices["prob"])),
            "slice_bleed_rate": op._r(slices["any"].mean()),
            "scan_log_loss": op._r(agg.log_loss(probs, labels)),
            "reliability": agg.reliability(probs, labels),
            "called_positive_share": op._r(called.mean()),
        },
    }


def validation_check(val_report, expected_auc=VAL_SCAN_AUC, tolerance=VAL_AUC_TOLERANCE,
                     sensitivity_floor=VAL_SENSITIVITY_FLOOR):
    """Problems with the validation pass, as sentences; empty means the test split may be opened."""
    pre = val_report["preregistered"]
    problems = []
    if pre["scan_auc"] is None or abs(pre["scan_auc"] - expected_auc) > tolerance:
        problems.append(f"validation scan AUC {pre['scan_auc']} is not EXP-6's {expected_auc} (±{tolerance})")
    if pre["sensitivity"] is None or pre["sensitivity"] < sensitivity_floor:
        problems.append(f"validation sensitivity at DEC-28's threshold is {pre['sensitivity']}, below {sensitivity_floor}")
    if any(v is None for v in pre["sensitivity_ci95"] + pre["specificity_ci95"]):
        problems.append("a bootstrap interval is empty")
    return problems


def summary_lines(rep):
    """Plain lines for the notebook and the CHANGELOG. Numbers only."""
    pre, ex = rep["preregistered"], rep["extras"]
    lines = [
        f"{rep['patients']:,} patients, {rep['scans']:,} scans (bleed rate {rep['scan_bleed_rate']}), {rep['slices']:,} slices.",
        f"Scan AUC {pre['scan_auc']} [{pre['scan_auc_ci95'][0]}, {pre['scan_auc_ci95'][1]}]; ECE {pre['scan_ece']}; "
        f"slice AUC {ex['slice_auc']} (extra).",
        f"At {rep['threshold_calibrated']}: sensitivity {pre['sensitivity']} [{pre['sensitivity_ci95'][0]}, "
        f"{pre['sensitivity_ci95'][1]}], specificity {pre['specificity']} [{pre['specificity_ci95'][0]}, "
        f"{pre['specificity_ci95'][1]}]; {ex['missed_bleeds_per_100_bleeds']} missed per 100 bleeds, "
        f"{ex['false_alarms_per_100_no_bleed']} false alarms per 100 non-bleeds.",
    ]
    for s, r in pre["subtype_sensitivity"].items():
        lo, hi = r["patient_bootstrap_ci95"]
        lines.append(f"  {s}: {r['caught']}/{r['bleeds']} = {r['sensitivity']} [{lo}, {hi}]")
    return lines


def ids_in(text, ids):
    """True if any of ids, or the RSNA ID prefix, appears in text."""
    return "ID_" in text or any(i in text for i in ids)
