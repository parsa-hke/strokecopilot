"""Vision B operating point: an honest threshold by cross-fitting on validation patients (plan §8.2, Phase 3).

EXP-6 chose its threshold (0.278) for 95% sensitivity on the validation split and measured it on the same
split, so its specificity is optimistic. Here the validation patients are split into two halves with a
fixed seed. The threshold is chosen on one half and measured on the other, then the halves swap. The two
held-out results are pooled. The RSNA test split stays closed (`rsna_split.rows` refuses it without
ALLOW_RSNA_TEST=1); nothing here reads it.

Thresholds are set on the raw scan score (EXP-6: top-2 slice mean). Platt calibration is monotonic, so a
threshold on the calibrated probability makes exactly the same calls, and the calibrator having been fit
on the full validation split does not leak into the held-out halves. The proposed threshold is converted
to the calibrated scale at the end, because that is what `calibration.json` and the contract use.

Halves are drawn by patient, so no patient's scans fall on both sides, and within two groups (patients with
a bleeding scan and patients without), so each half gets a similar share of bleeds. Patient IDs stay in
memory on Kaggle; every result here is counts, rates and thresholds (docs/DATA_USE.md).
"""

import math

import numpy as np
import pandas as pd

from strokecopilot.vision.haemorrhage import aggregate as agg

SEED = 20261108
REPEAT_SEEDS = tuple(range(SEED, SEED + 20))
TARGET = agg.TARGET_SENSITIVITY  # 0.95, as in the brief
SWEEP_TARGETS = (0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99)
CANDIDATE_TARGETS = (0.95, 0.96, 0.97, 0.98, 0.99)
BOOTSTRAP = 2000
Z95 = 1.959964


def scan_table(slices, method, k):
    """One row per scan: series_id, patient_id, label and the raw scan score (EXP-6's aggregation)."""
    scans = agg.scan_scores(slices, method, k)
    patients = slices.groupby("series_id")["patient_id"].first()
    if (slices.groupby("series_id")["patient_id"].nunique() > 1).any():
        raise ValueError("A scan belongs to more than one patient.")
    return scans.assign(patient_id=scans["series_id"].map(patients))


def patient_halves(patients, labels, seed=SEED):
    """0 or 1 per scan. Whole patients per half, stratified by whether the patient has any bleeding scan."""
    frame = pd.DataFrame({"patient": np.asarray(patients), "label": np.asarray(labels).astype(int)})
    per_patient = frame.groupby("patient")["label"].max().sort_index()
    rng = np.random.default_rng(seed)
    half = {}
    for _, group in per_patient.groupby(per_patient):
        ids = rng.permutation(group.index.to_numpy())
        # Alternate within each group; the group's first half-slot goes to a random side, so odd groups
        # do not always favour half 0.
        start = int(rng.integers(2))
        half.update({pid: (start + i) % 2 for i, pid in enumerate(ids)})
    return frame["patient"].map(half).to_numpy()


def counts(called, labels):
    called, labels = np.asarray(called).astype(bool), np.asarray(labels).astype(bool)
    return {"tp": int((called & labels).sum()), "fn": int((~called & labels).sum()),
            "tn": int((~called & ~labels).sum()), "fp": int((called & ~labels).sum())}


def wilson(k, n, z=Z95):
    """Wilson score interval for a proportion. (None, None) when n is 0."""
    if n == 0:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def rates(c):
    """Sensitivity and specificity with Wilson intervals, and the same as plain counts per 100 scans."""
    pos, neg = c["tp"] + c["fn"], c["tn"] + c["fp"]
    sens, spec = (c["tp"] / pos if pos else None), (c["tn"] / neg if neg else None)
    return {**c, "bleeds": pos, "no_bleeds": neg,
            "sensitivity": _r(sens), "sensitivity_ci95": [_r(v) for v in wilson(c["tp"], pos)],
            "specificity": _r(spec), "specificity_ci95": [_r(v) for v in wilson(c["tn"], neg)],
            "missed_bleeds_per_100_bleeds": _r(None if sens is None else 100 * (1 - sens), 2),
            "false_alarms_per_100_no_bleed": _r(None if spec is None else 100 * (1 - spec), 2)}


def _r(value, digits=4):
    return None if value is None else round(float(value), digits)


def cross_fit(scores, labels, patients, target=TARGET, seed=SEED):
    """Choose on one half, measure on the other, swap, pool. Returns (report, held-out calls per scan)."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    halves = patient_halves(patients, labels, seed)
    called = np.zeros(len(scores), dtype=bool)
    folds = []
    for measure_half in (0, 1):
        choose, measure = halves != measure_half, halves == measure_half
        chosen = agg.threshold_for_sensitivity(scores[choose], labels[choose], target)
        called[measure] = scores[measure] >= chosen["threshold"]
        folds.append({"chosen_on": 1 - measure_half, "measured_on": measure_half, "threshold_raw": _r(chosen["threshold"], 6),
                      "scans_chosen_on": int(choose.sum()), "sensitivity_where_chosen": _r(chosen["sensitivity"]),
                      "specificity_where_chosen": _r(chosen["specificity"]),
                      **rates(counts(called[measure], labels[measure]))})
    report = {"seed": seed, "target": target, "patients_per_half": [int(np.unique(np.asarray(patients)[halves == h]).size) for h in (0, 1)],
              "folds": folds, "pooled": rates(counts(called, labels))}
    return report, called


def patient_bootstrap(called, labels, patients, n=BOOTSTRAP, seed=SEED):
    """95% percentile intervals for pooled sensitivity and specificity, resampling whole patients.

    Scans of one patient are not independent, so this is the interval to quote next to the Wilson one.
    """
    frame = pd.DataFrame({"patient": np.asarray(patients), "called": np.asarray(called).astype(bool),
                          "label": np.asarray(labels).astype(bool)})
    frame["tp"] = frame["called"] & frame["label"]
    frame["tn"] = ~frame["called"] & ~frame["label"]
    per = frame.groupby("patient").agg(tp=("tp", "sum"), pos=("label", "sum"), tn=("tn", "sum"), n=("label", "size"))
    per["neg"] = per["n"] - per["pos"]
    arrays = per[["tp", "pos", "tn", "neg"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(per), np.full(len(per), 1 / len(per)), size=n)
    tp, pos, tn, neg = (weights @ arrays).T
    with np.errstate(invalid="ignore", divide="ignore"):
        sens, spec = tp / pos, tn / neg
    return {"replicates": n, "sensitivity_ci95": [_r(np.nanpercentile(sens, 2.5)), _r(np.nanpercentile(sens, 97.5))],
            "specificity_ci95": [_r(np.nanpercentile(spec, 2.5)), _r(np.nanpercentile(spec, 97.5))]}


def repeat(scores, labels, patients, target=TARGET, seeds=REPEAT_SEEDS):
    """The pooled result over many seeds: how much one random split can move the answer."""
    pooled = [cross_fit(scores, labels, patients, target, s)[0]["pooled"] for s in seeds]
    sens = np.array([p["sensitivity"] for p in pooled])
    spec = np.array([p["specificity"] for p in pooled])
    spread = lambda v: {"min": _r(v.min()), "median": _r(np.median(v)), "max": _r(v.max())}  # noqa: E731
    return {"target": target, "seeds": len(seeds), "sensitivity": spread(sens), "specificity": spread(spec),
            "seeds_below_target": int((sens < target).sum())}


def full_split_threshold(scores, labels, target, calibrator=None):
    """The threshold for target sensitivity on the whole validation split, raw and calibrated."""
    point = agg.threshold_for_sensitivity(scores, labels, target)
    out = {"target": target, "threshold_raw": _r(point["threshold"], 6),
           "same_split_sensitivity": _r(point["sensitivity"]), "same_split_specificity": _r(point["specificity"])}
    if calibrator is not None:
        out["threshold_calibrated"] = _r(float(calibrator.apply(np.array([point["threshold"]]))[0]), 6)
    return out


def sweep(scores, labels, patients, targets=SWEEP_TARGETS, seed=SEED, seeds=REPEAT_SEEDS, calibrator=None):
    """For each sensitivity target: the full-split threshold, the cross-fitted result and its spread."""
    rows = []
    for target in targets:
        report, _ = cross_fit(scores, labels, patients, target, seed)
        spread = repeat(scores, labels, patients, target, seeds)
        rows.append({**full_split_threshold(scores, labels, target, calibrator),
                     "cross_fitted": {k: report["pooled"][k] for k in ("sensitivity", "sensitivity_ci95", "specificity", "specificity_ci95",
                                                                        "missed_bleeds_per_100_bleeds", "false_alarms_per_100_no_bleed")},
                     "median_over_seeds": {"sensitivity": spread["sensitivity"]["median"], "specificity": spread["specificity"]["median"]},
                     "seeds_below_target": spread["seeds_below_target"]})
    return rows


def propose(sweep_rows, target=TARGET, candidates=CANDIDATE_TARGETS):
    """The proposal, by a rule fixed before the run.

    Median rule (proposed): the lowest candidate target whose cross-fitted sensitivity, median over seeds,
    is at least `target`. Its full-split threshold is the proposal.
    Strict rule (for comparison): the lowest candidate whose fixed-seed Wilson lower bound is at least
    `target`. It gives up more specificity; Parsa can choose it instead.
    Either rule can fail (no candidate passes); then the proposal is None and the report says so.
    """
    by_target = {round(row["target"], 4): row for row in sweep_rows}
    rows = [by_target[round(c, 4)] for c in candidates if round(c, 4) in by_target]

    def first(passes):
        for row in rows:
            if passes(row):
                return {"rule_target": row["target"], **{k: row.get(k) for k in ("threshold_raw", "threshold_calibrated")},
                        "cross_fitted": row["cross_fitted"], "median_over_seeds": row["median_over_seeds"]}
        return None

    median = first(lambda r: r["median_over_seeds"]["sensitivity"] >= target)
    strict = first(lambda r: r["cross_fitted"]["sensitivity_ci95"][0] is not None and r["cross_fitted"]["sensitivity_ci95"][0] >= target)
    return {"target_sensitivity": target, "proposed": median, "strict_alternative": strict,
            "rule": "lowest full-split target whose cross-fitted sensitivity (median over seeds) is at least the target"}


def check_monotonic(calibrator):
    """The equivalence of raw and calibrated thresholds needs a positive temperature."""
    if not calibrator.temperature > 0:
        raise ValueError(f"Calibration temperature {calibrator.temperature} is not positive, so calibrated and raw thresholds differ.")


def run(slices, calibrator, seed=SEED, seeds=REPEAT_SEEDS, targets=SWEEP_TARGETS, target=TARGET, bootstrap=BOOTSTRAP):
    """Everything Task 1 reports, from validation slice predictions. Numbers only, no IDs.

    slices needs series_id, patient_id, any and prob (one row per validation slice).
    """
    check_monotonic(calibrator)
    scans = scan_table(slices, calibrator.method, calibrator.k)
    scores, labels, patients = scans["score"].to_numpy(), scans["label"].to_numpy(), scans["patient_id"].to_numpy()
    probs = calibrator.apply(scores)
    report, called = cross_fit(scores, labels, patients, target, seed)
    rows = sweep(scores, labels, patients, targets, seed, seeds, calibrator)
    return {
        "scans": int(len(scans)), "patients": int(scans["patient_id"].nunique()), "bleed_scans": int(labels.sum()),
        "scan_auc": agg.roc_auc(labels, probs),
        "exp6_threshold": {"threshold_calibrated": calibrator.threshold,
                           **{f"same_split_{k}": _r(v) for k, v in agg.operating_point(probs, labels, calibrator.threshold).items()}},
        "main": {**report, "patient_bootstrap": patient_bootstrap(called, labels, patients, bootstrap, seed)},
        "repeats": repeat(scores, labels, patients, target, seeds),
        "sweep": rows,
        "proposal": propose(rows, target),
    }


def reasoning_text(metrics, isles_fp_rate=None, isles_threshold=None):
    """A first draft of the operating-point reasoning, filled with this run's numbers. Edit before use."""
    proposal = metrics["proposal"]["proposed"]
    exp6 = metrics["exp6_threshold"]
    main = metrics["main"]["pooled"]
    lines = [
        "A missed bleed is the worst failure in the system: thrombolysis given to a patient who is bleeding can kill them, "
        "while a false alarm holds treatment back only until a clinician reads the scan. So the threshold is set for at "
        f"least {metrics['proposal']['target_sensitivity']:.0%} sensitivity, and specificity is what we give up for it.",
        f"EXP-6's threshold ({exp6['threshold_calibrated']:.3f}) was chosen and measured on the same {metrics['scans']:,} validation scans; "
        f"recomputed here, it gives sensitivity {exp6['same_split_sensitivity']:.3f} and specificity {exp6['same_split_specificity']:.3f} "
        "on that split, which flatters it. Choosing the "
        f"threshold on one half of the patients and measuring it on the other (seed {metrics['main']['seed']}, both directions pooled) "
        f"gives sensitivity {main['sensitivity']:.3f} (95% CI {main['sensitivity_ci95'][0]:.3f}-{main['sensitivity_ci95'][1]:.3f}) and "
        f"specificity {main['specificity']:.3f} ({main['specificity_ci95'][0]:.3f}-{main['specificity_ci95'][1]:.3f}). Over "
        f"{metrics['repeats']['seeds']} seeds, held-out sensitivity ranged {metrics['repeats']['sensitivity']['min']:.3f}-"
        f"{metrics['repeats']['sensitivity']['max']:.3f} and fell below the target in {metrics['repeats']['seeds_below_target']}.",
    ]
    if proposal:
        cf = proposal["cross_fitted"]
        threshold = proposal.get("threshold_calibrated") or proposal["threshold_raw"]
        lines.append(
            f"Proposed threshold: {threshold:.3f} (calibrated), the full-split threshold for {proposal['rule_target']:.0%} sensitivity, "
            f"the lowest target whose held-out sensitivity reaches {metrics['proposal']['target_sensitivity']:.0%} (median over seeds). "
            f"Held out, it misses about {cf['missed_bleeds_per_100_bleeds']:.1f} of every 100 bleeds and wrongly flags about "
            f"{cf['false_alarms_per_100_no_bleed']:.1f} of every 100 scans without a bleed (RSNA validation)."
        )
        strict = metrics["proposal"]["strict_alternative"]
        if strict and strict["rule_target"] != proposal["rule_target"]:
            lines.append(
                f"A stricter choice, requiring the lower 95% bound to reach the target, is {strict.get('threshold_calibrated') or strict['threshold_raw']:.3f}: "
                f"about {strict['cross_fitted']['false_alarms_per_100_no_bleed']:.1f} false alarms per 100 scans without a bleed."
            )
    else:
        lines.append("No candidate threshold reached the target sensitivity held out; the proposal is open for Parsa.")
    if isles_fp_rate is not None:
        lines.append(
            f"On the deployment population, ISLES'24 patients who went on to treatment, Vision B flagged {isles_fp_rate:.1%} of scans "
            f"at {isles_threshold if isles_threshold is not None else 'the EXP-6 threshold'} (EXP-16): each of these would have had "
            "treatment held back until a clinician overrode the flag. That rate, not the RSNA one, is the best estimate of how often "
            "the gate delays treatment in practice."
        )
    return "\n\n".join(lines)


def tradeoff_figure(metrics, out_png=None):
    """Cross-fitted sensitivity and specificity against the sensitivity target, with 95% intervals."""
    import matplotlib

    if out_png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = metrics["sweep"]
    x = [r["target"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, colour, label in (("sensitivity", "#c44e52", "Sensitivity (held-out)"), ("specificity", "#4c72b0", "Specificity (held-out)")):
        y = [r["cross_fitted"][key] for r in rows]
        lo = [y_ - r["cross_fitted"][f"{key}_ci95"][0] for y_, r in zip(y, rows)]
        hi = [r["cross_fitted"][f"{key}_ci95"][1] - y_ for y_, r in zip(y, rows)]
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", color=colour, capsize=3, label=label)
    ax.axhline(metrics["proposal"]["target_sensitivity"], color="grey", linestyle=":", linewidth=1)
    proposed = metrics["proposal"]["proposed"]
    if proposed:
        ax.axvline(proposed["rule_target"], color="black", linestyle="--", linewidth=1,
                   label=f"proposed: target {proposed['rule_target']:g}, threshold {proposed.get('threshold_calibrated') or proposed['threshold_raw']:.3f}")
    lows = [r["cross_fitted"][f"{k}_ci95"][0] for r in rows for k in ("sensitivity", "specificity") if r["cross_fitted"][f"{k}_ci95"][0] is not None]
    bottom = max(0.0, min([0.8, *lows]) - 0.03)
    ax.set(xlabel="Sensitivity target used to choose the threshold", ylabel="Held-out rate", ylim=(bottom, 1.005),
           title=f"Vision B operating point, cross-fitted on {metrics['scans']:,} validation scans")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        return out_png
    return fig
