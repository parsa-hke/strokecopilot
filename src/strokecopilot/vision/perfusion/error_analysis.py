"""Vision A error analysis: where and why the infarct model fails (plan §7 Phase 3, §8.1; brief docs/team/lorna_vision_a.md).

Built one step at a time. Functions not built yet raise NotImplementedError.

1. Skeleton and synthetic tests (tests/test_error_analysis.py).
2. write_config(): runs/EXP-15_visionA_error_analysis/config.json, written before the run
   (docs/team/README.md §4). join_tables(): the nnU-Net (EXP-14) and threshold (EXP-1) per-case
   tables side by side.
3. case_features(): per-case possible causes of failure, from a loaded Case: size, occlusion site,
   territory, onset to imaging, and how much of the infarct has no Tmax data.
4. (Colab) run case_features() over the development cases and join it to the step-2 table.
5. group_summary(): scores per group of a feature, as counts and summaries.
6. (Colab, then docs) images of failed cases and the write-up.

Outputs that leave Drive carry counts and summaries only, never case IDs (docs/team/README.md §3).
Volume error matters more than Dice, because the treatment rules are volume cut-offs
(evaluate.DECISION_CUTOFFS_ML).
"""

import json
from pathlib import Path

import numpy as np

from strokecopilot.config import ISLES24_SLIM
from strokecopilot.nlp.notes.generate import minutes
from strokecopilot.vision.perfusion import volumes
from strokecopilot.vision.perfusion.evaluate import DECISION_CUTOFFS_ML
from strokecopilot.vision.perfusion.pod_job import FOLDS_SHA256, SLIM_SHA256
from strokecopilot.vision.perfusion.thresholds import reliable

EXPERIMENT = "EXP-15"
RUN_NAME = "EXP-15_visionA_error_analysis"
NNUNET_RUN = "EXP-14_nnunet_fold0"
THRESHOLD_RUN = "EXP-1_thresholds_baseline"

# Columns compared between the two models. Both tables come from evaluate.case_metrics().
SCORES = ("dice", "dice_reliable", "pred_ml", "error_ml", "abs_error_ml")
CROSSINGS = tuple(f"crosses_{cutoff:g}" for cutoff in DECISION_CUTOFFS_ML)
# Taken once, from the nnU-Net table. ref_ml is checked against the threshold table's.
SHARED = ("ref_ml", "infarct_reliable_share")
# Below this Dice a case counts as a miss when comparing the two models.
MISS_DICE = 0.05
# Two reference volumes closer than this (mL) count as the same.
REF_TOLERANCE_ML = 1e-6

# Size groups: under 5 mL (the brief's small lesions), then the 21 and 70 mL treatment cut-offs.
SIZE_EDGES_ML = (5.0, 21.0, 70.0)
# Tmax fill value where there is no perfusion data (docs/DATA.md).
TMAX_NO_DATA = -30.0
# Territory from the Circle of Willis label nearest the occlusion. TopCoW numbering, which docs/DATA.md
# notes is not confirmed for this dataset: 1 basilar, 2-3 PCAs; 4 and 6 ICAs, 5 and 7 MCAs, 10 Acom,
# 11-12 ACAs, 15 third A2. The posterior communicating arteries (8, 9) join both, so they stay unknown.
POSTERIOR_LABELS = frozenset({1, 2, 3})
ANTERIOR_LABELS = frozenset({4, 5, 6, 7, 10, 11, 12, 15})


def write_config(run_dir, git_commit, folds=(0,), notes=""):
    """Write config.json for EXP-15 before the run, and return it. Refuses to overwrite one.

    run_dir: the run folder on Drive, normally DRIVE_RUNS / RUN_NAME. git_commit: the commit the
    notebook installed. folds: the folds whose nnU-Net per-case tables are analysed. No case IDs.
    """
    run_dir = Path(run_dir)
    path = run_dir / "config.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists. config.json is written once, before the run.")
    if not git_commit:
        raise ValueError("git_commit is required: the commit the notebook installed.")
    config = {
        "experiment": EXPERIMENT,
        "name": RUN_NAME,
        "runner": "strokecopilot.vision.perfusion.error_analysis (notebooks/12_visionA_error_analysis.ipynb)",
        "git_commit": git_commit,
        "data_version": ISLES24_SLIM,
        "data_sha256": SLIM_SHA256,
        "folds_sha256": FOLDS_SHA256,
        "cases": "development only; held-out cases are refused by the loaders",
        "folds": list(folds),
        "inputs": {
            "nnunet": f"runs/{NNUNET_RUN}/per_case.csv",
            "threshold": f"runs/{THRESHOLD_RUN}/per_case.csv",
            "slim": f"slim/{ISLES24_SLIM}",
        },
        "decision_cutoffs_ml": list(DECISION_CUTOFFS_ML),
        "miss_dice": MISS_DICE,
        "outputs_policy": "counts and summaries only, never case IDs",
        "notes": notes,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1))
    return config


def join_tables(nnunet, threshold):
    """Step 2. One row per nnU-Net case with both models' scores side by side, and where they disagree.

    nnunet, threshold: the per_case.csv tables of EXP-14 and EXP-1 as pandas DataFrames. The
    threshold table may cover more cases (all folds); only the nnU-Net cases are kept. Raises if an
    nnU-Net case is missing from the threshold table, a case appears twice, or the two tables
    disagree on the reference volume. Columns:
    - case, fold, ref_ml, infarct_reliable_share;
    - nnunet_<score> and threshold_<score> for dice, dice_reliable, pred_ml, error_ml, abs_error_ml;
    - nnunet_<crossing> and threshold_<crossing> for each decision cut-off;
    - dice_gap and abs_error_gap (nnU-Net minus threshold);
    - better_volume: "nnunet", "threshold" or "tie", by absolute volume error;
    - miss: "both", "nnunet", "threshold" or "neither", by Dice below MISS_DICE;
    - crossings_disagree: how many cut-offs one model crosses and the other doesn't.
    """
    for name, table, extra in (("nnunet", nnunet, ("fold", *SHARED)), ("threshold", threshold, ("ref_ml",))):
        missing = [c for c in ("case", *SCORES, *CROSSINGS, *extra) if c not in table.columns]
        if missing:
            raise ValueError(f"The {name} table lacks columns {missing}.")
        if table["case"].duplicated().any():
            raise ValueError(f"The {name} table has {int(table['case'].duplicated().sum())} repeated cases.")
    absent = ~nnunet["case"].isin(threshold["case"])
    if absent.any():
        raise ValueError(f"{int(absent.sum())} nnU-Net cases are missing from the threshold table.")

    keep = ["case", *SCORES, *CROSSINGS]
    left = nnunet[["case", "fold", *SHARED, *SCORES, *CROSSINGS]].rename(columns={c: f"nnunet_{c}" for c in (*SCORES, *CROSSINGS)})
    right = threshold[[*keep, "ref_ml"]].rename(columns={c: f"threshold_{c}" for c in (*SCORES, *CROSSINGS, "ref_ml")})
    joined = left.merge(right, on="case", how="left", validate="one_to_one")

    ref_gap = (joined["ref_ml"] - joined["threshold_ref_ml"]).abs()
    if (ref_gap > REF_TOLERANCE_ML).any():
        raise ValueError(f"{int((ref_gap > REF_TOLERANCE_ML).sum())} cases have different reference volumes in the two tables.")
    joined = joined.drop(columns=["threshold_ref_ml"])

    joined["dice_gap"] = joined["nnunet_dice"] - joined["threshold_dice"]
    joined["abs_error_gap"] = joined["nnunet_abs_error_ml"] - joined["threshold_abs_error_ml"]
    joined["better_volume"] = np.select(
        [joined["abs_error_gap"] < 0, joined["abs_error_gap"] > 0], ["nnunet", "threshold"], default="tie")
    # A case with no Dice (both masks empty) is not a miss.
    nn_miss = joined["nnunet_dice"].lt(MISS_DICE)
    th_miss = joined["threshold_dice"].lt(MISS_DICE)
    joined["miss"] = np.select([nn_miss & th_miss, nn_miss, th_miss], ["both", "nnunet", "threshold"], default="neither")
    joined["crossings_disagree"] = sum(
        (joined[f"nnunet_{c}"].astype(bool) != joined[f"threshold_{c}"].astype(bool)).astype(int) for c in CROSSINGS)
    return joined


def size_band(infarct_ml):
    """Size group of a reference infarct volume: none, <5, 5-21, 21-70 or >=70 mL."""
    if infarct_ml is None:
        return None
    if infarct_ml <= 0:
        return "none"
    lower = 0.0
    for edge in SIZE_EDGES_ML:
        if infarct_ml < edge:
            return f"<{edge:g}" if lower == 0 else f"{lower:g}-{edge:g}"
        lower = edge
    return f">={lower:g}"


def territory_of_label(label):
    """anterior, posterior or unknown, from a Circle of Willis label (TopCoW numbering)."""
    if label is None:
        return "unknown"
    if int(label) in POSTERIOR_LABELS:
        return "posterior"
    if int(label) in ANTERIOR_LABELS:
        return "anterior"
    return "unknown"


def case_features(case):
    """Step 3. Possible causes of failure for one loaded isles24.Case, as a flat dict.

    The case needs tmax, cbf, infarct, occlusion and cow arrays, native_volumes["infarct_ml"] and
    the clinical row. Keys:
    - case (for joining to join_tables(); stays on Drive), infarct_ml (native reference), size_band;
    - occlusion_site and occlusion_reason (volumes.occlusion_site), occlusion_nearest_label,
      occlusion_distance_mm;
    - territory: anterior or posterior from the occlusion's nearest label, else unknown. No distance
      limit is applied, so check it against occlusion_distance_mm;
    - onset_to_imaging_min: onset to door + door to imaging, as the notes compute it. None when
      either is missing, which includes wake-up strokes;
    - infarct_tmax_missing_share: share of the infarct with Tmax at the -30 no-data value;
    - infarct_tmax_unreliable_share: share without reliable data (thresholds.reliable: Tmax below 0
      or CBF not finite), which plan §8.1 treats as no data. Both None for an empty infarct.
    """
    arrays = case.arrays
    infarct_ml = case.native_volumes.get("infarct_ml")
    evidence = volumes.occlusion_evidence(arrays["occlusion"], arrays["cow"], case.spacing_mm)
    site, reason = volumes.occlusion_site(evidence)

    onset_door = minutes(case.clinical.get("Onset to door"))
    door_imaging = minutes(case.clinical.get("Door to imaging"))
    onset_imaging = None if onset_door is None or door_imaging is None else onset_door + door_imaging

    infarct = np.asarray(arrays["infarct"], dtype=bool)
    n_infarct = int(np.count_nonzero(infarct))
    tmax = np.asarray(arrays["tmax"], dtype=np.float32)
    if n_infarct:
        missing_share = float(np.count_nonzero(infarct & (tmax == TMAX_NO_DATA)) / n_infarct)
        unreliable_share = float(np.count_nonzero(infarct & ~reliable(tmax, arrays["cbf"])) / n_infarct)
    else:
        missing_share = unreliable_share = None

    return {
        "case": case.id,
        "infarct_ml": infarct_ml,
        "size_band": size_band(infarct_ml),
        "occlusion_site": site,
        "occlusion_reason": reason,
        "occlusion_nearest_label": evidence["nearest_label"],
        "occlusion_distance_mm": evidence["distance_mm"],
        "territory": territory_of_label(evidence["nearest_label"]),
        "onset_to_imaging_min": onset_imaging,
        "infarct_tmax_missing_share": missing_share,
        "infarct_tmax_unreliable_share": unreliable_share,
    }


def group_summary(table, by):
    """Step 5. Cases, Dice and absolute volume error per group of column `by`. No case IDs."""
    raise NotImplementedError("Step 5")
