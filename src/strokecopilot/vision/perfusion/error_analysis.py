"""Vision A error analysis: where and why the infarct model fails (plan §7 Phase 3, §8.1; brief docs/team/lorna_vision_a.md).

Built one step at a time.

1. Skeleton and synthetic tests (tests/test_error_analysis.py).
2. write_config(): runs/EXP-15_visionA_error_analysis/config.json, written before the run
   (docs/team/README.md §4). join_tables(): the nnU-Net (EXP-14) and threshold (EXP-1) per-case
   tables side by side.
3. case_features(): per-case possible causes of failure, from a loaded Case: size, occlusion site,
   territory, onset to imaging, and how much of the infarct has no Tmax data.
4. (Colab) run case_features() over the development cases and join it to the step-2 table.
5. group_summary(): scores per group of a feature, as counts and summaries; miss_profile(): the
   Dice-0 misses and the largest one, as values only; build_metrics() / write_metrics(): metrics.json.
6. (Colab, then docs) images of failed cases and the write-up.

Outputs that leave Drive carry counts and summaries only, never case IDs (docs/team/README.md §3).
Volume error matters more than Dice, because the treatment rules are volume cut-offs
(evaluate.DECISION_CUTOFFS_ML).
"""

import json
from pathlib import Path

import numpy as np

from strokecopilot.config import ISLES24_SLIM
from strokecopilot.data import template
from strokecopilot.data.isles24_survey import PENUMBRA_TMAX_S
from strokecopilot.nlp.notes.generate import minutes
from strokecopilot.vision.perfusion import volumes
from strokecopilot.vision.perfusion.evaluate import DECISION_CUTOFFS_ML
from strokecopilot.vision.perfusion.nnunet_wrapper import FILE_ENDING
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
# EXP-1's per_case.csv holds two rows per case, one per `variant` (notebooks/11_visionA_thresholds.ipynb):
# the core restricted to Tmax > 6 s ("within_tmax6", the default and the one compared with EXP-14) or not.
THRESHOLD_VARIANT = "within_tmax6"
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

# Step 5. Bands of the share of the infarct with Tmax at -30.
TMAX_MISSING_EDGES = (0.05, 0.25)
# Step 6a. Bands of the share of the (reliable) infarct with Tmax > 6 s on admission.
TMAX6_EDGES = (0.25, 0.5, 0.75)
# The features the cases are grouped by. The last three are derived by add_groups().
GROUP_FEATURES = ("size_band", "territory", "occlusion_site", "tmax_missing_band", "onset_known", "tmax6_band")
# What the miss profile shows about the largest Dice-0 miss: values only, no case ID.
LARGEST_MISS_KEYS = ("ref_ml", "nnunet_pred_ml", "threshold_dice", "threshold_pred_ml", *GROUP_FEATURES,
                     "occlusion_distance_mm", "onset_to_imaging_min", "infarct_tmax_missing_share",
                     "infarct_tmax_unreliable_share", "infarct_reliable_share", "infarct_tmax6_share")
# Step 6a figures: the maps shown, with display windows (template.PANELS), and how many best cases.
FIGURE_PANELS = (("ncct", 0, 80), ("cbf", None, None), ("tmax", 0, 12))
BEST_CASES = 3
# Follow-ups step 4. Outline colours: the true final infarct, and nnU-Net's prediction (EXP-22 for fold 0).
TRUE_COLOUR = "red"
AI_COLOUR = "cyan"

# Follow-ups step 3. Artery type of each Circle of Willis label, side ignored (TopCoW numbering, as for
# POSTERIOR_LABELS above; not confirmed for this dataset). Labels not listed become "unknown".
ARTERY_OF_LABEL = {1: "basilar", 2: "PCA", 3: "PCA", 4: "ICA", 6: "ICA", 5: "MCA", 7: "MCA",
                   8: "Pcom", 9: "Pcom", 10: "Acom", 11: "ACA", 12: "ACA", 15: "ACA"}
NOT_AN_ARTERY = ("too far", "no occlusion found", "unknown")

# Follow-ups step 2. "Small" infarcts are under the first treatment cut-off, as in the write-up.
SMALL_ML = 21.0
# A case counts as having a no-data problem when at least this share of its infarct has Tmax -30 (Parsa's review).
NO_DATA_SHARE = 0.05
# Parsa's EXP-25 numbers (CHANGELOG, 13 Nov 2026) for the check in exp25_check(): means and stds over the five folds
# (mean Dice and mean absolute volume error per fold), cases on the wrong side of 21/31/51/70 mL out of 129, and
# each nnU-Net fold's own numbers (fold 0 is EXP-14, folds 1 to 4 are EXP-18 to EXP-21).
EXP25_REFERENCE = {
    "nnunet": {"dice_mean": 0.215, "dice_std": 0.046, "abs_error_mean": 23.5, "abs_error_std": 3.9, "crossings": (30, 28, 20, 17)},
    "threshold": {"dice_mean": 0.111, "dice_std": 0.031, "abs_error_mean": 25.2, "abs_error_std": 5.0, "crossings": (35, 29, 23, 17)},
    "nnunet_per_fold": {
        "0": {"cases": 26, "dice_mean": 0.218, "abs_error_mean": 19.6, "crossings": (5, 5, 3, 3)},
        "1": {"cases": 26, "dice_mean": 0.211, "abs_error_mean": 22.0, "crossings": (3, 4, 4, 3)},
        "2": {"cases": 26, "dice_mean": 0.173, "abs_error_mean": 27.0, "crossings": (9, 6, 3, 3)},
        "3": {"cases": 26, "dice_mean": 0.290, "abs_error_mean": 20.9, "crossings": (5, 5, 2, 3)},
        "4": {"cases": 25, "dice_mean": 0.184, "abs_error_mean": 28.3, "crossings": (8, 8, 8, 5)},
    },
}


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
            "threshold_variant": THRESHOLD_VARIANT,
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


def join_tables(nnunet, threshold, variant=THRESHOLD_VARIANT):
    """Step 2. One row per nnU-Net case with both models' scores side by side, and where they disagree.

    nnunet, threshold: the per_case.csv tables of EXP-14 and EXP-1 as pandas DataFrames. If the
    threshold table has a `variant` column, only the rows of `variant` are kept, and it raises if
    there are none. The threshold table may cover more cases (all folds); only the nnU-Net cases are
    kept. Raises if an
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
    if "variant" in threshold.columns:
        found = sorted(threshold["variant"].dropna().unique())
        threshold = threshold[threshold["variant"] == variant]
        if threshold.empty:
            raise ValueError(f"The threshold table has no {variant!r} rows; its variants are {found}.")
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


def five_fold_table(nnunet_folds, threshold, variant=THRESHOLD_VARIANT, expected_cases=None):
    """Follow-ups step 1. join_tables() over every fold: one row per development case, nnU-Net scored by the
    fold model that did not train on it.

    nnunet_folds: the per_case.csv tables of the five nnU-Net fold runs (EXP-14 for fold 0, EXP-18 to EXP-21 for
    folds 1 to 4), as pandas DataFrames, each holding one fold. threshold: EXP-1's per_case.csv, which covers all
    folds. Raises, with counts only, if a table holds more than one fold, a fold appears in two tables, a case
    appears twice, the threshold table puts a case in a different fold, or the total is not `expected_cases`.
    """
    import pandas as pd

    seen = []
    for table in nnunet_folds:
        folds = sorted(table["fold"].dropna().astype(int).unique())
        if len(folds) != 1:
            raise ValueError(f"An nnU-Net table holds {len(folds)} folds; each should hold exactly one.")
        if folds[0] in seen:
            raise ValueError(f"Fold {folds[0]} appears in two nnU-Net tables.")
        seen.append(folds[0])
    nnunet = pd.concat(nnunet_folds, ignore_index=True)
    nnunet["fold"] = nnunet["fold"].astype(int)
    if nnunet["case"].duplicated().any():
        raise ValueError(f"{int(nnunet['case'].duplicated().sum())} cases appear in more than one fold table.")
    if expected_cases is not None and len(nnunet) != expected_cases:
        raise ValueError(f"The fold tables hold {len(nnunet)} cases, not {expected_cases}.")

    joined = join_tables(nnunet, threshold, variant=variant)

    if "fold" in threshold.columns:
        base = threshold[threshold["variant"] == variant] if "variant" in threshold.columns else threshold
        fold_of = base.drop_duplicates("case").set_index("case")["fold"].astype(int)
        differ = int((joined["case"].map(fold_of) != joined["fold"]).sum())
        if differ:
            raise ValueError(f"{differ} cases are in a different fold in the threshold table.")
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
      or CBF not finite), which plan §8.1 treats as no data. Both None for an empty infarct;
    - infarct_tmax6_share: of the infarct voxels with reliable data, the share with Tmax > 6 s on
      the admission scan, i.e. how much of the final infarct the perfusion input shows as delayed.
      None when no infarct voxel has reliable data.
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
    valid = reliable(tmax, arrays["cbf"])
    if n_infarct:
        missing_share = float(np.count_nonzero(infarct & (tmax == TMAX_NO_DATA)) / n_infarct)
        unreliable_share = float(np.count_nonzero(infarct & ~valid) / n_infarct)
    else:
        missing_share = unreliable_share = None
    n_reliable = int(np.count_nonzero(infarct & valid))
    tmax6_share = float(np.count_nonzero(infarct & valid & (tmax > PENUMBRA_TMAX_S)) / n_reliable) if n_reliable else None

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
        "infarct_tmax6_share": tmax6_share,
    }


def tmax_missing_band(share):
    """Group of the share of the infarct with Tmax at -30: none, <5%, 5-25%, >=25%, or no infarct."""
    if share is None or (isinstance(share, float) and np.isnan(share)):
        return "no infarct"
    if share <= 0:
        return "none"
    lower = 0.0
    for edge in TMAX_MISSING_EDGES:
        if share < edge:
            return f"<{edge:.0%}" if lower == 0 else f"{lower:.0%}-{edge:.0%}"
        lower = edge
    return f">={lower:.0%}"


def tmax6_band(share):
    """Group of the share of the reliable infarct with Tmax > 6 s: <25%, 25%-50%, 50%-75%, >=75%, or no data."""
    if share is None or (isinstance(share, float) and np.isnan(share)):
        return "no data"
    lower = 0.0
    for edge in TMAX6_EDGES:
        if share < edge:
            return f"<{edge:.0%}" if lower == 0 else f"{lower:.0%}-{edge:.0%}"
        lower = edge
    return f">={lower:.0%}"


def add_groups(table):
    """A copy of the step-4 table with the derived group columns: tmax_missing_band, onset_known, tmax6_band."""
    table = table.copy()
    table["tmax_missing_band"] = [tmax_missing_band(s) for s in table["infarct_tmax_missing_share"]]
    table["onset_known"] = np.where(table["onset_to_imaging_min"].notna(), "known", "unknown")
    table["tmax6_band"] = [tmax6_band(s) for s in table["infarct_tmax6_share"]]
    return table


def crossing_counts(table):
    """Per decision cut-off, how many cases each model puts on the wrong side of it (prediction and
    reference on different sides), and how many both models do. Counts only, no case IDs."""
    out = {}
    for cutoff in DECISION_CUTOFFS_ML:
        nn = table[f"nnunet_crosses_{cutoff:g}"].astype(bool)
        th = table[f"threshold_crosses_{cutoff:g}"].astype(bool)
        out[f"{cutoff:g}"] = {"nnunet": int(nn.sum()), "threshold": int(th.sum()), "both": int((nn & th).sum()), "of": int(len(table))}
    nn_any = sum(table[f"nnunet_{c}"].astype(bool) for c in CROSSINGS) > 0
    th_any = sum(table[f"threshold_{c}"].astype(bool) for c in CROSSINGS) > 0
    out["any"] = {"nnunet": int(nn_any.sum()), "threshold": int(th_any.sum()), "both": int((nn_any & th_any).sum()), "of": int(len(table))}
    return out


def visibility_comparison(table):
    """infarct_tmax6_share for nnU-Net's Dice-0 misses vs the other cases: counts and medians, no IDs."""
    miss = table["nnunet_dice"] == 0
    out = {}
    for name, rows in (("dice0_misses", table[miss]), ("rest", table[~miss])):
        share = rows["infarct_tmax6_share"]
        out[name] = {"cases": int(len(rows)), "with_reliable_data": int(share.notna().sum()),
                     "median": _num(share.median()), "p25": _num(share.quantile(0.25)), "p75": _num(share.quantile(0.75))}
    return out


def is_overcall(table):
    """Over-call (agreed with Lorna, follow-ups): nnU-Net's predicted volume is larger than the true volume and
    crosses at least one decision cut-off, at any Dice. A boolean per row."""
    over = table["nnunet_pred_ml"] > table["ref_ml"]
    crosses_any = sum(table[f"nnunet_{c}"].astype(bool) for c in CROSSINGS) > 0
    return over & crosses_any


def is_undercall(table):
    """Under-call, the mirror of is_overcall(): nnU-Net's predicted volume is smaller than the true volume and
    crosses at least one decision cut-off, at any Dice. A boolean per row."""
    under = table["nnunet_pred_ml"] < table["ref_ml"]
    crosses_any = sum(table[f"nnunet_{c}"].astype(bool) for c in CROSSINGS) > 0
    return under & crosses_any


def _fold_block(rows):
    """The follow-up counts for one set of rows of the five-fold table: counts and medians only, no case IDs."""
    miss = rows["nnunet_dice"] == 0
    small = rows["ref_ml"] < SMALL_ML
    overcall = is_overcall(rows)
    undercall = is_undercall(rows)
    visible, missing = rows["infarct_tmax6_share"], rows["infarct_tmax_missing_share"]

    def side(mask):
        share = visible[mask]
        return {
            "cases": int(mask.sum()),
            "visible_with_reliable_data": int(share.notna().sum()),
            "visible_median": _num(share.median()),
            "visible_under_25pct": int((share < 0.25).sum()),
            "no_data_at_least_5pct": int((missing[mask] >= NO_DATA_SHARE).sum()),
        }

    return {
        "cases": int(len(rows)),
        "dice0": int(miss.sum()),
        "miss_rate_by_size": {
            f"under_{SMALL_ML:g}_ml": {"cases": int(small.sum()), "dice0": int((miss & small).sum())},
            f"{SMALL_ML:g}_ml_and_over": {"cases": int((~small).sum()), "dice0": int((miss & ~small).sum())},
        },
        "overcalls": int(overcall.sum()),
        "overcalls_dice0": int((overcall & miss).sum()),
        "undercalls": int(undercall.sum()),
        "undercalls_dice0": int((undercall & miss).sum()),
        "dice0_misses": side(miss),
        "rest": side(~miss),
    }


def five_fold_counts(table):
    """Follow-ups step 2. On the five-fold table with case_features() added: Dice-0 misses, the miss rate under and
    over 21 mL, over-calls and under-calls and how many of each are Dice 0, and the visible (Tmax > 6 s) and no-data comparisons
    for misses against the rest. Overall and per fold; counts and medians only, no case IDs."""
    return {
        "overall": _fold_block(table),
        "per_fold": {str(int(f)): _fold_block(table[table["fold"] == f]) for f in sorted(table["fold"].unique())},
    }


def artery_group(reason, nearest_label, distance_mm, limit_mm=None):
    """Follow-ups step 3. The artery the occlusion sits next to, counted only within the distance limit.

    Returns the artery type, side ignored (ARTERY_OF_LABEL), "too far" when the occlusion is more than `limit_mm`
    from its nearest label, "no occlusion found" when the occlusion mask is empty, and "unknown" otherwise
    (no labels, an unlisted label, or a missing value). The default limit is the contract's (volumes.DISTANCE_LIMIT_MM).
    """
    limit_mm = volumes.DISTANCE_LIMIT_MM if limit_mm is None else limit_mm
    if reason == "empty_mask":
        return "no occlusion found"
    if reason != "ok" or nearest_label is None or distance_mm is None:
        return "unknown"
    if isinstance(nearest_label, float) and np.isnan(nearest_label):
        return "unknown"
    if isinstance(distance_mm, float) and np.isnan(distance_mm):
        return "unknown"
    if distance_mm > limit_mm:
        return "too far"
    return ARTERY_OF_LABEL.get(int(nearest_label), "unknown")


def add_artery_groups(table, limit_mm=None):
    """A copy of the table with `artery_group` and `territory_within_limit` (anterior, posterior, or the same
    "too far" / "no occlusion found" / "unknown" as artery_group)."""
    table = table.copy()
    groups = [artery_group(r, l, d, limit_mm) for r, l, d in
              zip(table["occlusion_reason"], table["occlusion_nearest_label"], table["occlusion_distance_mm"])]
    table["artery_group"] = groups
    territory = []
    for group, label in zip(groups, table["occlusion_nearest_label"]):
        territory.append(territory_of_label(int(label)) if group not in NOT_AN_ARTERY else group)
    table["territory_within_limit"] = territory
    return table


def onset_summary(table):
    """Follow-ups step 3. Onset to scan time (minutes) for the cases that have it, misses against the rest:
    how many cases have it, the median and the range. Counts and values only, no case IDs."""
    known = table[table["onset_to_imaging_min"].notna()]
    miss = known["nnunet_dice"] == 0
    out = {"cases": int(len(table)), "onset_known": int(len(known))}
    for name, rows in (("dice0_misses", known[miss]), ("rest", known[~miss])):
        minutes_ = rows["onset_to_imaging_min"]
        out[name] = {"onset_known": int(len(rows)), "median_min": _num(minutes_.median(), 0),
                     "min_min": _num(minutes_.min(), 0), "max_min": _num(minutes_.max(), 0)}
    return out


def exp25_check(table, reference=None):
    """Compare the five-fold table with Parsa's EXP-25 numbers (CHANGELOG, 13 Nov 2026), rounded as EXP-25 reports
    them. Returns one row per number: what, ours, exp25, match. Counts and means only, no case IDs."""
    reference = EXP25_REFERENCE if reference is None else reference
    rows = []

    def add(what, ours, ref, digits):
        ours = None if ours is None else round(float(ours), digits)
        rows.append({"what": what, "ours": ours, "exp25": ref, "match": ours is not None and abs(ours - ref) < 1e-9})

    folds = sorted(table["fold"].unique())
    for model in ("nnunet", "threshold"):
        means = {m: [table.loc[table["fold"] == f, f"{model}_{m}"].mean() for f in folds] for m in ("dice", "abs_error_ml")}
        ref = reference[model]
        add(f"{model} Dice, mean over folds", np.mean(means["dice"]), ref["dice_mean"], 3)
        add(f"{model} Dice, std over folds", np.std(means["dice"], ddof=1), ref["dice_std"], 3)
        add(f"{model} abs volume error mL, mean over folds", np.mean(means["abs_error_ml"]), ref["abs_error_mean"], 1)
        add(f"{model} abs volume error mL, std over folds", np.std(means["abs_error_ml"], ddof=1), ref["abs_error_std"], 1)
        for cutoff, count in zip(DECISION_CUTOFFS_ML, ref["crossings"]):
            add(f"{model} wrong side of {cutoff:g} mL", table[f"{model}_crosses_{cutoff:g}"].astype(bool).sum(), count, 0)
    for fold, ref in reference["nnunet_per_fold"].items():
        rows_f = table[table["fold"] == int(fold)]
        add(f"fold {fold} cases", len(rows_f), ref["cases"], 0)
        add(f"fold {fold} nnunet Dice mean", rows_f["nnunet_dice"].mean(), ref["dice_mean"], 3)
        add(f"fold {fold} nnunet abs volume error mL", rows_f["nnunet_abs_error_ml"].mean(), ref["abs_error_mean"], 1)
        for cutoff, count in zip(DECISION_CUTOFFS_ML, ref["crossings"]):
            add(f"fold {fold} nnunet wrong side of {cutoff:g} mL", rows_f[f"nnunet_crosses_{cutoff:g}"].astype(bool).sum(), count, 0)
    return rows


def _num(value, digits=3):
    """A JSON-safe number: None for missing, int for counts, rounded float otherwise."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), digits)


def _scores(rows):
    """The step-5 numbers for a set of rows of the step-4 table."""
    dice = rows["nnunet_dice"]
    return {
        "cases": int(len(rows)),
        "nnunet_dice_mean": _num(dice.mean()),
        "nnunet_dice_median": _num(dice.median()),
        "nnunet_abs_error_ml_mean": _num(rows["nnunet_abs_error_ml"].mean(), 1),
        "threshold_abs_error_ml_mean": _num(rows["threshold_abs_error_ml"].mean(), 1),
        "dice0": int((dice == 0).sum()),
        "missed_nnunet_only": int((rows["miss"] == "nnunet").sum()),
        "missed_both": int((rows["miss"] == "both").sum()),
    }


def group_summary(table, by):
    """Step 5. One row per group of column `by` (missing values form their own group), as a list of dicts.

    Each row: group, cases, nnU-Net Dice mean and median (cases with no Dice left out), mean absolute
    volume error (mL), dice0 (nnU-Net Dice exactly 0), and missed_nnunet_only / missed_both by the
    `miss` column of join_tables() (Dice < MISS_DICE). Counts and summaries only, no case IDs.
    """
    groups = table[by].astype(object).where(table[by].notna(), "missing").astype(str)
    return [{"group": group, **_scores(table[groups == group])} for group in sorted(groups.unique())]


def summarise_features(table, features=GROUP_FEATURES):
    """group_summary() for each feature, on the step-4 table. Adds the derived group columns first."""
    table = add_groups(table)
    return {feature: group_summary(table, feature) for feature in features}


def miss_profile(table, features=GROUP_FEATURES):
    """Who the nnU-Net Dice-0 misses are, as counts and values only, never case IDs.

    Returns the number of misses, how many the threshold also missed, their reference and predicted
    volume ranges, counts per group of each feature, and the largest miss (by reference volume) as values.
    """
    table = add_groups(table)
    misses = table[table["nnunet_dice"] == 0]
    out = {"cases": int(len(table)), "dice0": int(len(misses))}
    if misses.empty:
        return {**out, "largest_miss": None}
    stats = lambda s: {"min": _num(s.min(), 1), "median": _num(s.median(), 1), "max": _num(s.max(), 1)}
    out.update({
        "threshold_also_dice0": int((misses["threshold_dice"] == 0).sum()),
        "ref_ml": stats(misses["ref_ml"]),
        "nnunet_pred_ml": stats(misses["nnunet_pred_ml"]),
        "nnunet_pred_empty": int((misses["nnunet_pred_ml"] == 0).sum()),
        "groups": {f: {str(k): int(v) for k, v in misses[f].value_counts(dropna=False).sort_index().items()} for f in features},
    })
    largest = misses.loc[misses["ref_ml"].idxmax()]
    out["largest_miss"] = {key: (_num(largest[key]) if key not in features else str(largest[key])) for key in LARGEST_MISS_KEYS}
    return out


def _strings(value):
    """Every key and text value inside nested dicts and lists."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def _ids_in(value, ids):
    """Case IDs found in `value`: a text equal to an ID, or containing an ID of 4 characters or more."""
    long_ids = [i for i in ids if len(i) >= 4]
    return sorted({i for s in _strings(value) for i in ([s] if s in ids else []) + [i for i in long_ids if i in s]})


def build_metrics(table, config, git_commit):
    """metrics.json for EXP-15: run details, group summaries and the miss profile.

    git_commit: the commit this run installed, which can differ from config.json's when config.json
    was kept from an earlier run; both are recorded. Raises if any case ID of the table appears
    anywhere in the result.
    """
    if not git_commit:
        raise ValueError("git_commit is required: the commit this run installed.")
    metrics = {
        "experiment": config.get("experiment", EXPERIMENT),
        "git_commit": git_commit,
        "config_git_commit": config.get("git_commit"),
        "data_version": config.get("data_version", ISLES24_SLIM),
        "folds": config.get("folds"),
        "threshold_variant": config.get("inputs", {}).get("threshold_variant", THRESHOLD_VARIANT),
        "overall": _scores(table),
        "who_misses": {str(k): int(v) for k, v in table["miss"].value_counts().sort_index().items()},
        "better_volume": {str(k): int(v) for k, v in table["better_volume"].value_counts().sort_index().items()},
        "groups": summarise_features(table),
        "wrong_side_of_cutoff": crossing_counts(table),
        "dice0_misses": miss_profile(table),
        "tmax6_share_misses_vs_rest": visibility_comparison(table),
    }
    leaked = _ids_in(metrics, set(table["case"].astype(str)))
    if leaked:
        raise ValueError(f"{len(leaked)} case IDs reached metrics.json. Nothing was written.")
    return metrics


def figure_cases(table, best=BEST_CASES):
    """Which cases to draw, with anonymous labels: every Dice-0 miss ("Miss 1", ... largest reference
    volume first), then the `best` cases with the highest nnU-Net Dice ("Best 1", ...).

    Returns dicts with label, case and size_band. The case ID is for loading only: keep it on Drive.
    """
    misses = table[table["nnunet_dice"] == 0].sort_values("ref_ml", ascending=False)
    top = table[table["nnunet_dice"] > 0].sort_values("nnunet_dice", ascending=False).head(best)
    picks = []
    for prefix, rows in (("Miss", misses), ("Best", top)):
        for i, (_, row) in enumerate(rows.iterrows(), 1):
            picks.append({"label": f"{prefix} {i}", "case": row["case"], "size_band": row["size_band"]})
    return picks


def figure_summary(table, picks):
    """One row per figure from figure_cases(): label, size band, Dice-0 flag and Tmax > 6 s share.

    Labels only: the case ID is used to look up the row and is not returned.
    """
    rows = table.set_index("case")
    out = []
    for pick in picks:
        row = rows.loc[pick["case"]]
        out.append({"label": pick["label"], "size_band": pick["size_band"], "dice0": bool(row["nnunet_dice"] == 0),
                    "infarct_tmax6_share": _num(row["infarct_tmax6_share"], 2)})
    return out


def load_prediction(pred_dir, case_id, shape):
    """Follow-ups step 4. nnU-Net's predicted mask for one case from a folder of validation masks (EXP-22 for
    fold 0: `<case>.nii.gz`, on the slim grid), as booleans. Raises if the shape differs from `shape`."""
    import nibabel as nib

    path = Path(pred_dir) / f"{case_id}{FILE_ENDING}"
    pred = np.asanyarray(nib.load(str(path)).dataobj) > 0
    if pred.shape != tuple(shape):
        raise ValueError(f"A predicted mask has shape {pred.shape}, expected {tuple(shape)}.")
    return pred


def case_figure(case, label, size_band=None, z=None, prediction=None):
    """NCCT, CBF and Tmax at the slice with the most infarct, the final infarct outlined in red.

    With `prediction` (nnU-Net's mask, same grid), its outline is drawn in cyan on the same slice, with a small
    legend. The legend says when the AI marked nothing on this slice, and on how many other slices it did.
    Titled only with the anonymous label and size band, never the case ID. Returns the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    z = template.largest_infarct_slice(case) if z is None else z
    infarct = np.asarray(case.arrays["infarct"][:, :, z], dtype=bool)
    pred = None
    if prediction is not None:
        prediction = np.asarray(prediction, dtype=bool)
        if prediction.shape != case.arrays["infarct"].shape:
            raise ValueError(f"The predicted mask has shape {prediction.shape}, expected {case.arrays['infarct'].shape}.")
        pred = prediction[:, :, z]
    fig, axes = plt.subplots(1, len(FIGURE_PANELS), figsize=(3.2 * len(FIGURE_PANELS), 3.9))
    for ax, (name, low, high) in zip(axes, FIGURE_PANELS):
        low, high = template._window(case.arrays[name], low, high)
        ax.imshow(case.arrays[name][:, :, z].T, cmap="gray", vmin=low, vmax=high, origin="lower")
        if infarct.any():
            ax.contour(infarct.T, levels=[0.5], colors=TRUE_COLOUR, linewidths=0.8)
        if pred is not None and pred.any():
            ax.contour(pred.T, levels=[0.5], colors=AI_COLOUR, linewidths=0.8)
        ax.set_title(name.upper(), fontsize=9)
        ax.axis("off")
    band = f", {size_band} mL" if size_band else ""
    if prediction is None:
        fig.suptitle(f"{label}{band}: final infarct in red", fontsize=10)
        fig.tight_layout()
        return fig
    fig.suptitle(f"{label}{band}: final infarct in red, AI outline in cyan", fontsize=10)
    ai_label = "AI outline"
    if not pred.any():
        others = int(np.count_nonzero(prediction.any(axis=(0, 1))))
        ai_label = "AI outline: none on this slice" + (f" (marks on {others} other slices)" if others else ", none anywhere")
    handles = [Line2D([0], [0], color=TRUE_COLOUR, lw=1.5, label="Final infarct"),
               Line2D([0], [0], color=AI_COLOUR, lw=1.5, label=ai_label)]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def figure_texts(fig):
    """Every piece of text drawn in a figure, for checking that no case ID is in it."""
    from matplotlib.text import Text

    return [t.get_text() for t in fig.findobj(Text)]


def write_metrics(run_dir, metrics):
    """Write metrics.json in the run folder, last, as it marks a finished run."""
    path = Path(run_dir) / "metrics.json"
    path.write_text(json.dumps(metrics, indent=1))
    return path
