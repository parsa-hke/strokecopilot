"""The held-out contract path (DEC-31 D1; Phase 4): vision_output.json for cases no model has seen, built and checked on
development cases first.

For the 20 held-out cases nothing is precomputed: EXP-33's development contracts read each field from a finished run,
and every one of those runs covered development cases only. So every field is computed here, case by case, the way
those runs computed it:
- core_ml: the nnU-Net prediction (DEC-25). Held-out cases use the five fold models' ensemble (D1: softmax averaged,
  nnU-Net's own `-f 0 1 2 3 4`, each fold's checkpoint_final.pth). The development check predicts each case with the
  one fold model that validated it, so the result can be compared with that fold's recorded prediction.
- penumbra_ml: the Tmax > 6 s rule (DEC-7, DEC-26), thresholds.threshold_masks, as EXP-1.
- occlusion_site: volumes.occlusion_evidence then the contract rule, as EXP-1 and build.py.
- haemorrhage: EXP-16's preparation (isles_check.prepare: 5 mm slabs, windows, JPEG), EXP-6's model and calibration,
  DEC-28's threshold (build.build_case applies it).
- confidence: confidence.case_confidence with EXP-32's mapping fitted on all 129 development cases (DEC-27). For a
  development case that mapping is in-sample, so the development check compares confidence.core with EXP-33 only for
  information; the penumbra score must match.
- aspects null (DEC-9); warnings from confidence.core_warnings (DEC-27).
Ground-truth contracts (DEC-11) are built the EXP-34 way: the native reference infarct volume as core, no bleed,
confidence 1.0.

nnU-Net needs a results folder with plans.json and dataset.json beside each fold, and the Drive copies hold only
fold_N/. They are taken from the checkpoints themselves (init_args, the exact plans each model was trained with;
Parsa, 19 Nov 2026), after checking that the five agree and match each fold's debug.json and recorded plan. DEC-31's
rebuild with nnUNetv2_plan_and_preprocess is a cross-check (plan_differences).

Held-out cases are refused unless ALLOW_HELDOUT=1 (splits.check_ids, in isles24.load_case). Files with case IDs stay on
Drive; summaries hold counts only. The notebook is 60_heldout_contracts.ipynb.
"""

import ast
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from strokecopilot.contract import build
from strokecopilot.data import splits
from strokecopilot.vision.perfusion import confidence as conf
from strokecopilot.vision.perfusion import evaluate, thresholds, volumes
from strokecopilot.vision.perfusion import nnunet_wrapper as nw

EXPECTED_HELDOUT = 20
EXPECTED_EPOCH = 300
FOLDS = (0, 1, 2, 3, 4)
CHECKPOINT = "checkpoint_final.pth"
TRAINER_DIR = f"{nw.TRAINER}__{nw.PLANS}__{nw.CONFIGURATION}"
VISION_A_NAMES = ("ncct", "cbf", "tmax", "occlusion", "cow", "infarct")
# The runs, named exactly (a pattern also matched EXP-3's older fold 0 folder, found in EXP-61's first try, 19 Nov 2026).
CHECKPOINT_RUNS = dict(build.NNUNET_RUNS)  # EXP-14, EXP-18 to EXP-21: checkpoint_final.pth, debug.json, plan_3d_fullres.json
# Each fold's recorded validation masks. EXP-14 kept none; EXP-22 wrote them from EXP-14's checkpoint_final.pth (its
# config.json records that checkpoint's sha256, which the notebook checks).
VALIDATION_RUNS = {**CHECKPOINT_RUNS, 0: "EXP-22_nnunet_fold0_masks"}


def fold_path(runs_root, fold, runs=None):
    """A fold's nnU-Net folder in a named run (CHECKPOINT_RUNS by default). Raises if it is missing."""
    runs = CHECKPOINT_RUNS if runs is None else runs
    path = Path(runs_root) / runs[fold] / "nnUNet_results" / nw.DATASET_NAME / TRAINER_DIR / f"fold_{fold}"
    if not path.is_dir():
        raise FileNotFoundError(f"Fold {fold}'s folder is missing from {runs[fold]}.")
    return path


def validation_path(runs_root, fold):
    """The folder holding a fold's recorded validation masks."""
    path = fold_path(runs_root, fold, VALIDATION_RUNS) / "validation"
    if not path.is_dir():
        raise FileNotFoundError(f"Fold {fold}'s validation masks are missing from {VALIDATION_RUNS[fold]}.")
    return path


def recorded_plan_path(runs_root, fold):
    return Path(runs_root) / CHECKPOINT_RUNS[fold] / "plan_3d_fullres.json"


# The fields a development-check contract must reproduce exactly from EXP-33's. haemorrhage.confidence is compared as a
# largest difference instead (GPU arithmetic and the JPEG library can move the fourth decimal).
REPRODUCED_FIELDS = ("haemorrhage.present", "core_ml", "penumbra_ml", "mismatch_ratio", "mismatch_volume_ml", "occlusion_site",
                     "aspects", "warnings")


def canonical(obj):
    """A JSON round trip, so tuples and lists compare equal."""
    return json.loads(json.dumps(obj))


def differences(a, b, path=""):
    """Paths where two JSON-like objects differ (after a JSON round trip). Empty when they are equal."""
    a, b = canonical(a), canonical(b)
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            where = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                out.append(f"{where} (only in {'second' if key not in a else 'first'})")
            else:
                out.extend(differences(a[key], b[key], where))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} vs {len(b)})"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        try:
            return [] if np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12) else [path]
        except (TypeError, ValueError):
            return [path]
    return [] if a == b else [path]


def read_checkpoint(path):
    """What nnU-Net saved in a checkpoint besides the weights: init_args (plans, dataset_json, fold, configuration),
    the trainer's name and the epoch."""
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    return {"init_args": saved["init_args"], "trainer_name": saved.get("trainer_name"),
            "epoch": saved.get("current_epoch"), "mirroring": saved.get("inference_allowed_mirroring_axes")}


def debug_plans(debug_json):
    """The plans and dataset.json a fold's debug.json recorded (nnU-Net writes them as Python reprs)."""
    data = json.loads(Path(debug_json).read_text())

    def parse(value):
        return ast.literal_eval(value) if isinstance(value, str) else value

    return {"plans": parse(data["plans_manager"]), "configuration": parse(data["configuration_manager"]),
            "dataset_json": parse(data["dataset_json"]), "fold": int(data["fold"])}


def check_folds(saved, debug=None, recorded=None, epoch=EXPECTED_EPOCH):
    """Problems, as sentences, with the five folds' saved settings; empty means they agree.

    saved: {fold: read_checkpoint()}; debug: {fold: debug_plans()}; recorded: {fold: the run's plan_3d_fullres.json}.
    Each fold must be its own fold, our trainer, 3d_fullres and the expected epoch; all five must share plans and
    dataset.json; and each must match its debug.json and recorded plan when given.
    """
    problems = []
    if sorted(saved) != list(FOLDS):
        problems.append(f"folds {sorted(saved)} given, expected {list(FOLDS)}")
        return problems
    first = saved[FOLDS[0]]["init_args"]
    for k, s in saved.items():
        args = s["init_args"]
        if int(args["fold"]) != k:
            problems.append(f"fold {k}'s checkpoint says fold {args['fold']}")
        if args["configuration"] != nw.CONFIGURATION:
            problems.append(f"fold {k} is {args['configuration']}, not {nw.CONFIGURATION}")
        if s["trainer_name"] != nw.TRAINER:
            problems.append(f"fold {k} was trained by {s['trainer_name']}, not {nw.TRAINER}")
        if epoch is not None and s["epoch"] != epoch:
            problems.append(f"fold {k} stopped at epoch {s['epoch']}, not {epoch}")
        if differences(args["plans"], first["plans"]):
            problems.append(f"fold {k}'s plans differ from fold {FOLDS[0]}'s")
        if differences(args["dataset_json"], first["dataset_json"]):
            problems.append(f"fold {k}'s dataset.json differs from fold {FOLDS[0]}'s")
        if debug and k in debug:
            d = debug[k]
            if d["fold"] != k or differences(d["plans"], args["plans"]) or differences(d["dataset_json"], args["dataset_json"]):
                problems.append(f"fold {k}'s checkpoint and debug.json disagree")
        if recorded and k in recorded and differences(recorded[k], args["plans"]["configurations"][nw.CONFIGURATION]):
            problems.append(f"fold {k}'s checkpoint and recorded plan_3d_fullres.json disagree")
    return problems


def assemble_results(checkpoints, results_root, saved=None, copy=shutil.copyfile):
    """nnU-Net's results folder for the ensemble: plans.json and dataset.json from the checkpoints, and each fold's
    checkpoint_final.pth in fold_k/. checkpoints: {fold: path}. Raises if check_folds finds a problem.
    Returns counts and hashes only."""
    import hashlib

    saved = saved or {k: read_checkpoint(p) for k, p in checkpoints.items()}
    problems = check_folds(saved, epoch=None)
    if problems:
        raise ValueError("The fold checkpoints do not agree: " + "; ".join(problems) + ".")
    args = saved[FOLDS[0]]["init_args"]
    trainer_dir = Path(results_root) / nw.DATASET_NAME / TRAINER_DIR
    trainer_dir.mkdir(parents=True, exist_ok=True)
    (trainer_dir / "plans.json").write_text(json.dumps(args["plans"], indent=1))
    (trainer_dir / "dataset.json").write_text(json.dumps(args["dataset_json"], indent=1))
    for k, path in checkpoints.items():
        fold = trainer_dir / f"fold_{k}"
        fold.mkdir(exist_ok=True)
        copy(str(path), str(fold / CHECKPOINT))

    def sha(obj):
        return hashlib.sha256(json.dumps(canonical(obj), sort_keys=True).encode()).hexdigest()

    return {"trainer_dir": str(trainer_dir), "folds": sorted(checkpoints), "plans_sha256": sha(args["plans"]),
            "dataset_json_sha256": sha(args["dataset_json"]), "epochs": {k: saved[k]["epoch"] for k in sorted(saved)}}


def plan_differences(saved_plans, rebuilt_plans):
    """DEC-31's cross-check: where the plans rebuilt by nnUNetv2_plan_and_preprocess on the development data differ
    from the saved ones. Only the parts that change what a model sees at inference are compared: the 3d_fullres
    configuration, the per-channel intensity properties (CT normalisation), and the transposes."""
    keys = ("transpose_forward", "transpose_backward", "foreground_intensity_properties_per_channel")
    out = differences(saved_plans["configurations"][nw.CONFIGURATION], rebuilt_plans["configurations"][nw.CONFIGURATION],
                      f"configurations.{nw.CONFIGURATION}")
    for key in keys:
        out.extend(differences(saved_plans.get(key), rebuilt_plans.get(key), key))
    return out


def write_inputs(slim_root, cases, images_dir, folds=None, log_every=10):
    """nnU-Net's input files for `cases` (images only, no labels), named <case>_<channel>.nii.gz as in training
    (nnunet_wrapper.channel_arrays). Held-out cases are refused unless ALLOW_HELDOUT=1 (isles24.load_case)."""
    import nibabel as nib

    from strokecopilot.data import isles24

    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(cases, 1):
        loaded = isles24.load_case(case, slim_root, names=nw.INPUT_NAMES[:-1], folds=folds)
        for c, array in enumerate(nw.channel_arrays(loaded.arrays)):
            nib.save(nib.Nifti1Image(array, loaded.affine), str(images_dir / f"{case}_{c:04d}{nw.FILE_ENDING}"))
        if log_every and (i % log_every == 0 or i == len(cases)):
            print(f"Inputs written for {i}/{len(cases)} cases", flush=True)
    return images_dir


def predict_command(images_dir, out_dir, folds=FOLDS, device="cuda", checkpoint=CHECKPOINT, n_proc=2):
    """nnU-Net's own prediction: with several folds it averages their softmax (D1). Defaults as in training's
    validation: step size 0.5 and mirroring, so a single fold reproduces its validation prediction."""
    return ["nnUNetv2_predict", "-i", str(images_dir), "-o", str(out_dir), "-d", str(nw.DATASET_ID),
            "-c", nw.CONFIGURATION, "-tr", nw.TRAINER, "-p", nw.PLANS, "-f", *[str(f) for f in folds],
            "-chk", checkpoint, "-device", device, "-npp", str(n_proc), "-nps", str(n_proc)]


def load_prediction(pred_dir, case, shape=None):
    """A predicted mask as booleans; checks its shape against the case's arrays."""
    import nibabel as nib

    mask = np.asanyarray(nib.load(str(Path(pred_dir) / f"{case}{nw.FILE_ENDING}")).dataobj) > 0
    if shape is not None and mask.shape != tuple(shape):
        raise ValueError(f"A prediction has shape {mask.shape}, expected {tuple(shape)}.")
    return mask


def vision_a_inputs(case, pred):
    """Vision A's contract inputs and scores for one loaded case (isles24.Case with VISION_A_NAMES) and its predicted
    core mask. Unrounded volumes; build.build_case rounds them."""
    arrays, spacing = case.arrays, case.spacing_mm
    masks = thresholds.threshold_masks(arrays["cbf"], arrays["tmax"], arrays.get("ncct"))  # EXP-1, within Tmax > 6 s
    threshold_ml = volumes.mask_ml(masks["core"], spacing) if masks["core"] is not None else None
    penumbra_ml = volumes.mask_ml(masks["penumbra"], spacing)
    core_ml = volumes.mask_ml(pred, spacing)
    scores = evaluate.case_metrics(pred, arrays["infarct"], masks["valid"], spacing, case.native_volumes["infarct_ml"])
    return {"core_ml": core_ml, "threshold_core_ml": threshold_ml,
            "penumbra_ml": volumes.round_ml(penumbra_ml),  # as EXP-1's contract_penumbra_ml
            "evidence": volumes.occlusion_evidence(arrays["occlusion"], arrays["cow"], spacing),
            "ref_ml": float(case.native_volumes["infarct_ml"]), "scores": scores}


def haemorrhage_confidence(native_ncct, predict, calibrator):
    """EXP-16's primary reading of one native NCCT: 5 mm slabs, EXP-6's model, its calibration. The calibrated
    probability; build.build_case compares it with DEC-28's threshold."""
    from strokecopilot.vision.haemorrhage import isles_check as ic

    images, _ = ic.prepare(native_ncct)
    return float(ic.score(images, predict, calibrator)["confidence"])


def case_inputs(case, pred, mapping, haem_conf):
    """Everything build_case and build_ground_truth_case need for one case, plus its scores (for D3)."""
    a = vision_a_inputs(case, pred)
    threshold_ml = a["threshold_core_ml"]
    if threshold_ml is None:
        # DEC-27's mapping reads only the cut-off distance of nnU-Net's core; the threshold core enters only the
        # disagreement signal, which that mapping does not use. A mapping that does use it cannot go on.
        if "log_disagreement" in mapping["features"]:
            raise ValueError("The threshold core could not be computed, and this mapping needs it.")
        threshold_ml = a["core_ml"]
    confidence = conf.case_confidence(case, mapping, a["core_ml"], threshold_ml)
    return {**a, "haemorrhage_confidence": float(haem_conf), "confidence": confidence}


def build_pair(case_id, inputs, model_version, gt_model_version):
    """(predicted contract, ground-truth contract), both validated against the schema."""
    predicted = build.build_case(case_id, model_version, inputs["core_ml"], inputs["penumbra_ml"], inputs["evidence"],
                                 inputs["haemorrhage_confidence"], inputs["confidence"])
    truth = build.build_ground_truth_case(case_id, gt_model_version, inputs["ref_ml"], inputs["penumbra_ml"],
                                          inputs["evidence"])
    return predicted, truth


def check_case_set(cases, mode, folds=None):
    """Refuse a case list that does not fit the mode. "development": development cases only. "heldout": exactly the
    held-out cases, and only with ALLOW_HELDOUT=1. Messages carry counts only."""
    folds = folds or splits.load_folds()
    if mode == "development":
        splits.check_ids(cases, folds)
        if set(cases) - set(splits.development_ids(folds)):
            raise ValueError(f"{len(set(cases) - set(splits.development_ids(folds)))} cases are not development cases.")
    elif mode == "heldout":
        expected = set(splits.heldout_ids(folds))  # raises unless ALLOW_HELDOUT=1
        if set(cases) != expected or len(cases) != EXPECTED_HELDOUT:
            raise ValueError(f"{len(cases)} cases given; the held-out set is {len(expected)}, expected {EXPECTED_HELDOUT}.")
    else:
        raise ValueError(f"Unknown mode {mode!r}.")


def compare_with_reference(new, old, fields=REPRODUCED_FIELDS):
    """Per field, how many cases differ between two sets of contracts ({case: dict}), plus the largest core and
    penumbra differences in mL. Counts only."""
    common = sorted(set(new) & set(old))
    mismatched = Counter()
    worst = {"core_ml": 0.0, "penumbra_ml": 0.0, "haemorrhage_confidence": 0.0}
    def get(obj, field):
        for part in field.split("."):
            obj = obj.get(part) if isinstance(obj, dict) else None
        return obj

    for case in common:
        for f in fields:
            if differences(get(new[case], f), get(old[case], f)):
                mismatched[f] += 1
        for f in ("core_ml", "penumbra_ml"):
            a, b = new[case].get(f), old[case].get(f)
            if a is not None and b is not None:
                worst[f] = max(worst[f], abs(a - b))
        worst["haemorrhage_confidence"] = max(worst["haemorrhage_confidence"], abs(
            new[case]["haemorrhage"]["confidence"] - old[case]["haemorrhage"]["confidence"]))
    return {"cases": len(common), "mismatched": {f: mismatched.get(f, 0) for f in fields},
            "largest_difference": {k: round(v, 4) for k, v in worst.items()}}


def mask_agreement(a, b):
    """Voxels that differ between two masks, and their Dice (1.0 when both are empty)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError(f"Masks of shape {a.shape} and {b.shape}.")
    total = int(a.sum() + b.sum())
    return {"differing_voxels": int((a ^ b).sum()), "dice": 1.0 if total == 0 else float(2 * (a & b).sum() / total)}


def write_contracts(contracts, out_dir):
    """One <case>.json per contract (case IDs: Drive only). Refuses to overwrite an existing folder's files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clash = [c for c in contracts if (out / f"{c}.json").exists()]
    if clash:
        raise FileExistsError(f"{len(clash)} contracts already exist in the output folder; use a new folder.")
    for case, obj in contracts.items():
        (out / f"{case}.json").write_text(json.dumps(obj, indent=1) + "\n")
    return out


def on_drive(path):
    """True for a path inside Google Drive (Colab's mount or Drive for desktop)."""
    p = str(Path(os.path.expanduser(str(path))).resolve())
    return p.startswith("/content/drive/") or "GoogleDrive" in p
