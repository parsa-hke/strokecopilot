"""nnU-Net v2 for Vision A: dataset conversion, our folds, a short-run trainer, training and scoring (plan §8.1).

Only development cases go into the nnU-Net dataset, and `splits_final.json` holds our five folds, so
nnU-Net never makes its own split and never sees a held-out case (DEC-5). Inputs are six channels on
the slim grid: NCCT, CBF, CBV, MTT, Tmax and a Tmax-valid mask. Tmax below 0 is no reliable data
(docs/DATA.md), so it is set to 0 there and the mask tells the network where that happened. The label
is the whole final infarct; scoring reports Dice on the reliable part as well (evaluate.py).

The trainer `nnUNetTrainer_StrokeCopilot` reads its epoch count, iterations per epoch and checkpoint
interval from the environment, so a Colab session can resume a run from Drive with the same settings.
"""

import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np

from strokecopilot.data import isles24, splits
from strokecopilot.vision.perfusion import evaluate, thresholds

NNUNET_VERSION = "2.8.1"
DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID}_ISLES24Perf"
CONFIGURATION = "3d_fullres"
PLANS = "nnUNetPlans"
TRAINER = "nnUNetTrainer_StrokeCopilot"
CHANNELS = ("ncct", "cbf", "cbv", "mtt", "tmax", "tmax_valid")
CHANNEL_NAMES = {"0": "CT", "1": "zscore", "2": "zscore", "3": "zscore", "4": "zscore", "5": "noNorm"}
FILE_ENDING = ".nii.gz"
INPUT_NAMES = ("ncct", "cbf", "cbv", "mtt", "tmax", "infarct")

TRAINER_SOURCE = '''import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_StrokeCopilot(nnUNetTrainer):
    """Fewer epochs and frequent checkpoints, set by strokecopilot notebooks through the environment."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ["SC_NUM_EPOCHS"])
        self.num_iterations_per_epoch = int(os.environ["SC_ITERS_PER_EPOCH"])
        self.save_every = int(os.environ["SC_SAVE_EVERY"])
'''


def channel_arrays(arrays):
    """The six input channels, float32, from load_case arrays."""
    tmax = np.asarray(arrays["tmax"], dtype=np.float32)
    valid = np.isfinite(tmax) & (tmax >= 0)
    out = [np.asarray(arrays["ncct"], dtype=np.float32)]
    for name in ("cbf", "cbv", "mtt"):
        out.append(np.clip(np.nan_to_num(np.asarray(arrays[name], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0, None))
    out.append(np.where(valid, tmax, 0.0).astype(np.float32))
    out.append(valid.astype(np.float32))
    return out


def dataset_json(n_cases):
    return {
        "channel_names": CHANNEL_NAMES,
        "labels": {"background": 0, "infarct": 1},
        "numTraining": n_cases,
        "file_ending": FILE_ENDING,
        "name": DATASET_NAME,
        "description": "ISLES'24 slim v1 development cases: NCCT, CBF, CBV, MTT, Tmax (negative set to 0), Tmax-valid mask",
        "licence": "CC BY-NC-SA 4.0",
    }


def _case_done(images, labels, case):
    files = [os.path.join(images, f"{case}_{i:04d}{FILE_ENDING}") for i in range(len(CHANNELS))]
    return all(os.path.exists(f) for f in files + [os.path.join(labels, f"{case}{FILE_ENDING}")])


def convert(slim_root, cases, raw_root, folds=None, log_every=20):
    """Write the nnU-Net raw dataset for `cases` (development only). Cases already written are skipped."""
    splits.check_ids(cases, folds)
    dataset = os.path.join(raw_root, DATASET_NAME)
    images, labels = os.path.join(dataset, "imagesTr"), os.path.join(dataset, "labelsTr")
    os.makedirs(images, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    for i, case in enumerate(cases, 1):
        if not _case_done(images, labels, case):
            loaded = isles24.load_case(case, slim_root, names=INPUT_NAMES, folds=folds)
            for c, array in enumerate(channel_arrays(loaded.arrays)):
                nib.save(nib.Nifti1Image(array, loaded.affine), os.path.join(images, f"{case}_{c:04d}{FILE_ENDING}"))
            label = loaded.arrays["infarct"].astype(np.uint8)
            nib.save(nib.Nifti1Image(label, loaded.affine), os.path.join(labels, f"{case}{FILE_ENDING}"))
        if log_every and (i % log_every == 0 or i == len(cases)):
            print(f"Converted {i}/{len(cases)} cases", flush=True)
    present = sorted(f[: -len(FILE_ENDING)] for f in os.listdir(labels) if f.endswith(FILE_ENDING))
    if present != sorted(cases):
        raise ValueError(f"The raw dataset holds {len(present)} cases, expected {len(cases)}. Remove {dataset} and convert again.")
    with open(os.path.join(dataset, "dataset.json"), "w") as f:
        json.dump(dataset_json(len(cases)), f, indent=1)
    return dataset


def splits_final(folds=None):
    """Our five folds in nnU-Net's format: a list of {"train": [...], "val": [...]}, fold k at index k."""
    folds = folds or splits.load_folds()
    out = []
    for k in sorted(folds["folds"], key=int):
        train, val = splits.train_val_ids(k, folds)
        out.append({"train": train, "val": val})
    return out


def write_splits(preprocessed_root, folds=None):
    path = os.path.join(preprocessed_root, DATASET_NAME, "splits_final.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits_final(folds), f, indent=1)
    return path


def install_trainer(package_dir=None):
    """Put the trainer where nnU-Net looks for trainers: inside its installed package."""
    if package_dir is None:
        import nnunetv2

        package_dir = nnunetv2.__path__[0]
    path = os.path.join(package_dir, "training", "nnUNetTrainer", "variants", "nnUNetTrainer_StrokeCopilot.py")
    with open(path, "w") as f:
        f.write(TRAINER_SOURCE)
    return path


def environment(raw_root, preprocessed_root, results_root, epochs, iterations, save_every, n_proc_da=None):
    env = dict(os.environ)
    env.update({
        "nnUNet_raw": str(raw_root), "nnUNet_preprocessed": str(preprocessed_root), "nnUNet_results": str(results_root),
        "SC_NUM_EPOCHS": str(epochs), "SC_ITERS_PER_EPOCH": str(iterations), "SC_SAVE_EVERY": str(save_every),
    })
    if n_proc_da:
        env["nnUNet_n_proc_DA"] = str(n_proc_da)
    return env


def fold_dir(results_root, fold):
    return os.path.join(results_root, DATASET_NAME, f"{TRAINER}__{PLANS}__{CONFIGURATION}", f"fold_{fold}")


def train_command(fold, resume):
    command = ["nnUNetv2_train", str(DATASET_ID), CONFIGURATION, str(fold), "-tr", TRAINER]
    return command + (["--c"] if resume else [])


def run_streaming(command, env, quiet=("it/s]",)):
    """Run a command and print its output as it comes, without progress-bar noise. Returns the exit code."""
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        if not any(q in line for q in quiet):
            sys.stdout.write(line)
            sys.stdout.flush()
    return process.wait()


def checkpoint_state(fold_path):
    """Which checkpoints a fold folder holds: final means training finished."""
    names = ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth")
    return {name: os.path.exists(os.path.join(fold_path, name)) for name in names}


def epoch_times(fold_path):
    """Epoch times in seconds, from nnU-Net's training logs in a fold folder."""
    times = []
    for name in sorted(os.listdir(fold_path)):
        if name.startswith("training_log") and name.endswith(".txt"):
            with open(os.path.join(fold_path, name)) as f:
                for line in f:
                    if "Epoch time:" in line:
                        times.append(float(line.split("Epoch time:")[1].split("s")[0]))
    return times


def score_predictions(pred_dir, slim_root, cases, fold, folds=None):
    """Our metrics (evaluate.case_metrics) for nnU-Net's predictions of `cases`, one row per case."""
    rows = []
    for case in cases:
        loaded = isles24.load_case(case, slim_root, names=("cbf", "tmax", "infarct"), folds=folds)
        pred = np.asanyarray(nib.load(os.path.join(pred_dir, f"{case}{FILE_ENDING}")).dataobj) > 0
        if pred.shape != loaded.arrays["infarct"].shape:
            raise ValueError(f"Prediction for {case} has shape {pred.shape}, expected {loaded.arrays['infarct'].shape}.")
        valid = thresholds.reliable(loaded.arrays["tmax"], loaded.arrays["cbf"])
        row = {"case": case, "fold": fold}
        row.update(evaluate.case_metrics(pred, loaded.arrays["infarct"], valid, loaded.spacing_mm, loaded.native_volumes["infarct_ml"]))
        rows.append(row)
    return rows
