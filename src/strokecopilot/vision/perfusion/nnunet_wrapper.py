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
import re
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
FINGERPRINT_EXTRACTOR = "StrokeCopilotFingerprintExtractor"
# nnU-Net samples 1e8 foreground voxels per channel for its intensity statistics, which peaked at 5.7 GiB on
# synthetic cases of our size and ran a 12.7 GB Colab runtime out of memory. 1e7 is still about 77,000 per case.
FOREGROUND_SAMPLES = 10_000_000
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


FINGERPRINT_SOURCE = f'''from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor


class StrokeCopilotFingerprintExtractor(DatasetFingerprintExtractor):
    """nnU-Net's extractor with fewer foreground samples for the intensity statistics, to fit Colab memory."""

    def __init__(self, dataset_name_or_id, num_processes: int = 8, verbose: bool = False):
        super().__init__(dataset_name_or_id, num_processes, verbose)
        self.num_foreground_voxels_for_intensitystats = {FOREGROUND_SAMPLES}
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


def splits_final(folds=None, val_limit=None):
    """Our five folds in nnU-Net's format: a list of {"train": [...], "val": [...]}, fold k at index k.
    `val_limit` keeps only the first n validation cases of each fold, for a dummy validation run."""
    folds = folds or splits.load_folds()
    out = []
    for k in sorted(folds["folds"], key=int):
        train, val = splits.train_val_ids(k, folds)
        out.append({"train": train, "val": val[:val_limit] if val_limit else val})
    return out


def write_splits(preprocessed_root, folds=None, val_limit=None):
    path = os.path.join(preprocessed_root, DATASET_NAME, "splits_final.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits_final(folds, val_limit), f, indent=1)
    return path


def install_extensions(package_dir=None):
    """Put our trainer and fingerprint extractor where nnU-Net looks for them: inside its installed package."""
    if package_dir is None:
        import nnunetv2

        package_dir = nnunetv2.__path__[0]
    paths = {
        os.path.join(package_dir, "training", "nnUNetTrainer", "variants", f"{TRAINER}.py"): TRAINER_SOURCE,
        os.path.join(package_dir, "experiment_planning", "dataset_fingerprint", "strokecopilot_fingerprint.py"): FINGERPRINT_SOURCE,
    }
    for path, source in paths.items():
        with open(path, "w") as f:
            f.write(source)
    return list(paths)


def environment(raw_root, preprocessed_root, results_root, epochs, iterations, save_every, n_proc=None, n_proc_da=None):
    """nnU-Net's paths and our trainer's settings. n_proc caps nnU-Net's default worker count, and n_proc_da
    the training data workers: each worker starts a fresh Python that imports PyTorch, so the defaults of
    8 and up to 12 ran a Colab L4 runtime out of memory on 4 November 2026 (exit code -9)."""
    env = dict(os.environ)
    env.update({
        "nnUNet_raw": str(raw_root), "nnUNet_preprocessed": str(preprocessed_root), "nnUNet_results": str(results_root),
        "SC_NUM_EPOCHS": str(epochs), "SC_ITERS_PER_EPOCH": str(iterations), "SC_SAVE_EVERY": str(save_every),
    })
    if n_proc:
        env["nnUNet_def_n_proc"] = str(n_proc)
    if n_proc_da:
        env["nnUNet_n_proc_DA"] = str(n_proc_da)
    return env


def plan_command(n_proc):
    """Fingerprint, plan and preprocess 3d_fullres with n_proc workers for each step."""
    return ["nnUNetv2_plan_and_preprocess", "-d", str(DATASET_ID), "-c", CONFIGURATION, "--verify_dataset_integrity",
            "-fpe", FINGERPRINT_EXTRACTOR, "-npfp", str(n_proc), "-np", str(n_proc)]


def memory_gib(meminfo="/proc/meminfo"):
    """Total and available RAM in GiB on Linux, or None elsewhere."""
    try:
        with open(meminfo) as f:
            values = {line.split(":")[0]: int(line.split()[1]) for line in f if ":" in line}
    except OSError:
        return None
    return {"total": round(values["MemTotal"] / 2**20, 1), "available": round(values["MemAvailable"] / 2**20, 1)}


def explain_exit(code, step):
    """A message for a failed nnU-Net command; -9 means the system killed it, almost always for memory."""
    if code == -9:
        return (f"{step} was killed (exit code -9), almost always because the runtime ran out of memory. "
                f"RAM now: {memory_gib()}. Lower N_PROC or N_PROC_DA in section 1 and Run all again.")
    return f"{step} failed with exit code {code}. Send the output above."


def fold_dir(results_root, fold):
    return os.path.join(results_root, DATASET_NAME, f"{TRAINER}__{PLANS}__{CONFIGURATION}", f"fold_{fold}")


def train_command(fold, resume):
    command = ["nnUNetv2_train", str(DATASET_ID), CONFIGURATION, str(fold), "-tr", TRAINER]
    return command + (["--c"] if resume else [])


def validate_command(fold, device="cuda"):
    """nnU-Net's own validation of a finished fold: loads fold_k/checkpoint_final.pth and writes fold_k/validation/,
    the same masks a training run writes at its end."""
    return ["nnUNetv2_train", str(DATASET_ID), CONFIGURATION, str(fold), "-tr", TRAINER, "--val", "-device", device]


def checkpoint_plan(checkpoint):
    """(3d_fullres plan, fold) a checkpoint was trained with, from the init arguments nnU-Net saves in it."""
    import torch

    init = torch.load(checkpoint, map_location="cpu", weights_only=False)["init_args"]
    return init["plans"]["configurations"][init["configuration"]], int(init["fold"])


def same_plan(a, b):
    """Plans compared after a JSON round trip, so tuples and lists compare equal."""
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


_PERCENT = re.compile(r"(\d+)%\|")


def run_streaming(command, env, every_percent=10):
    """Run a command and print its output as it comes. Progress bars print every `every_percent` percent,
    so long steps stay visible without flooding the notebook. Returns the exit code."""
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = {}
    for line in process.stdout:
        match = _PERCENT.search(line)
        if match:
            label, percent = line[: match.start()].strip(), int(match.group(1))
            if percent < 100 and percent // every_percent == last.get(label, -1):
                continue
            last[label] = percent // every_percent
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
