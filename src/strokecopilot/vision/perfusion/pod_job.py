"""Notebook 10 without the notebook: the nnU-Net run a rented pod does (DEC-21, plan §4.6).

    python -m strokecopilot.vision.perfusion.pod_job --exp EXP-13 --mode dummy --fold 0 --commit <sha> \
        --tar /root/sc/data/isles24_slim_v1.tar --run-dir /root/sc/runs/EXP-13 --work /root/sc/work

Same steps and checks as notebooks/10_visionA_nnunet.ipynb: tarball and split checked by sha256, development cases
only (counts printed, never IDs), our folds in splits_final.json, plan saved, train with resume, our metrics on the
fold's validation cases, a held-out leak check, then config.json, per_case.csv and metrics.json (last: it marks a
finished run). It also samples GPU use every 10 s, which shows whether the data workers keep the GPU busy.

`--mode predict` trains nothing: it puts a finished run's checkpoint_final.pth (`--checkpoint`) in the fold folder
and runs nnU-Net's own validation, which writes the fold's validation masks, then scores them like a training run.
It refuses a checkpoint whose plan or fold differs from this run's. The copied checkpoint is deleted afterwards, so
collect does not bring a second copy back. `--limit n` keeps the first n validation cases, and `--device cpu|mps`
lets that dummy run on a laptop.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

from strokecopilot.data import archive_listing as al
from strokecopilot.data import isles24, splits
from strokecopilot.vision.perfusion import evaluate
from strokecopilot.vision.perfusion import nnunet_wrapper as nw

SLIM_SHA256 = "b0afd16e4cbfb16a69fb5190083fc3869f5dcc6c92f86f10135b0de1f59f4728"
FOLDS_SHA256 = "023813f2dba930b7521653f1549b1fa1369b8bd29b63658175c5cea58410d291"
EXPECTED_DEVELOPMENT = 129
ITERATIONS_PER_EPOCH = 250
MODES = {
    "dummy": {"epochs": 2, "resume_epochs": 3, "save_every": 1},
    "train": {"epochs": None, "resume_epochs": None, "save_every": 10},
    "predict": {"epochs": 1, "resume_epochs": None, "save_every": 1},
}


def effective_cpus(root="/sys/fs/cgroup"):
    """CPUs this container may use: the cgroup quota when there is one (os.cpu_count() shows the whole host on a
    rented pod), else the affinity mask."""
    try:
        quota, period = open(os.path.join(root, "cpu.max")).read().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota = int(open(os.path.join(root, "cpu", "cpu.cfs_quota_us")).read())
        period = int(open(os.path.join(root, "cpu", "cpu.cfs_period_us")).read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)


def workers(cpus):
    """(n_proc, n_proc_da): augmentation workers leave two CPUs for the main process and the GPU feeder."""
    return max(2, min(8, cpus // 2)), max(2, min(16, cpus - 2))


class GpuSampler:
    """Samples GPU utilisation and memory every `every` seconds in a thread, for the worker-bottleneck reading."""

    def __init__(self, every=10):
        self.every, self.samples, self._stop = every, [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True)
            try:
                util, mem = (float(x) for x in out.stdout.strip().splitlines()[0].split(","))
                self.samples.append((round(time.time()), util, mem))
            except (ValueError, IndexError):
                pass
            self._stop.wait(self.every)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=30)
        return summarise_gpu(self.samples)


def gpu_name():
    """The GPU's name and memory from nvidia-smi, or "" where there is none (a laptop dummy)."""
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def summarise_gpu(samples):
    if not samples:
        return None
    util = sorted(s[1] for s in samples)
    return {"samples": len(util), "util_mean": round(sum(util) / len(util), 1), "util_median": util[len(util) // 2],
            "share_below_50": round(sum(u < 50 for u in util) / len(util), 3), "mem_max_mib": max(s[2] for s in samples)}


def unpack(tar, parent):
    """Check the tarball, unpack it beside itself into `parent`, and delete it to free the disk."""
    root = os.path.join(parent, isles24.SLIM_NAME)
    if os.path.exists(os.path.join(root, "manifest.json")):
        return root
    if al.sha256_of(tar) != SLIM_SHA256:
        raise ValueError("The tarball is not the one the build recorded. Stop and check it.")
    os.makedirs(parent, exist_ok=True)
    subprocess.run(["tar", "-xf", tar, "-C", parent], check=True)
    os.remove(tar)
    return root


def leaked(texts, folds):
    """How many held-out IDs appear in any of the texts (0 is the only acceptable answer)."""
    return sum(case in text for case in folds["heldout"] for text in texts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-proc-da", type=int)
    ap.add_argument("--checkpoint", help="predict: a finished run's checkpoint_final.pth")
    ap.add_argument("--source-exp", help="predict: the run that checkpoint came from, for the record")
    ap.add_argument("--limit", type=int, help="predict: keep only the first n validation cases (a dummy)")
    ap.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    a = ap.parse_args(argv)
    settings = dict(MODES[a.mode])
    if a.mode == "train":
        if not a.epochs:
            raise SystemExit("--epochs is required in train mode.")
        settings["epochs"] = a.epochs
    if a.mode == "predict" and not (a.checkpoint and os.path.isfile(a.checkpoint)):
        raise SystemExit("--mode predict needs --checkpoint pointing at a checkpoint_final.pth.")
    if a.mode != "predict" and (a.checkpoint or a.limit or a.device != "cuda"):
        raise SystemExit("--checkpoint, --limit and --device are for --mode predict only.")
    os.makedirs(a.run_dir, exist_ok=True)
    if os.path.exists(os.path.join(a.run_dir, "metrics.json")):
        raise SystemExit(f"{a.run_dir} already holds a finished run.")

    gpu = gpu_name()
    import torch
    cpus = effective_cpus()
    n_proc, n_proc_da = workers(cpus)
    n_proc_da = a.n_proc_da or n_proc_da
    print(f"{a.exp} {a.mode} at {a.commit[:7]}; nnU-Net {nw.NNUNET_VERSION}; torch {torch.__version__}; GPU {gpu or 'none'}; "
          f"{cpus} CPUs usable ({os.cpu_count()} on the host); RAM {nw.memory_gib()} GiB; workers {n_proc}/{n_proc_da}.", flush=True)
    if a.device == "cuda" and (not gpu or not torch.cuda.is_available()):
        raise SystemExit("No GPU visible to torch.")

    if al.sha256_of(splits.FOLDS_PATH) != FOLDS_SHA256:
        raise ValueError("The installed split is not the fixed one. Stop and check it.")
    folds = splits.load_folds()
    print(f"Split: {len(folds['heldout'])} held out; folds of " + ", ".join(str(len(v)) for v in folds["folds"].values()) + ".", flush=True)

    timings = {}
    started = time.time()
    slim_root = unpack(a.tar, a.work)
    development = isles24.list_cases(slim_root, folds)
    if len(development) != EXPECTED_DEVELOPMENT or set(development) != set(splits.development_ids(folds)):
        raise ValueError(f"The loader lists {len(development)} cases, not the {EXPECTED_DEVELOPMENT} development cases.")
    timings["unpack_s"] = round(time.time() - started)

    started = time.time()
    raw, preprocessed = f"{a.work}/nnunet/raw", f"{a.work}/nnunet/preprocessed"
    results = os.path.join(a.run_dir, "nnUNet_results")
    dataset = nw.convert(slim_root, development, raw, folds)
    timings["convert_s"] = round(time.time() - started)
    print(f"Unpacked in {timings['unpack_s']} s, converted {len(development)} cases in {timings['convert_s']} s; "
          f"free disk {shutil.disk_usage(a.work).free / 2**30:.0f} GiB.", flush=True)

    env = nw.environment(raw, preprocessed, results, epochs=settings["epochs"], iterations=ITERATIONS_PER_EPOCH,
                         save_every=settings["save_every"], n_proc=n_proc, n_proc_da=n_proc_da)
    nw.install_extensions()
    started = time.time()
    code = nw.run_streaming(nw.plan_command(n_proc), env)
    if code:
        raise RuntimeError(nw.explain_exit(code, "Planning and preprocessing"))
    timings["preprocess_s"] = round(time.time() - started)
    plan = json.load(open(f"{preprocessed}/{nw.DATASET_NAME}/nnUNetPlans.json"))["configurations"][nw.CONFIGURATION]
    saved_plan = os.path.join(a.run_dir, "plan_3d_fullres.json")
    if os.path.exists(saved_plan) and json.load(open(saved_plan)) != plan:
        raise ValueError("This run planned a different network from the one it started with.")
    json.dump(plan, open(saved_plan, "w"), indent=1)
    splits_path = nw.write_splits(preprocessed, folds, val_limit=a.limit)
    print(f"Preprocessed in {timings['preprocess_s']} s. Patch {plan['patch_size']}, batch {plan['batch_size']}, "
          f"spacing {plan['spacing']}. Our folds written to splits_final.json.", flush=True)

    fold_path = nw.fold_dir(results, a.fold)
    if a.mode == "predict":
        return predict(a, settings, plan, fold_path, env, slim_root, folds, splits_path, dataset, timings, gpu, cpus, torch)
    resume = os.path.isdir(fold_path) and any(nw.checkpoint_state(fold_path).values())
    sampler = GpuSampler().start()
    started = time.time()
    code = nw.run_streaming(nw.train_command(a.fold, resume=resume), env)
    if code:
        raise RuntimeError(nw.explain_exit(code, "Training"))
    if settings["resume_epochs"]:
        env = nw.environment(raw, preprocessed, results, epochs=settings["resume_epochs"], iterations=ITERATIONS_PER_EPOCH,
                             save_every=settings["save_every"], n_proc=n_proc, n_proc_da=n_proc_da)
        print("Resuming from the checkpoint for one more epoch.", flush=True)
        code = nw.run_streaming(nw.train_command(a.fold, resume=True), env)
        if code:
            raise RuntimeError(nw.explain_exit(code, "Resuming") + " This is what the dummy run checks.")
    timings["train_s"] = round(time.time() - started)
    gpu_use = sampler.stop()
    if not nw.checkpoint_state(fold_path)["checkpoint_final.pth"]:
        raise RuntimeError("Training ended without checkpoint_final.pth.")

    val = list(folds["folds"][str(a.fold)])
    rows = nw.score_predictions(os.path.join(fold_path, "validation"), slim_root, val, a.fold, folds)
    summary = evaluate.summarise(rows)
    times = nw.epoch_times(fold_path)
    s = summary["per_fold"][str(a.fold)]

    config = {
        "experiment": a.exp, "mode": a.mode, "runner": "strokecopilot.vision.perfusion.pod_job (DEC-21 pod)", "git_commit": a.commit,
        "data_version": "isles24_slim_v1.tar", "data_sha256": SLIM_SHA256, "folds_sha256": FOLDS_SHA256, "fold": a.fold,
        "nnunet_version": nw.NNUNET_VERSION, "dataset": nw.DATASET_NAME, "configuration": nw.CONFIGURATION, "trainer": nw.TRAINER,
        "channels": nw.CHANNELS, "channel_normalisation": nw.CHANNEL_NAMES, "label": "final infarct, whole mask",
        "epochs": settings["resume_epochs"] or settings["epochs"], "iterations_per_epoch": ITERATIONS_PER_EPOCH,
        "save_every": settings["save_every"], "patch_size": plan["patch_size"], "batch_size": plan["batch_size"],
        "spacing": plan["spacing"], "torch": torch.__version__, "gpu": gpu, "cpus_usable": cpus, "cpus_host": os.cpu_count(),
        "ram_gib": nw.memory_gib(), "n_proc": n_proc, "n_proc_da": n_proc_da,
    }
    metrics = {
        "experiment": a.exp, "git_commit": a.commit, "data_version": "isles24_slim_v1.tar", "fold": a.fold, "summary": summary,
        "epoch_seconds": times, "epoch_seconds_median": round(sorted(times)[len(times) // 2], 1) if times else None,
        "gpu_use": gpu_use, "timings_s": timings,
    }
    save(a, config, metrics, rows, folds, splits_path, dataset, fold_path)
    print(f"OK: {a.exp} {a.mode} at {a.commit[:7]}; {gpu}; epoch median {metrics['epoch_seconds_median']} s; "
          f"GPU use {gpu_use}; fold {a.fold} Dice {s['dice']['mean']:.3f} ± {s['dice']['std']:.3f} over {s['dice']['n']} cases; "
          f"timings {timings}.", flush=True)
    return 0


def save(a, config, metrics, rows, folds, splits_path, dataset, fold_path):
    """Leak-check, then write config.json, per_case.csv and metrics.json (last: it marks a finished run)."""
    import pandas as pd
    texts = {"config.json": json.dumps(config, indent=1), "per_case.csv": pd.DataFrame(rows).to_csv(index=False),
             "metrics.json": json.dumps(metrics, indent=1)}
    validation = os.path.join(fold_path, "validation")
    names = os.listdir(validation) if os.path.isdir(validation) else []
    checked = [*texts.values(), open(splits_path).read(), open(f"{dataset}/dataset.json").read(), *names]
    if leaked(checked, folds) or any(case in os.listdir(f"{dataset}/labelsTr") for case in folds["heldout"]):
        raise ValueError("A held-out ID reached the outputs or the nnU-Net dataset. Nothing was saved. Stop and check.")
    for name in ("config.json", "per_case.csv", "metrics.json"):
        with open(os.path.join(a.run_dir, name), "w") as f:
            f.write(texts[name])


def predict(a, settings, plan, fold_path, env, slim_root, folds, splits_path, dataset, timings, gpu, cpus, torch):
    """Validation masks from a finished run's checkpoint, scored like a training run (see the module docstring)."""
    trained_plan, trained_fold = nw.checkpoint_plan(a.checkpoint)
    if trained_fold != a.fold or not nw.same_plan(trained_plan, plan):
        raise ValueError(f"The checkpoint was trained on fold {trained_fold} with a different plan or fold from this run "
                         f"(fold {a.fold}). Its masks would not be the run's. Stop and check.")
    ckpt_sha = al.sha256_of(a.checkpoint)
    os.makedirs(fold_path, exist_ok=True)
    target = os.path.join(fold_path, "checkpoint_final.pth")
    shutil.copyfile(a.checkpoint, target)
    val = list(folds["folds"][str(a.fold)])[: a.limit or None]
    print(f"Checkpoint matches this run's plan and fold {a.fold}; validating {len(val)} cases on {a.device}.", flush=True)
    sampler = GpuSampler().start() if a.device == "cuda" else None
    started = time.time()
    code = nw.run_streaming(nw.validate_command(a.fold, a.device), env)
    timings["validate_s"] = round(time.time() - started)
    gpu_use = sampler.stop() if sampler else None
    os.remove(target)
    if code:
        raise RuntimeError(nw.explain_exit(code, "Validation"))
    rows = nw.score_predictions(os.path.join(fold_path, "validation"), slim_root, val, a.fold, folds)
    summary = evaluate.summarise(rows)
    s = summary["per_fold"][str(a.fold)]
    config = {
        "experiment": a.exp, "mode": a.mode, "runner": "strokecopilot.vision.perfusion.pod_job (DEC-21 pod)", "git_commit": a.commit,
        "data_version": "isles24_slim_v1.tar", "data_sha256": SLIM_SHA256, "folds_sha256": FOLDS_SHA256, "fold": a.fold,
        "source_experiment": a.source_exp, "checkpoint": "checkpoint_final.pth", "checkpoint_sha256": ckpt_sha,
        "cases_limit": a.limit, "device": a.device, "nnunet_version": nw.NNUNET_VERSION, "dataset": nw.DATASET_NAME,
        "configuration": nw.CONFIGURATION, "trainer": nw.TRAINER, "channels": nw.CHANNELS, "label": "final infarct, whole mask",
        "patch_size": plan["patch_size"], "batch_size": plan["batch_size"], "spacing": plan["spacing"],
        "torch": torch.__version__, "gpu": gpu, "cpus_usable": cpus,
    }
    metrics = {"experiment": a.exp, "git_commit": a.commit, "data_version": "isles24_slim_v1.tar", "fold": a.fold,
               "source_experiment": a.source_exp, "summary": summary, "gpu_use": gpu_use, "timings_s": timings}
    save(a, config, metrics, rows, folds, splits_path, dataset, fold_path)
    print(f"OK: {a.exp} predict at {a.commit[:7]} from {a.source_exp or 'a checkpoint'}; fold {a.fold} Dice "
          f"{s['dice']['mean']:.3f} ± {s['dice']['std']:.3f} over {s['dice']['n']} cases; timings {timings}.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
