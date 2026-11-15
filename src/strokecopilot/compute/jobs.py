"""The runs a pod can do: what each runs, its hour limit and its cost estimate (DEC-21).

Each entry is one Experiments row. `hours` is the watchdog's hard limit; `est_hours` sizes the balance check and the
approval message. EXP-13 measured 37-44 s per epoch on a 4090 after the first (69 s, with compilation), so 300
epochs take about 3.3-3.7 h, plus about 15 min of setup and a few minutes of validation.

A run may name others to run first on the same pod (`with`): each keeps its own run folder, log and exit code,
`collect` brings all of them back, and a failure in one does not stop the next. `inputs` are files sent up from
Drive for Desktop before the job starts (Drive-relative path -> path under the pod's /root/sc).
"""

NNUNET_VERSION = "2.8.1"

# EXP-14's final weights, from which EXP-22 regenerates fold 0's validation masks (they stayed on its destroyed pod).
EXP14_CHECKPOINT = ("runs/EXP-14_nnunet_fold0/nnUNet_results/Dataset501_ISLES24Perf/"
                    "nnUNetTrainer_StrokeCopilot__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth")


def _fold(k):
    return {"what": f"nnU-Net fold {k}, 300 epochs (DEC-21)", "run_dir": f"EXP-{17 + k}_nnunet_fold{k}",
            "args": f"--mode train --fold {k} --epochs 300", "hours": 5.5, "est_hours": 4.0}


# Open reasoner models served with vLLM on one rented 48 GB GPU (DEC-24, hosting amended 15 Nov 2026). A `serve` run
# carries no data and no job: the pod only serves one model at a time to the Mac through an SSH tunnel, and the
# pipeline on the Mac sends the requests (pipeline/paid_runs.py). The watchdog still enforces the hour limit.
SERVE_FILTERS = {
    "num_gpus": {"eq": 1}, "gpu_ram": {"gte": 44_000}, "verified": {"eq": True}, "rentable": {"eq": True},
    "rented": {"eq": False}, "reliability": {"gte": 0.98}, "cpu_cores_effective": {"gte": 8}, "cpu_ram": {"gte": 48_000},
    "disk_space": {"gte": 200}, "inet_down": {"gte": 500}, "dph_total": {"lte": 0.90}, "cuda_max_good": {"gte": 12.8},
}
SERVE_DISK_GB = 200  # two 27B models in 16-bit weights are about 110 GB of downloads
# The pair served. If Parsa rules the Gemma pair out, this becomes ("qwen2.5-32b", "baichuan-m2-32b") (DEC-24).
SERVE_MODELS = ("gemma-3-27b", "medgemma-27b-text")
# Policy switches, off until Parsa decides; a commit that turns one on is the record of that decision.
# allow_gated: a Hugging Face token (read-only, for these repositories) goes to the pod, which DEC-21 forbade.
# allow_medgemma: Parsa has read the HAI-DEF terms section 3.2.
# Both decided by Parsa on 16 Nov 2026 (DEC-21 amended: a fine-grained read-only token for the two gated repositories,
# stored only root-readable on the pod, removed once the weights are downloaded, and revoked after the run; DEC-24:
# MedGemma's HAI-DEF terms cleared).
SERVE_POLICY = {"allow_gated": True, "allow_medgemma": True}


def _serve(what, run_dir, hours, est_hours):
    return {"what": what, "run_dir": run_dir, "hours": hours, "est_hours": est_hours, "filters": SERVE_FILTERS,
            "disk_gb": SERVE_DISK_GB, "serve": {"models": SERVE_MODELS, **SERVE_POLICY}}


RUNS = {
    "EXP-36": _serve("Open reasoner models on a 48 GB GPU: one test request per model (C8 dummy)",
                     "EXP-36_paid_runs_test", 1.5, 0.8),
    "EXP-37": _serve("Open reasoner models on a 48 GB GPU: the paid runs' reasoner pass (C8)",
                     "EXP-37_paid_runs", 5.0, 3.5),
    "EXP-13": {"what": "nnU-Net fold 0 dummy: 2 epochs, resume to 3, predict and score (timing, workers, resume, collect, down)",
               "run_dir": "EXP-13_nnunet_pod_dummy", "args": "--mode dummy --fold 0", "hours": 1.5, "est_hours": 0.6},
    "EXP-14": {"what": "nnU-Net fold 0, 300 epochs (DEC-21)",
               "run_dir": "EXP-14_nnunet_fold0", "args": "--mode train --fold 0 --epochs 300", "hours": 5.5, "est_hours": 4.0},
    # Fold 1 also carries EXP-22 (a few minutes of inference) before training, hence the extra half hour.
    "EXP-18": {**_fold(1), "what": "nnU-Net fold 1, 300 epochs (DEC-21), after EXP-22 on the same pod", "with": ["EXP-22"],
               "hours": 6.0, "est_hours": 4.2},
    "EXP-19": _fold(2),
    "EXP-20": _fold(3),
    "EXP-21": _fold(4),
    "EXP-22": {"what": "Fold 0 validation masks regenerated from EXP-14's checkpoint_final.pth (nnU-Net --val)",
               "run_dir": "EXP-22_nnunet_fold0_masks", "args": "--mode predict --fold 0 --source-exp EXP-14 "
               "--checkpoint /root/sc/inputs/EXP-22/checkpoint_final.pth",
               "inputs": {EXP14_CHECKPOINT: "inputs/EXP-22/checkpoint_final.pth"}, "hours": 1.0, "est_hours": 0.2},
}

# What collect brings back: never the preprocessed data or the tarball.
COLLECT = ["config.json", "metrics.json", "per_case.csv", "plan_3d_fullres.json", "job.log", "job_part*.log", "job.exit", "validation/*.nii.gz",
           "checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth", "training_log_*.txt", "progress.png", "debug.json"]


def get(exp):
    if exp not in RUNS:
        raise KeyError(f"{exp} is not a pod run. Known: {', '.join(RUNS)}.")
    return RUNS[exp]


def chain(exp):
    """The runs one pod does for `exp`, in order: the ones it carries first, then itself."""
    return [*get(exp).get("with", []), exp]


def inputs(exp):
    """Every input file the pod needs for `exp` and the runs it carries: {Drive-relative path: pod path}."""
    out = {}
    for e in chain(exp):
        out.update({src: f"/root/sc/{dst}" for src, dst in get(e).get("inputs", {}).items()})
    return out


def _single(exp, commit):
    spec = get(exp)
    return (f"python -m strokecopilot.vision.perfusion.pod_job --exp {exp} {spec['args']} --commit {commit} "
            f"--tar /root/sc/data/isles24_slim_v1.tar --run-dir /root/sc/runs/{exp} --work /root/sc/work")


def command(exp, commit):
    """The pod's command line. Carried runs write their own job.log and job.exit, and the main run follows
    whatever their exit code; the main run's output goes to its job.log through remote.start_job."""
    parts = []
    for e in get(exp).get("with", []):
        run = f"/root/sc/runs/{e}"
        parts.append(f"mkdir -p {run} && ( {_single(e, commit)} ) > {run}/job.log 2>&1; echo $? > {run}/job.exit")
    parts.append(_single(exp, commit))
    return "; ".join(parts)
