"""The runs a pod can do: what each runs, its hour limit and its cost estimate (DEC-21).

Each entry is one Experiments row. `hours` is the watchdog's hard limit; `est_hours` sizes the balance check and the
approval message. Epoch times are estimates until EXP-13 measures them on a 4090.
"""

NNUNET_VERSION = "2.8.1"

RUNS = {
    "EXP-13": {"what": "nnU-Net fold 0 dummy: 2 epochs, resume to 3, predict and score (timing, workers, resume, collect, down)",
               "run_dir": "EXP-13_nnunet_pod_dummy", "args": "--mode dummy --fold 0", "hours": 1.5, "est_hours": 0.6},
    "EXP-14": {"what": "nnU-Net fold 0, 300 epochs (DEC-21)",
               "run_dir": "EXP-14_nnunet_fold0", "args": "--mode train --fold 0 --epochs 300", "hours": 9.0, "est_hours": 6.5},
}

# What collect brings back: never the preprocessed data or the tarball.
COLLECT = ["config.json", "metrics.json", "per_case.csv", "plan_3d_fullres.json", "job.log", "job.exit",
           "checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth", "training_log_*.txt", "progress.png", "debug.json"]


def get(exp):
    if exp not in RUNS:
        raise KeyError(f"{exp} is not a pod run. Known: {', '.join(RUNS)}.")
    return RUNS[exp]


def command(exp, commit):
    spec = get(exp)
    return (f"python -m strokecopilot.vision.perfusion.pod_job --exp {exp} {spec['args']} --commit {commit} "
            f"--tar /root/sc/data/isles24_slim_v1.tar --run-dir /root/sc/runs/{exp} --work /root/sc/work")
