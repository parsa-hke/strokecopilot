"""Phase 5 task H (DEC-37): EXP-6's ResNet-50 exported to ONNX, with parity, size and CPU latency.

- **Export:** the network as trained (one "any haemorrhage" logit), input float32 N x 3 x 320 x 320 already
  normalised the way `model.normalise` does it (ImageNet mean and std on the uint8 windows), batch size dynamic, opset 17.
- **Parity:** PyTorch and ONNX Runtime (CPU) on the same slabs: every 5 mm slab of 20 ISLES'24 development scans drawn by
  seed (EXP-50's preparation), compared as probabilities, and as scan scores and DEC-28 calls for those scans. No RSNA
  image is used here (they stay on Kaggle), and no case ID is saved.
- **Size and latency:** file sizes, and the median of 30 timed runs at batch 1 and batch 32 on the Mac's CPU for both
  runtimes, after warm-up, single-threaded and with all threads.

The model weights stay private to the team (docs/DATA_USE.md): the .onnx file is written to the Drive run folder, never
to the repo.
"""

import json
import time
from pathlib import Path

import numpy as np

SEED = 20261120
N_SCANS = 20
OPSET = 17


def export(net, path, size=320):
    import torch
    net = net.eval().float().to("cpu")
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(net, (dummy,), str(path), input_names=["image"], output_names=["logit"],
                      dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}}, opset_version=OPSET, dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def normalised(images):
    """uint8 N x H x W x 3 -> float32 N x 3 x H x W, as model.normalise does on the GPU."""
    from strokecopilot.vision.haemorrhage import model as hmodel
    x = np.asarray(images, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
    mean = np.asarray(hmodel.MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(hmodel.STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return np.ascontiguousarray((x - mean) / std)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def timed(fn, x, runs=30, warmup=3):
    for _ in range(warmup):
        fn(x)
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        fn(x)
        t.append(time.perf_counter() - s)
    return float(np.median(t) * 1000)


def parity(p_torch, p_onnx, scans, calibrator):
    """Slab and scan agreement. scans: list of (start, stop) slab ranges per scan."""
    from strokecopilot.vision.haemorrhage import slab_rules as sr
    d = np.abs(np.asarray(p_torch) - np.asarray(p_onnx))
    s_t = np.array([sr.scan_score(p_torch[a:b], None, "none", None) for a, b in scans])
    s_o = np.array([sr.scan_score(p_onnx[a:b], None, "none", None) for a, b in scans])
    c_t = np.asarray(calibrator.apply(s_t)) >= calibrator.threshold
    c_o = np.asarray(calibrator.apply(s_o)) >= calibrator.threshold
    return {"slabs": int(len(d)), "scans": int(len(scans)), "max_abs_prob_difference": float(d.max()),
            "mean_abs_prob_difference": float(d.mean()), "max_abs_scan_score_difference": float(np.abs(s_t - s_o).max()),
            "same_dec28_call": int((c_t == c_o).sum()), "flagged_torch": int(c_t.sum()), "flagged_onnx": int(c_o.sum())}


def main(argv=None):
    import argparse
    import hashlib
    import os
    import random

    import onnxruntime as ort
    import torch

    from strokecopilot.config import local_drive_root
    from strokecopilot.data import isles24, splits
    from strokecopilot.vision.haemorrhage import infer, train
    from strokecopilot.vision.haemorrhage import isles_check as ic
    from strokecopilot.vision.haemorrhage import isles_false_alarms as fa

    ap = argparse.ArgumentParser(description="Task H: EXP-6 to ONNX")
    ap.add_argument("--out", default="runs/EXP-78_onnx_export")
    ap.add_argument("--model-dir", default=f"{fa.DRIVE}/runs/EXP-6_visionB_resnet50")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    drive_out = local_drive_root() / "runs" / out.name
    drive_out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(os.path.expanduser(args.model_dir))
    net, state = train.load_best(model_dir / "best.pt", torch.device("cpu"))
    calibrator = infer.contract_calibrator(model_dir / "calibration.json")
    onnx_path = drive_out / "exp6_resnet50.onnx"  # private weights: Drive only
    export(net, onnx_path)

    if hashlib.sha256(splits.FOLDS_PATH.read_bytes()).hexdigest() != fa.FOLDS_SHA256:
        raise ValueError("The installed split is not the fixed one. Stop and tell Parsa.")
    folds = splits.load_folds()
    root = Path(__file__).resolve().parents[3] / "data" / "local" / isles24.SLIM_NAME
    cases = isles24.list_cases(root, folds)
    chosen = random.Random(SEED).sample(sorted(cases), N_SCANS)
    images, scans = [], []
    for case in chosen:
        slabs, _ = ic.prepare(isles24.load_native_ncct(case, root, folds))
        scans.append((sum(len(x) for x in images), sum(len(x) for x in images) + len(slabs)))
        images.append(slabs)
    x = normalised(np.concatenate(images))

    def torch_logits(batch):
        with torch.no_grad():
            return net(torch.from_numpy(batch)).float().squeeze(1).numpy()

    sessions = {}
    for threads in (1, 0):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        sessions[threads] = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    onnx_logits = lambda batch, s=sessions[0]: s.run(["logit"], {"image": batch})[0].reshape(-1)
    p_torch = np.concatenate([sigmoid(torch_logits(x[i:i + 32])) for i in range(0, len(x), 32)])
    p_onnx = np.concatenate([sigmoid(onnx_logits(x[i:i + 32])) for i in range(0, len(x), 32)])
    result = {"experiment": "EXP-78", "torch": torch.__version__, "onnxruntime": ort.__version__, "opset": OPSET,
              "model_epoch": state["epoch"], "parity": parity(p_torch, p_onnx, scans, calibrator),
              "size_mb": {"best_pt": round((model_dir / "best.pt").stat().st_size / 1e6, 1),
                          "onnx": round(onnx_path.stat().st_size / 1e6, 1)},
              "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(), "latency_ms": {}}
    for threads in (1, 0):
        torch.set_num_threads(1 if threads == 1 else os.cpu_count())
        s = sessions[threads]
        run = lambda b, s=s: s.run(["logit"], {"image": b})
        label = "1 thread" if threads == 1 else f"{os.cpu_count()} threads"
        result["latency_ms"][label] = {f"batch {b}": {"pytorch": round(timed(torch_logits, x[:b]), 1),
                                                     "onnxruntime": round(timed(run, x[:b]), 1)} for b in (1, 32)}
    text = json.dumps(result, indent=1)
    if ic.ids_in(text, cases + list(folds["heldout"])):
        raise ValueError("A case ID reached metrics.json. Nothing was saved.")
    for folder in (out, drive_out):
        (folder / "metrics.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
