"""SHAP survey row 1 (DEC-35): exact Shapley values of Vision A's input channels, on the 129 development cases.

nnU-Net reads six channels (`nnunet_wrapper.channel_arrays`): NCCT, CBF, CBV, MTT, Tmax and a Tmax-valid mask. The
game has five players, Tmax and its mask being one: NCCT, CBF, CBV, MTT, Tmax. Every one of the 32 coalitions is
predicted, so the values are exact. Each case is predicted by the one fold model that validated it (DEC-25), with
nnU-Net's validation settings (step 0.5, mirroring), so the full coalition must reproduce the recorded prediction.

**An absent input** (DEC-35's baselines), filled only inside the case's nonzero region so that nnU-Net crops the same
box in every coalition:
- CBF, CBV, MTT: the channel's own mean over that region. nnU-Net normalises these per image (z-score), so the channel
  becomes flat: no spatial information, the case's own level.
- NCCT: nnU-Net's dataset foreground mean (from the plans), which its CT normalisation maps to 0.
- Tmax: Tmax 0 and its mask 0, which is how the network was trained to see Tmax with no reliable data.

**Outputs per coalition:** predicted volume (mL), Dice against the reference final infarct, and whether the volume is
on the wrong side of any of 21, 31, 51 and 70 mL. **The limit:** no model was trained with an input missing, so these
values measure how much the trained network relies on an input, not what a model trained without it would do.

Colab (`notebooks/15_visionA_channel_shap.ipynb`) writes one row per case and coalition to a JSON-lines file on Drive
(case IDs: Drive only). `summarise` on the Mac turns the rows into Shapley values and counts.
"""

import json
from pathlib import Path

import numpy as np

PLAYERS = ("NCCT", "CBF", "CBV", "MTT", "Tmax")
CHANNELS = {"NCCT": (0,), "CBF": (1,), "CBV": (2,), "MTT": (3,), "Tmax": (4, 5)}
CUTOFFS = (21.0, 31.0, 51.0, 70.0)
FULL = (1 << len(PLAYERS)) - 1


def masked(img, mask, ncct_mean):
    """img: (6, ...) as nnU-Net's reader returns it. mask: bit i set = PLAYERS[i] present."""
    region = np.any(img != 0, axis=0)
    out = img.copy()
    for i, player in enumerate(PLAYERS):
        if mask >> i & 1:
            continue
        if player == "Tmax":
            out[4] = 0
            out[5] = 0
        else:
            c = CHANNELS[player][0]
            fill = ncct_mean if player == "NCCT" else float(img[c][region].mean())
            out[c] = np.where(region, fill, 0).astype(out.dtype)
    return out, region


def wrong_side(pred_ml, ref_ml, cutoffs=CUTOFFS):
    return int(any((pred_ml < c) != (ref_ml < c) for c in cutoffs))


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def done_keys(rows_path):
    path = Path(rows_path)
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add((r["case"], r["mask"]))
    return keys


def run_case(case, files, label_file, predictor, ncct_mean, rows_path, done=frozenset(), masks=None):
    """Predict one case for every coalition not yet done, appending a row per coalition to `rows_path`."""
    import time

    reader = predictor.plans_manager.image_reader_writer_class()
    img, props = reader.read_images(files)
    ref, _ = reader.read_seg(label_file)
    ref = ref[0] > 0
    voxel_ml = float(np.prod(props["spacing"])) / 1000.0
    ref_ml = float(ref.sum()) * voxel_ml
    full_region = np.any(img != 0, axis=0)
    out = []
    for mask in (range(FULL + 1) if masks is None else masks):
        if (case, mask) in done:
            continue
        x, region = masked(img, mask, ncct_mean)
        if not np.array_equal(np.any(x != 0, axis=0), full_region):
            raise ValueError(f"coalition {mask}: the nonzero region changed, so nnU-Net would crop differently")
        t = time.time()
        seg = predictor.predict_single_npy_array(x, props, None, None, False) > 0
        row = {"case": case, "mask": mask, "pred_ml": float(seg.sum()) * voxel_ml, "ref_ml": ref_ml,
               "dice": dice(seg, ref), "seconds": round(time.time() - t, 2)}
        row["wrong_side"] = wrong_side(row["pred_ml"], ref_ml)
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------- the Mac

def load_rows(rows_path):
    rows = {}
    for line in Path(rows_path).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[(r["case"], r["mask"])] = r
    return rows


def summarise(rows, recorded_ml=None, size_bands=((0, 5), (5, 21), (21, 70), (70, float("inf")))):
    """Shapley values per case (pred_ml, dice, wrong_side) and the summary. recorded_ml: {case: EXP-27 nnU-Net mL}, to
    check the full coalition reproduces the recorded prediction. Counts and means only; no case ID is returned."""
    from strokecopilot.analysis.shapley import shapley_from_values

    outputs = ("pred_ml", "dice", "wrong_side")
    cases = sorted({c for c, _ in rows})
    complete = [c for c in cases if all((c, m) in rows for m in range(FULL + 1))]
    per_case = []
    for c in complete:
        table = np.array([[rows[(c, m)][k] for k in outputs] for m in range(FULL + 1)], dtype=float)
        phi = shapley_from_values(table)  # (5, 3)
        per_case.append({"ref_ml": rows[(c, FULL)]["ref_ml"], "phi": phi, "full": table[-1], "none": table[0]})
    out = {"cases": len(cases), "complete": len(complete), "outputs": list(outputs), "players": list(PLAYERS)}
    if recorded_ml:
        diffs = [abs(rows[(c, FULL)]["pred_ml"] - recorded_ml[c]) for c in complete if c in recorded_ml]
        out["full_coalition_reproduces_recorded"] = {"compared": len(diffs), "max_ml_difference": max(diffs) if diffs else None,
                                                     "within_0.05_ml": sum(d <= 0.05 for d in diffs)}
    for k, name in enumerate(outputs):
        mag = np.array([np.abs(p["phi"][:, k]) for p in per_case])
        signed = np.array([p["phi"][:, k] for p in per_case])
        out[name] = {"mean_abs": dict(zip(PLAYERS, mag.mean(0).tolist())) if len(mag) else None,
                     "mean": dict(zip(PLAYERS, signed.mean(0).tolist())) if len(signed) else None,
                     "share": dict(zip(PLAYERS, (mag.mean(0) / mag.mean(0).sum()).tolist())) if len(mag) and mag.sum() else None,
                     "full_mean": float(np.mean([p["full"][k] for p in per_case])) if per_case else None,
                     "none_mean": float(np.mean([p["none"][k] for p in per_case])) if per_case else None}
        bands = {}
        for lo, hi in size_bands:
            sel = [p for p in per_case if lo <= p["ref_ml"] < hi]
            if sel:
                bands[f"{lo:g}-{hi:g} mL"] = {"cases": len(sel), "mean_abs": dict(zip(
                    PLAYERS, np.mean([np.abs(p["phi"][:, k]) for p in sel], axis=0).tolist()))}
        out[name]["by_reference_size"] = bands
    return out


def main(argv=None):
    """On the Mac, after notebook 15: Shapley values from the rows on Drive, checked against EXP-27's recorded volumes."""
    import argparse
    import csv

    from strokecopilot.config import local_drive_root
    ap = argparse.ArgumentParser(description="SHAP row 1: Vision A channels")
    ap.add_argument("--run", default="EXP-75_visionA_channel_shap", help="the run folder name on Drive")
    ap.add_argument("--out", default="runs/EXP-75_visionA_channel_shap")
    args = ap.parse_args(argv)
    drive = local_drive_root()
    rows = load_rows(drive / "runs" / args.run / "rows.jsonl")
    recorded = {r["case"]: float(r["nnunet_pred_ml"]) for r in
                csv.DictReader(open(drive / "runs" / "EXP-27_visionA_error_followups" / "per_case.csv"))}
    summary = summarise(rows, recorded)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"experiment": "EXP-75", **summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
