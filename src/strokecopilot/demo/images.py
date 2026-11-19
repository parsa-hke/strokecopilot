"""ISLES'24 pictures for the Space's story (DEC-32 §6): one development case, shown by its bank alias, never by ID.

ISLES'24 is CC BY-NC-SA 4.0, so images may appear with attribution (the page and the Space README carry it). Only a
development case is used before Phase 4's results; the case is picked in code and only its alias is printed.

Four panels at the axial slice where the reference infarct is largest: plain CT; CBF; Tmax; and the plain CT with
nnU-Net's predicted core (the validation mask of the fold model that never trained on the case), the Tmax > 6 s area
(the penumbra rule, DEC-7) and the reference final infarct outlined. Written into the bank folder (gitignored build).

    python -m strokecopilot.demo.images --bank demo/space/build/bank
"""

import argparse
import json
from pathlib import Path

import numpy as np

PANEL_PX = 360


def pick_slice(*masks):
    """The axial slice (last axis) with the most voxels in any of the masks."""
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union |= np.asarray(m, bool)
    return int(np.argmax(union.sum(axis=(0, 1))))


def _orient(a):
    """RAS array slice -> image rows (anterior at the top, the patient's right on the left, radiological view)."""
    return np.rot90(a)


def render(case, pred, out_dir, alias, volumes=None):
    """Write the four panels and a small JSON caption for one loaded isles24.Case and its predicted core mask."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from strokecopilot.vision.perfusion import thresholds

    a = case.arrays
    masks = thresholds.threshold_masks(a["cbf"], a["tmax"], a["ncct"])
    z = pick_slice(a["infarct"])
    ncct = _orient(np.clip(a["ncct"][..., z], 0, 80))
    cbf = _orient(np.where(np.isfinite(a["cbf"][..., z]), a["cbf"][..., z], 0))
    tmax = _orient(np.where(a["tmax"][..., z] >= 0, a["tmax"][..., z], np.nan))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, draw):
        fig = plt.figure(figsize=(PANEL_PX / 100, PANEL_PX / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        fig.patch.set_facecolor("#0b0b0b")
        draw(ax)
        fig.savefig(out / name, facecolor=fig.get_facecolor())
        plt.close(fig)

    save("story_ncct.png", lambda ax: ax.imshow(ncct, cmap="gray", interpolation="bilinear"))
    save("story_cbf.png", lambda ax: ax.imshow(cbf, cmap="magma", vmin=0, vmax=np.nanpercentile(cbf[cbf > 0], 98) if (cbf > 0).any() else 1,
                                               interpolation="bilinear"))
    save("story_tmax.png", lambda ax: ax.imshow(np.nan_to_num(tmax, nan=0), cmap="viridis", vmin=0, vmax=12, interpolation="bilinear"))

    def overlay(ax):
        ax.imshow(ncct, cmap="gray", interpolation="bilinear")
        pen = _orient(masks["penumbra"][..., z])
        ax.imshow(np.ma.masked_where(~pen, pen), cmap=matplotlib.colors.ListedColormap(["#eda100"]), alpha=0.35)
        core = _orient(np.asarray(pred, bool)[..., z])
        ax.imshow(np.ma.masked_where(~core, core), cmap=matplotlib.colors.ListedColormap(["#e34948"]), alpha=0.75)
        ref = _orient(np.asarray(a["infarct"], bool)[..., z]).astype(float)
        if ref.any():
            ax.contour(ref, levels=[0.5], colors=["#ffffff"], linewidths=1.2)

    save("story_overlay.png", overlay)
    caption = {"alias": alias, "slice_mm": round(float(z * case.spacing_mm), 1), "licence": "ISLES'24, CC BY-NC-SA 4.0",
               **(volumes or {}),
               "panels": {"story_ncct.png": "Plain CT", "story_cbf.png": "Cerebral blood flow (CBF)",
                          "story_tmax.png": "Time to maximum (Tmax), 0-12 s",
                          "story_overlay.png": "Predicted core (red), Tmax > 6 s (amber), reference final infarct (white outline)"}}
    (out / "story.json").write_text(json.dumps(caption, indent=1))
    return caption


def main(argv=None):
    from strokecopilot.config import local_drive_root
    from strokecopilot.contract.heldout import validation_path
    from strokecopilot.data import isles24, splits
    from strokecopilot.pipeline import heldout_analysis as ha

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", required=True)
    parser.add_argument("--slim", default="data/local/isles24_slim_v1")
    args = parser.parse_args(argv)
    bank = Path(args.bank)
    index = json.loads((bank / "index.json").read_text())["cases"]
    runs = local_drive_root() / "runs"
    exp37 = Path("runs/EXP-37_paid_runs/claude-sonnet-5-5/real_predicted/metrics_rescore.json")
    ids = sorted(r["case"] for r in json.loads(exp37.read_text())["per_case"])
    splits.check_ids(ids)  # development cases only
    aliases = ha.aliases_for(ids, "D")
    folds = splits.load_folds()
    fold_of = {c: int(k) for k, members in folds["folds"].items() for c in members}
    # The development case with the clearest picture: a sizeable reference infarct that nnU-Net also found.
    gt = ha.load_contracts(local_drive_root() / "artifacts" / "vision_outputs_gt_v1", ids)
    pred = ha.load_contracts(local_drive_root() / "artifacts" / "vision_outputs_v1", ids)
    ranked = sorted(ids, key=lambda c: -(min(gt[c]["core_ml"], 80) * (pred[c]["core_ml"] >= 5) * (not pred[c]["haemorrhage"]["present"])))
    case_id = ranked[0]
    loaded = isles24.load_case(case_id, args.slim, names=("ncct", "cbf", "tmax", "infarct"), folds=folds)
    import nibabel as nib

    mask = np.asanyarray(nib.load(str(validation_path(runs, fold_of[case_id]) / f"{case_id}.nii.gz")).dataobj) > 0
    caption = render(loaded, mask, bank / "img", aliases[case_id],
                     {"core_ml_predicted": pred[case_id]["core_ml"], "core_ml_reference": gt[case_id]["core_ml"]})
    if any(aliases[case_id] != c and c in json.dumps(caption) for c in ids):
        raise ValueError("A case ID reached the caption.")
    if aliases[case_id] not in {r["id"] for r in index}:
        raise ValueError("The chosen case is not in the bank.")
    print(f"OK: story images for {aliases[case_id]} in {bank / 'img'}.")


if __name__ == "__main__":
    main()
