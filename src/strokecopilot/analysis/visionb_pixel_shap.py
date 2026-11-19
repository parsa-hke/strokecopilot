"""SHAP survey row 6 (DEC-35): pixel attributions for Vision B's false alarms on ISLES'24, beside Item 2's Grad-CAM.

EXP-50 (Item 2) found that the 68 development scans Vision B flags at DEC-28's 0.175 have their top slabs in brain
tissue (Grad-CAM: median 76% of the heat in brain). This asks the same question with a second method, expected
gradients (`shap.GradientExplainer`), on the same slabs and with the same tissue masks, so the two can be compared:

- **Scans and slabs:** EXP-50's preparation (`isles_false_alarms.scan_case`: 5 mm slabs, windows) and EXP-6's model; a
  scan is flagged when its top-2 score reaches DEC-28's threshold, exactly as EXP-50 called "none".
- **Explained:** the logit of the top 2 slabs of each flagged scan. **Background:** the top slab of 50 unflagged scans,
  drawn by seed, so an attribution says what makes this slab look more like a bleed than a typical unflagged top slab.
- **Read-out per slab** (no case ID): the share of |attribution| (summed over the three windows) in brain, bone, other
  head tissue and outside the head (`explain.heat_shares` on `explain.tissue_masks`), Grad-CAM's shares on the same
  slab, and the rank correlation of the two maps at Grad-CAM's resolution.

ISLES'24 images may appear with attribution (CC BY-NC-SA 4.0); no RSNA image is used.
"""

import json
from pathlib import Path

import numpy as np

SEED = 20261120
BACKGROUND = 50
NSAMPLES = 100
KEYS = ("brain", "bone", "other", "outside")


def _pool(x, size=10):
    """Mean-pool a H x W map to size x size (Grad-CAM's layer4 grid at 320 px), for a rank correlation."""
    h, w = x.shape
    return x[: h - h % size, : w - w % size].reshape(size, h // size, size, w // size).mean(axis=(1, 3))


def slab_rows(images, attributions, cams):
    """Per slab: SHAP and Grad-CAM tissue shares, and their maps' Spearman correlation (pooled to 10 x 10)."""
    from scipy import stats

    from strokecopilot.vision.haemorrhage import explain as ex
    rows = []
    for image, att, cam in zip(images, attributions, cams):
        masks = ex.tissue_masks(image)
        mag = np.abs(att).sum(axis=-1)
        shap_sh, cam_sh = ex.heat_shares(mag, masks), ex.heat_shares(cam, masks)
        rho = stats.spearmanr(_pool(mag).ravel(), _pool(cam).ravel()).statistic
        pos = np.clip(att.sum(axis=-1), 0, None)
        rows.append({"shap": shap_sh, "shap_positive": ex.heat_shares(pos, masks), "gradcam": cam_sh,
                     "spearman_shap_gradcam": float(rho)})
    return rows


def summarise(rows):
    med = lambda key, k: float(np.median([r[key][k] for r in rows if r[key][k] is not None]))
    return {"slabs": len(rows),
            "median_share": {m: {k: med(m, k) for k in KEYS} for m in ("shap", "shap_positive", "gradcam")},
            "slabs_brain_over_bone": {m: int(sum(r[m]["brain"] > r[m]["bone"] for r in rows if r[m]["brain"] is not None))
                                      for m in ("shap", "gradcam")},
            "median_spearman": float(np.median([r["spearman_shap_gradcam"] for r in rows])),
            "spearman_iqr": [float(np.percentile([r["spearman_shap_gradcam"] for r in rows], q)) for q in (25, 75)]}


def figure(examples, path):
    """Up to four flagged top slabs: the brain window, Grad-CAM, and |SHAP| (ISLES'24, CC BY-NC-SA 4.0)."""
    from strokecopilot.pipeline.figures import INK_2, _plt
    plt = _plt()
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.4 * n))
    axes = np.atleast_2d(axes)
    for row, ex_ in zip(axes, examples):
        brain = ex_["image"][..., 0]
        mag = np.abs(ex_["shap"]).sum(axis=-1)
        row[0].imshow(brain, cmap="gray")
        row[1].imshow(brain, cmap="gray")
        row[1].imshow(ex_["cam"], cmap="magma", alpha=0.5)
        row[2].imshow(brain, cmap="gray")
        row[2].imshow(mag / (np.percentile(mag, 99.5) or 1), cmap="magma", alpha=0.6, vmin=0, vmax=1)
        for ax, title in zip(row, ("brain window", "Grad-CAM (Item 2)", "|SHAP|, expected gradients")):
            ax.set_title(title, fontsize=8, color=INK_2, loc="left")
            ax.axis("off")
    fig.text(0.01, 0.0, "ISLES'24 (CC BY-NC-SA 4.0); flagged development scans' top slabs, no case ID.", fontsize=7,
             color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    import argparse
    import hashlib
    import os
    import random
    import time

    import shap
    import torch

    from strokecopilot.data import isles24, splits
    from strokecopilot.vision.haemorrhage import explain as ex
    from strokecopilot.vision.haemorrhage import infer
    from strokecopilot.vision.haemorrhage import isles_check as ic
    from strokecopilot.vision.haemorrhage import isles_false_alarms as fa
    from strokecopilot.vision.haemorrhage import model as hmodel
    from strokecopilot.vision.haemorrhage import slab_rules as sr
    from strokecopilot.vision.haemorrhage import train

    ap = argparse.ArgumentParser(description="SHAP row 6: Vision B pixel attributions on ISLES'24's false alarms")
    ap.add_argument("--out", default="runs/EXP-72_visionB_pixel_shap")
    ap.add_argument("--model-dir", default=f"{fa.DRIVE}/runs/EXP-6_visionB_resnet50")
    ap.add_argument("--limit", type=int, default=0, help="dummy run: stop after this many flagged scans")
    args = ap.parse_args(argv)
    started = time.time()
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    if hashlib.sha256(splits.FOLDS_PATH.read_bytes()).hexdigest() != fa.FOLDS_SHA256:
        raise ValueError("The installed split is not the fixed one. Stop and tell Parsa.")
    folds = splits.load_folds()
    root = Path(__file__).resolve().parents[3] / "data" / "local" / isles24.SLIM_NAME
    if not (root / "manifest.json").exists():
        raise SystemExit("the slim ISLES'24 folder is not in data/local (EXP-50 unpacks it)")
    cases = isles24.list_cases(root, folds)
    if len(cases) != fa.EXPECTED_DEVELOPMENT:
        raise ValueError(f"{len(cases)} development cases, expected {fa.EXPECTED_DEVELOPMENT}.")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_dir = Path(os.path.expanduser(args.model_dir))
    net, _ = train.load_best(model_dir / "best.pt", device)
    net.eval()
    contract = infer.contract_calibrator(model_dir / "calibration.json")
    fa.check_aggregation(contract)
    threshold = fa.thresholds(contract)["none"][2]

    flagged, unflagged_top = [], []
    for case in cases:
        images, probs, _, _ = fa.scan_case(isles24.load_native_ncct(case, root, folds),
                                           lambda x: infer.slice_probs(net, x, device))
        top = np.argsort(-probs, kind="stable")[: fa.TOP]
        if sr.scan_score(probs, None, "none", None) >= threshold:
            flagged.append(images[top])
        else:
            unflagged_top.append(images[top[0]])
    n_flagged = len(flagged)
    if args.limit:
        flagged = flagged[: args.limit]
    rng = random.Random(SEED)
    background = np.stack(rng.sample(unflagged_top, min(BACKGROUND, len(unflagged_top))))

    class Logit(torch.nn.Module):  # normalised input -> the single logit, as Grad-CAM explains it
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).float()

    to_x = lambda imgs: hmodel.normalise(torch.as_tensor(np.ascontiguousarray(imgs)), device).contiguous()
    explainer = shap.GradientExplainer(Logit(net), to_x(background))
    images = np.concatenate(flagged)
    attributions = []
    for i in range(0, len(images), 4):
        sv = explainer.shap_values(to_x(images[i : i + 4]), nsamples=NSAMPLES, rseed=SEED)
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
        if sv.ndim == 5:
            sv = sv[..., 0]
        attributions.append(np.transpose(sv, (0, 2, 3, 1)))  # N x H x W x 3
    attributions = np.concatenate(attributions)
    cams, _ = ex.grad_cam(net, images, device)
    rows = slab_rows(images, attributions, cams)
    metrics = {"experiment": "EXP-72", "flagged_scans": n_flagged, "explained_scans": len(flagged),
               "background_slabs": len(background), "nsamples": NSAMPLES, "device": str(device),
               **summarise(rows), "minutes": round((time.time() - started) / 60, 1)}
    text = json.dumps(metrics, indent=1)
    if ic.ids_in(text, cases + list(folds["heldout"])):
        raise ValueError("A case ID reached metrics.json. Nothing was saved.")
    (out / "metrics.json").write_text(text)
    (out / "per_slab.json").write_text(json.dumps(rows, indent=1))
    order = np.argsort([-r["shap"]["brain"] for r in rows[::2]])[:4] * 2  # four top slabs (rank 0), most brain first
    figure([{"image": images[k], "cam": cams[k], "shap": attributions[k]} for k in order], out / "pixel_shap_examples.png")
    print(text)


if __name__ == "__main__":
    main()
