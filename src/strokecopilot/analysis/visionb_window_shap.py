"""SHAP survey row 5 (DEC-35): exact Shapley values of Vision B's three CT windows (brain, subdural, bone).

The network reads three windows of the same HU values (`prep_rsna.WINDOWS`), so the question is which window it uses.
Three players, 8 coalitions, all evaluated: exact. **An absent window** is that channel set to its RSNA training mean
(a constant, measured on a seeded sample of training slices). The value of a coalition, per scan, is the calibrated
scan probability (EXP-6's top-2 aggregation and Platt scaling, as the contract applies them). Globally, the scan AUC
and the sensitivity and specificity at DEC-28's threshold are also read for each coalition.

- **RSNA validation (Kaggle, `notebooks/26_visionB_window_shap.ipynb`, EXP-76):** every validation slice under each
  coalition; summaries for bleeds (and per subtype), non-bleeds and false positives at DEC-28's threshold. Only
  aggregates leave Kaggle, and the RSNA training means, which the ISLES'24 half needs.
- **ISLES'24 development (Mac, `isles_main`, EXP-77):** the same coalitions on the 5 mm slabs of the 129 scans (EXP-50's
  preparation), with the means from EXP-76; summaries for the 68 flagged and the 61 unflagged scans. No case ID saved.

The test split is never read. The limit: the network never saw a window missing, so this measures reliance.
"""

import json
from pathlib import Path

import numpy as np

PLAYERS = ("brain", "subdural", "bone")
FULL = (1 << len(PLAYERS)) - 1
SUBTYPES = ("epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural")
SEED = 20261120


def fill_image(image, mask, fill):
    """uint8 (..., H, W, 3) image with the absent windows set to their fill values (bit i set = PLAYERS[i] present)."""
    out = image.copy() if hasattr(image, "copy") else image.clone()
    for c in range(len(PLAYERS)):
        if not mask >> c & 1:
            out[..., c] = int(round(fill[c]))
    return out


class MaskedSlices:
    """A map-style dataset (`train.SliceDataset` wrapped) whose images have the absent windows filled."""

    def __init__(self, base, mask, fill):
        self.base, self.mask, self.fill = base, mask, fill

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, label = self.base[i]
        return fill_image(image, self.mask, self.fill), label


def channel_means(images):
    """Mean of each channel over a stack of uint8 H x W x 3 images."""
    return [float(np.mean([np.asarray(im)[..., c].mean() for im in images])) for c in range(len(PLAYERS))]


def scan_values(probs_by_mask, slices, calibrator):
    """{mask: calibrated scan probability per scan} and the scan labels, in one scan order. slices: series_id, any and
    subtype columns; probs_by_mask: {mask: slice probabilities in slices' order}."""
    from strokecopilot.vision.haemorrhage import aggregate as agg
    out, labels = {}, None
    for mask, probs in probs_by_mask.items():
        scans = agg.scan_scores(slices.assign(p=probs), calibrator.method, calibrator.k, prob="p")
        out[mask] = np.asarray(calibrator.apply(scans["score"]), dtype=float)
        if labels is None:
            labels = scans[["series_id", "label"]]
            sub = slices.groupby("series_id")[[s for s in SUBTYPES if s in slices]].max()
            labels = labels.join(sub, on="series_id")
    return out, labels


def summarise(values, labels, threshold):
    """Shapley per scan and window, summarised by group; plus each coalition's AUC, sensitivity and specificity, and the
    Shapley values of the AUC. labels: label and subtype columns (no ID is returned)."""
    from strokecopilot.analysis.shapley import shapley_from_values
    from strokecopilot.vision.haemorrhage.aggregate import roc_auc
    table = np.stack([values[m] for m in range(FULL + 1)], axis=1)  # scans x 8
    phi = np.stack([shapley_from_values(row) for row in table])  # scans x 3
    y = labels["label"].to_numpy().astype(bool)
    called = table[:, FULL] >= threshold
    groups = {"bleed": y, "no bleed": ~y, "false positive": ~y & called, "true positive": y & called}
    for s in SUBTYPES:
        if s in labels:
            groups[f"bleed: {s}"] = labels[s].to_numpy().astype(bool)
    out = {"scans": int(len(y)), "bleeds": int(y.sum()), "threshold": threshold, "groups": {}}
    for g, sel in groups.items():
        if sel.any():
            out["groups"][g] = {"scans": int(sel.sum()),
                                "mean_abs_phi": dict(zip(PLAYERS, np.abs(phi[sel]).mean(0).tolist())),
                                "mean_phi": dict(zip(PLAYERS, phi[sel].mean(0).tolist())),
                                "mean_prob_full": float(table[sel, FULL].mean()), "mean_prob_none": float(table[sel, 0].mean())}
    coalitions = {}
    for m in range(FULL + 1):
        p = table[:, m]
        coalitions[str(m)] = {"present": [PLAYERS[i] for i in range(len(PLAYERS)) if m >> i & 1],
                              "auc": roc_auc(y, p) if y.any() and (~y).any() else None,
                              "sensitivity": float((p[y] >= threshold).mean()) if y.any() else None,
                              "specificity": float((p[~y] < threshold).mean()) if (~y).any() else None}
    if all(c["auc"] is not None for c in coalitions.values()):
        auc_phi = shapley_from_values(np.array([coalitions[str(m)]["auc"] for m in range(FULL + 1)]))
        out["auc_shapley"] = dict(zip(PLAYERS, auc_phi.tolist()))
    out["coalitions"] = coalitions
    return out


def isles_main(argv=None):
    """EXP-77 on the Mac: the same game on ISLES'24's 129 development scans, with EXP-76's RSNA means."""
    import argparse
    import hashlib
    import os

    import pandas as pd
    import torch

    from strokecopilot.data import isles24, splits
    from strokecopilot.vision.haemorrhage import infer
    from strokecopilot.vision.haemorrhage import isles_check as ic
    from strokecopilot.vision.haemorrhage import isles_false_alarms as fa
    from strokecopilot.vision.haemorrhage import slab_rules as sr
    from strokecopilot.vision.haemorrhage import train

    ap = argparse.ArgumentParser(description="SHAP row 5, ISLES'24 half (EXP-77)")
    ap.add_argument("--means", required=True, help="EXP-76's metrics.json (holds rsna_train_channel_means)")
    ap.add_argument("--out", default="runs/EXP-77_visionB_window_shap_isles")
    ap.add_argument("--model-dir", default=f"{fa.DRIVE}/runs/EXP-6_visionB_resnet50")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    fill = json.loads(Path(args.means).read_text())["rsna_train_channel_means"]
    if hashlib.sha256(splits.FOLDS_PATH.read_bytes()).hexdigest() != fa.FOLDS_SHA256:
        raise ValueError("The installed split is not the fixed one. Stop and tell Parsa.")
    folds = splits.load_folds()
    root = Path(__file__).resolve().parents[3] / "data" / "local" / isles24.SLIM_NAME
    cases = isles24.list_cases(root, folds)
    if len(cases) != fa.EXPECTED_DEVELOPMENT:
        raise ValueError(f"{len(cases)} development cases, expected {fa.EXPECTED_DEVELOPMENT}.")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_dir = Path(os.path.expanduser(args.model_dir))
    net, _ = train.load_best(model_dir / "best.pt", device)
    calibrator = infer.contract_calibrator(model_dir / "calibration.json")
    fa.check_aggregation(calibrator)
    values = {m: [] for m in range(FULL + 1)}
    for case in cases:
        images, _ = ic.prepare(isles24.load_native_ncct(case, root, folds))
        for m in range(FULL + 1):
            probs = infer.slice_probs(net, fill_image(images, m, fill), device)
            score = sr.scan_score(probs, None, "none", None)
            values[m].append(float(calibrator.apply(pd.Series([score]))[0]))
    values = {m: np.array(v) for m, v in values.items()}
    labels = pd.DataFrame({"label": values[FULL] >= calibrator.threshold})  # flagged under the full input
    result = summarise(values, labels, calibrator.threshold)
    result["groups"] = {{"bleed": "flagged (presumed bleed-free)", "no bleed": "not flagged"}.get(k, k): v
                        for k, v in result["groups"].items() if k in ("bleed", "no bleed")}
    result.pop("auc_shapley", None)
    result["coalitions"] = {k: {"present": v["present"], "flagged": int((values[int(k)] >= calibrator.threshold).sum())}
                            for k, v in result["coalitions"].items()}
    text = json.dumps({"experiment": "EXP-77", "fill": fill, **result}, indent=1)
    if ic.ids_in(text, cases + list(folds["heldout"])):
        raise ValueError("A case ID reached metrics.json. Nothing was saved.")
    (out / "metrics.json").write_text(text)
    print(text)


if __name__ == "__main__":
    isles_main()
