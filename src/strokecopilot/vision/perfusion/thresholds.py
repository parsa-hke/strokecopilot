"""Clinical threshold baseline: core = rCBF < 30% of the other hemisphere, penumbra = Tmax > 6 s (plan §8.1).

Tmax below 0 means no reliable perfusion data, not only the -30 fill value (docs/DATA.md), so those
voxels can be neither core nor penumbra. The reference CBF is the median over reliable voxels of the
hemisphere with less Tmax > 6 s, leaving out voxels with Tmax > 4 s. The midline is the left-right
centre of soft tissue on NCCT, since the slim grid is RAS and cropped to the brain; it falls back to the
centre of the reliable voxels without NCCT, which shifts if one side lacks perfusion data. By default core
must also lie inside Tmax > 6 s, which keeps low-flow CSF and white matter out of the core.
`penumbra` is the whole Tmax > 6 s region, core included (DEC-7).
"""

import numpy as np

from strokecopilot.data.isles24_survey import PENUMBRA_TMAX_S, TISSUE_HU
from strokecopilot.vision.perfusion import evaluate, volumes

RCBF_CORE = 0.30
REFERENCE_MAX_TMAX_S = 4.0
CORE_WITHIN_PENUMBRA = True


def reliable(tmax, cbf):
    """Voxels with usable perfusion data: Tmax 0 or more and finite CBF."""
    return np.isfinite(tmax) & (tmax >= 0) & np.isfinite(cbf)


def hemispheres(centre_of):
    """Boolean masks (low x, high x) split at the left-right centre of a mask. RAS: high x is right."""
    xs = np.nonzero(centre_of)[0]
    midline = float(xs.mean()) if xs.size else centre_of.shape[0] / 2
    x = np.arange(centre_of.shape[0])[:, None, None]
    return np.broadcast_to(x < midline, centre_of.shape), np.broadcast_to(x >= midline, centre_of.shape), midline


def threshold_masks(cbf, tmax, ncct=None, rcbf=RCBF_CORE, tmax_s=PENUMBRA_TMAX_S, core_within_penumbra=CORE_WITHIN_PENUMBRA):
    """Core and penumbra masks for one case, with the values they rest on.

    `core` is None when no reference CBF can be found (no reliable voxels on the healthier side, or
    their median is not positive); the contract then carries core_ml as null.
    """
    cbf = np.asarray(cbf, dtype=np.float32)
    tmax = np.asarray(tmax, dtype=np.float32)
    valid = reliable(tmax, cbf)
    penumbra = valid & (tmax > tmax_s)
    tissue = valid if ncct is None else (ncct >= TISSUE_HU[0]) & (ncct <= TISSUE_HU[1])
    low, high, midline = hemispheres(tissue)
    affected_high = np.count_nonzero(penumbra & high) > np.count_nonzero(penumbra & low)
    healthy = low if affected_high else high
    reference_voxels = valid & healthy & (tmax <= REFERENCE_MAX_TMAX_S)
    if not reference_voxels.any():
        reference_voxels = valid & healthy
    reference = float(np.median(cbf[reference_voxels])) if reference_voxels.any() else None
    core = None
    if reference is not None and reference > 0:
        core = valid & (cbf < rcbf * reference)
        if core_within_penumbra:
            core &= penumbra
    return {
        "core": core,
        "penumbra": penumbra,
        "valid": valid,
        "reference_cbf": reference,
        "affected_side": "right" if affected_high else "left",
        "midline_x": midline,
    }


def baseline_row(case, fold, rcbf=RCBF_CORE, core_within_penumbra=CORE_WITHIN_PENUMBRA):
    """Scores, contract volumes and occlusion evidence for one loaded case (data.isles24.Case)."""
    arrays, spacing = case.arrays, case.spacing_mm
    masks = threshold_masks(arrays["cbf"], arrays["tmax"], arrays.get("ncct"), rcbf=rcbf, core_within_penumbra=core_within_penumbra)
    core = masks["core"] if masks["core"] is not None else np.zeros_like(masks["valid"])
    row = {"case": case.id, "fold": fold, "core_computed": masks["core"] is not None}
    row.update(evaluate.case_metrics(core, arrays["infarct"], masks["valid"], spacing, case.native_volumes["infarct_ml"]))
    penumbra_ml = volumes.mask_ml(masks["penumbra"], spacing)
    row.update({
        "penumbra_ml": penumbra_ml,
        "penumbra_ref_ml": case.native_volumes["tmax6_ml"],
        "penumbra_error_ml": penumbra_ml - case.native_volumes["tmax6_ml"],
        "reference_cbf": masks["reference_cbf"],
        "affected_side": masks["affected_side"],
    })
    contract = volumes.contract_volumes(row["pred_ml"] if row["core_computed"] else None, penumbra_ml)
    row.update({f"contract_{k}": v for k, v in contract.items()})
    evidence = volumes.occlusion_evidence(arrays["occlusion"], arrays["cow"], spacing)
    row.update({f"occlusion_{k}": v for k, v in evidence.items()})
    return row


def montage(cases, out_png):
    """One row per loaded development case: NCCT and Tmax at the largest infarct slice with the infarct
    (red) and threshold core (cyan), then a projection of the labels with the occlusion in black."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(cases), 3, figsize=(11, 3.5 * len(cases)), squeeze=False)
    for row, case in zip(axes, cases):
        arrays = case.arrays
        masks = threshold_masks(arrays["cbf"], arrays["tmax"], arrays.get("ncct"))
        infarct = arrays["infarct"]
        z = int(infarct.sum(axis=(0, 1)).argmax()) if infarct.any() else infarct.shape[2] // 2
        for ax, name, low, high in ((row[0], "ncct", 0, 80), (row[1], "tmax", 0, 12)):
            ax.imshow(arrays[name][:, :, z].T, cmap="gray", vmin=low, vmax=high, origin="lower")
            ax.contour(infarct[:, :, z].T, levels=[0.5], colors="red", linewidths=0.7)
            if masks["core"] is not None and masks["core"][:, :, z].any():
                ax.contour(masks["core"][:, :, z].T, levels=[0.5], colors="cyan", linewidths=0.7)
            ax.set_title(f"{case.id} z={z} {name}", fontsize=8)
        row[2].imshow(arrays["cow"].max(axis=2).T, cmap="tab20", vmin=0, vmax=20, interpolation="nearest", origin="lower")
        if arrays["occlusion"].any():
            row[2].contour(arrays["occlusion"].max(axis=2).T, levels=[0.5], colors="black", linewidths=1.0)
        row[2].set_title(f"{case.id} labels, occlusion in black", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


DIAGNOSTIC_RCBF = (0.20, 0.30, 0.38, 0.45)


def diagnose_case(case, rcbfs=DIAGNOSTIC_RCBF):
    """Checks for a low baseline Dice: is the infarct where the perfusion deficit is, and what rCBF it has.

    For one loaded case: the share of the infarct inside Tmax > 6 s and on the side the baseline calls
    affected, Dice of Tmax > 6 s against the infarct, rCBF percentiles inside the infarct, and Dice and
    volume at several rCBF cut-offs. The baseline itself stays at 30%; this only shows how sensitive it is.
    """
    arrays, spacing = case.arrays, case.spacing_mm
    masks = threshold_masks(arrays["cbf"], arrays["tmax"], arrays.get("ncct"))
    infarct, valid, penumbra = arrays["infarct"], masks["valid"], masks["penumbra"]
    low, high, _ = hemispheres((arrays["ncct"] >= TISSUE_HU[0]) & (arrays["ncct"] <= TISSUE_HU[1]))
    affected = high if masks["affected_side"] == "right" else low
    n_infarct = np.count_nonzero(infarct)
    row = {
        "case": case.id,
        "infarct_ml_grid": volumes.mask_ml(infarct, spacing),
        "infarct_in_tmax6_share": float(np.count_nonzero(infarct & penumbra) / n_infarct) if n_infarct else None,
        "infarct_on_affected_side_share": float(np.count_nonzero(infarct & affected) / n_infarct) if n_infarct else None,
        "tmax6_dice": evaluate.dice(penumbra, infarct),
        "tmax6_ml": volumes.mask_ml(penumbra, spacing),
    }
    reference = masks["reference_cbf"]
    inside = infarct & valid
    if reference and inside.any():
        rcbf = arrays["cbf"][inside] / reference
        row.update({f"infarct_rcbf_p{p}": float(np.percentile(rcbf, p)) for p in (25, 50, 75)})
    for cut in rcbfs:
        if reference and reference > 0:
            core = valid & penumbra & (arrays["cbf"] < cut * reference)
            row[f"dice_rcbf{cut:g}"] = evaluate.dice(core, infarct)
            row[f"core_ml_rcbf{cut:g}"] = volumes.mask_ml(core, spacing)
    return row
