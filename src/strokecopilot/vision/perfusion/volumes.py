"""Mask -> mL, contract volumes, and occlusion site from the Circle of Willis labels (plan §8.1).

Mismatch arithmetic is defined once in strokecopilot.contract.schema.derive_mismatch.

Occlusion site is a rule, not a model. Each occlusion voxel takes its nearest Circle of Willis label.
The labels follow TopCoW (1-12 and 15, kept in TopBrain 2025's table): 1 is the basilar, 4 and 6
the ICAs, 5 and 7 the M1 segments of the MCAs. The masks are generated automatically and cannot
separate an M1 from an M2 occlusion (docs/DATA.md), so anything nearest an MCA label is `unknown`.
So is an empty mask, a case without labels, an occlusion too far from its label, or one whose voxels
point at different sites. ICA and basilar are the only sites this returns besides `unknown`.
"""

from collections import Counter

import numpy as np
from scipy import ndimage

from strokecopilot.contract.schema import VOLUME_DECIMALS, derive_mismatch

ML_PER_MM3 = 1e-3

LABEL_SITE = {1: "basilar", 4: "ICA", 6: "ICA", 5: "MCA", 7: "MCA"}  # other labels are "other"
RETURNED_SITES = ("ICA", "basilar")
# Set from EXP-1 on the 129 development cases: ICA-nearest occlusions sit at a median 0 mm from the
# label (p95 6.0 mm), basilar-nearest ones at 17-51 mm except one under 3 mm, so nothing lies between
# about 6 and 17 mm. 5 mm keeps 13 of 15 ICA cases; 10 mm would add the other 2.
DISTANCE_LIMIT_MM = 5.0
# Below this share of voxels pointing at one site, the occlusion is ambiguous.
MIN_SITE_SHARE = 0.75


def mask_ml(mask, spacing_mm):
    """Volume of a boolean mask in mL, unrounded. spacing_mm is one isotropic value or three."""
    voxel_mm3 = float(np.prod(np.broadcast_to(np.asarray(spacing_mm, dtype=float), (3,))))
    return float(np.count_nonzero(mask)) * voxel_mm3 * ML_PER_MM3


def round_ml(value):
    return None if value is None else round(float(value), VOLUME_DECIMALS)


def contract_volumes(core_ml, penumbra_ml):
    """The four contract volume fields from unrounded mL. Penumbra includes the core (DEC-7)."""
    core, penumbra = round_ml(core_ml), round_ml(penumbra_ml)
    ratio, mismatch = derive_mismatch(core, penumbra)
    return {"core_ml": core, "penumbra_ml": penumbra, "mismatch_ratio": ratio, "mismatch_volume_ml": mismatch}


def site_of_label(label):
    return LABEL_SITE.get(int(label), "other")


def occlusion_evidence(occlusion, cow, spacing_mm):
    """Where an occlusion mask sits relative to the labels, before any limit is applied.

    Returns a dict: `reason` (`ok`, `empty_mask`, `no_labels`, `shape_mismatch`), the nearest label
    by voxel majority, its site, the share of voxels whose nearest label has that site, the median
    distance in mm from occlusion voxels to their nearest label, and the share outside every label.
    """
    occlusion = np.asarray(occlusion, dtype=bool)
    cow = np.asarray(cow).astype(np.int32)
    empty = {"nearest_label": None, "site": None, "site_share": None, "distance_mm": None, "outside_share": None}
    if not occlusion.any():
        return {"reason": "empty_mask", **empty}
    if occlusion.shape != cow.shape:
        return {"reason": "shape_mismatch", **empty}
    if not (cow > 0).any():
        return {"reason": "no_labels", **empty}
    box = ndimage.find_objects((occlusion | (cow > 0)).astype(np.uint8))[0]
    cow_box, occ_box = cow[box], occlusion[box]
    sampling = np.broadcast_to(np.asarray(spacing_mm, dtype=float), (3,))
    dist, idx = ndimage.distance_transform_edt(cow_box == 0, sampling=sampling, return_indices=True)
    nearest = cow_box[tuple(i[occ_box] for i in idx)]
    label = Counter(nearest.tolist()).most_common(1)[0][0]
    sites = Counter(site_of_label(n) for n in nearest.tolist())
    site = site_of_label(label)
    return {
        "reason": "ok",
        "nearest_label": int(label),
        "site": site,
        "site_share": sites[site] / len(nearest),
        "distance_mm": float(np.median(dist[occ_box])),
        "outside_share": float((cow_box[occ_box] == 0).mean()),
    }


def occlusion_site(evidence, limit_mm=DISTANCE_LIMIT_MM, min_share=MIN_SITE_SHARE):
    """Contract `occlusion_site` from occlusion_evidence(): ICA, basilar or unknown, and why."""
    if evidence["reason"] != "ok":
        return "unknown", evidence["reason"]
    if evidence["site"] not in RETURNED_SITES:
        return "unknown", f"nearest_{evidence['site']}"
    if evidence["site_share"] < min_share:
        return "unknown", "ambiguous"
    if evidence["distance_mm"] > limit_mm:
        return "unknown", "too_far"
    return evidence["site"], "ok"


def limit_table(evidence_rows, limits_mm=(3.0, 5.0, 7.5, 10.0, 15.0), min_share=MIN_SITE_SHARE):
    """How many cases each candidate distance limit makes ICA, basilar or unknown, and distances per site.

    evidence_rows: occlusion_evidence() dicts, one per case. Counts only, no case IDs.
    """
    table = {}
    for limit in limits_mm:
        sites = Counter(occlusion_site(e, limit_mm=limit, min_share=min_share)[0] for e in evidence_rows)
        table[f"{limit:g}"] = dict(sorted(sites.items()))
    distances = {}
    for site in ("ICA", "basilar", "MCA", "other"):
        found = sorted(e["distance_mm"] for e in evidence_rows if e["reason"] == "ok" and e["site"] == site)
        distances[site] = {"cases": len(found), "distance_mm_p5_p50_p95": [float(np.percentile(found, p)) for p in (5, 50, 95)] if found else None}
    return {
        "cases": len(evidence_rows),
        "reasons": dict(Counter(e["reason"] for e in evidence_rows)),
        "nearest_label": dict(Counter(e["nearest_label"] for e in evidence_rows if e["reason"] == "ok").most_common()),
        "ambiguous_at_min_share": sum(e["reason"] == "ok" and e["site_share"] < min_share for e in evidence_rows),
        "distance_by_site": distances,
        "sites_by_limit_mm": table,
    }
