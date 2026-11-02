"""One-off checks of the unpacked ISLES'24 data (plan §7, Phase 1): anomalies, spot-check, labels, grid.

Reports hold counts and percentiles. Case IDs appear only for development cases, which the caller
passes in, so a held-out ID can never reach a report. Images are drawn for development cases only.
"""

import os
import tempfile
from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from strokecopilot.data.clinical_survey import read_tables
from strokecopilot.data.isles24_layout import STRAY_LESION, path_of

GRID_NAMES = ("cbf", "cbv", "mtt", "tmax", "lvo", "cow", "lesion", "dwi")
MAP_NAMES = ("cbf", "cbv", "mtt", "tmax")
TISSUE_HU = (-10, 100)
PENUMBRA_TMAX_S = 6.0
ML_PER_MM3 = 1e-3


def _load(root, case, name):
    return nib.load(path_of(root, case, name))


def _array(img, dtype=np.float32):
    return np.nan_to_num(np.asanyarray(img.dataobj).astype(dtype))


def _zooms(img):
    return tuple(float(z) for z in img.header.get_zooms()[:3])


def _volume_ml(mask, zooms):
    return float(mask.sum()) * float(np.prod(zooms)) * ML_PER_MM3


def _percentiles(values, points=(5, 50, 95)):
    if len(values) == 0:
        return None
    return [round(float(p), 3) for p in np.percentile(values, points)]


# Anomalies, checked before the split exists


def extra_rows(tables):
    """Workbooks with more than one row: which columns the sparser row fills and whether they conflict."""
    found = []
    for kind, by_case in tables.items():
        for case, book in by_case.items():
            df = next(iter(book.values()))
            if len(df) < 2:
                continue
            counts = df.notna().sum(axis=1)
            order = counts.sort_values(ascending=False, kind="stable").index  # rows with equal counts stay two rows
            main, extra = df.loc[order[0]], df.loc[order[-1]]
            filled = [c for c in df.columns if pd.notna(extra[c])]
            found.append(
                {
                    "kind": kind,
                    "case": case,
                    "rows": len(df),
                    "non_null_per_row": [int(n) for n in counts],
                    "sparser_row_fills": [str(c) for c in filled],
                    "sparser_row_only_on": [str(c) for c in filled if pd.isna(main[c])],
                    "sparser_row_conflicts_with_main_on": [str(c) for c in filled if pd.notna(main[c]) and main[c] != extra[c]],
                }
            )
    return found


def centres_of(tables):
    """{case: centre}, from the fullest row of each baseline workbook."""
    centres = {}
    for case, book in tables["baseline"].items():
        df = next(iter(book.values()))
        centres[case] = int(df.loc[df.notna().sum(axis=1).idxmax(), "Center"])
    return centres


def stray_mask_check(root, case):
    """Compare a case's extra lesion mask with its normal one, numbers only."""
    stray_img = nib.load(os.path.join(root, STRAY_LESION.format(c=case)))
    normal_img = _load(root, case, "lesion")
    stray, normal = _array(stray_img) > 0, _array(normal_img) > 0
    result = {
        "stray_shape": list(stray.shape),
        "normal_shape": list(normal.shape),
        "stray_spacing": _zooms(stray_img),
        "normal_spacing": _zooms(normal_img),
        "stray_ml": round(_volume_ml(stray, _zooms(stray_img)), 2),
        "normal_ml": round(_volume_ml(normal, _zooms(normal_img)), 2),
        "same_grid": stray.shape == normal.shape and bool(np.allclose(stray_img.affine, normal_img.affine, atol=1e-2)),
    }
    if stray.shape == normal.shape:
        overlap = int((stray & normal).sum())
        total = int(stray.sum() + normal.sum())
        result["dice_voxelwise"] = round(2 * overlap / total, 3) if total else None
    return result


# Split input


def infarct_volumes(root, cases):
    """{case: infarct mL} from the mask in NCCT space, plus how many cases hold each mask value."""
    volumes, labels = {}, Counter()
    for case in cases:
        img = _load(root, case, "lesion")
        mask = np.asanyarray(img.dataobj)
        volumes[case] = round(_volume_ml(mask > 0, _zooms(img)), 3)
        labels.update(str(v) for v in np.unique(mask))
    return volumes, dict(labels)


def split_balance(folds, volumes, centres):
    """Volumes and centres per group, without any ID."""
    groups = {"heldout": folds["heldout"], **{f"fold{k}": v for k, v in folds["folds"].items()}}
    return {
        name: {
            "cases": len(cases),
            "infarct_ml_p5_p50_p95": _percentiles([volumes[c] for c in cases]),
            "centres": dict(Counter(str(centres[c]) for c in cases)),
        }
        for name, cases in groups.items()
    }


def sample_by_volume(volumes, cases, n):
    """n cases spread evenly across the non-zero infarct volumes."""
    ranked = sorted((c for c in cases if volumes[c] > 0), key=lambda c: (volumes[c], c))
    if len(ranked) <= n:
        return ranked
    return [ranked[i] for i in np.linspace(0, len(ranked) - 1, n).round().astype(int)]


# Headers, all cases, counts only


def header_survey(root, cases, printable):
    """Shape and spacing per file type, and whether each image shares the NCCT grid."""
    report = {}
    shapes, spacings, dtypes = {}, {}, {}
    mismatches = {name: [] for name in GRID_NAMES}
    for case in cases:
        reference = _load(root, case, "ncct")
        for name in ("ncct", *GRID_NAMES):
            img = reference if name == "ncct" else _load(root, case, name)
            shapes.setdefault(name, []).append(img.shape)
            spacings.setdefault(name, []).append(_zooms(img))
            dtypes.setdefault(name, Counter())[str(img.get_data_dtype())] += 1
            if name != "ncct" and (img.shape[:3] != reference.shape[:3] or not np.allclose(img.affine, reference.affine, atol=1e-2)):
                mismatches[name].append(case)
    for name in shapes:
        spacing = np.array(spacings[name])
        shape = np.array([s[:3] for s in shapes[name]])
        report[name] = {
            "ndim": dict(Counter(str(len(s)) for s in shapes[name])),
            "dtype": dict(dtypes[name]),
            "spacing_min": spacing.min(axis=0).round(3).tolist(),
            "spacing_median": np.median(spacing, axis=0).round(3).tolist(),
            "spacing_max": spacing.max(axis=0).round(3).tolist(),
            "shape_min": shape.min(axis=0).tolist(),
            "shape_median": np.median(shape, axis=0).tolist(),
            "shape_max": shape.max(axis=0).tolist(),
        }
        if name != "ncct":
            report[name]["cases_off_ncct_grid"] = len(mismatches[name])
            report[name]["development_cases_off_ncct_grid"] = sorted(set(mismatches[name]) & set(printable))[:20]
    return report


# Spot-check, development cases only


def lesion_in_tissue(root, cases):
    """Share of each infarct mask on NCCT voxels in the soft-tissue HU range. Low suggests misalignment."""
    fractions = {}
    for case in cases:
        lesion = np.asanyarray(_load(root, case, "lesion").dataobj) > 0
        if not lesion.any():
            continue
        ncct = _array(_load(root, case, "ncct"))
        if ncct.shape != lesion.shape:
            continue
        fractions[case] = float(((ncct[lesion] >= TISSUE_HU[0]) & (ncct[lesion] <= TISSUE_HU[1])).mean())
    worst = sorted(fractions, key=fractions.get)[:5]
    return {
        "cases_measured": len(fractions),
        "p5_p50_p95": _percentiles(list(fractions.values())),
        "lowest": {c: round(fractions[c], 3) for c in worst},
    }


def cow_lvo_survey(root, cases):
    """Which Circle of Willis labels exist and where occlusion masks sit relative to them."""
    labels_present, label_ml = Counter(), {}
    nearest, outside_fractions, distances = Counter(), [], []
    empty_lvo = 0
    for case in cases:
        cow_img = _load(root, case, "cow")
        zooms = _zooms(cow_img)
        cow = np.asanyarray(cow_img.dataobj).astype(np.int32)
        lvo = np.asanyarray(_load(root, case, "lvo").dataobj) > 0
        for label in np.unique(cow):
            labels_present[str(label)] += 1
            label_ml.setdefault(str(label), []).append(_volume_ml(cow == label, zooms))
        if not lvo.any():
            empty_lvo += 1
            continue
        if lvo.shape != cow.shape or not cow.any():
            nearest["no comparable labels"] += 1
            continue
        box = ndimage.find_objects((lvo | (cow > 0)).astype(np.uint8))[0]
        cow_box, lvo_box = cow[box], lvo[box]
        dist, idx = ndimage.distance_transform_edt(cow_box == 0, sampling=zooms, return_indices=True)
        hits = cow_box[tuple(i[lvo_box] for i in idx)]
        nearest[str(Counter(hits.tolist()).most_common(1)[0][0])] += 1
        outside_fractions.append(float((cow_box[lvo_box] == 0).mean()))
        distances.append(float(np.median(dist[lvo_box])))
    return {
        "cases": len(cases),
        "cases_with_each_label": dict(sorted(labels_present.items(), key=lambda kv: int(kv[0]))),
        "label_ml_p50": {k: round(float(np.median(v)), 3) for k, v in sorted(label_ml.items(), key=lambda kv: int(kv[0]))},
        "empty_lvo_masks": empty_lvo,
        "lvo_dominant_or_nearest_label": dict(nearest.most_common()),
        "lvo_share_outside_labels_p5_p50_p95": _percentiles(outside_fractions),
        "lvo_median_distance_to_labels_mm_p5_p50_p95": _percentiles(distances),
    }


def value_survey(root, cases):
    """Ranges of NCCT and each map, to confirm HU and units. Median across cases of each percentile."""
    report = {}
    for name in ("ncct", *MAP_NAMES):
        rows, non_finite = [], 0
        for case in cases:
            raw = np.asanyarray(_load(root, case, name).dataobj)
            non_finite += int((~np.isfinite(raw)).sum())
            rows.append(np.percentile(raw[np.isfinite(raw)], (0.1, 1, 50, 99, 99.9)))
        report[name] = {
            "median_of_p0.1_p1_p50_p99_p99.9": np.median(rows, axis=0).round(3).tolist(),
            "non_finite_voxels": non_finite,
        }
    return report


def resample(array, zooms, spacing, order):
    factors = [z / spacing for z in zooms]
    return ndimage.zoom(array, factors, order=order, prefilter=False, grid_mode=True, mode="nearest")


def brain_box_mm(ncct, zooms, coarse_mm=3.0, margin_mm=5.0):
    """Bounding box of the largest soft-tissue component, as (start, stop) in mm per axis."""
    coarse = resample(ncct, zooms, coarse_mm, order=1)
    tissue = (coarse >= TISSUE_HU[0]) & (coarse <= TISSUE_HU[1])
    labelled, n = ndimage.label(ndimage.binary_opening(tissue, iterations=2))
    if n == 0:
        return [(0.0, s * z) for s, z in zip(ncct.shape, zooms)]
    sizes = ndimage.sum_labels(tissue, labelled, range(1, n + 1))
    box = ndimage.find_objects((labelled == int(np.argmax(sizes)) + 1).astype(np.uint8))[0]
    limits = [s * z for s, z in zip(ncct.shape, zooms)]
    return [
        (max(0.0, b.start * coarse_mm - margin_mm), min(limit, b.stop * coarse_mm + margin_mm))
        for b, limit in zip(box, limits)
    ]


def _crop(array, box_mm, spacing):
    slices = tuple(slice(int(start / spacing), int(np.ceil(stop / spacing))) for start, stop in box_mm)
    return array[slices]


def _gz_bytes(array, spacing, dtype, tmp):
    path = os.path.join(tmp, "x.nii.gz")
    nib.save(nib.Nifti1Image(array.astype(dtype), np.diag([spacing, spacing, spacing, 1.0])), path)
    return os.path.getsize(path)


def grid_candidates(root, cases, spacings, n_all_cases):
    """What each isotropic spacing costs: volume errors against the native grid, crop size, file size."""
    rows = {s: {"lesion_err": [], "lesion_rel": [], "tmax_err": [], "lost": 0, "bytes": [], "shape": []} for s in spacings}
    boxes, native_bytes = [], []
    for case in cases:
        ncct_img = _load(root, case, "ncct")
        zooms = _zooms(ncct_img)
        ncct = _array(ncct_img)
        lesion = _array(_load(root, case, "lesion"))
        tmax = _array(_load(root, case, "tmax"))
        box = brain_box_mm(ncct, zooms)
        boxes.append([stop - start for start, stop in box])
        lesion_ml = _volume_ml(lesion > 0, zooms)
        tmax_ml = _volume_ml(tmax > PENUMBRA_TMAX_S, zooms)
        native_bytes.append(os.path.getsize(path_of(root, case, "ncct")))
        for spacing in spacings:
            row = rows[spacing]
            small_lesion = resample((lesion > 0).astype(np.float32), zooms, spacing, 1) >= 0.5
            small_tmax = resample(tmax, zooms, spacing, 1)
            small_ml = _volume_ml(small_lesion, (spacing,) * 3)
            row["lesion_err"].append(abs(small_ml - lesion_ml))
            if lesion_ml > 0:
                row["lesion_rel"].append(abs(small_ml - lesion_ml) / lesion_ml)
                row["lost"] += small_ml == 0
            row["tmax_err"].append(abs(_volume_ml(small_tmax > PENUMBRA_TMAX_S, (spacing,) * 3) - tmax_ml))
            cropped_shape = _crop(small_tmax, box, spacing).shape
            row["shape"].append(cropped_shape)
            with tempfile.TemporaryDirectory() as tmp:
                total = _gz_bytes(_crop(resample(ncct, zooms, spacing, 1), box, spacing), spacing, np.int16, tmp)
                total += _gz_bytes(_crop(small_tmax, box, spacing), spacing, np.float32, tmp)
                for name in ("cbf", "cbv", "mtt"):
                    total += _gz_bytes(_crop(resample(_array(_load(root, case, name)), zooms, spacing, 1), box, spacing), spacing, np.float32, tmp)
                total += _gz_bytes(_crop(small_lesion, box, spacing), spacing, np.uint8, tmp)
                total += _gz_bytes(_crop(resample((_array(_load(root, case, "lvo")) > 0).astype(np.float32), zooms, spacing, 1) >= 0.5, box, spacing), spacing, np.uint8, tmp)
                total += _gz_bytes(_crop(resample(_array(_load(root, case, "cow")), zooms, spacing, 0), box, spacing), spacing, np.uint8, tmp)
            row["bytes"].append(total)
    native_gib = float(np.mean(native_bytes)) * n_all_cases / 1024 ** 3
    return {
        "cases_sampled": len(cases),
        "brain_box_mm_p50": np.median(boxes, axis=0).round(1).tolist(),
        "brain_box_mm_max": np.max(boxes, axis=0).round(1).tolist(),
        "ncct_native_gib_all_cases": round(native_gib, 2),
        "spacings": {
            str(s): {
                "lesion_abs_error_ml_p50_p95": _percentiles(r["lesion_err"], (50, 95)),
                "lesion_relative_error_p50_p95": _percentiles(r["lesion_rel"], (50, 95)),
                "lesions_lost": int(r["lost"]),
                "tmax6_abs_error_ml_p50_p95": _percentiles(r["tmax_err"], (50, 95)),
                "cropped_shape_p50": np.median(r["shape"], axis=0).tolist(),
                "cropped_shape_max": np.max(r["shape"], axis=0).tolist(),
                "projected_slim_gib_all_cases": round(float(np.mean(r["bytes"])) * n_all_cases / 1024 ** 3 + native_gib, 2),
            }
            for s, r in rows.items()
        },
    }


def montage(root, cases, out_png):
    """One row per development case: NCCT, DWI and Tmax at the largest infarct slice, then labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(cases), 4, figsize=(14, 3.5 * len(cases)), squeeze=False)
    for row, case in zip(axes, cases):
        lesion = np.asanyarray(_load(root, case, "lesion").dataobj) > 0
        z = int(lesion.sum(axis=(0, 1)).argmax())
        panels = [("ncct", 0, 80, "NCCT 0-80 HU"), ("dwi", None, None, "DWI"), ("tmax", 0, 12, "Tmax 0-12")]
        for ax, (name, low, high, title) in zip(row, panels):
            image = _array(_load(root, case, name))[:, :, z]
            if low is None:
                low, high = np.percentile(image, (1, 99.5))
            ax.imshow(image.T, cmap="gray", vmin=low, vmax=high, origin="lower")
            ax.contour(lesion[:, :, z].T, levels=[0.5], colors="red", linewidths=0.7)
            ax.set_title(f"{case} z={z} {title}", fontsize=8)
        cow = np.asanyarray(_load(root, case, "cow").dataobj).astype(np.int32)
        lvo = np.asanyarray(_load(root, case, "lvo").dataobj) > 0
        row[3].imshow(cow.max(axis=2).T, cmap="tab20", interpolation="nearest", origin="lower")
        if lvo.any():
            row[3].contour(lvo.max(axis=2).T, levels=[0.5], colors="black", linewidths=1.0)
        row[3].set_title(f"{case} labels, occlusion in black", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def read_clinical_tables(root, cases):
    return read_tables(os.path.join(root, "phenotype"), cases)


