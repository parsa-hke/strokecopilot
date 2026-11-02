"""ISLES'24 working copy -> slim tarball, run once (plan §5.4, §7 Phase 1).

Every volume is turned to RAS, cropped to the brain and resampled to one isotropic grid. Only NCCT is
averaged before shrinking. Maps and masks are not, because volumes are read from them by threshold:
averaging shrank small masks in tests on irregular shapes, and it would change how much of a speckled
map crosses Tmax > 6 s. Map background stays exactly at its fill value, because Tmax -30 marks tissue
without perfusion data and must not be blended into real values. `ncct_native` is an unchanged copy of
the source file.

Reports hold counts and percentiles. Percentiles and case IDs come from development cases only, so
nothing about a held-out case can be read from a report.
"""

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter

import nibabel as nib
import nibabel.processing
import numpy as np
import pandas as pd
from scipy import ndimage

from strokecopilot.data import clinical_columns as cc
from strokecopilot.data.isles24_layout import path_of
from strokecopilot.data.isles24_survey import PENUMBRA_TMAX_S, brain_box_mm

SLIM_NAME = "isles24_slim_v1"
SLIM_VERSION = 1
SPACING_MM = 1.5
PART_SIZE = 25
MAP_BACKGROUND = {"cbf": 0.0, "cbv": 0.0, "mtt": 0.0, "tmax": -30.0}
LOW_COVERAGE = 0.9
UNKNOWN_CODES = {"Hyperlipidemia": (99,)}
TIME_COLUMNS = ("Onset to door", "Alert to door", "Door to imaging")
ML_PER_MM3 = 1e-3


def _canonical(root, case, name):
    return nib.as_closest_canonical(nib.load(path_of(root, case, name)))


def _zooms(img):
    return tuple(float(z) for z in img.header.get_zooms()[:3])


def _align(img, reference, order, cval=0.0):
    """Array of img on the reference grid, and whether it had to be moved there."""
    if img.shape[:3] == reference.shape[:3] and np.allclose(img.affine, reference.affine, atol=1e-2):
        return np.asanyarray(img.dataobj), False
    moved = nibabel.processing.resample_from_to(img, reference, order=order, cval=cval)
    return np.asanyarray(moved.dataobj), True


def crop_box(ncct, zooms, masks):
    """Voxel (start, stop) per axis: the brain box, grown to hold every mask, and whether it grew."""
    box = brain_box_mm(ncct, zooms)
    starts = [int(start / z) for (start, _), z in zip(box, zooms)]
    stops = [min(math.ceil(stop / z), n) for (_, stop), z, n in zip(box, zooms, ncct.shape)]
    grew = False
    for mask in masks:
        if not mask.any():
            continue
        for axis, found in enumerate(ndimage.find_objects(mask.astype(np.uint8))[0]):
            if found.start < starts[axis] or found.stop > stops[axis]:
                grew = True
            starts[axis] = min(starts[axis], found.start)
            stops[axis] = max(stops[axis], found.stop)
    return list(zip(starts, stops)), grew


class Grid:
    """Sampling from a cropped native array onto the isotropic grid. Voxel edges of the crop line up."""

    def __init__(self, affine, zooms, box, spacing):
        self.step = np.array([spacing / z for z in zooms])
        self.offset = 0.5 * self.step - 0.5
        self.slices = tuple(slice(start, stop) for start, stop in box)
        self.shape = tuple(math.ceil((stop - start) * z / spacing) for (start, stop), z in zip(box, zooms))
        self.sigma = tuple(max(s - 1.0, 0.0) / 2 for s in self.step)
        to_native = np.diag([*self.step, 1.0])
        to_native[:3, 3] = [start + off for (start, _), off in zip(box, self.offset)]
        self.affine = affine @ to_native

    def _smooth(self, array):
        return ndimage.gaussian_filter(array, self.sigma, truncate=3.0, mode="nearest") if any(self.sigma) else array

    def _sample(self, array, order=1):
        return ndimage.affine_transform(array, self.step, offset=self.offset, output_shape=self.shape, order=order, mode="nearest")

    def image(self, array):
        return self._sample(self._smooth(array[self.slices].astype(np.float32)))

    def map(self, array, background, smooth=False):
        """Interpolate valid voxels only. Where under half the weight is valid, the output is background."""
        cropped = array[self.slices].astype(np.float32)
        valid = np.isfinite(cropped) & (np.abs(cropped - background) > 1e-3)
        prepare = self._smooth if smooth else (lambda a: a)
        values = self._sample(prepare(np.where(valid, cropped, 0.0).astype(np.float32)))
        weights = self._sample(prepare(valid.astype(np.float32)))
        out = np.full(self.shape, background, dtype=np.float32)
        keep = weights >= 0.5
        out[keep] = values[keep] / weights[keep]
        return out

    def mask(self, array):
        """Binary mask at half coverage. A mask that would vanish keeps its best voxel, and says so."""
        fraction = self._sample(array[self.slices].astype(np.float32))
        out = fraction >= 0.5
        rescued = bool(array.any() and not out.any())
        if rescued:
            out[np.unravel_index(int(np.argmax(fraction)), self.shape)] = True
        return out, rescued

    def labels(self, array):
        return self._sample(array[self.slices].astype(np.uint8), order=0)


def _save(array, affine, path):
    img = nib.Nifti1Image(array, affine)
    img.header.set_xyzt_units("mm")
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, path)


def _ml(mask, zooms):
    return float(mask.sum()) * float(np.prod(zooms)) * ML_PER_MM3


def _valid_tmax(tmax):
    return np.isfinite(tmax) & (np.abs(tmax - MAP_BACKGROUND["tmax"]) > 1e-3)


def process_case(root, case, out_dir, spacing=SPACING_MM):
    """Write one case's slim files into out_dir and return its QC row."""
    os.makedirs(out_dir, exist_ok=True)
    original_codes = "".join(nib.aff2axcodes(nib.load(path_of(root, case, "ncct")).affine))
    reference = _canonical(root, case, "ncct")
    zooms = _zooms(reference)
    ncct = np.nan_to_num(np.asanyarray(reference.dataobj).astype(np.float32))

    moved = []
    arrays = {}
    for name, order, cval in (("lesion", 0, 0.0), ("lvo", 0, 0.0), ("cow", 0, 0.0), *((m, 1, b) for m, b in MAP_BACKGROUND.items())):
        arrays[name], was_moved = _align(_canonical(root, case, name), reference, order, cval)
        if was_moved:
            moved.append(name)
    lesion, lvo = arrays["lesion"] > 0, arrays["lvo"] > 0

    box, grew = crop_box(ncct, zooms, [lesion, lvo])
    grid = Grid(reference.affine, zooms, box, spacing)

    rescued, non_finite = [], {}
    _save(np.clip(np.rint(grid.image(ncct)), -32768, 32767).astype(np.int16), grid.affine, os.path.join(out_dir, "ncct.nii.gz"))
    for name, background in MAP_BACKGROUND.items():
        non_finite[name] = int((~np.isfinite(arrays[name])).sum())
        arrays[f"small_{name}"] = grid.map(arrays[name], background)
        _save(arrays[f"small_{name}"], grid.affine, os.path.join(out_dir, f"{name}.nii.gz"))
    small = {}
    for name, target, array in (("lesion", "infarct_followup", lesion), ("lvo", "cta_occlusion", lvo)):
        small[name], was_rescued = grid.mask(array)
        if was_rescued:
            rescued.append(name)
        _save(small[name].astype(np.uint8), grid.affine, os.path.join(out_dir, f"{target}.nii.gz"))
    cow = arrays["cow"].astype(np.uint8)
    small_cow = grid.labels(cow)
    _save(small_cow, grid.affine, os.path.join(out_dir, "cow.nii.gz"))
    shutil.copyfile(path_of(root, case, "ncct"), os.path.join(out_dir, "ncct_native.nii.gz"))

    tmax = arrays["tmax"]
    covered = _valid_tmax(tmax)
    small_tmax = arrays["small_tmax"]
    row = {
        "case": case,
        "original_axes": original_codes,
        "shape": list(grid.shape),
        "moved_to_ncct_grid": moved,
        "crop_grew_for_mask": grew,
        "masks_rescued": rescued,
        "non_finite_voxels": non_finite,
        "lesion_ml": _ml(lesion, zooms),
        "lesion_ml_slim": _ml(small["lesion"], (spacing,) * 3),
        "tmax6_ml": _ml(covered & (np.nan_to_num(tmax) > PENUMBRA_TMAX_S), zooms),
        "tmax6_ml_slim": _ml(small_tmax > PENUMBRA_TMAX_S, (spacing,) * 3),
        "tmax6_ml_if_averaged": _ml(grid.map(tmax, MAP_BACKGROUND["tmax"], smooth=True) > PENUMBRA_TMAX_S, (spacing,) * 3),
        "lesion_with_tmax": float(covered[lesion].mean()) if lesion.any() else None,
        "tmax_negative_share": float((tmax[covered] < 0).mean()) if covered.any() else None,
        "cow_labels_lost": sorted(int(v) for v in set(np.unique(cow)) - set(np.unique(small_cow))),
    }
    with open(os.path.join(out_dir, "qc.json"), "w") as f:
        json.dump(row, f)
    return row


def build_parts(root, cases, parts_dir, part_size=PART_SIZE, spacing=SPACING_MM, tmp_root=None):
    """Process cases in batches, saving each batch as a tar. Batches already saved are skipped."""
    os.makedirs(parts_dir, exist_ok=True)
    started = time.time()
    batches = [cases[i : i + part_size] for i in range(0, len(cases), part_size)]
    done = 0
    for number, batch in enumerate(batches):
        part = os.path.join(parts_dir, f"part_{number:03d}.tar")
        if os.path.exists(part):
            print(f"Part {number + 1} of {len(batches)} already saved.", flush=True)
            continue
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            for case in batch:
                process_case(root, case, os.path.join(tmp, case), spacing)
                done += 1
                print(f"Done {done} cases this session, {(time.time() - started) / done:.0f} s each.", flush=True)
            local = os.path.join(tmp, "part.tar")
            subprocess.run(["tar", "-cf", local, "-C", tmp, *batch], check=True)
            shutil.copyfile(local, f"{part}.part")
        if os.path.getsize(f"{part}.part") == 0:
            raise OSError(f"Part {number} was saved empty.")
        os.replace(f"{part}.part", part)
        print(f"Saved part {number + 1} of {len(batches)}.", flush=True)
    return len(batches)


def merge_rows(df):
    """One row from a workbook's rows, and what merging found. Disagreeing values become missing."""
    row, only_on_later, conflicts = {}, [], []
    for column in df.columns:
        values = [v for v in df[column] if pd.notna(v)]
        unique = list(dict.fromkeys(values))
        row[column] = unique[0] if len(unique) == 1 else None
        if len(unique) > 1:
            conflicts.append(str(column))
        elif unique and pd.isna(df[column].iloc[0]):
            only_on_later.append(str(column))
    return row, {"rows": len(df), "only_on_later_rows": only_on_later, "conflicts_set_missing": conflicts}


def build_clinical(tables, cases):
    """One row per case, with every column checked against its note tag, and a report on merged rows."""
    columns = (*cc.BASELINE_COLUMNS, *cc.OUTCOME_COLUMNS)
    rows, merged = [], []
    for case in cases:
        row = {"case": case}
        for kind in ("baseline", "outcome"):
            df = next(iter(tables[kind][case].values()))
            for column in df.columns:
                cc.tag_of(column)
            values, found = merge_rows(df)
            row.update(values)
            if found["rows"] > 1:
                merged.append({"kind": kind, **found})
        rows.append(row)
    clinical = pd.DataFrame(rows).reindex(columns=["case", *columns]).astype(object)
    unknown_codes = {}
    for column, codes in UNKNOWN_CODES.items():
        hits = clinical[column].isin(codes)
        unknown_codes[column] = int(hits.sum())
        clinical[column] = clinical[column].where(~hits, None)
    return clinical, {"merged_workbooks": merged, "unknown_codes_set_missing": unknown_codes}


def time_formats(clinical, cases):
    """Shape of the time text, digits replaced, for the given cases."""
    subset = clinical[clinical["case"].isin(cases)]
    return {
        column: dict(Counter(re.sub(r"\d", "9", str(v)) for v in subset[column] if pd.notna(v)).most_common(10))
        for column in TIME_COLUMNS
    }


def assemble(parts_dir, cases, clinical, slim_parent, spacing=SPACING_MM, source=None):
    """Unpack saved parts into the slim folder, write clinical.csv and manifest.json, return QC rows."""
    slim_dir = os.path.join(slim_parent, SLIM_NAME)
    if os.path.exists(slim_dir):
        shutil.rmtree(slim_dir)
    os.makedirs(slim_dir)
    for part in sorted(os.listdir(parts_dir)):
        if part.endswith(".tar"):
            subprocess.run(["tar", "-xf", os.path.join(parts_dir, part), "-C", slim_dir], check=True)
    missing = [case for case in cases if not os.path.exists(os.path.join(slim_dir, case, "qc.json"))]
    if missing:
        raise ValueError(f"{len(missing)} cases are not in the saved parts.")
    rows = []
    for case in cases:
        path = os.path.join(slim_dir, case, "qc.json")
        with open(path) as f:
            rows.append(json.load(f))
        os.remove(path)
    clinical.to_csv(os.path.join(slim_dir, "clinical.csv"), index=False)
    pd.DataFrame([{"case": r["case"], "infarct_ml": r["lesion_ml"], "tmax6_ml": r["tmax6_ml"]} for r in rows]).to_csv(os.path.join(slim_dir, "native_volumes.csv"), index=False)
    manifest = {
        "name": SLIM_NAME,
        "version": SLIM_VERSION,
        "spacing_mm": spacing,
        "orientation": "RAS, except ncct_native, which is the unchanged source file",
        "crop": "brain bounding box from NCCT with a 5 mm margin, grown to hold the infarct and occlusion masks",
        "resampling": "linear; NCCT averaged before shrinking; map background kept exact; labels nearest; masks kept at half coverage",
        "maps": list(MAP_BACKGROUND),
        "map_background": MAP_BACKGROUND,
        "source": source or {},
        "licence": "CC BY-NC-SA 4.0",
        "cases": list(cases),
    }
    with open(os.path.join(slim_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return rows


SLIM_FILES = {
    "ncct.nii.gz": np.int16,
    "ncct_native.nii.gz": None,
    "cbf.nii.gz": np.float32,
    "cbv.nii.gz": np.float32,
    "mtt.nii.gz": np.float32,
    "tmax.nii.gz": np.float32,
    "cta_occlusion.nii.gz": np.uint8,
    "cow.nii.gz": np.uint8,
    "infarct_followup.nii.gz": np.uint8,
}


def check_slim(slim_dir, rows):
    """Problems in the finished folder, as counts. Reads headers and every mask."""
    with open(os.path.join(slim_dir, "manifest.json")) as f:
        manifest = json.load(f)
    spacing = manifest["spacing_mm"]
    problems = Counter()
    if len(pd.read_csv(os.path.join(slim_dir, "clinical.csv"))) != len(manifest["cases"]):
        problems["clinical.csv row count"] += 1
    by_case = {row["case"]: row for row in rows}
    for case in manifest["cases"]:
        reference = None
        for name, dtype in SLIM_FILES.items():
            path = os.path.join(slim_dir, case, name)
            if not os.path.exists(path):
                problems[f"missing {name}"] += 1
                continue
            if dtype is None:
                continue
            img = nib.load(path)
            if reference is None:
                reference = img
            if img.shape != reference.shape or not np.allclose(img.affine, reference.affine, atol=1e-4):
                problems[f"{name} off the case grid"] += 1
            if not np.allclose(img.header.get_zooms()[:3], spacing, atol=1e-3):
                problems[f"{name} wrong spacing"] += 1
            if nib.aff2axcodes(img.affine) != ("R", "A", "S"):
                problems[f"{name} not RAS"] += 1
            if img.get_data_dtype() != dtype:
                problems[f"{name} wrong type"] += 1
        row = by_case[case]
        lesion = nib.load(os.path.join(slim_dir, case, "infarct_followup.nii.gz"))
        if bool(row["lesion_ml"]) != bool(np.asanyarray(lesion.dataobj).any()):
            problems["infarct present in one version only"] += 1
    return dict(problems)


def _percentiles(values, points=(5, 50, 95)):
    values = [v for v in values if v is not None]
    return [round(float(p), 3) for p in np.percentile(values, points)] if values else None


def summarise(rows, development):
    """Counts over all cases; percentiles and IDs over development cases only."""
    dev = [row for row in rows if row["case"] in set(development)]
    with_lesion = [row for row in dev if row["lesion_ml"] > 0]
    covered = sorted((row for row in with_lesion if row["lesion_with_tmax"] is not None), key=lambda r: r["lesion_with_tmax"])
    return {
        "cases": len(rows),
        "original_axes": dict(Counter(row["original_axes"] for row in rows)),
        "moved_to_ncct_grid": dict(Counter(name for row in rows for name in row["moved_to_ncct_grid"])),
        "crop_grew_for_mask": sum(row["crop_grew_for_mask"] for row in rows),
        "masks_rescued": dict(Counter(name for row in rows for name in row["masks_rescued"])),
        "non_finite_voxels": {name: sum(row["non_finite_voxels"][name] for row in rows) for name in MAP_BACKGROUND},
        "cases_losing_a_cow_label": sum(bool(row["cow_labels_lost"]) for row in rows),
        "cow_labels_lost": dict(Counter(str(label) for row in rows for label in row["cow_labels_lost"])),
        "shape_p50": np.median([row["shape"] for row in rows], axis=0).tolist(),
        "shape_max": np.max([row["shape"] for row in rows], axis=0).tolist(),
        "development": {
            "cases": len(dev),
            "lesion_abs_error_ml_p50_p95": _percentiles([abs(r["lesion_ml_slim"] - r["lesion_ml"]) for r in with_lesion], (50, 95)),
            "lesion_relative_error_p50_p95": _percentiles([abs(r["lesion_ml_slim"] - r["lesion_ml"]) / r["lesion_ml"] for r in with_lesion], (50, 95)),
            "tmax6_abs_error_ml_p50_p95": _percentiles([abs(r["tmax6_ml_slim"] - r["tmax6_ml"]) for r in dev], (50, 95)),
            "tmax6_abs_error_ml_p50_p95_if_averaged": _percentiles([abs(r["tmax6_ml_if_averaged"] - r["tmax6_ml"]) for r in dev], (50, 95)),
            "lesion_with_tmax_p5_p50_p95": _percentiles([r["lesion_with_tmax"] for r in covered]),
            f"cases_under_{LOW_COVERAGE}_lesion_with_tmax": sum(r["lesion_with_tmax"] < LOW_COVERAGE for r in covered),
            "lowest_lesion_with_tmax": {r["case"]: round(r["lesion_with_tmax"], 3) for r in covered[:5]},
            "tmax_negative_share_p50_p95": _percentiles([r["tmax_negative_share"] for r in dev], (50, 95)),
        },
    }


def pack_slim(slim_parent, tar_path):
    """Tar the slim folder, uncompressed: the images are already gzipped."""
    subprocess.run(["tar", "-cf", tar_path, "-C", slim_parent, SLIM_NAME], check=True)


def montage(slim_dir, cases, out_png):
    """One row per development case from the finished files: NCCT and Tmax at the largest infarct slice, then labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(cases), 3, figsize=(10.5, 3.5 * len(cases)), squeeze=False)
    for row, case in zip(axes, cases):
        load = lambda name: np.asanyarray(nib.load(os.path.join(slim_dir, case, f"{name}.nii.gz")).dataobj)  # noqa: E731
        lesion = load("infarct_followup") > 0
        z = int(lesion.sum(axis=(0, 1)).argmax())
        for ax, name, low, high in ((row[0], "ncct", 0, 80), (row[1], "tmax", 0, 12)):
            ax.imshow(load(name)[:, :, z].T, cmap="gray", vmin=low, vmax=high, origin="lower")
            if lesion[:, :, z].any():
                ax.contour(lesion[:, :, z].T, levels=[0.5], colors="red", linewidths=0.7)
            ax.set_title(f"{case} z={z} {name}", fontsize=8)
        occlusion = load("cta_occlusion") > 0
        row[2].imshow(load("cow").max(axis=2).T, cmap="tab20", interpolation="nearest", origin="lower")
        if occlusion.any():
            row[2].contour(occlusion.max(axis=2).T, levels=[0.5], colors="black", linewidths=1.0)
        row[2].set_title(f"{case} labels, occlusion in black", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
