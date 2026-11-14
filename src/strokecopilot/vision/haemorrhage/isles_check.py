"""Vision B on ISLES'24 native NCCT: the domain-shift check (plan §7 Phase 3, §8.2; docs/team/mohamed_vision_b.md).

ISLES'24 patients went on to reperfusion therapy, so their plain CT is presumed to show no bleed, and
every scan Vision B calls positive counts as a presumed false positive. There are no bleed labels,
so reports say "presumed" throughout.

Each native NCCT goes through the same steps as an RSNA training slice, so the check measures the new
data rather than a difference in preprocessing:
1. turn to RAS, then lay out axial slices as DICOM stores them: rows run front to back and columns
   from the patient's right to left (front of the head at the top, patient's right on the left);
2. average neighbouring native slices into slabs of the training thickness, in HU, weighting each
   native slice by its overlap with the slab, because thin slices are noisier than RSNA's 5 mm ones;
3. pad to a square with air, so resizing keeps the aspect ratio, then `rsna.window_hu`: brain,
   subdural and bone windows, resized to 320 px with the same interpolation as the shards;
4. JPEG at the shards' quality and back, because the model only ever saw JPEG-compressed slices.

Scan scores use EXP-6's aggregation and calibration (`calibration.json`) through `infer`, so the
threshold is the calibrated one. Reports hold counts, rates and percentiles only. The per-case table
holds case IDs, so it stays on Drive (docs/team/README.md).
"""

import math
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from strokecopilot.data import rsna, splits
from strokecopilot.data.prep_rsna import IMAGE_SIZE, JPEG_QUALITY
from strokecopilot.vision.haemorrhage import infer

# RSNA's dataset description gives axial 5 mm slices, with one site resampled to 5 mm (plan §5.2). The
# shards' z_mm could confirm the spacing on Kaggle; until then this is the documented value.
TARGET_SLICE_MM = 5.0
AIR_HU = -1000.0
# The primary reading reformats to the training thickness. "native" scores the unchanged thin
# slices, to show how much of any false-positive rate the reformatting itself accounts for.
VARIANTS = {"slab_5mm": TARGET_SLICE_MM, "native": None}
PRIMARY = "slab_5mm"
# RSNA training scans' median field of view, 512 x pixel spacing (EXP-28, notebook 23: p5-p95 216.0-253.3 mm).
RSNA_FOV_MM = 250.0
# The follow-up to EXP-16 (PR #3 review, item 3): the same 5 mm slabs with the field of view left as it is
# (reproduces EXP-16) and matched to RSNA's, so the change in positive calls is paired scan by scan.
FOV_VARIANTS = {"slab_5mm": TARGET_SLICE_MM, "slab_5mm_fov250": {"target_mm": TARGET_SLICE_MM, "fov_mm": RSNA_FOV_MM}}
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
PERCENTILES = (5, 25, 50, 75, 90, 95, 99)
HISTOGRAM_BINS = 20
TOP_SLABS = 2
Z95 = 1.959964


def axial_stack(img):
    """nibabel image -> float32 HU array (slices, rows, cols) in DICOM layout, and (slice, row, col) mm.

    Slices run from foot to head. Missing values become air.
    """
    import nibabel as nib

    canonical = nib.as_closest_canonical(img)
    hu = np.asarray(canonical.get_fdata(dtype=np.float32))
    if hu.ndim == 4 and hu.shape[3] == 1:
        hu = hu[..., 0]
    if hu.ndim != 3:
        raise ValueError(f"Expected a 3D CT volume, got {hu.ndim} dimensions.")
    x_mm, y_mm, z_mm = (float(v) for v in canonical.header.get_zooms()[:3])
    # RAS index i runs to the patient's right and j to the front. Flipping both and swapping gives
    # rows running to the back and columns to the patient's left, as in an RSNA DICOM slice.
    stack = np.transpose(hu[::-1, ::-1, :], (2, 1, 0))
    return np.ascontiguousarray(np.nan_to_num(stack, nan=AIR_HU)), (z_mm, y_mm, x_mm)


def variant_spec(spec):
    """A variant's (target_mm, fov_mm). A number or None is a slab thickness alone; a dict may add fov_mm."""
    if isinstance(spec, dict):
        unknown = set(spec) - {"target_mm", "fov_mm"}
        if unknown:
            raise ValueError(f"Unknown variant settings {sorted(unknown)}.")
        return spec.get("target_mm"), spec.get("fov_mm")
    return spec, None


def slab_weights(n_slices, slice_mm, target_mm):
    """(slabs, n_slices) weights that average native slices into slabs of target_mm.

    Each native slice covers slice_mm and counts by its overlap with a slab. Whole slabs only, centred
    on the covered range. With target_mm None, or not thicker than the native slices, nothing changes.
    """
    if n_slices < 1:
        raise ValueError("A volume needs at least one slice.")
    if target_mm is None or target_mm <= slice_mm * (1 + 1e-6):
        return np.eye(n_slices, dtype=np.float32)
    total = n_slices * slice_mm
    n_slabs = max(1, int(math.floor(total / target_mm + 1e-6)))
    offset = max(0.0, (total - n_slabs * target_mm) / 2)
    native_lo = np.arange(n_slices) * slice_mm
    native_hi = native_lo + slice_mm
    slab_lo = offset + np.arange(n_slabs) * target_mm
    slab_hi = slab_lo + min(target_mm, total)
    overlap = np.minimum(slab_hi[:, None], native_hi[None, :]) - np.maximum(slab_lo[:, None], native_lo[None, :])
    weights = np.clip(overlap, 0, None)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def to_slabs(stack, slice_mm, target_mm):
    """(slices, rows, cols) HU -> (slabs, rows, cols) HU, averaged in HU before any windowing."""
    stack = stack.astype(np.float32, copy=False)
    if target_mm is None or target_mm <= slice_mm * (1 + 1e-6):
        return stack
    return np.tensordot(slab_weights(stack.shape[0], slice_mm, target_mm), stack, axes=(1, 0))


def _fit_axis(stack, axis, length, value=AIR_HU):
    """Centre-crop or centre-pad one axis to length."""
    size = stack.shape[axis]
    if size > length:
        start = (size - length) // 2
        return np.take(stack, np.arange(start, start + length), axis=axis)
    if size < length:
        before = (length - size) // 2
        pad = [(0, 0)] * stack.ndim
        pad[axis] = (before, length - size - before)
        return np.pad(stack, pad, constant_values=value)
    return stack


def square(stack, value=AIR_HU):
    """Pad rows or columns with air so every slice is square."""
    side = max(stack.shape[1], stack.shape[2])
    return _fit_axis(_fit_axis(stack, 1, side, value), 2, side, value)


def set_field_of_view(stack, row_mm, col_mm, fov_mm):
    """Crop or pad (with air) to a fixed field of view in mm, about the centre.

    Off in VARIANTS (EXP-16); FOV_VARIANTS sets it to RSNA_FOV_MM, so a head covers as much of the image as in
    RSNA. The centre is the image centre, not the head's: a head far off centre in a large field could be cut.
    """
    stack = _fit_axis(stack, 1, max(1, round(fov_mm / row_mm)))
    return _fit_axis(stack, 2, max(1, round(fov_mm / col_mm)))


def jpeg_roundtrip(image, quality=JPEG_QUALITY):
    """Encode and decode as the shards were written (prep_rsna), keeping the channel order."""
    encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not encoded:
        raise ValueError("JPEG encoding failed.")
    return rsna.decode(jpeg.tobytes())


def window_slices(slabs, size=IMAGE_SIZE, jpeg_quality=JPEG_QUALITY):
    """(slabs, L, L) HU -> uint8 (slabs, size, size, 3): brain, subdural and bone, as in training."""
    out = np.empty((slabs.shape[0], size, size, 3), dtype=np.uint8)
    for i, slab in enumerate(slabs):
        image = rsna.window_hu(slab, size)
        out[i] = jpeg_roundtrip(image, jpeg_quality) if jpeg_quality else image
    return out


def prepare(img, target_mm=TARGET_SLICE_MM, size=IMAGE_SIZE, jpeg_quality=JPEG_QUALITY, fov_mm=None):
    """Native NCCT image -> (uint8 slabs x size x size x 3 for the model, info with numbers only)."""
    import nibabel as nib

    stack, (slice_mm, row_mm, col_mm) = axial_stack(img)
    slabs = to_slabs(stack, slice_mm, target_mm)
    if fov_mm:
        slabs = set_field_of_view(slabs, row_mm, col_mm, fov_mm)
    slabs = square(slabs)
    sample = stack[:, ::4, ::4]
    info = {
        "orientation": "".join(nib.aff2axcodes(img.affine)),
        "native_slices": int(stack.shape[0]),
        "slice_mm": round(slice_mm, 3),
        "row_mm": round(row_mm, 4),
        "col_mm": round(col_mm, 4),
        "fov_mm": round(max(stack.shape[1] * row_mm, stack.shape[2] * col_mm), 1),
        "slabs": int(slabs.shape[0]),
        "slab_mm": round(target_mm if target_mm and target_mm > slice_mm else slice_mm, 3),
        "thicker_than_target": bool(target_mm is not None and slice_mm > target_mm * (1 + 1e-6)),
        "hu_p01": round(float(np.percentile(sample, 1)), 1),
        "hu_p99": round(float(np.percentile(sample, 99)), 1),
    }
    return window_slices(slabs, size, jpeg_quality), info


def score(images, predict, calibrator):
    """Windowed slabs -> scan score, calibrated confidence, the call, and the slabs that drove it.

    predict maps uint8 (n, size, size, 3) images to n slice probabilities, e.g. infer.slice_probs.
    """
    probs = np.asarray(predict(images), dtype=float)
    if probs.shape != (len(images),):
        raise ValueError(f"predict returned {probs.shape} for {len(images)} slabs.")
    raw = infer.scan_score(probs, calibrator.method, calibrator.k)
    confidence = float(calibrator.apply(np.array([raw]))[0])
    top = np.argsort(-probs, kind="stable")[: max(TOP_SLABS, calibrator.k or 1)]
    return {"raw_score": raw, "confidence": confidence, "present": bool(confidence >= calibrator.threshold),
            "max_slab_prob": float(probs.max()), "top_slabs": [int(i) for i in top]}


def wilson(count, n, z=Z95):
    """Wilson score interval for a proportion. (None, None) when n is 0."""
    if n == 0:
        return None, None
    p = count / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _percentiles(values, points=PERCENTILES):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return None
    return {**{f"p{p}": round(float(np.percentile(values, p)), 4) for p in points},
            "min": round(float(values.min()), 4), "max": round(float(values.max()), 4), "mean": round(float(values.mean()), 4)}


def rate_at(confidences, threshold):
    """Scans called positive at threshold, as a presumed false-positive rate with a 95% Wilson interval."""
    confidences = np.asarray(confidences, dtype=float)
    called = int((confidences >= threshold).sum())
    low, high = wilson(called, len(confidences))
    rate = called / len(confidences) if len(confidences) else None
    return {"threshold": round(float(threshold), 4), "called_positive": called, "scans": int(len(confidences)),
            "presumed_fp_rate": None if rate is None else round(rate, 4),
            "ci95": [None if low is None else round(low, 4), None if high is None else round(high, 4)]}


def summarise(confidences, threshold, thresholds=THRESHOLDS, bins=HISTOGRAM_BINS):
    """Numbers only: the rate at the chosen threshold, at fixed thresholds, percentiles and a histogram."""
    confidences = np.asarray(confidences, dtype=float)
    counts, edges = np.histogram(confidences, bins=bins, range=(0.0, 1.0))
    return {
        "scans": int(len(confidences)),
        "at_threshold": rate_at(confidences, threshold),
        "by_threshold": [rate_at(confidences, t) for t in sorted({*thresholds, round(float(threshold), 4)})],
        "confidence": _percentiles(confidences),
        "histogram": {"edges": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts]},
    }


def run_check(cases, load_image, predict, calibrator, variants=VARIANTS, primary=PRIMARY, progress=None):
    """Score every case under every variant. Returns (per-case table with case IDs, metrics without IDs).

    load_image(case) returns the native NCCT as a nibabel image (isles24.load_native_ncct, which
    refuses held-out cases; that refusal stops the run). Any other failure is kept in the table with
    its error and counted by type in the metrics; a case is never silently dropped.
    """
    if primary not in variants:
        raise ValueError(f"The primary variant {primary!r} is not among {sorted(variants)}.")
    rows = []
    for i, case in enumerate(cases, 1):
        row = {"case": case, "error": ""}
        try:
            img = load_image(case)
            for name, spec in variants.items():
                target_mm, fov_mm = variant_spec(spec)
                images, info = prepare(img, target_mm, fov_mm=fov_mm)
                if name == primary:
                    row.update(info)
                else:
                    row[f"{name}_slabs"] = info["slabs"]
                result = score(images, predict, calibrator)
                row.update({f"{name}_{key}": value for key, value in result.items() if key != "top_slabs"})
                row[f"{name}_top_slabs"] = ";".join(str(s) for s in result["top_slabs"])
        except splits.HeldoutError:
            raise  # never a per-case failure: the whole run stops
        except Exception as error:  # reported, not hidden: see metrics["failed"]
            row["error"] = f"{type(error).__name__}: {str(error)[:200]}"
        rows.append(row)
        if progress:
            progress(i, len(cases))
    table = pd.DataFrame(rows)
    ok = table[table["error"] == ""].copy()
    if not ok.empty:
        ok = ok.sort_values(f"{primary}_confidence", ascending=False, kind="stable")
        ok["rank"] = np.arange(1, len(ok) + 1)
        table = table.merge(ok[["case", "rank"]], on="case", how="left")
    else:
        table["rank"] = np.nan
    table = table.sort_values(["rank", "case"], na_position="last").reset_index(drop=True)
    return table, metrics_of(table, calibrator, variants, primary)


def metrics_of(table, calibrator, variants=VARIANTS, primary=PRIMARY):
    """Metrics from the per-case table: counts, rates and percentiles, no case IDs."""
    ok = table[table["error"] == ""]
    failed = table[table["error"] != ""]
    metrics = {
        "cases": int(len(table)),
        "scored": int(len(ok)),
        "failed": int(len(failed)),
        "failure_types": dict(Counter(e.split(":")[0] for e in failed["error"])),
        "label_note": "No bleed labels: ISLES'24 patients went on to reperfusion therapy, so every positive call is a presumed false positive.",
        "calibrator": {"method": calibrator.method, "k": calibrator.k, "kind": calibrator.kind, "threshold": calibrator.threshold},
        "primary": primary,
        "target_slice_mm": variant_spec(variants[primary])[0],
        "variant_settings": {name: dict(zip(("target_mm", "fov_mm"), variant_spec(spec))) for name, spec in variants.items()},
        "variants": {name: summarise(ok[f"{name}_confidence"], calibrator.threshold) for name in variants} if len(ok) else {},
    }
    if len(ok):
        metrics["native_ncct"] = {
            "slice_mm": _percentiles(ok["slice_mm"]),
            "native_slices": _percentiles(ok["native_slices"]),
            "slabs": _percentiles(ok["slabs"]),
            "in_plane_mm": _percentiles(ok["row_mm"]),
            "fov_mm": _percentiles(ok["fov_mm"]),
            "hu_p01": _percentiles(ok["hu_p01"]),
            "thicker_than_target": int(ok["thicker_than_target"].sum()),
            "orientation": dict(Counter(ok["orientation"])),
        }
        main = ok[f"{primary}_present"].astype(bool)
        metrics["agreement"] = {}
        for other in variants:
            if other == primary:
                continue
            alt = ok[f"{other}_present"].astype(bool)
            metrics["agreement"][other] = {"both_positive": int((main & alt).sum()),
                                           f"only_{primary}": int((main & ~alt).sum()), f"only_{other}": int((~main & alt).sum())}
    return metrics


def ids_in(text, ids):
    """How many of ids appear in text. Used to check that saved metrics hold no case IDs."""
    return sum(1 for i in ids if i and i in text)


def histogram_figure(metrics, out_png=None):
    """Bar chart of calibrated scan confidence per variant, with the threshold marked. Counts only."""
    import matplotlib

    if out_png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = metrics["variants"]
    fig, axes = plt.subplots(1, len(variants), figsize=(4.8 * len(variants), 3.6), squeeze=False)
    threshold = metrics["calibrator"]["threshold"]
    for ax, (name, summary) in zip(axes[0], variants.items()):
        edges, counts = summary["histogram"]["edges"], summary["histogram"]["counts"]
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#4c72b0", edgecolor="white")
        ax.axvline(threshold, color="#c44e52", linestyle="--", linewidth=1.2, label=f"threshold {threshold:.3f}")
        at = summary["at_threshold"]
        ax.set(xlim=(0, 1), xlabel="Calibrated bleed probability", ylabel="Scans",
               title=f"{name}: {at['called_positive']}/{at['scans']} called positive")
        ax.legend(fontsize=8)
    fig.suptitle("Vision B on ISLES'24 development NCCT (presumed no bleed)", fontsize=10)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        return out_png
    return fig


def top_scans_figure(table, load_image, n=8, primary=PRIMARY, out_png=None, variants=VARIANTS):
    """The brain window of the slabs that drove each of the n highest-scoring scans.

    Panels are labelled by rank and score, never by case ID; the table maps rank to case on Drive.
    """
    import matplotlib

    if out_png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = table[(table["error"] == "") & table["rank"].notna()].nsmallest(n, "rank")
    fig, axes = plt.subplots(max(1, len(top)), TOP_SLABS, figsize=(3.0 * TOP_SLABS, 3.0 * max(1, len(top))), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for row_axes, (_, row) in zip(axes, top.iterrows()):
        target_mm, fov_mm = variant_spec(variants.get(primary, TARGET_SLICE_MM))
        images, _ = prepare(load_image(row["case"]), target_mm, fov_mm=fov_mm)
        slabs = [int(s) for s in str(row[f"{primary}_top_slabs"]).split(";") if s][:TOP_SLABS]
        for ax, slab in zip(row_axes, slabs):
            ax.imshow(images[slab][:, :, 0], cmap="gray", vmin=0, vmax=255)  # channel 0 is the brain window
            ax.set_title(f"#{int(row['rank'])}  p={row[f'{primary}_confidence']:.3f}  slab {slab}", fontsize=8)
    fig.suptitle("Highest-scoring scans, brain window (40/80)", fontsize=10)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        return out_png
    return fig
