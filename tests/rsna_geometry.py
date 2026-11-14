"""RSNA scan geometry from the shard index: slice spacing and field of view (PR #3 review, open items 2 and 3).

The ISLES'24 domain-shift check (EXP-16) reformats native NCCT to 5 mm slabs, the spacing RSNA documents, and
leaves the field of view unmatched. This measures both on the RSNA training scans, from index columns only:

- spacing: the gap between neighbouring slice centres, from `z_mm` (ImagePositionPatient z). It is the slice
  interval, not the slice thickness: slices can overlap, and RSNA's files carry no SliceThickness (plan §5.2).
  With a tilted gantry the z gap is the table step, slightly more than the true interval;
- field of view: matrix width times `pixel_spacing_mm` (PixelSpacing, row direction). The matrix is not in the
  index, so it is taken as RSNA's documented 512 pixels.

No image is read. Results are counts, distributions and percentiles; no slice, scan or patient ID.
"""

import numpy as np
import pandas as pd

MATRIX = 512  # RSNA's documented image size; the index does not store it
DOCUMENTED_SPACING_MM = 5.0
TOLERANCE_MM = 0.25  # a scan's median gap within this of a value counts as that value
IRREGULAR_RATIO = 1.1  # largest gap over smallest above this: the spacing changes within the scan
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def per_scan(index):
    """One row per scan: slices, gaps between neighbouring slice centres, extent and pixel spacing.

    Slices with no z_mm are dropped and counted. Duplicate positions give a zero gap and are counted, not
    dropped, because they say something about the scan.
    """
    frame = index[["series_id", "z_mm", "pixel_spacing_mm"]].copy()
    frame["z_mm"] = pd.to_numeric(frame["z_mm"], errors="coerce")
    frame["pixel_spacing_mm"] = pd.to_numeric(frame["pixel_spacing_mm"], errors="coerce")
    rows = []
    for series, group in frame.groupby("series_id", sort=False):
        z = np.sort(group["z_mm"].dropna().to_numpy())
        gaps = np.diff(z)
        positive = gaps[gaps > 1e-3]
        spacing = group["pixel_spacing_mm"].dropna()
        rows.append({
            "series_id": series,
            "slices": int(len(group)),
            "missing_z": int(group["z_mm"].isna().sum()),
            "duplicate_positions": int((gaps <= 1e-3).sum()),
            "median_gap_mm": float(np.median(positive)) if positive.size else np.nan,
            "min_gap_mm": float(positive.min()) if positive.size else np.nan,
            "max_gap_mm": float(positive.max()) if positive.size else np.nan,
            "extent_mm": float(z[-1] - z[0]) if z.size > 1 else np.nan,
            "pixel_spacing_mm": float(spacing.median()) if len(spacing) else np.nan,
            "pixel_spacing_varies": bool(spacing.nunique() > 1),
        })
    table = pd.DataFrame(rows)
    table["irregular"] = table["max_gap_mm"] > IRREGULAR_RATIO * table["min_gap_mm"]
    table["fov_mm"] = MATRIX * table["pixel_spacing_mm"]
    return table


def _percentiles(values, digits=3):
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return None
    return {**{f"p{p}": round(float(np.percentile(values, p)), digits) for p in PERCENTILES},
            "min": round(float(values.min()), digits), "max": round(float(values.max()), digits),
            "mean": round(float(values.mean()), digits), "n": int(len(values))}


def _common_values(values, step, top=8):
    """The most common values after rounding to step, as {value: scans}, most common first."""
    rounded = (pd.Series(values, dtype=float).dropna() / step).round() * step
    counts = rounded.round(3).value_counts().head(top)
    return {f"{v:g}": int(c) for v, c in counts.items()}


def summarise(scans, documented_mm=DOCUMENTED_SPACING_MM, tolerance_mm=TOLERANCE_MM):
    """Numbers only: how RSNA scans are spaced and how much of the head each image covers."""
    gap = scans["median_gap_mm"]
    near = lambda v: int(((gap - v).abs() <= tolerance_mm).sum())  # noqa: E731
    return {
        "scans": int(len(scans)),
        "slices": int(scans["slices"].sum()),
        "matrix_assumed": MATRIX,
        "spacing": {
            "note": "Gap between neighbouring slice centres (slice interval), not slice thickness.",
            "per_scan_median_gap_mm": _percentiles(gap),
            "common_gaps_mm": _common_values(gap, 0.25),
            "scans_near_documented": {"value_mm": documented_mm, "tolerance_mm": tolerance_mm, "scans": near(documented_mm),
                                      "share": round(near(documented_mm) / len(scans), 4) if len(scans) else None},
            "scans_without_gap": int(gap.isna().sum()),
            "irregular_scans": int(scans["irregular"].sum()),
            "scans_with_duplicate_positions": int((scans["duplicate_positions"] > 0).sum()),
            "slices_missing_z": int(scans["missing_z"].sum()),
            "extent_mm": _percentiles(scans["extent_mm"], 1),
        },
        "field_of_view": {
            "note": f"{MATRIX} x PixelSpacing; the matrix is RSNA's documented size, not read from each file.",
            "pixel_spacing_mm": _percentiles(scans["pixel_spacing_mm"], 4),
            "fov_mm": _percentiles(scans["fov_mm"], 1),
            "common_fov_mm": _common_values(scans["fov_mm"], 10),
            "scans_without_spacing": int(scans["pixel_spacing_mm"].isna().sum()),
            "scans_where_spacing_varies": int(scans["pixel_spacing_varies"].sum()),
        },
    }
