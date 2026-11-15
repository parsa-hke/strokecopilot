"""RSNA scan geometry from index columns, on synthetic indexes. No RSNA data."""

import json

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from strokecopilot.data import rsna_geometry as geo  # noqa: E402


def _scan(series, n, gap, spacing=0.488, start=0.0, shuffle=True, seed=0):
    z = start + gap * np.arange(n)
    if shuffle:
        z = np.random.default_rng(seed).permutation(z)
    return pd.DataFrame({"series_id": series, "z_mm": z, "pixel_spacing_mm": spacing})


def _index():
    return pd.concat([
        _scan("ID_a", 32, 5.0),
        _scan("ID_b", 30, 5.0, spacing=0.43),
        _scan("ID_c", 60, 2.5, spacing=0.5),
        _scan("ID_d", 40, 4.8),
    ], ignore_index=True)


def test_per_scan_measures_gaps_extent_and_fov_whatever_the_slice_order():
    scans = geo.per_scan(_index()).set_index("series_id")
    assert scans.loc["ID_a", "median_gap_mm"] == pytest.approx(5.0) and scans.loc["ID_c", "median_gap_mm"] == pytest.approx(2.5)
    assert scans.loc["ID_a", "extent_mm"] == pytest.approx(31 * 5.0) and scans.loc["ID_a", "slices"] == 32
    assert scans.loc["ID_b", "fov_mm"] == pytest.approx(512 * 0.43)
    assert not scans["irregular"].any() and (scans["duplicate_positions"] == 0).all()


def test_per_scan_flags_irregular_spacing_duplicates_and_missing_positions():
    irregular = pd.DataFrame({"series_id": "ID_e", "z_mm": [0, 5, 10, 15, 25, 35], "pixel_spacing_mm": 0.5})
    duplicate = pd.DataFrame({"series_id": "ID_f", "z_mm": [0, 5, 5, 10], "pixel_spacing_mm": 0.5})
    missing = pd.DataFrame({"series_id": "ID_g", "z_mm": [0, 5, None, 10], "pixel_spacing_mm": [0.5, 0.5, 0.5, 0.45]})
    lone = pd.DataFrame({"series_id": "ID_h", "z_mm": [3.0], "pixel_spacing_mm": None})
    scans = geo.per_scan(pd.concat([irregular, duplicate, missing, lone], ignore_index=True)).set_index("series_id")
    assert bool(scans.loc["ID_e", "irregular"]) and scans.loc["ID_e", "median_gap_mm"] == pytest.approx(5.0)
    assert scans.loc["ID_f", "duplicate_positions"] == 1 and scans.loc["ID_f", "median_gap_mm"] == pytest.approx(5.0)
    assert scans.loc["ID_g", "missing_z"] == 1 and bool(scans.loc["ID_g", "pixel_spacing_varies"])
    assert np.isnan(scans.loc["ID_h", "median_gap_mm"]) and np.isnan(scans.loc["ID_h", "fov_mm"])


def test_summarise_counts_scans_at_the_documented_spacing_and_holds_no_ids():
    summary = geo.summarise(geo.per_scan(_index()))
    spacing, fov = summary["spacing"], summary["field_of_view"]
    assert summary["scans"] == 4 and summary["slices"] == 162
    assert spacing["scans_near_documented"]["scans"] == 3 and spacing["scans_near_documented"]["share"] == 0.75  # 4.8 mm is within 0.25
    assert spacing["common_gaps_mm"] == {"5": 2, "2.5": 1, "4.75": 1}
    assert spacing["per_scan_median_gap_mm"]["p50"] == pytest.approx(4.9)
    assert fov["fov_mm"]["max"] == pytest.approx(round(512 * 0.5, 1)) and fov["scans_without_spacing"] == 0
    assert "ID_" not in json.dumps(summary)


def test_summarise_handles_missing_values():
    lone = pd.DataFrame({"series_id": ["ID_x"], "z_mm": [1.0], "pixel_spacing_mm": [None]})
    summary = geo.summarise(geo.per_scan(lone))
    assert summary["spacing"]["scans_without_gap"] == 1 and summary["spacing"]["per_scan_median_gap_mm"] is None
    assert summary["field_of_view"]["scans_without_spacing"] == 1 and summary["field_of_view"]["fov_mm"] is None
