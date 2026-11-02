"""ISLES'24 slim preparation on small synthetic NIfTI files. Needs the [isles] extra."""

import json
import os

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")
pytest.importorskip("scipy")

from strokecopilot.data import clinical_columns as cc  # noqa: E402
from strokecopilot.data import isles24_layout as layout  # noqa: E402
from strokecopilot.data import prep_isles24 as prep  # noqa: E402

SHAPE = (120, 130, 40)
ZOOMS = (0.5, 0.5, 2.0)
AFFINE = np.diag([*ZOOMS, 1.0])
CASE = "sub-stroke0001"


def _save(root, case, name, array, affine=AFFINE):
    path = layout.path_of(str(root), case, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), path)


def _sphere(centre_mm, radius_mm):
    grid = np.indices(SHAPE).astype(np.float32)
    distance = sum(((grid[axis] + 0.5) * ZOOMS[axis] - centre_mm[axis]) ** 2 for axis in range(3))
    return distance <= radius_mm ** 2


def _case(root, case=CASE, lesion_radius=10.0, affine=AFFINE, lvo=True, cow_affine=None):
    ncct = np.full(SHAPE, -1000, dtype=np.float32)
    ncct[10:110, 10:120, 4:36] = 35
    _save(root, case, "ncct", ncct, affine)
    for name in ("cbf", "cbv", "mtt"):
        values = np.zeros(SHAPE, dtype=np.float32)
        values[10:110, 10:120, 4:36] = 20
        _save(root, case, name, values, affine)
    tmax = np.full(SHAPE, -30.0, dtype=np.float32)
    tmax[10:110, 10:120, 4:30] = 2.0
    tmax[20:60, 30:70, 10:20] = 9.0
    _save(root, case, "tmax", tmax, affine)
    lesion = _sphere((25.0, 30.0, 30.0), lesion_radius) if lesion_radius else np.zeros(SHAPE, dtype=bool)
    _save(root, case, "lesion", lesion.astype(np.float64), affine)
    occlusion = np.zeros(SHAPE, dtype=np.uint8)
    if lvo:
        occlusion[60:66, 60:66, 20:22] = 1
    _save(root, case, "lvo", occlusion, affine)
    cow = np.zeros(SHAPE, dtype=np.uint8)
    cow[50:60, 60:66, 20:22] = 5
    cow[70:80, 60:66, 20:22] = 7
    _save(root, case, "cow", cow, cow_affine if cow_affine is not None else affine)
    (root / "marker").write_text("")


def _loaded(out, name):
    return nib.load(os.path.join(out, f"{name}.nii.gz"))


def test_process_case_keeps_volumes_grid_and_types(tmp_path):
    _case(tmp_path)
    out = str(tmp_path / "out")
    row = prep.process_case(str(tmp_path), CASE, out)

    for name in ("ncct", "cbf", "tmax", "infarct_followup", "cta_occlusion", "cow"):
        img = _loaded(out, name)
        assert img.shape == tuple(row["shape"]) and np.allclose(img.header.get_zooms()[:3], 1.5)
    assert _loaded(out, "ncct").get_data_dtype() == np.int16
    assert _loaded(out, "infarct_followup").get_data_dtype() == np.uint8
    assert abs(row["lesion_ml_slim"] - row["lesion_ml"]) / row["lesion_ml"] < 0.03
    assert abs(row["tmax6_ml_slim"] - row["tmax6_ml"]) / row["tmax6_ml"] < 0.03
    assert (row["moved_to_ncct_grid"], row["masks_rescued"]) == ([], [])
    assert row["shape"][0] < SHAPE[0] * ZOOMS[0] / 1.5 + 8
    assert row["lesion_with_tmax"] == 1.0


def test_native_ncct_is_an_unchanged_copy(tmp_path):
    _case(tmp_path)
    out = str(tmp_path / "out")
    prep.process_case(str(tmp_path), CASE, out)
    with open(layout.path_of(str(tmp_path), CASE, "ncct"), "rb") as a, open(os.path.join(out, "ncct_native.nii.gz"), "rb") as b:
        assert a.read() == b.read()


def test_lesion_stays_at_the_same_place_in_the_scanner(tmp_path):
    flipped = AFFINE.copy()
    flipped[0, 0] = -ZOOMS[0]
    flipped[0, 3] = SHAPE[0] * ZOOMS[0]
    _case(tmp_path, affine=flipped)
    out = str(tmp_path / "out")
    row = prep.process_case(str(tmp_path), CASE, out)
    assert row["original_axes"] == "LAS"

    def centre(img):
        return nib.affines.apply_affine(img.affine, np.argwhere(np.asanyarray(img.dataobj) > 0).mean(axis=0))

    small = _loaded(out, "infarct_followup")
    assert nib.aff2axcodes(small.affine) == ("R", "A", "S")
    assert np.allclose(centre(small), centre(nib.load(layout.path_of(str(tmp_path), CASE, "lesion"))), atol=0.3)


def test_tmax_background_stays_exact_and_is_not_blended(tmp_path):
    _case(tmp_path)
    out = str(tmp_path / "out")
    prep.process_case(str(tmp_path), CASE, out)
    tmax = np.asanyarray(_loaded(out, "tmax").dataobj)
    outside = tmax[tmax < 1.9]
    assert outside.size and np.all(outside == -30.0)
    assert tmax[tmax > -30].min() >= 1.99


def test_non_finite_map_values_are_counted_and_become_background(tmp_path):
    _case(tmp_path)
    mtt = np.full(SHAPE, 20.0, dtype=np.float32)
    mtt[0:30, 0:30, 0:10] = np.nan
    _save(tmp_path, CASE, "mtt", mtt)
    out = str(tmp_path / "out")
    row = prep.process_case(str(tmp_path), CASE, out)
    assert row["non_finite_voxels"]["mtt"] == 9000
    small = np.asanyarray(_loaded(out, "mtt").dataobj)
    assert np.isfinite(small).all()


def test_lesion_outside_tmax_is_measured(tmp_path):
    _case(tmp_path)
    tmax = np.full(SHAPE, -30.0, dtype=np.float32)
    _save(tmp_path, CASE, "tmax", tmax)
    row = prep.process_case(str(tmp_path), CASE, str(tmp_path / "out"))
    assert row["lesion_with_tmax"] == 0.0 and row["tmax_negative_share"] is None


def test_tiny_lesion_is_rescued_and_empty_masks_stay_empty(tmp_path):
    _case(tmp_path, lesion_radius=0.0, lvo=False)
    lesion = np.zeros(SHAPE, dtype=np.float64)
    lesion[40, 40, 15] = 1
    _save(tmp_path, CASE, "lesion", lesion)
    out = str(tmp_path / "out")
    row = prep.process_case(str(tmp_path), CASE, out)
    assert row["masks_rescued"] == ["lesion"]
    assert np.asanyarray(_loaded(out, "infarct_followup").dataobj).sum() == 1
    assert np.asanyarray(_loaded(out, "cta_occlusion").dataobj).sum() == 0

    _save(tmp_path, CASE, "lesion", np.zeros(SHAPE, dtype=np.float64))
    row = prep.process_case(str(tmp_path), CASE, str(tmp_path / "out2"))
    assert row["masks_rescued"] == [] and row["lesion_with_tmax"] is None


def test_off_grid_labels_are_moved_by_world_position(tmp_path):
    shifted = AFFINE.copy()
    shifted[1, 3] = 1.0
    _case(tmp_path, cow_affine=shifted)
    out = str(tmp_path / "out")
    row = prep.process_case(str(tmp_path), CASE, out)
    assert row["moved_to_ncct_grid"] == ["cow"]
    small = _loaded(out, "cow")
    assert set(np.unique(np.asanyarray(small.dataobj))) == {0, 5, 7}


def test_crop_grows_for_a_mask_outside_the_brain_box(tmp_path):
    _case(tmp_path, lvo=False)
    lvo = np.zeros(SHAPE, dtype=np.uint8)
    lvo[1:3, 1:3, 1:2] = 1
    _save(tmp_path, CASE, "lvo", lvo)
    out = str(tmp_path / "out")
    row = prep.process_case(str(tmp_path), CASE, out)
    assert row["crop_grew_for_mask"] is True
    assert np.asanyarray(_loaded(out, "cta_occlusion").dataobj).any()


def _tables(extra_baseline=None, extra_outcome=None):
    baseline = {column: [1] for column in cc.BASELINE_COLUMNS}
    outcome = {column: [2] for column in cc.OUTCOME_COLUMNS}
    return {
        "baseline": {
            "a": {"s": pd.DataFrame(baseline, dtype=object)},
            "b": {"s": extra_baseline if extra_baseline is not None else pd.DataFrame(baseline, dtype=object)},
        },
        "outcome": {
            "a": {"s": pd.DataFrame(outcome, dtype=object)},
            "b": {"s": extra_outcome if extra_outcome is not None else pd.DataFrame(outcome, dtype=object)},
        },
    }


def test_merge_rows_fills_gaps_and_blanks_conflicts():
    df = pd.DataFrame({"x": [1, 1], "y": [None, 5], "z": [1, 2], "w": [None, None]}, dtype=object)
    row, found = prep.merge_rows(df)
    assert row["x"] == 1 and row["y"] == 5 and row["z"] is None and row["w"] is None
    assert found == {"rows": 2, "only_on_later_rows": ["y"], "conflicts_set_missing": ["z"]}


def test_clinical_has_one_row_per_case_and_blanks_unknown_codes():
    two = pd.DataFrame({column: [1, 1] for column in cc.BASELINE_COLUMNS}, dtype=object)
    two["Hyperlipidemia"] = [99, 99]
    two["INR"] = [1.0, 1.1]
    clinical, report = prep.build_clinical(_tables(extra_baseline=two), ["a", "b"])
    assert list(clinical["case"]) == ["a", "b"]
    assert list(clinical.columns) == ["case", *cc.BASELINE_COLUMNS, *cc.OUTCOME_COLUMNS]
    b = clinical[clinical["case"] == "b"].iloc[0]
    assert pd.isna(b["Hyperlipidemia"]) and pd.isna(b["INR"])
    assert report["unknown_codes_set_missing"] == {"Hyperlipidemia": 1}
    assert report["merged_workbooks"] == [{"kind": "baseline", "rows": 2, "only_on_later_rows": [], "conflicts_set_missing": ["INR"]}]


def test_clinical_refuses_an_untagged_column():
    bad = pd.DataFrame({"mTICI": [1]}, dtype=object)
    with pytest.raises(KeyError, match="no note tag"):
        prep.build_clinical(_tables(extra_outcome=bad), ["a", "b"])


def test_time_formats_hide_digits():
    clinical = pd.DataFrame({"case": ["a", "b"], "Onset to door": ["01:30", None], "Alert to door": [None, None], "Door to imaging": ["00:20", "1:05"]})
    found = prep.time_formats(clinical, ["a"])
    assert found == {"Onset to door": {"99:99": 1}, "Alert to door": {}, "Door to imaging": {"99:99": 1}}


@pytest.fixture
def built(tmp_path):
    cases = [f"sub-stroke{i:04d}" for i in (1, 2, 3)]
    root = tmp_path / "train"
    root.mkdir()
    for case in cases:
        _case(root, case)
    parts = str(tmp_path / "parts")
    assert prep.build_parts(str(root), cases, parts, part_size=2) == 2
    clinical = pd.DataFrame({"case": cases, "Age": [70, 71, 72]})
    rows = prep.assemble(parts, cases, clinical, str(tmp_path / "slim"), source={"record_id": 1})
    return {"cases": cases, "root": str(root), "parts": parts, "rows": rows, "slim": str(tmp_path / "slim" / prep.SLIM_NAME)}


def test_parts_assemble_into_the_slim_layout(built):
    assert sorted(os.listdir(built["parts"])) == ["part_000.tar", "part_001.tar"]
    assert sorted(os.listdir(os.path.join(built["slim"], built["cases"][0]))) == sorted(prep.SLIM_FILES)
    with open(os.path.join(built["slim"], "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["spacing_mm"] == 1.5 and manifest["cases"] == built["cases"]
    assert manifest["map_background"]["tmax"] == -30.0
    volumes = pd.read_csv(os.path.join(built["slim"], "native_volumes.csv"))
    assert list(volumes["case"]) == built["cases"] and volumes["infarct_ml"][0] == built["rows"][0]["lesion_ml"]
    assert prep.check_slim(built["slim"], built["rows"]) == {}


def test_saved_parts_are_not_built_again(built, capsys):
    capsys.readouterr()
    stamps = [os.path.getmtime(os.path.join(built["parts"], p)) for p in sorted(os.listdir(built["parts"]))]
    prep.build_parts(built["root"], built["cases"], built["parts"], part_size=2)
    assert stamps == [os.path.getmtime(os.path.join(built["parts"], p)) for p in sorted(os.listdir(built["parts"]))]
    assert "already saved" in capsys.readouterr().out


def test_assemble_refuses_missing_cases(built, tmp_path):
    with pytest.raises(ValueError, match="1 cases"):
        prep.assemble(built["parts"], [*built["cases"], "sub-stroke0099"], pd.DataFrame(), str(tmp_path / "other"))


def test_check_slim_finds_problems(built):
    case = built["cases"][1]
    os.remove(os.path.join(built["slim"], case, "cow.nii.gz"))
    img = nib.load(os.path.join(built["slim"], case, "cbf.nii.gz"))
    nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj)[:-1], img.affine), os.path.join(built["slim"], case, "cbf.nii.gz"))
    problems = prep.check_slim(built["slim"], built["rows"])
    assert problems == {"missing cow.nii.gz": 1, "cbf.nii.gz off the case grid": 1}


def test_summary_shows_only_development_cases(built):
    development = built["cases"][:2]
    summary = prep.summarise(built["rows"], development)
    text = json.dumps(summary)
    assert summary["cases"] == 3 and summary["development"]["cases"] == 2
    assert built["cases"][2] not in text and built["cases"][0] in text
    assert summary["development"]["lesion_relative_error_p50_p95"][1] < 0.03


def test_montage_draws_finished_cases(built, tmp_path):
    out = str(tmp_path / "figure.png")
    prep.montage(built["slim"], built["cases"][:2], out)
    assert os.path.getsize(out) > 0


def test_small_irregular_mask_keeps_its_volume(tmp_path):
    """The reason masks are not averaged: on a 1 mL shape, averaging loses more than plain sampling."""
    _case(tmp_path, lesion_radius=6.0)
    row = prep.process_case(str(tmp_path), CASE, str(tmp_path / "out"))
    assert abs(row["lesion_ml_slim"] - row["lesion_ml"]) / row["lesion_ml"] < 0.05


def test_speckled_tmax_keeps_its_volume_above_threshold(tmp_path):
    _case(tmp_path)
    rng = np.random.default_rng(0)
    tmax = np.where(rng.random(SHAPE) < 0.2, 9.0, 3.0).astype(np.float32)
    _save(tmp_path, CASE, "tmax", tmax)
    row = prep.process_case(str(tmp_path), CASE, str(tmp_path / "out"))
    assert abs(row["tmax6_ml_slim"] - row["tmax6_ml"]) < abs(row["tmax6_ml_if_averaged"] - row["tmax6_ml"])
