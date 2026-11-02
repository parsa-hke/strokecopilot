"""ISLES'24 layout and survey checks on small synthetic NIfTI files. Needs the [isles] extra."""

import json
import os

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")
pytest.importorskip("scipy")

from strokecopilot.data import isles24_layout as layout  # noqa: E402
from strokecopilot.data import isles24_survey as survey  # noqa: E402
from strokecopilot.data import splits  # noqa: E402

SHAPE = (40, 40, 20)
ZOOMS = (1.0, 1.0, 2.0)
AFFINE = np.diag([*ZOOMS, 1.0])
CASES = [f"sub-stroke{i:04d}" for i in (1, 2, 5, 9, 14, 20)]


def _save(root, case, name, array, affine=AFFINE):
    path = layout.path_of(str(root), case, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), path)


def _case(root, case, i):
    ncct = np.full(SHAPE, -1000, dtype=np.int16)
    ncct[5:35, 5:35, 2:18] = 35
    _save(root, case, "ncct", ncct)
    for name in ("cbf", "cbv", "mtt", "dwi"):
        _save(root, case, name, np.full(SHAPE, 10.0, dtype=np.float32))
    tmax = np.zeros(SHAPE, dtype=np.float32)
    tmax[10:20, 10:20, 5:10] = 8.0
    _save(root, case, "tmax", tmax)
    lesion = np.zeros(SHAPE, dtype=np.uint8)
    lesion[10:10 + i + 2, 10:14, 6:9] = 1
    _save(root, case, "lesion", lesion)
    cow = np.zeros(SHAPE, dtype=np.uint8)
    cow[20:25, 20:22, 9:11] = 5
    cow[12:15, 20:22, 9:11] = 7
    _save(root, case, "cow", cow)
    lvo = np.zeros(SHAPE, dtype=np.uint8)
    lvo[26:28, 20:22, 9:11] = 1  # just beyond label 5
    _save(root, case, "lvo", lvo)
    for name, row in (("baseline", {"Center": 1 + i % 2, "Age": 70}), ("outcome", {"mRS 24h": 3})):
        path = layout.path_of(str(root), case, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame([row]).to_excel(path, index=False)


@pytest.fixture
def root(tmp_path):
    for i, case in enumerate(CASES):
        _case(tmp_path, case, i)
    return str(tmp_path)


def test_layout_finds_cases_and_missing_files(root):
    assert layout.find_cases(root) == CASES
    assert layout.missing_files(root, CASES) == {}
    os.remove(layout.path_of(root, CASES[0], "tmax"))
    assert layout.missing_files(root, CASES) == {"tmax": 1}


def test_pack_and_unpack(tmp_path):
    parent = tmp_path / "a"
    (parent / "train").mkdir(parents=True)
    (parent / "train" / "f.txt").write_text("x")
    layout.pack(str(parent), str(tmp_path / "w.tar"))
    dest = tmp_path / "b"
    dest.mkdir()
    layout.unpack(str(tmp_path / "w.tar"), str(dest))
    assert (dest / "train" / "f.txt").read_text() == "x"


def test_volumes_use_real_spacing(root):
    volumes, labels = survey.infarct_volumes(root, CASES)
    assert volumes[CASES[0]] == pytest.approx(2 * 4 * 3 * 2 / 1000)
    assert set(labels) == {"0", "1"}


def test_centres_and_extra_rows():
    tables = {
        "baseline": {
            "a": {"Sheet1": pd.DataFrame({"Center": [1], "Age": [70], "Sex": ["F"]})},
            "b": {"Sheet1": pd.DataFrame({"Center": [2, 2], "Age": [60, 61], "Sex": ["M", None]})},
        }
    }
    assert survey.centres_of(tables) == {"a": 1, "b": 2}
    [found] = survey.extra_rows(tables)
    assert found["case"] == "b" and found["non_null_per_row"] == [3, 2]
    assert found["sparser_row_fills"] == ["Center", "Age"]
    assert found["sparser_row_conflicts_with_main_on"] == ["Age"]


def test_stray_mask_check(root):
    case = CASES[3]
    stray = np.zeros(SHAPE, dtype=np.uint8)
    stray[10:15, 10:14, 6:9] = 1
    path = os.path.join(root, layout.STRAY_LESION.format(c=case))
    nib.save(nib.Nifti1Image(stray, AFFINE), path)
    result = survey.stray_mask_check(root, case)
    assert result["same_grid"] is True
    assert result["stray_ml"] == result["normal_ml"]
    assert result["dice_voxelwise"] == 1.0


def test_header_survey_names_only_development_cases(root):
    off = np.zeros((30, 40, 20), dtype=np.float32)
    _save(root, CASES[0], "cbf", off)
    _save(root, CASES[1], "cbv", off)
    report = survey.header_survey(root, CASES, printable=[CASES[1]])
    assert report["cbf"]["cases_off_ncct_grid"] == 1
    assert report["cbf"]["development_cases_off_ncct_grid"] == []
    assert report["cbv"]["development_cases_off_ncct_grid"] == [CASES[1]]
    assert report["ncct"]["spacing_median"] == list(ZOOMS)
    assert CASES[0] not in json.dumps(report)


def test_lesion_in_tissue_detects_misalignment(root):
    assert survey.lesion_in_tissue(root, CASES)["p5_p50_p95"][1] == 1.0
    lesion = np.zeros(SHAPE, dtype=np.uint8)
    lesion[0:3, 0:3, 0:3] = 1  # in air
    _save(root, CASES[0], "lesion", lesion)
    report = survey.lesion_in_tissue(root, CASES)
    assert report["lowest"][CASES[0]] == 0.0


def test_cow_lvo_survey_finds_nearest_label(root):
    report = survey.cow_lvo_survey(root, CASES)
    assert report["cases_with_each_label"] == {"0": 6, "5": 6, "7": 6}
    assert report["lvo_dominant_or_nearest_label"] == {"5": 6}
    assert report["lvo_share_outside_labels_p5_p50_p95"][1] == 1.0
    assert report["lvo_median_distance_to_labels_mm_p5_p50_p95"][1] > 0
    assert report["empty_lvo_masks"] == 0


def test_cow_lvo_survey_counts_empty_masks(root):
    _save(root, CASES[0], "lvo", np.zeros(SHAPE, dtype=np.uint8))
    assert survey.cow_lvo_survey(root, CASES)["empty_lvo_masks"] == 1


def test_value_survey_counts_non_finite(root):
    tmax = np.ones(SHAPE, dtype=np.float32)
    tmax[0, 0, 0] = np.nan
    _save(root, CASES[0], "tmax", tmax)
    report = survey.value_survey(root, CASES[:1])
    assert report["tmax"]["non_finite_voxels"] == 1
    assert report["ncct"]["non_finite_voxels"] == 0


def test_resample_keeps_physical_size():
    array = np.zeros((40, 40, 20), dtype=np.float32)
    out = survey.resample(array, ZOOMS, 2.0, order=1)
    assert out.shape == (20, 20, 20)


def test_grid_candidates_report_errors_and_sizes(root):
    report = survey.grid_candidates(root, CASES[:2], [1.0, 2.0], n_all_cases=149)
    assert report["cases_sampled"] == 2
    fine, coarse = report["spacings"]["1.0"], report["spacings"]["2.0"]
    assert np.prod(fine["cropped_shape_p50"]) > np.prod(coarse["cropped_shape_p50"])
    assert fine["projected_slim_gib_all_cases"] >= coarse["projected_slim_gib_all_cases"]
    assert fine["lesions_lost"] == 0
    assert fine["tmax6_abs_error_ml_p50_p95"][0] == pytest.approx(0, abs=0.2)
    assert len(report["brain_box_mm_p50"]) == 3
    json.dumps(report)


def test_sample_by_volume_spreads_and_skips_zero():
    volumes = {f"c{i}": float(i) for i in range(10)}
    assert survey.sample_by_volume(volumes, list(volumes), 3) == ["c1", "c5", "c9"]
    assert "c0" not in survey.sample_by_volume(volumes, list(volumes), 20)


def test_split_balance_has_no_ids():
    volumes = {f"sub-stroke{i:04d}": float(i) for i in range(1, 41)}
    centres = {c: 1 + i % 2 for i, c in enumerate(volumes)}
    folds = splits.make_splits(volumes, centres, n_heldout=5)
    balance = survey.split_balance(folds, volumes, centres)
    assert balance["heldout"]["cases"] == 5
    assert "sub-stroke" not in json.dumps(balance)


def test_montage_writes_a_png(root, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "m.png"
    survey.montage(root, CASES[:2], str(out))
    assert out.stat().st_size > 1000
