"""ISLES'24 slim reader on a small synthetic tarball. Needs the [isles] extra."""

import json
import os
import subprocess

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from strokecopilot.data import clinical_columns as cc  # noqa: E402
from strokecopilot.data import isles24, splits  # noqa: E402
from strokecopilot.data import isles24_layout as layout  # noqa: E402
from strokecopilot.data import prep_isles24 as prep  # noqa: E402

SHAPE = (60, 70, 24)
ZOOMS = (0.5, 0.5, 2.0)
AFFINE = np.diag([*ZOOMS, 1.0])
CASES = ["sub-stroke0001", "sub-stroke0002", "sub-stroke0003"]
FOLDS = {"heldout": ["sub-stroke0003"], "folds": {"0": ["sub-stroke0001"], "1": ["sub-stroke0002"]}}


def _save(root, case, name, array):
    path = layout.path_of(str(root), case, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(array, AFFINE), path)


def _source_case(root, case):
    ncct = np.full(SHAPE, -1000, dtype=np.float32)
    ncct[6:54, 6:64, 3:21] = 35
    _save(root, case, "ncct", ncct)
    for name in ("cbf", "cbv", "mtt"):
        _save(root, case, name, np.where(ncct > 0, 20.0, 0.0).astype(np.float32))
    tmax = np.full(SHAPE, -30.0, dtype=np.float32)
    tmax[6:54, 6:64, 3:15] = 8.0
    _save(root, case, "tmax", tmax)
    lesion = np.zeros(SHAPE)
    lesion[20:40, 20:40, 6:12] = 1
    _save(root, case, "lesion", lesion)
    occlusion = np.zeros(SHAPE, dtype=np.uint8)
    occlusion[30:36, 30:36, 8:10] = 1
    _save(root, case, "lvo", occlusion)
    cow = np.zeros(SHAPE, dtype=np.uint8)
    cow[20:30, 30:36, 8:10] = 5
    _save(root, case, "cow", cow)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    base = tmp_path_factory.mktemp("slim")
    source = base / "train"
    for case in CASES:
        _source_case(source, case)
    prep.build_parts(str(source), CASES, str(base / "parts"))
    row = {column: 1 for column in (*cc.BASELINE_COLUMNS, *cc.OUTCOME_COLUMNS)}
    row["INR"] = None
    row["Onset to door"] = "01:30:00"
    clinical = pd.DataFrame([{"case": case, **row} for case in CASES])
    prep.assemble(str(base / "parts"), CASES, clinical, str(base / "out"))
    return base / "out" / prep.SLIM_NAME


def test_list_cases_leaves_out_held_out(root):
    assert isles24.list_cases(root, folds=FOLDS) == CASES[:2]


def test_list_cases_includes_held_out_when_allowed(root, monkeypatch):
    monkeypatch.setenv("ALLOW_HELDOUT", "1")
    assert isles24.list_cases(root, folds=FOLDS) == CASES


def test_load_case_returns_arrays_clinical_spacing_and_volumes(root):
    case = isles24.load_case(CASES[0], root, folds=FOLDS)
    assert set(case.arrays) == set(isles24.FILES)
    assert case.spacing_mm == 1.5
    assert case.arrays["infarct"].dtype == bool and case.arrays["occlusion"].dtype == bool
    assert case.arrays["tmax"].dtype == np.float32 and case.arrays["ncct"].dtype == np.int16
    assert len({a.shape for a in case.arrays.values()}) == 1
    assert nib.aff2axcodes(case.affine) == ("R", "A", "S")
    assert case.clinical["case"] == CASES[0] and case.clinical["INR"] is None
    assert case.native_volumes["infarct_ml"] == pytest.approx(20 * 20 * 6 * 0.5 * 1e-3)


def test_load_case_can_read_a_subset(root):
    case = isles24.load_case(CASES[0], root, names=("tmax",), folds=FOLDS)
    assert set(case.arrays) == {"tmax"}


def test_held_out_case_is_refused(root):
    with pytest.raises(splits.HeldoutError):
        isles24.load_case(CASES[2], root, folds=FOLDS)
    with pytest.raises(splits.HeldoutError):
        isles24.load_native_ncct(CASES[2], root, folds=FOLDS)


def test_unknown_case_is_refused(root):
    with pytest.raises(KeyError, match="not in the tarball"):
        isles24.load_case("sub-stroke9999", root, folds=FOLDS)


def test_note_fields_leave_out_forbidden_columns(root):
    fields = isles24.load_case(CASES[0], root, folds=FOLDS).note_fields
    assert "Age" in fields and "Onset to door" in fields
    assert not set(fields) & set(cc.FORBIDDEN_FOR_NOTES) and "case" not in fields


def test_native_ncct_is_the_source_grid(root):
    assert isles24.load_native_ncct(CASES[0], root, folds=FOLDS).shape == SHAPE


def test_fetch_slim_unpacks_once(root, tmp_path):
    tar = tmp_path / "isles24_slim_v1.tar"
    subprocess.run(["tar", "-cf", str(tar), "-C", str(root.parent), prep.SLIM_NAME], check=True)
    parent = tmp_path / "local"
    out = isles24.fetch_slim(str(tar), parent=str(parent))
    assert json.load(open(os.path.join(out, "manifest.json")))["cases"] == CASES
    assert not (parent / "isles24_slim_v1.tar").exists()
    tar.unlink()
    assert isles24.fetch_slim(str(tar), parent=str(parent)) == out
