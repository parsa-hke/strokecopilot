"""Threshold baseline and Vision A metrics on synthetic perfusion maps. Needs numpy and scipy."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from strokecopilot.data.isles24 import Case  # noqa: E402
from strokecopilot.vision.perfusion import evaluate, thresholds  # noqa: E402

SHAPE = (40, 30, 20)
SPACING = 1.5


def _maps():
    """Brain at x 4-35. The right side (high x) has a delay with a low-flow centre."""
    cbf = np.zeros(SHAPE, dtype=np.float32)
    tmax = np.full(SHAPE, -30.0, dtype=np.float32)
    cbf[4:36, 4:26, 4:16] = 50.0
    tmax[4:36, 4:26, 4:16] = 1.0
    tmax[24:34, 8:20, 6:14] = 10.0   # Tmax > 6 s: 10 x 12 x 8 voxels
    cbf[26:30, 10:14, 8:12] = 10.0   # rCBF 20%: 4 x 4 x 4 voxels, inside the delay
    cbf[6:10, 10:14, 8:12] = 10.0    # low flow on the healthy side, no delay
    return cbf, tmax


def test_core_and_penumbra():
    cbf, tmax = _maps()
    masks = thresholds.threshold_masks(cbf, tmax)
    assert masks["affected_side"] == "right" and masks["reference_cbf"] == 50.0
    assert np.count_nonzero(masks["penumbra"]) == 10 * 12 * 8
    assert np.count_nonzero(masks["core"]) == 64  # healthy-side low flow is outside Tmax > 6 s
    unrestricted = thresholds.threshold_masks(cbf, tmax, core_within_penumbra=False)
    assert np.count_nonzero(unrestricted["core"]) == 128


def test_negative_tmax_is_not_reliable():
    cbf, tmax = _maps()
    tmax[26:30, 10:14, 8:12] = -5.0  # failed fit over the core
    masks = thresholds.threshold_masks(cbf, tmax)
    assert np.count_nonzero(masks["core"]) == 0
    assert not masks["valid"][27, 11, 9]


def test_midline_from_ncct_and_no_reference():
    cbf, tmax = _maps()
    ncct = np.full(SHAPE, -1000.0, dtype=np.float32)
    ncct[4:36, 4:26, 4:16] = 35.0
    tmax[4:20] = -30.0  # the healthy side has no perfusion data
    masks = thresholds.threshold_masks(cbf, tmax, ncct)
    assert masks["midline_x"] == pytest.approx(19.5)  # from NCCT, not shifted by the missing data
    assert masks["reference_cbf"] is None and masks["core"] is None
    assert np.count_nonzero(masks["penumbra"]) == 960
    cbf[:] = 0.0
    assert thresholds.threshold_masks(cbf, _maps()[1])["core"] is None  # reference not positive


def test_dice_and_crossings():
    a = np.zeros(SHAPE, bool)
    a[0:10] = True
    b = np.zeros(SHAPE, bool)
    b[5:15] = True
    assert evaluate.dice(a, b) == pytest.approx(0.5)
    assert evaluate.dice(np.zeros(SHAPE, bool), np.zeros(SHAPE, bool)) is None
    assert evaluate.dice(a, np.zeros(SHAPE, bool)) == 0.0
    assert evaluate.crosses(69.9, 70.0, 70.0) and not evaluate.crosses(10.0, 20.0, 21.0)


def test_baseline_row_and_summary():
    cbf, tmax = _maps()
    infarct = np.zeros(SHAPE, bool)
    infarct[26:30, 10:14, 8:12] = True
    occlusion = np.zeros(SHAPE, bool)
    occlusion[30, 15, 10] = True
    cow = np.zeros(SHAPE, np.uint8)
    cow[31:33, 15, 10] = 4
    arrays = {"cbf": cbf, "tmax": tmax, "infarct": infarct, "occlusion": occlusion, "cow": cow}
    case = Case(id="sub-stroke9999", arrays=arrays, spacing_mm=SPACING, affine=np.eye(4), clinical={},
                native_volumes={"infarct_ml": 0.2, "tmax6_ml": 3.0})
    row = thresholds.baseline_row(case, fold=0)
    assert row["dice"] == 1.0 and row["dice_reliable"] == 1.0
    assert row["pred_ml"] == pytest.approx(64 * SPACING**3 / 1000)
    assert row["contract_core_ml"] == 0.2 and row["contract_penumbra_ml"] == 3.2
    assert row["penumbra_error_ml"] == pytest.approx(960 * SPACING**3 / 1000 - 3.0)
    assert row["occlusion_site"] == "ICA" and row["infarct_reliable_share"] == 1.0
    rows = [row, {**row, "fold": 1, "dice": None, "dice_reliable": None}]
    summary = evaluate.summarise(rows)
    assert summary["cases"] == 2 and summary["cases_without_dice"] == 1
    assert summary["mean_std_over_folds"]["dice"]["n"] == 1
    assert summary["threshold_crossings"]["70"] == {"cases": 0, "of": 2}


def test_montage_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    cbf, tmax = _maps()
    infarct = np.zeros(SHAPE, bool)
    infarct[26:30, 10:14, 8:12] = True
    arrays = {"cbf": cbf, "tmax": tmax, "ncct": np.full(SHAPE, 35.0), "infarct": infarct,
              "occlusion": infarct, "cow": np.zeros(SHAPE, np.uint8)}
    case = Case(id="sub-stroke9999", arrays=arrays, spacing_mm=SPACING, affine=np.eye(4), clinical={})
    out = tmp_path / "m.png"
    thresholds.montage([case], out)
    assert out.stat().st_size > 1000


def test_diagnose_case():
    cbf, tmax = _maps()
    infarct = np.zeros(SHAPE, bool)
    infarct[26:30, 10:14, 8:12] = True
    arrays = {"cbf": cbf, "tmax": tmax, "ncct": np.where(cbf > 0, 35.0, -1000.0), "infarct": infarct}
    case = Case(id="sub-stroke9999", arrays=arrays, spacing_mm=SPACING, affine=np.eye(4), clinical={})
    row = thresholds.diagnose_case(case)
    assert row["infarct_in_tmax6_share"] == 1.0 and row["infarct_on_affected_side_share"] == 1.0
    assert row["infarct_rcbf_p50"] == pytest.approx(0.2)
    assert row["dice_rcbf0.3"] == 1.0 and row["dice_rcbf0.2"] == 0.0  # strict <: nothing below 20%
    assert row["tmax6_dice"] == pytest.approx(2 * 64 / (960 + 64))


def test_paired_comparison_and_loader(tmp_path):
    model = [{"case": f"c{i}", "fold": 0, "dice": 0.5, "abs_error_ml": 5.0, "crosses_21": i == 0, "crosses_31": False,
              "crosses_51": False, "crosses_70": False} for i in range(6)]
    base = [{**row, "dice": 0.2, "abs_error_ml": 10.0, "crosses_21": i in (1, 2)} for i, row in enumerate(model)]
    base[5]["dice"] = None
    out = evaluate.paired_comparison(model, base + [{**base[0], "case": "other"}], n_boot=200)
    assert out["cases"] == 6
    d = out["metrics"]["dice"]
    assert d["n"] == 5 and d["model_better"] == 5 and d["mean_difference"] == pytest.approx(0.3)
    assert d["ci95"][0] == pytest.approx(0.3) and d["ci95"][1] == pytest.approx(0.3)
    a = out["metrics"]["abs_error_ml"]
    assert a["n"] == 6 and a["model_better"] == 6 and a["mean_difference"] == pytest.approx(-5.0)
    c = out["crossings"]["21"]
    assert (c["model"], c["baseline"], c["only_model"], c["only_baseline"], c["both"]) == (1, 2, 1, 2, 0)
    assert c["mcnemar_p"] == pytest.approx(1.0) and out["crossings"]["31"]["mcnemar_p"] is None
    path = tmp_path / "per_case.csv"
    path.write_text("variant,case,fold,dice,abs_error_ml,crosses_21\nx,c0,1,0.5,2.0,True\ny,c0,1,,3.0,False\n")
    rows = evaluate.load_per_case(path, variant="y")
    assert rows == [{"variant": "y", "case": "c0", "fold": 1, "dice": None, "abs_error_ml": 3.0, "crosses_21": False}]
