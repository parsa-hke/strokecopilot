"""Vision A error analysis on synthetic cases. Needs numpy and scipy; no real data.

Step 1 sets up the synthetic case builder. Later steps add tests for each function in
strokecopilot.vision.perfusion.error_analysis as it is filled in.
"""

import io

import pytest

np =pytest.importorskip("numpy")
pytest.importorskip("scipy")

from strokecopilot.data.isles24 import Case  # noqa: E402
from strokecopilot.vision.perfusion import error_analysis  # noqa: E402
from strokecopilot.vision.perfusion.thresholds import reliable  # noqa: E402
from strokecopilot.vision.perfusion.volumes import mask_ml  # noqa: E402

SHAPE = (40, 30, 20)
SPACING = 2.0  # 8 mm3 per voxel, so 125 voxels = 1 mL


def synthetic_case(infarct_box=(slice(24, 34), slice(8, 20), slice(6, 14)), no_data_box=None, occlusion_label=6):
    """A small fake case in the shape isles24.load_case() returns.

    Brain at x 4-35 with Tmax 1 s; a delay (Tmax 10 s) over the infarct box on the right side.
    no_data_box sets Tmax to -30 there (no perfusion data). The occlusion is a few voxels next to
    one Circle of Willis label (6 = right ICA by default).
    """
    cbf = np.zeros(SHAPE, dtype=np.float32)
    tmax = np.full(SHAPE, -30.0, dtype=np.float32)
    brain = (slice(4, 36), slice(4, 26), slice(4, 16))
    cbf[brain], tmax[brain] = 50.0, 1.0
    infarct = np.zeros(SHAPE, dtype=bool)
    if infarct_box is not None:
        infarct[infarct_box] = True
        tmax[infarct_box], cbf[infarct_box] = 10.0, 10.0
    if no_data_box is not None:
        tmax[no_data_box] = -30.0
    cow = np.zeros(SHAPE, dtype=np.int16)
    cow[20:24, 14:16, 8:10] = occlusion_label
    occlusion = np.zeros(SHAPE, dtype=bool)
    occlusion[22:24, 14:16, 10:11] = True
    return Case(
        id="synthetic",
        arrays={"cbf": cbf, "cbv": cbf / 10, "mtt": np.full(SHAPE, 4.0, np.float32), "tmax": tmax,
                "infarct": infarct, "occlusion": occlusion, "cow": cow},
        spacing_mm=SPACING,
        affine=np.diag([SPACING, SPACING, SPACING, 1.0]),
        # Time text as clinical.csv has it: hh:mm:ss, sometimes with microseconds.
        clinical={"case": "synthetic", "Onset to door": "01:30:00", "Door to imaging": "00:30:00.250000"},
        native_volumes={"infarct_ml": mask_ml(infarct, SPACING), "tmax6_ml": mask_ml(tmax > 6, SPACING)},
    )


def test_synthetic_case_volumes():
    case = synthetic_case()
    assert case.native_volumes["infarct_ml"] == pytest.approx(10 * 12 * 8 / 125)
    assert synthetic_case(infarct_box=None).native_volumes["infarct_ml"] == 0.0


def test_synthetic_case_no_data_region():
    box = (slice(24, 29), slice(8, 20), slice(6, 14))  # half the infarct
    case = synthetic_case(no_data_box=box)
    valid = reliable(case.arrays["tmax"], case.arrays["cbf"])
    infarct = case.arrays["infarct"]
    assert np.count_nonzero(infarct & valid) / np.count_nonzero(infarct) == pytest.approx(0.5)


pd = pytest.importorskip("pandas")
from strokecopilot.vision.perfusion import evaluate  # noqa: E402


def _row(case, pred, truth, ref_ml, fold=0):
    """One per_case.csv row, scored the way EXP-1 and EXP-14 score theirs."""
    valid = np.ones(SHAPE, dtype=bool)
    return {"case": case, "fold": fold, **evaluate.case_metrics(pred, truth, valid, SPACING, ref_ml)}


def _box(x0, x1):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[x0:x1, 8:20, 6:14] = True  # 96 voxels per x slice, 0.768 mL
    return mask


def _tables():
    """Three fake cases. a: nnU-Net exact, threshold empty. b: both empty on a real infarct. c: nnU-Net
    far too big (crosses 31, 51 and 70 mL), threshold exact. The threshold table has one extra case."""
    small, big = _box(24, 34), _box(0, 40)   # 7.68 mL and 30.72 mL
    huge = np.ones(SHAPE, dtype=bool)        # 192 mL
    empty = np.zeros(SHAPE, dtype=bool)
    ref = {"a": mask_ml(small, SPACING), "b": mask_ml(small, SPACING), "c": mask_ml(big, SPACING)}
    nnunet = pd.DataFrame([_row("a", small, small, ref["a"]), _row("b", empty, small, ref["b"]), _row("c", huge, big, ref["c"])])
    threshold = pd.DataFrame([_row("a", empty, small, ref["a"]), _row("b", empty, small, ref["b"]),
                              _row("c", big, big, ref["c"]), _row("d", small, small, ref["a"], fold=1)])
    # Through CSV and back, as the tables arrive from Drive.
    return [pd.read_csv(io.StringIO(t.to_csv(index=False))) for t in (nnunet, threshold)]


def test_join_tables_side_by_side():
    joined = error_analysis.join_tables(*_tables()).set_index("case")
    assert list(joined.index) == ["a", "b", "c"]  # only nnU-Net's cases
    assert joined.loc["a", "nnunet_dice"] == 1.0 and joined.loc["a", "threshold_dice"] == 0.0
    assert joined.loc["a", "dice_gap"] == 1.0
    assert joined.loc["a", "better_volume"] == "nnunet"
    assert joined.loc["c", "better_volume"] == "threshold"
    assert joined.loc["b", "better_volume"] == "tie"


def test_join_tables_misses_and_crossings():
    joined = error_analysis.join_tables(*_tables()).set_index("case")
    assert joined["miss"].to_dict() == {"a": "threshold", "b": "both", "c": "neither"}
    # c: nnU-Net predicts 192 mL against 30.72; it crosses 31, 51 and 70 mL, the exact threshold none.
    assert joined.loc["c", "crossings_disagree"] == 3
    assert joined.loc["b", "crossings_disagree"] == 0


def test_join_tables_refuses_bad_tables():
    nnunet, threshold = _tables()
    with pytest.raises(ValueError, match="missing from the threshold"):
        error_analysis.join_tables(nnunet, threshold[threshold["case"] != "a"])
    with pytest.raises(ValueError, match="repeated"):
        error_analysis.join_tables(pd.concat([nnunet, nnunet.iloc[:1]]), threshold)
    shifted = threshold.copy()
    shifted.loc[shifted["case"] == "a", "ref_ml"] += 1.0
    with pytest.raises(ValueError, match="reference volumes"):
        error_analysis.join_tables(nnunet, shifted)
    with pytest.raises(ValueError, match="lacks columns"):
        error_analysis.join_tables(nnunet.drop(columns=["abs_error_ml"]), threshold)


def test_write_config(tmp_path):
    run_dir = tmp_path / error_analysis.RUN_NAME
    config = error_analysis.write_config(run_dir, "abc1234")
    saved = (run_dir / "config.json").read_text()
    assert config["experiment"] == "EXP-15" and config["git_commit"] == "abc1234"
    assert config["inputs"]["nnunet"] == "runs/EXP-14_nnunet_fold0/per_case.csv"
    assert '"EXP-15"' in saved
    with pytest.raises(FileExistsError):
        error_analysis.write_config(run_dir, "def5678")
    with pytest.raises(ValueError):
        error_analysis.write_config(tmp_path / "other", "")


def test_case_features_default_case():
    f = error_analysis.case_features(synthetic_case())
    assert f["case"] == "synthetic"
    assert f["infarct_ml"] == pytest.approx(7.68) and f["size_band"] == "5-21"
    # The occlusion sits 2 mm from label 6, the left ICA: within the 5 mm limit.
    assert (f["occlusion_site"], f["occlusion_reason"]) == ("ICA", "ok")
    assert f["occlusion_nearest_label"] == 6 and f["occlusion_distance_mm"] == pytest.approx(2.0)
    assert f["territory"] == "anterior"
    assert f["onset_to_imaging_min"] == pytest.approx(120.0)
    assert f["infarct_tmax_missing_share"] == 0.0 and f["infarct_tmax_unreliable_share"] == 0.0


@pytest.mark.parametrize("label, site, territory", [
    (1, "basilar", "posterior"),
    (5, "unknown", "anterior"),   # MCA: M1 and M2 can't be told apart
    (8, "unknown", "unknown"),    # posterior communicating artery joins both
])
def test_case_features_territory(label, site, territory):
    f = error_analysis.case_features(synthetic_case(occlusion_label=label))
    assert (f["occlusion_site"], f["territory"]) == (site, territory)


def test_case_features_tmax_no_data():
    half = (slice(24, 29), slice(8, 20), slice(6, 14))
    case = synthetic_case(no_data_box=half)
    case.arrays["tmax"][29:34, 8:20, 6:8] = -5.0  # failed fit: not -30, but still no reliable data
    f = error_analysis.case_features(case)
    assert f["infarct_tmax_missing_share"] == pytest.approx(0.5)
    assert f["infarct_tmax_unreliable_share"] == pytest.approx(0.5 + 0.5 * 2 / 8)


def test_case_features_missing_inputs():
    case = synthetic_case(infarct_box=None)
    case.clinical["Onset to door"] = None  # unknown onset, as in wake-up strokes
    case.arrays["occlusion"][:] = False
    f = error_analysis.case_features(case)
    assert f["size_band"] == "none" and f["infarct_tmax_missing_share"] is None
    assert f["onset_to_imaging_min"] is None
    assert (f["occlusion_site"], f["occlusion_reason"], f["territory"]) == ("unknown", "empty_mask", "unknown")


@pytest.mark.parametrize("ml, band", [(None, None), (0.0, "none"), (4.9, "<5"), (5.0, "5-21"), (20.99, "5-21"),
                                      (21.0, "21-70"), (70.0, ">=70"), (101.0, ">=70")])
def test_size_band(ml, band):
    assert error_analysis.size_band(ml) == band


def test_later_steps_not_built_yet():
    with pytest.raises(NotImplementedError):
        error_analysis.group_summary(None, "size_band")
