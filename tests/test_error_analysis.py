"""Vision A error analysis on synthetic cases. Needs numpy and scipy; no real data.

Step 1 sets up the synthetic case builder. Later steps add tests for each function in
strokecopilot.vision.perfusion.error_analysis as it is filled in.
"""

import io
import json

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


def test_join_tables_keeps_one_threshold_variant():
    """EXP-1 has a row per case for each core variant; only within_tmax6 is joined."""
    nnunet, within = _tables()
    unrestricted = within.copy()
    unrestricted["dice"] = 0.5  # different scores, same cases and reference volumes
    threshold = pd.concat([within.assign(variant="within_tmax6"), unrestricted.assign(variant="unrestricted")])
    joined = error_analysis.join_tables(nnunet, threshold).set_index("case")
    assert len(joined) == 3
    assert joined.loc["a", "threshold_dice"] == 0.0  # the within_tmax6 row, not 0.5
    other = error_analysis.join_tables(nnunet, threshold, variant="unrestricted").set_index("case")
    assert (other["threshold_dice"] == 0.5).all()
    with pytest.raises(ValueError, match="no 'missing' rows"):
        error_analysis.join_tables(nnunet, threshold, variant="missing")
    with pytest.raises(ValueError, match="repeated"):
        error_analysis.join_tables(nnunet, threshold.drop(columns=["variant"]))


def test_write_config(tmp_path):
    run_dir = tmp_path / error_analysis.RUN_NAME
    config = error_analysis.write_config(run_dir, "abc1234")
    saved = (run_dir / "config.json").read_text()
    assert config["experiment"] == "EXP-15" and config["git_commit"] == "abc1234"
    assert config["inputs"]["nnunet"] == "runs/EXP-14_nnunet_fold0/per_case.csv"
    assert config["inputs"]["threshold_variant"] == "within_tmax6"
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


def test_notebook_table_merge():
    """What notebook 12 does: join the two tables, then add case_features for each case."""
    import dataclasses

    joined = error_analysis.join_tables(*_tables())
    rows = [error_analysis.case_features(dataclasses.replace(synthetic_case(), id=case)) for case in joined["case"]]
    table = joined.merge(pd.DataFrame(rows), on="case", how="left", validate="one_to_one")
    assert len(table) == 3 and table["territory"].notna().all()
    assert {"nnunet_dice", "threshold_abs_error_ml", "size_band", "infarct_tmax_missing_share"} <= set(table.columns)


@pytest.mark.parametrize("ml, band", [(None, None), (0.0, "none"), (4.9, "<5"), (5.0, "5-21"), (20.99, "5-21"),
                                      (21.0, "21-70"), (70.0, ">=70"), (101.0, ">=70")])
def test_size_band(ml, band):
    assert error_analysis.size_band(ml) == band


def _step4_table():
    """A step-4 table by hand: six cases with known scores and features."""
    return pd.DataFrame({
        "case": ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"],
        "ref_ml": [2.0, 3.0, 101.0, 40.0, 0.0, 120.0],
        "nnunet_dice": [0.0, 0.0, 0.0, 0.5, np.nan, 0.7],
        "nnunet_pred_ml": [0.0, 1.5, 0.0, 30.0, 0.0, 110.0],
        "nnunet_abs_error_ml": [2.0, 1.5, 101.0, 10.0, 0.0, 10.0],
        "threshold_dice": [0.0, 0.2, 0.0, 0.3, np.nan, 0.6],
        "threshold_pred_ml": [0.0, 4.0, 5.0, 20.0, 0.0, 90.0],
        "miss": ["both", "nnunet", "both", "neither", "neither", "neither"],
        "better_volume": ["tie", "nnunet", "threshold", "nnunet", "tie", "nnunet"],
        "size_band": ["<5", "<5", ">=70", "21-70", "none", ">=70"],
        "territory": ["anterior", "posterior", "anterior", "anterior", "unknown", "anterior"],
        "occlusion_site": ["unknown", "basilar", "ICA", "unknown", "unknown", "ICA"],
        "occlusion_distance_mm": [3.0, 20.0, 1.0, 2.0, np.nan, 0.0],
        "onset_to_imaging_min": [120.0, np.nan, 300.0, np.nan, np.nan, 90.0],
        "infarct_tmax_missing_share": [0.0, 0.1, 0.6, 0.02, np.nan, 0.0],
        "infarct_tmax_unreliable_share": [0.0, 0.1, 0.7, 0.02, np.nan, 0.0],
        "infarct_reliable_share": [1.0, 0.9, 0.3, 0.98, np.nan, 1.0],
    })


@pytest.mark.parametrize("share, band", [(None, "no infarct"), (float("nan"), "no infarct"), (0.0, "none"),
                                         (0.01, "<5%"), (0.05, "5%-25%"), (0.25, ">=25%"), (1.0, ">=25%")])
def test_tmax_missing_band(share, band):
    assert error_analysis.tmax_missing_band(share) == band


def test_group_summary_by_size():
    rows = {r["group"]: r for r in error_analysis.group_summary(_step4_table(), "size_band")}
    assert set(rows) == {"<5", ">=70", "21-70", "none"}
    small = rows["<5"]
    assert small["cases"] == 2 and small["dice0"] == 2
    assert small["missed_nnunet_only"] == 1 and small["missed_both"] == 1
    assert small["nnunet_dice_mean"] == 0.0 and small["nnunet_abs_error_ml_mean"] == 1.8  # (2.0 + 1.5) / 2, 1 decimal
    big = rows[">=70"]
    assert big["cases"] == 2 and big["nnunet_dice_mean"] == 0.35 and big["nnunet_dice_median"] == 0.35
    assert rows["none"]["nnunet_dice_mean"] is None  # no Dice when both masks are empty


def test_summarise_features_derived_groups():
    groups = error_analysis.summarise_features(_step4_table())
    assert set(groups) == set(error_analysis.GROUP_FEATURES)
    onset = {r["group"]: r["cases"] for r in groups["onset_known"]}
    assert onset == {"known": 3, "unknown": 3}
    tmax = {r["group"]: r["cases"] for r in groups["tmax_missing_band"]}
    assert tmax == {"none": 2, "5%-25%": 1, ">=25%": 1, "<5%": 1, "no infarct": 1}
    for feature in groups.values():  # every case is in exactly one group
        assert sum(r["cases"] for r in feature) == 6


def test_miss_profile():
    profile = error_analysis.miss_profile(_step4_table())
    assert profile["dice0"] == 3 and profile["threshold_also_dice0"] == 2
    assert profile["nnunet_pred_empty"] == 2
    assert profile["ref_ml"] == {"min": 2.0, "median": 3.0, "max": 101.0}
    assert profile["groups"]["size_band"] == {"<5": 2, ">=70": 1}
    largest = profile["largest_miss"]
    assert largest["ref_ml"] == 101.0 and largest["nnunet_pred_ml"] == 0.0
    assert largest["occlusion_site"] == "ICA" and largest["tmax_missing_band"] == ">=25%"
    assert largest["onset_to_imaging_min"] == 300.0
    assert "case" not in largest


def test_build_and_write_metrics(tmp_path):
    table = _step4_table()
    config = {"experiment": "EXP-15", "git_commit": "abc1234", "folds": [0], "inputs": {"threshold_variant": "within_tmax6"}}
    metrics = error_analysis.build_metrics(table, config, "def5678")
    # The commit this run installed, and config.json's own, kept from an earlier run.
    assert metrics["git_commit"] == "def5678" and metrics["config_git_commit"] == "abc1234"
    with pytest.raises(ValueError, match="git_commit"):
        error_analysis.build_metrics(table, config, "")
    assert metrics["overall"]["cases"] == 6 and metrics["overall"]["dice0"] == 3
    assert metrics["who_misses"] == {"both": 2, "neither": 3, "nnunet": 1}
    assert metrics["threshold_variant"] == "within_tmax6"
    path = error_analysis.write_metrics(tmp_path, metrics)
    saved = path.read_text()
    assert "sub-0" not in saved
    assert json.loads(saved)["dice0_misses"]["largest_miss"]["ref_ml"] == 101.0


def test_build_metrics_refuses_case_ids():
    table = _step4_table()
    table.loc[0, "territory"] = "sub-03"  # an ID where a group label should be
    with pytest.raises(ValueError, match="case IDs"):
        error_analysis.build_metrics(table, {}, "abc1234")
