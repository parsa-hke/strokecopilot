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


def _fold_tables():
    """Three fake fold tables (folds 0, 1 and 2, two cases each), as the five nnU-Net runs would give, and a
    threshold table over all six cases with both variants, as EXP-1 gives."""
    small, big, empty = _box(24, 34), _box(0, 40), np.zeros(SHAPE, dtype=bool)
    ref_small, ref_big = mask_ml(small, SPACING), mask_ml(big, SPACING)
    cases = {  # case: (fold, nnU-Net prediction, truth, reference mL)
        "f0a": (0, small, small, ref_small), "f0b": (0, empty, small, ref_small),
        "f1a": (1, big, big, ref_big), "f1b": (1, small, big, ref_big),
        "f2a": (2, big, small, ref_small), "f2b": (2, small, small, ref_small),
    }
    nnunet = [pd.DataFrame([_row(c, p, t, r, fold=f) for c, (f, p, t, r) in cases.items() if f == fold]) for fold in (0, 1, 2)]
    within = [_row(c, empty, t, r, fold=f) for c, (f, p, t, r) in cases.items()]
    threshold = pd.concat([pd.DataFrame(within).assign(variant="within_tmax6"),
                           pd.DataFrame(within).assign(variant="unrestricted", dice=0.5)], ignore_index=True)
    # Through CSV and back, as the tables arrive from Drive.
    return ([pd.read_csv(io.StringIO(t.to_csv(index=False))) for t in nnunet],
            pd.read_csv(io.StringIO(threshold.to_csv(index=False))))


def test_five_fold_table_joins_every_fold():
    nnunet, threshold = _fold_tables()
    table = error_analysis.five_fold_table(nnunet, threshold, expected_cases=6).set_index("case")
    assert len(table) == 6
    assert table["fold"].value_counts().sort_index().to_dict() == {0: 2, 1: 2, 2: 2}
    assert table.loc["f0a", "nnunet_dice"] == 1.0 and table.loc["f0b", "nnunet_dice"] == 0.0
    assert (table["threshold_dice"] == 0.0).all()  # the within_tmax6 rows, not the unrestricted 0.5
    # f2a: nnU-Net predicts 30.72 mL for a 7.68 mL infarct, crossing 21 mL (an over-call, counted in step 2).
    assert bool(table.loc["f2a", "nnunet_crosses_21"]) and not bool(table.loc["f2a", "nnunet_crosses_31"])
    assert table.loc["f0b", "miss"] == "both"


def test_five_fold_table_refuses_bad_inputs():
    nnunet, threshold = _fold_tables()
    two_folds = pd.concat([nnunet[0], nnunet[1]], ignore_index=True)
    with pytest.raises(ValueError, match="holds 2 folds"):
        error_analysis.five_fold_table([two_folds, nnunet[2]], threshold)
    with pytest.raises(ValueError, match="Fold 0 appears in two"):
        error_analysis.five_fold_table([nnunet[0], nnunet[0].assign(case=["x", "y"])], threshold)
    repeated = nnunet[1].copy()
    repeated.loc[0, "case"] = "f0a"
    with pytest.raises(ValueError, match="more than one fold table"):
        error_analysis.five_fold_table([nnunet[0], repeated, nnunet[2]], threshold)
    with pytest.raises(ValueError, match="not 129"):
        error_analysis.five_fold_table(nnunet, threshold, expected_cases=129)
    moved = threshold.copy()
    moved.loc[moved["case"] == "f2b", "fold"] = 1
    with pytest.raises(ValueError, match="1 cases are in a different fold"):
        error_analysis.five_fold_table(nnunet, moved)
    with pytest.raises(ValueError, match="missing from the threshold"):
        error_analysis.five_fold_table(nnunet, threshold[threshold["case"] != "f1a"])


def _five_fold_features():
    """A hand-made five-fold table with features: 8 cases over folds 0 and 1, known answers."""
    table = pd.DataFrame({
        "case": [f"c{i}" for i in range(8)],
        "fold": [0, 0, 0, 0, 1, 1, 1, 1],
        "ref_ml": [2.2, 10.0, 101.0, 40.0, 1.1, 15.0, 80.0, 5.0],
        "nnunet_pred_ml": [0.0, 0.8, 0.0, 30.0, 32.6, 25.0, 60.0, 4.1],
        "nnunet_dice": [0.0, 0.0, 0.0, 0.52, 0.0, 0.3, 0.6, np.nan],
        "nnunet_abs_error_ml": [2.2, 9.2, 101.0, 10.0, 31.5, 10.0, 20.0, 0.9],
        "threshold_pred_ml": [0.0, 5.0, 5.0, 20.0, 0.0, 10.0, 50.0, 3.5],
        "threshold_dice": [0.0, 0.1, 0.0, 0.3, 0.0, 0.2, 0.4, np.nan],
        "threshold_abs_error_ml": [2.2, 5.0, 96.0, 20.0, 1.1, 5.0, 30.0, 1.5],
        "infarct_tmax6_share": [0.1, 0.9, 0.22, 0.7, 0.75, 0.6, np.nan, 0.5],
        "infarct_tmax_missing_share": [0.0, 0.0, 0.02, 0.0, 0.0, 0.5, 0.9, 0.06],
        "occlusion_reason": ["ok", "ok", "ok", "ok", "empty_mask", "ok", "ok", "no_labels"],
        "occlusion_nearest_label": [7, 5, 4, 1, np.nan, 2, 8, np.nan],
        "occlusion_distance_mm": [1.0, 12.0, 0.0, 20.0, np.nan, 3.0, 2.0, np.nan],
        "onset_to_imaging_min": [120.0, np.nan, 300.0, np.nan, 90.0, np.nan, 60.0, 200.0],
    })
    for model in ("nnunet", "threshold"):
        for cutoff in evaluate.DECISION_CUTOFFS_ML:
            table[f"{model}_crosses_{cutoff:g}"] = [evaluate.crosses(p, r, cutoff) for p, r in zip(table[f"{model}_pred_ml"], table["ref_ml"])]
    return table


def test_is_overcall():
    # c4: 32.6 mL for 1.1 mL, crosses 21 and 31 (the Miss 8 type, Dice 0). c5: 25.0 for 15.0, crosses 21, Dice 0.3.
    # c3: 30.0 for 40.0 crosses 31 but under-calls, so it is not an over-call.
    flags = error_analysis.is_overcall(_five_fold_features())
    assert flags.tolist() == [False, False, False, False, True, True, False, False]


def test_is_undercall():
    # c2: 0 for 101 mL, crosses all four (Dice 0). c3: 30 for 40, crosses 31. c6: 60 for 80, crosses 70.
    # c4 crosses by over-calling, so it is not an under-call; c0, c1 and c7 under-call but cross nothing.
    flags = error_analysis.is_undercall(_five_fold_features())
    assert flags.tolist() == [False, False, True, True, False, False, True, False]
    assert not (flags & error_analysis.is_overcall(_five_fold_features())).any()


@pytest.mark.parametrize("reason, label, distance, group", [
    ("ok", 7, 1.0, "MCA"), ("ok", 4, 5.0, "ICA"),          # 5 mm is still within the limit
    ("ok", 1, 5.01, "too far"), ("ok", 2, 3.0, "PCA"),
    ("ok", 8, 2.0, "Pcom"), ("ok", 13, 1.0, "unknown"),    # 13 is not a listed label
    ("empty_mask", None, None, "no occlusion found"), ("no_labels", None, None, "unknown"),
    ("ok", float("nan"), float("nan"), "unknown"),
])
def test_artery_group(reason, label, distance, group):
    assert error_analysis.artery_group(reason, label, distance) == group


def test_artery_group_limit_is_adjustable():
    assert error_analysis.artery_group("ok", 1, 20.0) == "too far"
    assert error_analysis.artery_group("ok", 1, 20.0, limit_mm=25.0) == "basilar"


def test_add_artery_groups_and_group_summary():
    table = error_analysis.add_artery_groups(_five_fold_features())
    assert table["artery_group"].tolist() == ["MCA", "too far", "ICA", "too far", "no occlusion found", "PCA", "Pcom", "unknown"]
    assert table["territory_within_limit"].tolist() == [
        "anterior", "too far", "anterior", "too far", "no occlusion found", "posterior", "unknown", "unknown"]
    table["miss"] = np.where(table["nnunet_dice"] == 0, "both", "neither")
    rows = {r["group"]: r for r in error_analysis.group_summary(table, "artery_group")}
    assert rows["too far"]["cases"] == 2 and rows["too far"]["dice0"] == 1
    assert sum(r["cases"] for r in rows.values()) == 8


def test_onset_summary():
    s = error_analysis.onset_summary(_five_fold_features())
    assert s["cases"] == 8 and s["onset_known"] == 5
    # Misses with onset: c0 120, c2 300, c4 90. Others: c6 60, c7 200 (c7 has no Dice, so it is not a miss).
    assert s["dice0_misses"] == {"onset_known": 3, "median_min": 120.0, "min_min": 90.0, "max_min": 300.0}
    assert s["rest"] == {"onset_known": 2, "median_min": 130.0, "min_min": 60.0, "max_min": 200.0}
    assert not error_analysis._ids_in(s, set(_five_fold_features()["case"]))


def test_five_fold_counts_overall_and_per_fold():
    counts = error_analysis.five_fold_counts(_five_fold_features())
    o = counts["overall"]
    assert o["cases"] == 8 and o["dice0"] == 4
    assert o["miss_rate_by_size"] == {"under_21_ml": {"cases": 5, "dice0": 3}, "21_ml_and_over": {"cases": 3, "dice0": 1}}
    assert o["overcalls"] == 2 and o["overcalls_dice0"] == 1
    assert o["undercalls"] == 3 and o["undercalls_dice0"] == 1
    assert o["dice0_misses"] == {"cases": 4, "visible_with_reliable_data": 4, "visible_median": 0.485,
                                 "visible_under_25pct": 2, "no_data_at_least_5pct": 0}
    assert o["rest"] == {"cases": 4, "visible_with_reliable_data": 3, "visible_median": 0.6,
                         "visible_under_25pct": 0, "no_data_at_least_5pct": 3}
    assert set(counts["per_fold"]) == {"0", "1"}
    assert counts["per_fold"]["0"]["dice0"] == 3 and counts["per_fold"]["1"]["overcalls"] == 2
    assert sum(f["cases"] for f in counts["per_fold"].values()) == 8
    assert not error_analysis._ids_in(counts, set(_five_fold_features()["case"]))


def test_exp25_check_matches_and_reports_differences():
    table = _five_fold_features()
    # Worked out by hand from the table, rounded as EXP-25 reports, so it should match everywhere.
    # nnU-Net Dice per fold: 0.13 and 0.3 (c7 has no Dice); mean 0.215, std 0.17/sqrt(2) = 0.120.
    # nnU-Net abs error per fold: 122.4/4 = 30.6 and 62.4/4 = 15.6; mean 23.1, std 15/sqrt(2) = 10.6.
    # Threshold Dice per fold: 0.1 and 0.2; mean 0.15, std 0.071. Abs error: 30.8 and 9.4; mean 20.1, std 15.1.
    # nnU-Net crossings: c2 all four, c3 31, c4 21+31, c5 21, c6 70. Threshold: c2 all four, c3 21+31, c6 51+70.
    ref = {
        "nnunet": {"dice_mean": 0.215, "dice_std": 0.12, "abs_error_mean": 23.1, "abs_error_std": 10.6, "crossings": (3, 3, 1, 2)},
        "threshold": {"dice_mean": 0.15, "dice_std": 0.071, "abs_error_mean": 20.1, "abs_error_std": 15.1, "crossings": (2, 2, 2, 2)},
        "nnunet_per_fold": {"0": {"cases": 4, "dice_mean": 0.13, "abs_error_mean": 30.6, "crossings": (1, 2, 1, 1)},
                            "1": {"cases": 4, "dice_mean": 0.3, "abs_error_mean": 15.6, "crossings": (2, 1, 0, 1)}},
    }
    rows = error_analysis.exp25_check(table, ref)
    assert len(rows) == 2 * 8 + 2 * 7
    assert [r["what"] for r in rows if not r["match"]] == []
    ref["nnunet_per_fold"]["1"]["crossings"] = (2, 1, 0, 0)  # one wrong number
    bad = [r for r in error_analysis.exp25_check(table, ref) if not r["match"]]
    assert bad == [{"what": "fold 1 nnunet wrong side of 70 mL", "ours": 1, "exp25": 0, "match": False}]


def test_write_followup_config(tmp_path):
    run_dir = tmp_path / "EXP-99_visionA_error_followups"
    inputs = {"nnunet_fold0": "runs/EXP-14_nnunet_fold0/per_case.csv"}
    config = error_analysis.write_followup_config(run_dir, "abc1234", "EXP-99", inputs)
    assert config["experiment"] == "EXP-99" and config["name"] == "EXP-99_visionA_error_followups"
    assert "smaller than the true volume" in config["definitions"]["undercall"]
    assert "larger than the true volume" in config["definitions"]["overcall"]
    assert config["definitions"]["artery_distance_limit_mm"] == 5.0
    assert json.loads((run_dir / "config.json").read_text())["git_commit"] == "abc1234"
    with pytest.raises(FileExistsError):
        error_analysis.write_followup_config(run_dir, "def5678", "EXP-99", inputs)
    for placeholder in ("EXP-NN", "EXP-[your number]", ""):
        with pytest.raises(ValueError, match="Notion row"):
            error_analysis.write_followup_config(tmp_path / "other", "abc1234", placeholder, inputs)


def test_find_mask_dir(tmp_path):
    nested = tmp_path / "fold_0" / "validation"
    nested.mkdir(parents=True)
    for case in ("sub-a", "sub-b"):
        (nested / f"{case}.nii.gz").write_bytes(b"")
    (tmp_path / "sub-a.nii.gz").write_bytes(b"")  # one stray copy higher up
    folder, found = error_analysis.find_mask_dir(tmp_path, ["sub-a", "sub-b", "sub-c"])
    assert folder == nested and found == 2
    with pytest.raises(FileNotFoundError):
        error_analysis.find_mask_dir(tmp_path, ["sub-x"])


def test_compare_with_per_case():
    table = _five_fold_features()
    other = pd.DataFrame({"case": table["case"], "dice": table["nnunet_dice"],
                          "pred_ml": table["nnunet_pred_ml"], "ref_ml": table["ref_ml"]})
    out = error_analysis.compare_with_per_case(table, other)
    assert out["in_both"] == 8 and out["only_ours"] == 0 and out["only_theirs"] == 0
    assert out["columns"] == {"dice": {"compared": 8, "differ": 0}, "pred_ml": {"compared": 8, "differ": 0},
                              "ref_ml": {"compared": 8, "differ": 0}}
    other.loc[2, "pred_ml"] = 5.0
    other = other.iloc[1:]  # one case missing from theirs
    out = error_analysis.compare_with_per_case(table, other)
    assert out["only_ours"] == 1 and out["columns"]["pred_ml"]["differ"] == 1
    assert not error_analysis._ids_in(out, set(table["case"]))
    with pytest.raises(ValueError, match="repeated"):
        error_analysis.compare_with_per_case(table, pd.concat([other, other]))


def test_followup_metrics_refuses_case_ids():
    table = _five_fold_features()
    counts = error_analysis.five_fold_counts(table)
    groups = error_analysis.group_summary(
        error_analysis.add_artery_groups(table).assign(miss="neither"), "artery_group")
    onset = error_analysis.onset_summary(table)
    metrics = error_analysis.followup_metrics(table, {"experiment": "EXP-99", "git_commit": "abc"}, "def",
                                              {"artery_group": groups}, onset, counts, [])
    assert metrics["git_commit"] == "def" and metrics["config_git_commit"] == "abc" and metrics["cases"] == 8
    json.dumps(metrics)  # must be writable as JSON
    onset["note"] = "c3"  # a case ID where none should be
    with pytest.raises(ValueError, match="case IDs"):
        error_analysis.followup_metrics(table, {}, "def", {}, onset, counts, [])


def test_followup_notebook_flow(tmp_path):
    """What notebook 13 does after loading data, on fake tables: join, add features, count, group, check, save."""
    nnunet, threshold = _fold_tables()
    table = error_analysis.five_fold_table(nnunet, threshold, expected_cases=6)
    features = pd.DataFrame({
        "case": table["case"], "size_band": "5-21", "infarct_tmax6_share": 0.5, "infarct_tmax_missing_share": 0.0,
        "occlusion_reason": "ok", "occlusion_nearest_label": 7, "occlusion_distance_mm": 2.0,
        "onset_to_imaging_min": [100.0, np.nan, 150.0, np.nan, 200.0, np.nan]})
    table = table.merge(features, on="case", how="left", validate="one_to_one")
    counts = error_analysis.five_fold_counts(table)
    grouped = error_analysis.add_artery_groups(table)
    groups = {g: error_analysis.group_summary(grouped, g) for g in ("artery_group", "territory_within_limit")}
    onset = error_analysis.onset_summary(table)
    exp25_rows = error_analysis.exp25_check(table)  # the real reference: fake data will not match, that is fine
    assert len(exp25_rows) > 0 and not all(r["match"] for r in exp25_rows)
    picks = error_analysis.figure_cases(table[table["fold"] == 0])
    assert [p["label"] for p in picks] == ["Miss 1", "Best 1"]
    summary = error_analysis.figure_summary(table[table["fold"] == 0], picks)
    config = error_analysis.write_followup_config(tmp_path / "EXP-99_x", "abc1234", "EXP-99", {"a": "b"})
    metrics = error_analysis.followup_metrics(table, config, "def5678", groups, onset, counts, exp25_rows,
                                              {"x.csv": error_analysis.compare_with_per_case(table, table[["case"]])})
    path = error_analysis.write_metrics(tmp_path / "EXP-99_x", metrics)
    saved = json.loads(path.read_text())
    assert saved["counts"]["overall"]["cases"] == 6 and saved["git_commit"] == "def5678"
    assert not error_analysis._ids_in(saved, set(table["case"])) and not error_analysis._ids_in(summary, set(table["case"]))


def test_exp25_reference_is_consistent():
    """The recorded EXP-25 per-fold crossings and case counts add up to its totals."""
    ref = error_analysis.EXP25_REFERENCE
    folds = ref["nnunet_per_fold"].values()
    assert sum(f["cases"] for f in folds) == 129
    assert tuple(sum(f["crossings"][i] for f in folds) for i in range(4)) == ref["nnunet"]["crossings"]


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


def test_case_features_tmax6_share():
    # Default case: the whole infarct is delayed (Tmax 10 s).
    assert error_analysis.case_features(synthetic_case())["infarct_tmax6_share"] == 1.0
    case = synthetic_case()
    case.arrays["tmax"][24:29, 8:20, 6:14] = 3.0     # half the infarct: data, but no delay
    case.arrays["tmax"][29:34, 8:20, 6:10] = -30.0   # a quarter: no data, left out of the share
    f = error_analysis.case_features(case)
    assert f["infarct_tmax6_share"] == pytest.approx(1 / 3)  # 1 delayed quarter of 3 reliable quarters
    case.arrays["tmax"][case.arrays["infarct"]] = -30.0
    assert error_analysis.case_features(case)["infarct_tmax6_share"] is None


@pytest.mark.parametrize("share, band", [(None, "no data"), (float("nan"), "no data"), (0.0, "<25%"),
                                         (0.25, "25%-50%"), (0.6, "50%-75%"), (0.75, ">=75%"), (1.0, ">=75%")])
def test_tmax6_band(share, band):
    assert error_analysis.tmax6_band(share) == band


def test_visibility_comparison():
    v = error_analysis.visibility_comparison(_step4_table())
    assert v["dice0_misses"] == {"cases": 3, "with_reliable_data": 3, "median": 0.2, "p25": 0.15, "p75": 0.25}
    assert v["rest"]["cases"] == 3 and v["rest"]["with_reliable_data"] == 2 and v["rest"]["median"] == 0.85


def test_figure_cases_labels():
    picks = error_analysis.figure_cases(_step4_table())
    assert [p["label"] for p in picks] == ["Miss 1", "Miss 2", "Miss 3", "Best 1", "Best 2"]
    assert picks[0]["size_band"] == ">=70"  # the 101 mL miss comes first
    assert picks[3]["case"] == "sub-06"     # highest Dice; only 2 cases have Dice above 0


def test_figure_summary_has_labels_not_ids():
    table = _step4_table()
    summary = error_analysis.figure_summary(table, error_analysis.figure_cases(table))
    assert summary[0] == {"label": "Miss 1", "size_band": ">=70", "dice0": True, "infarct_tmax6_share": 0.3}
    assert summary[3] == {"label": "Best 1", "size_band": ">=70", "dice0": False, "infarct_tmax6_share": 0.9}
    assert all("case" not in row for row in summary)
    assert not error_analysis._ids_in(summary, set(table["case"]))


def test_case_figure_is_anonymous():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    case = synthetic_case()
    case.id = case.clinical["case"] = "sub-stroke9999"
    case.arrays["ncct"] = np.full(SHAPE, 30.0, dtype=np.float32)
    fig = error_analysis.case_figure(case, "Miss 1", "5-21")
    texts = error_analysis.figure_texts(fig)
    assert "Miss 1, 5-21 mL: final infarct in red" in texts
    assert {"NCCT", "CBF", "TMAX"} <= set(texts)
    assert not any("sub-stroke9999" in t for t in texts)
    assert len(fig.axes) == 3
    plt.close(fig)


def _figure_case():
    """The synthetic case named like a real one, with an NCCT map, for the figure tests."""
    case = synthetic_case()
    case.id = case.clinical["case"] = "sub-stroke9999"
    case.arrays["ncct"] = np.full(SHAPE, 30.0, dtype=np.float32)
    return case


def _outline_colours(fig):
    """The edge colours of every outline (contour) drawn on the figure's panels, as RGBA tuples."""
    colours = set()
    for ax in fig.axes:
        for collection in ax.collections:
            for rgba in np.atleast_2d(collection.get_edgecolor()):
                colours.add(tuple(np.round(rgba, 3)))
    return colours


def test_case_figure_draws_both_outlines():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    case = _figure_case()
    prediction = np.zeros(SHAPE, dtype=bool)
    prediction[20:30, 10:18, 6:14] = True  # overlaps part of the infarct, on the same slices
    fig = error_analysis.case_figure(case, "Miss 8", "under 5", prediction=prediction)
    colours = _outline_colours(fig)
    assert tuple(np.round(to_rgba("red"), 3)) in colours and tuple(np.round(to_rgba("cyan"), 3)) in colours
    texts = error_analysis.figure_texts(fig)
    assert "Miss 8, under 5 mL: final infarct in red, AI outline in cyan" in texts
    assert {"Final infarct", "AI outline"} <= set(texts)
    assert not any("sub-stroke9999" in t for t in texts)
    plt.close(fig)


def test_case_figure_says_when_the_ai_marked_elsewhere_or_nothing():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    case = _figure_case()
    z = 10  # the infarct's largest slice in the synthetic case is one of z 6 to 13
    elsewhere = np.zeros(SHAPE, dtype=bool)
    elsewhere[5:9, 5:9, 16:19] = True  # 3 slices, none with infarct
    fig = error_analysis.case_figure(case, "Miss 8", prediction=elsewhere, z=z)
    assert "AI outline: none on this slice (marks on 3 other slices)" in error_analysis.figure_texts(fig)
    assert tuple(np.round(to_rgba("cyan"), 3)) not in _outline_colours(fig)
    plt.close(fig)
    fig = error_analysis.case_figure(case, "Miss 1", prediction=np.zeros(SHAPE, dtype=bool))
    assert "AI outline: none on this slice, none anywhere" in error_analysis.figure_texts(fig)
    plt.close(fig)
    with pytest.raises(ValueError, match="shape"):
        error_analysis.case_figure(case, "Miss 1", prediction=np.zeros((4, 4, 4), dtype=bool))


def test_load_prediction(tmp_path):
    nib = pytest.importorskip("nibabel")
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(tmp_path / "sub-stroke9999.nii.gz"))
    pred = error_analysis.load_prediction(tmp_path, "sub-stroke9999", SHAPE)
    assert pred.dtype == bool and int(pred.sum()) == 8
    with pytest.raises(ValueError, match="shape"):
        error_analysis.load_prediction(tmp_path, "sub-stroke9999", (4, 4, 4))


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
    """A step-4 table by hand: six cases with known scores and features. Crossings follow from the volumes."""
    table = pd.DataFrame({
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
        "infarct_tmax6_share": [0.1, 0.2, 0.3, 0.8, np.nan, 0.9],
        "threshold_abs_error_ml": [2.0, 1.0, 96.0, 20.0, 0.0, 30.0],
    })
    for model in ("nnunet", "threshold"):
        for cutoff in evaluate.DECISION_CUTOFFS_ML:
            table[f"{model}_crosses_{cutoff:g}"] = [evaluate.crosses(p, r, cutoff) for p, r in zip(table[f"{model}_pred_ml"], table["ref_ml"])]
    return table


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


def test_crossing_counts():
    # The 101 mL miss (predicted 0 and 5 mL) crosses every cut-off for both models. The 40 mL case
    # crosses 31 mL for nnU-Net (30 mL) and 21 and 31 mL for the threshold (20 mL).
    c = error_analysis.crossing_counts(_step4_table())
    assert c["21"] == {"nnunet": 1, "threshold": 2, "both": 1, "of": 6}
    assert c["31"] == {"nnunet": 2, "threshold": 2, "both": 2, "of": 6}
    assert c["51"] == c["70"] == {"nnunet": 1, "threshold": 1, "both": 1, "of": 6}
    assert c["any"] == {"nnunet": 2, "threshold": 2, "both": 2, "of": 6}


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
    assert metrics["wrong_side_of_cutoff"]["70"]["nnunet"] == 1
    big = {r["group"]: r for r in metrics["groups"]["size_band"]}[">=70"]
    assert big["nnunet_abs_error_ml_mean"] == 55.5 and big["threshold_abs_error_ml_mean"] == 63.0
    path = error_analysis.write_metrics(tmp_path, metrics)
    saved = path.read_text()
    assert "sub-0" not in saved
    assert json.loads(saved)["dice0_misses"]["largest_miss"]["ref_ml"] == 101.0


def test_build_metrics_refuses_case_ids():
    table = _step4_table()
    table.loc[0, "territory"] = "sub-03"  # an ID where a group label should be
    with pytest.raises(ValueError, match="case IDs"):
        error_analysis.build_metrics(table, {}, "abc1234")
