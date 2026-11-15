"""Vision A confidence on fake data. Needs numpy, scipy and pandas; no real data."""

import json

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from strokecopilot.data.isles24 import Case  # noqa: E402
from strokecopilot.vision.perfusion import confidence as conf  # noqa: E402
from strokecopilot.vision.perfusion import evaluate  # noqa: E402

SHAPE = (40, 30, 20)
SPACING = 2.0


def _maps():
    """Brain at x 4-35 (tissue 30 HU, Tmax 1 s). Tmax 10 s over x 24-33 (the at-risk area)."""
    ncct = np.full(SHAPE, -1000.0, dtype=np.float32)
    cbf = np.zeros(SHAPE, dtype=np.float32)
    tmax = np.full(SHAPE, -30.0, dtype=np.float32)
    brain = (slice(4, 36), slice(4, 26), slice(4, 16))
    ncct[brain], cbf[brain], tmax[brain] = 30.0, 50.0, 1.0
    tmax[24:34, 8:20, 6:14] = 10.0
    return ncct, cbf, tmax


# ------------------------------------------------------------------------------- Signals


def test_brain_mask_uses_the_tissue_range():
    ncct, _, _ = _maps()
    assert int(conf.brain_mask(ncct).sum()) == 32 * 22 * 12


def test_at_risk_no_data_share_counts_only_the_zone():
    ncct, cbf, tmax = _maps()
    brain = conf.brain_mask(ncct)
    assert conf.at_risk_no_data_share(tmax, cbf, SPACING, brain) == 0.0
    # No data far from the at-risk area (x 4-7, more than 10 mm away) does not count.
    far = tmax.copy()
    far[4:8, 4:26, 4:16] = -30.0
    assert conf.at_risk_no_data_share(far, cbf, SPACING, brain) == 0.0
    # No data right next to it (within the 10 mm margin) does.
    near = tmax.copy()
    near[34:36, 8:20, 6:14] = -30.0
    share = conf.at_risk_no_data_share(near, cbf, SPACING, brain)
    assert 0.0 < share < 0.2
    # A wider margin reaches the far patch too.
    assert conf.at_risk_no_data_share(far, cbf, SPACING, brain, margin_mm=60.0) > 0.0


def test_at_risk_no_data_share_without_an_at_risk_area():
    ncct, cbf, tmax = _maps()
    tmax[tmax > 6] = 1.0
    tmax[4:12, 4:26, 4:16] = -30.0  # a quarter of the brain without data
    brain = conf.brain_mask(ncct)
    assert conf.at_risk_no_data_share(tmax, cbf, SPACING, brain) == pytest.approx(0.25)
    assert conf.at_risk_no_data_share(tmax, cbf, SPACING, np.zeros(SHAPE, dtype=bool)) is None


def test_brain_share_and_penumbra_confidence():
    ncct, cbf, tmax = _maps()
    tmax[4:12, 4:26, 4:16] = -30.0
    brain = conf.brain_mask(ncct)
    share = conf.brain_no_data_share(tmax, cbf, brain)
    assert share == pytest.approx(0.25)
    assert conf.penumbra_confidence(share) == pytest.approx(0.75)
    assert conf.penumbra_confidence(None) == 0.0 and conf.penumbra_confidence(float("nan")) == 0.0
    assert conf.brain_no_data_share(tmax, cbf, np.zeros(SHAPE, dtype=bool)) is None


def test_case_signals():
    ncct, cbf, tmax = _maps()
    case = Case(id="sub-stroke9999", arrays={"ncct": ncct, "cbf": cbf, "tmax": tmax}, spacing_mm=SPACING,
                affine=np.eye(4), clinical={})
    s = conf.case_signals(case)
    assert s == {"case": "sub-stroke9999", "at_risk_no_data_share": 0.0, "brain_no_data_share": 0.0,
                 "confidence_penumbra": 1.0}


def test_volume_signals_use_log1p():
    table = pd.DataFrame({"nnunet_pred_ml": [0.0, 9.0, 99.0], "threshold_pred_ml": [0.0, 0.0, 9.0]})
    t = conf.add_volume_signals(table)
    assert t["log_pred_ml"].tolist() == pytest.approx([0.0, np.log(10), np.log(100)])
    assert t["log_disagreement"].tolist() == pytest.approx([0.0, np.log(10), np.log(10)])


# ------------------------------------------------------------------------------- The mapping


def _fake_table(n_per_fold=20, seed=0):
    """Five fake folds where large errors follow small predictions and big disagreement, as the design expects."""
    rng = np.random.default_rng(seed)
    n = 5 * n_per_fold
    pred = rng.uniform(0, 120, n)
    thr = np.clip(pred * rng.uniform(0.2, 1.8, n), 0, None)
    share = rng.uniform(0, 0.3, n)
    table = pd.DataFrame({"case": [f"sub-{i:03d}" for i in range(n)], "fold": np.repeat(range(5), n_per_fold),
                          "nnunet_pred_ml": pred, "threshold_pred_ml": thr, "at_risk_no_data_share": share})
    table = conf.add_volume_signals(table)
    risk = -0.8 * table["log_pred_ml"] + 1.5 * table["log_disagreement"] + 6 * share + 2.0
    error = np.abs(rng.normal(0, 4, n)) + 25 * (rng.uniform(size=n) < 1 / (1 + np.exp(-risk)))
    ref = pred + error * np.where(rng.uniform(size=n) < 0.5, -1, 1)
    table["ref_ml"] = np.clip(ref, 0, None)
    table["nnunet_abs_error_ml"] = (table["nnunet_pred_ml"] - table["ref_ml"]).abs()
    for cutoff in evaluate.DECISION_CUTOFFS_ML:
        table[f"nnunet_crosses_{cutoff:g}"] = [evaluate.crosses(p, r, cutoff) for p, r in zip(table["nnunet_pred_ml"], table["ref_ml"])]
    return conf.add_targets(table)


def test_fit_mapping_learns_the_direction():
    table = _fake_table()
    mapping = conf.fit_mapping(table[list(conf.FEATURES)].to_numpy(), table["large_error"].to_numpy())
    w = dict(zip(conf.FEATURES, mapping["weights"]))
    assert w["log_pred_ml"] < 0 < w["log_disagreement"]  # small or disputed predictions mean more risk
    json.dumps(mapping)


def test_fit_mapping_handles_missing_and_refuses_one_class():
    X = np.array([[np.nan, 1.0, 0.1], [0.2, 2.0, 0.3], [0.1, 3.0, 0.0], [0.0, 0.5, 1.0]])
    mapping = conf.fit_mapping(X, [True, False, False, True])
    assert np.isfinite(conf.predict_confidence(mapping, X)).all()
    with pytest.raises(ValueError, match="one class"):
        conf.fit_mapping(X, [False, False, False, False])


def test_cross_fitted_confidence_never_sees_its_own_case():
    table = _fake_table()
    scores, mappings = conf.cross_fitted_confidence(table)
    assert set(mappings) == {"0", "1", "2", "3", "4"} and ((scores >= 0) & (scores <= 1)).all()
    assert mappings["0"]["n"] == 80  # fitted on the other four folds only
    # Flip one case's own label: its own score must not move, because its fold's mapping never saw it.
    flipped = table.copy()
    flipped.loc[0, "large_error"] = not flipped.loc[0, "large_error"]
    again, _ = conf.cross_fitted_confidence(flipped)
    assert again[0] == pytest.approx(scores[0])
    assert not np.allclose(again[table["fold"] != 0], scores[table["fold"] != 0])  # other folds did see it


def test_confidence_core_single_case_matches_the_table(tmp_path):
    table = _fake_table()
    mapping = conf.fit_mapping(table[list(conf.FEATURES)].to_numpy(), table["large_error"].to_numpy())
    path = conf.save_mapping(tmp_path / "mapping.json", mapping)
    loaded = conf.load_mapping(path)
    row = table.iloc[3]
    single = conf.confidence_core(loaded, row["at_risk_no_data_share"], row["nnunet_pred_ml"], row["threshold_pred_ml"])
    assert single == pytest.approx(conf.predict_confidence(mapping, table[list(conf.FEATURES)].to_numpy())[3])
    assert 0.0 <= conf.confidence_core(loaded, None, 0.0, 0.0) <= 1.0


# ------------------------------------------------------------------------------- Bands and the check


def test_bands_are_three_equal_groups():
    b = conf.bands(np.linspace(0, 1, 9))
    assert b.tolist() == ["low"] * 3 + ["middle"] * 3 + ["high"] * 3
    assert sorted(pd.Series(conf.bands(np.zeros(12))).value_counts().tolist()) == [4, 4, 4]  # ties split by order


def test_band_table_and_done_check_on_a_hand_table():
    # Six cases: the two lowest-confidence ones carry the large errors.
    table = pd.DataFrame({
        "case": [f"c{i}" for i in range(6)], "confidence_core": [0.1, 0.2, 0.5, 0.6, 0.8, 0.9],
        "nnunet_abs_error_ml": [40.0, 15.0, 5.0, 12.0, 2.0, 1.0],
        "nnunet_pred_ml": [0.0, 30.0, 10.0, 50.0, 80.0, 5.0], "ref_ml": [40.0, 15.0, 15.0, 62.0, 82.0, 6.0],
    })
    for cutoff in evaluate.DECISION_CUTOFFS_ML:
        table[f"nnunet_crosses_{cutoff:g}"] = [evaluate.crosses(p, r, cutoff) for p, r in zip(table["nnunet_pred_ml"], table["ref_ml"])]
    table = conf.add_targets(table)
    rows = conf.band_table(table)
    # c0: 0 vs 40 crosses 21 and 31; c1: 30 vs 15 crosses 21; c3: 50 vs 62 crosses 51.
    assert rows["low"]["cases"] == 2 and rows["low"]["wrong_side_any"] == 2 and rows["low"]["error_10ml_or_more"] == 2
    assert rows["middle"]["wrong_side_any"] == 1 and rows["high"]["wrong_side_any"] == 0
    assert rows["low"]["wrong_side_21"] == 2 and rows["middle"]["wrong_side_51"] == 1
    assert rows["all"]["cases"] == 6 and rows["low"]["abs_error_ml_mean"] == 27.5
    check = conf.done_check(rows, table)
    assert check["passes"] and check["spearman_confidence_vs_abs_error"] < 0
    json.dumps(check)


# ------------------------------------------------------------------------------- Run record


def test_check_table():
    table = _fake_table()
    conf.check_table(table, expected_cases=100)
    with pytest.raises(ValueError, match="not 129"):
        conf.check_table(table, expected_cases=129)
    with pytest.raises(ValueError, match="lacks columns"):
        conf.check_table(table.drop(columns=["threshold_pred_ml"]))
    with pytest.raises(ValueError, match="repeats"):
        conf.check_table(pd.concat([table, table.iloc[:1]]))


def test_write_config_refuses_placeholders(tmp_path):
    for placeholder in ("EXP-NN", "EXP-[your number]", ""):
        with pytest.raises(ValueError, match="Notion row"):
            conf.write_config(tmp_path / "x", "abc1234", placeholder, {})
    config = conf.write_config(tmp_path / "EXP-99_visionA_confidence", "abc1234", "EXP-99", {"a": "b"})
    assert config["experiment"] == "EXP-99" and "log(1 + nnU-Net predicted mL)" in config["signals"]["log_pred_ml"]
    with pytest.raises(FileExistsError):
        conf.write_config(tmp_path / "EXP-99_visionA_confidence", "abc1234", "EXP-99", {})


def test_build_metrics_end_to_end_and_refuses_case_ids(tmp_path):
    table = _fake_table()
    table["confidence_core"], mappings = conf.cross_fitted_confidence(table)
    table["confidence_penumbra"] = 1 - table["at_risk_no_data_share"]
    rows = conf.band_table(table)
    check = conf.done_check(rows, table)
    final = conf.fit_mapping(table[list(conf.FEATURES)].to_numpy(), table["large_error"].to_numpy())
    metrics = conf.build_metrics(table, {"experiment": "EXP-99", "git_commit": "abc"}, "def", rows, check, mappings, final)
    path = conf.write_metrics(tmp_path, metrics)
    saved = json.loads(path.read_text())
    assert saved["cases"] == 100 and saved["git_commit"] == "def" and saved["config_git_commit"] == "abc"
    assert saved["done_check"]["passes"]  # the fake data has the expected pattern
    rows["low"]["note"] = "sub-007"
    with pytest.raises(ValueError, match="case IDs"):
        conf.build_metrics(table, {}, "def", rows, check, mappings, final)
