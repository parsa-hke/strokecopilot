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
SPACING = 2.0  # 8 mm3 per voxel, 125 voxels per mL


def _maps():
    """Brain at x 4-35 (tissue 30 HU, Tmax 1 s, CBF 50) on every slice. Tmax 10 s over x 24-33 (the at-risk area)."""
    ncct = np.full(SHAPE, -1000.0, dtype=np.float32)
    cbf = np.zeros(SHAPE, dtype=np.float32)
    tmax = np.full(SHAPE, -30.0, dtype=np.float32)
    brain = (slice(4, 36), slice(4, 26), slice(4, 16))
    ncct[brain], cbf[brain], tmax[brain] = 30.0, 50.0, 1.0
    tmax[24:34, 8:20, 6:14] = 10.0
    return ncct, cbf, tmax


def _slab_head():
    """A head whose plain CT covers z 2-17 with a scalp ring, while perfusion covers only the brain on z 6-13."""
    ncct = np.full(SHAPE, -1000.0, dtype=np.float32)
    ncct[2:38, 2:28, 2:18] = 30.0                   # scalp and soft tissue, every slice of the head
    cbf = np.zeros(SHAPE, dtype=np.float32)
    tmax = np.full(SHAPE, -30.0, dtype=np.float32)
    cbf[6:34, 6:24, 6:14], tmax[6:34, 6:24, 6:14] = 50.0, 1.0  # the perfusion slab, brain only
    return ncct, cbf, tmax


# ---------------------------------------------------------------------------- Where missing data is counted


def test_brain_mask_uses_the_tissue_range():
    ncct, _, _ = _maps()
    assert int(conf.brain_mask(ncct).sum()) == 32 * 22 * 12


def test_slab_and_scalp_are_not_counted_as_missing():
    ncct, cbf, tmax = _slab_head()
    tissue = conf.brain_mask(ncct)
    # The old measure counts the scalp and the slices outside the slab as missing.
    assert conf.no_data_share(tmax, cbf, tissue) > 0.5
    area = conf.measurement_area(ncct, tmax, cbf)
    assert np.flatnonzero(area.any(axis=(0, 1))).tolist() == list(range(6, 14))
    assert conf.no_data_share(tmax, cbf, area) == 0.0


def test_real_gaps_inside_the_slab_still_count():
    ncct, cbf, tmax = _slab_head()
    hole = tmax.copy()
    hole[15:25, 10:20, 8:10] = -30.0                # a region with no data, enclosed in the brain
    area = conf.measurement_area(ncct, hole, cbf)
    assert conf.no_data_share(hole, cbf, area) == pytest.approx(10 * 10 * 2 / (28 * 18 * 8))
    gap = tmax.copy()
    gap[:, :, 9] = -30.0                            # a whole slice missing inside the slab
    gap_cbf = cbf.copy()
    gap_cbf[:, :, 9] = 0.0
    area = conf.measurement_area(ncct, gap, gap_cbf)
    assert area[:, :, 9].any()                      # the gap slice takes the slab's outline
    assert conf.no_data_share(gap, gap_cbf, area) == pytest.approx(1 / 8)


def test_perfusion_region_without_any_perfusion_is_empty():
    region = conf.perfusion_region(np.full(SHAPE, -30.0), np.zeros(SHAPE))
    assert not region.any()


# ------------------------------------------------------------------------------- Signals


def test_at_risk_no_data_share_counts_only_the_zone():
    ncct, cbf, tmax = _maps()
    brain = conf.brain_mask(ncct)
    assert conf.at_risk_no_data_share(tmax, cbf, SPACING, brain) == 0.0
    far = tmax.copy()
    far[4:8, 4:26, 4:16] = -30.0                    # more than 10 mm away: not counted
    assert conf.at_risk_no_data_share(far, cbf, SPACING, brain) == 0.0
    near = tmax.copy()
    near[34:36, 8:20, 6:14] = -30.0                 # within the 10 mm margin: counted
    assert 0.0 < conf.at_risk_no_data_share(near, cbf, SPACING, brain) < 0.2
    assert conf.at_risk_no_data_share(far, cbf, SPACING, brain, margin_mm=60.0) > 0.0


def test_at_risk_no_data_share_without_an_at_risk_area():
    ncct, cbf, tmax = _maps()
    tmax[tmax > 6] = 1.0
    tmax[4:12, 4:26, 4:16] = -30.0
    brain = conf.brain_mask(ncct)
    assert conf.at_risk_no_data_share(tmax, cbf, SPACING, brain) == pytest.approx(0.25)
    assert conf.at_risk_no_data_share(tmax, cbf, SPACING, np.zeros(SHAPE, dtype=bool)) is None


def test_penumbra_confidence():
    assert conf.penumbra_confidence(0.25) == pytest.approx(0.75)
    assert conf.penumbra_confidence(None) == 0.0 and conf.penumbra_confidence(float("nan")) == 0.0


def test_tmax6_ml():
    _, cbf, tmax = _maps()
    assert conf.tmax6_ml(tmax, cbf, SPACING) == pytest.approx(10 * 12 * 8 / 125)


def test_case_signals():
    ncct, cbf, tmax = _slab_head()
    tmax[24:30, 8:20, 6:14] = 10.0
    case = Case(id="sub-stroke9999", arrays={"ncct": ncct, "cbf": cbf, "tmax": tmax}, spacing_mm=SPACING,
                affine=np.eye(4), clinical={})
    s = conf.case_signals(case)
    assert s["case"] == "sub-stroke9999"
    assert s["at_risk_no_data_share"] == 0.0 and s["brain_no_data_share"] == 0.0 and s["confidence_penumbra"] == 1.0
    assert s["tissue_no_data_share"] > 0.5            # the old measure, kept for the before/after check
    assert s["area_slices"] == 8 and s["tissue_slices"] == 16
    assert s["tmax6_ml"] == pytest.approx(6 * 12 * 8 / 125)


def test_volume_signals_use_log1p():
    table = pd.DataFrame({"nnunet_pred_ml": [0.0, 9.0, 99.0], "threshold_pred_ml": [0.0, 0.0, 9.0],
                          "tmax6_ml": [0.0, 9.0, 99.0]})
    t = conf.add_volume_signals(table)
    assert t["log_pred_ml"].tolist() == pytest.approx([0.0, np.log(10), np.log(100)])
    assert t["log_disagreement"].tolist() == pytest.approx([0.0, np.log(10), np.log(10)])
    assert t["log_tmax6_ml"].tolist() == pytest.approx([0.0, np.log(10), np.log(100)])
    assert "log_tmax6_ml" not in conf.add_volume_signals(table.drop(columns=["tmax6_ml"])).columns


def test_log_cutoff_distance():
    # Nearest on the log scale, not in plain mL: 40 mL is nearer 51 than 31 (|log 41 - log 52| < |log 41 - log 32|),
    # and 60 mL is nearer 70 than 51. So: 0 -> 21, 21 -> 21, 40 -> 51, 60 -> 70, 200 -> 70.
    d = conf.log_cutoff_distance(np.array([0.0, 21.0, 40.0, 60.0, 200.0]))
    expected = [np.log(22), 0.0, np.log(52 / 41), np.log(71 / 61), np.log(201 / 71)]
    assert d.tolist() == pytest.approx(expected)
    assert float(conf.log_cutoff_distance(21.0)) == 0.0


def test_add_targets_marks_undercalls():
    table = pd.DataFrame({"nnunet_pred_ml": [0.0, 40.0, 5.0], "ref_ml": [101.0, 10.0, 6.0],
                          "nnunet_abs_error_ml": [101.0, 30.0, 1.0]})
    for cutoff in evaluate.DECISION_CUTOFFS_ML:
        table[f"nnunet_crosses_{cutoff:g}"] = [evaluate.crosses(p, r, cutoff) for p, r in zip(table["nnunet_pred_ml"], table["ref_ml"])]
    t = conf.add_targets(table)
    assert t["large_error"].tolist() == [True, True, False]
    assert t["undercall"].tolist() == [True, False, False]   # the second is an over-call
    assert t["error_10ml"].tolist() == [True, True, False]


# ------------------------------------------------------------------------------- The mapping


def _fake_table(n_per_fold=20, seed=0):
    """Five fake folds where large errors follow big or disputed predictions, with Tmax > 6 s volumes."""
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
    # Tmax > 6 s volumes from their own random stream, so they do not change the data above.
    table["tmax6_ml"] = pred * np.random.default_rng(seed + 1).uniform(1, 4, n)
    table = conf.add_volume_signals(table)
    return conf.add_targets(table)


def test_fit_mapping_learns_the_direction():
    table = _fake_table()
    mapping = conf.fit_mapping(table[list(conf.FEATURES)].to_numpy(), table["large_error"].to_numpy())
    w = dict(zip(conf.FEATURES, mapping["weights"]))
    assert w["log_disagreement"] > 0  # disputed predictions mean more risk in this fake data
    json.dumps(mapping)


def test_fit_mapping_handles_missing_and_refuses_bad_input():
    X = np.array([[np.nan, 1.0, 0.1], [0.2, 2.0, 0.3], [0.1, 3.0, 0.0], [0.0, 0.5, 1.0]])
    mapping = conf.fit_mapping(X, [True, False, False, True])
    assert np.isfinite(conf.predict_confidence(mapping, X)).all()
    with pytest.raises(ValueError, match="one class"):
        conf.fit_mapping(X, [False, False, False, False])
    with pytest.raises(ValueError, match="columns"):
        conf.fit_mapping(X, [True, False, False, True], features=conf.VERSIONS["v2"])


def test_cross_fitted_confidence_never_sees_its_own_case():
    table = _fake_table()
    for features in conf.VERSIONS.values():
        scores, mappings = conf.cross_fitted_confidence(table, features)
        assert set(mappings) == {"0", "1", "2", "3", "4"} and ((scores >= 0) & (scores <= 1)).all()
        assert mappings["0"]["n"] == 80 and mappings["0"]["features"] == list(features)
        flipped = table.copy()
        flipped.loc[0, "large_error"] = not flipped.loc[0, "large_error"]
        again, _ = conf.cross_fitted_confidence(flipped, features)
        assert again[0] == pytest.approx(scores[0])  # its own fold's mapping never saw it
        assert not np.allclose(again[table["fold"] != 0], scores[table["fold"] != 0])


def test_confidence_core_single_case_matches_the_table(tmp_path):
    table = _fake_table()
    for features in conf.VERSIONS.values():
        mapping = conf.fit_mapping(table[list(features)].to_numpy(), table["large_error"].to_numpy(), features=features)
        loaded = conf.load_mapping(conf.save_mapping(tmp_path / "mapping.json", mapping))
        row = table.iloc[3]
        single = conf.confidence_core(loaded, row["at_risk_no_data_share"], row["nnunet_pred_ml"],
                                      row["threshold_pred_ml"], row["tmax6_ml"])
        assert single == pytest.approx(conf.predict_confidence(mapping, table[list(features)].to_numpy())[3])
        assert 0.0 <= conf.confidence_core(loaded, None, 0.0, 0.0) <= 1.0


# ---------------------------------------------------------------------------- Bands and the checks


def test_bands_are_three_equal_groups():
    b = conf.bands(np.linspace(0, 1, 9))
    assert b.tolist() == ["low"] * 3 + ["middle"] * 3 + ["high"] * 3
    assert sorted(pd.Series(conf.bands(np.zeros(12))).value_counts().tolist()) == [4, 4, 4]


def test_band_table_and_done_check_on_a_hand_table():
    # Six cases: the two lowest-confidence ones carry the large errors; c4 is an under-call in the high band.
    table = pd.DataFrame({
        "case": [f"c{i}" for i in range(6)], "confidence_core": [0.1, 0.2, 0.5, 0.6, 0.8, 0.9],
        "nnunet_abs_error_ml": [40.0, 15.0, 5.0, 12.0, 2.0, 1.0],
        "nnunet_pred_ml": [0.0, 30.0, 10.0, 50.0, 69.0, 5.0], "ref_ml": [40.0, 15.0, 15.0, 62.0, 71.0, 6.0],
    })
    for cutoff in evaluate.DECISION_CUTOFFS_ML:
        table[f"nnunet_crosses_{cutoff:g}"] = [evaluate.crosses(p, r, cutoff) for p, r in zip(table["nnunet_pred_ml"], table["ref_ml"])]
    table = conf.add_targets(table)
    rows = conf.band_table(table)
    # c0: 0 vs 40 crosses 21 and 31 (under); c1: 30 vs 15 crosses 21 (over); c3: 50 vs 62 crosses 51 (under);
    # c4: 69 vs 71 crosses 70 (under).
    assert rows["low"]["cases"] == 2 and rows["low"]["wrong_side_any"] == 2 and rows["low"]["error_10ml_or_more"] == 2
    assert rows["low"]["undercalls"] == 1 and rows["middle"]["undercalls"] == 1 and rows["high"]["undercalls"] == 1
    assert rows["high"]["wrong_side_any"] == 1 and rows["high"]["wrong_side_70"] == 1
    assert rows["all"]["undercalls"] == 3 and rows["low"]["abs_error_ml_mean"] == 27.5
    check = conf.done_check(rows, table)
    assert check["passes"] and check["undercalls_in_high_band"] == 1 and check["undercalls_total"] == 3
    assert check["spearman_confidence_vs_abs_error"] < 0
    json.dumps(check)


def test_signals_alone():
    table = _fake_table()
    alone = conf.signals_alone(table)
    assert set(alone) == set(conf.ALL_SIGNALS)
    assert alone["log_disagreement"]["weight_alone"] > 0 and not alone["log_disagreement"]["raises_confidence"]
    json.dumps(alone)


def test_compare_versions_keeps_every_version():
    table = _fake_table()
    results, confidences = conf.compare_versions(table)
    assert list(results) == ["v1", "v2", "v3"] and set(confidences) == {"v1", "v2", "v3"}
    for name, r in results.items():
        assert r["features"] == list(conf.VERSIONS[name])
        assert r["bands"]["all"]["cases"] == 100 and len(r["fold_mappings"]) == 5
        assert r["done_check"]["undercalls_total"] == int(table["undercall"].sum())
    json.dumps(results)


def test_slab_check():
    table = pd.DataFrame({"tissue_no_data_share": [0.3, 0.4, 0.2], "brain_no_data_share": [0.01, 0.2, 0.0],
                          "tissue_slices": [60, 70, 65], "area_slices": [20, 22, 18]})
    s = conf.slab_check(table)
    assert s["tissue_no_data_share"] == {"median": 0.3, "min": 0.2, "max": 0.4}
    assert s["area_no_data_share"]["median"] == 0.01 and s["cases_area_share_over_0_1"] == 1
    assert s["area_slices"]["median"] == 20.0


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
    assert config["experiment"] == "EXP-99" and set(config["versions"]) == {"v1", "v2", "v3"}
    assert "nearest cut-off" in config["signals"]["log_cutoff_distance"]
    with pytest.raises(FileExistsError):
        conf.write_config(tmp_path / "EXP-99_visionA_confidence", "abc1234", "EXP-99", {})


def test_build_metrics_end_to_end_and_refuses_case_ids(tmp_path):
    table = _fake_table()
    table["confidence_penumbra"] = 1 - table["at_risk_no_data_share"]
    table["tissue_no_data_share"], table["brain_no_data_share"] = 0.3, table["at_risk_no_data_share"]
    table["tissue_slices"], table["area_slices"] = 60, 20
    versions, _ = conf.compare_versions(table)
    metrics = conf.build_metrics(table, {"experiment": "EXP-99", "git_commit": "abc"}, "def", versions,
                                 conf.signals_alone(table), conf.slab_check(table))
    saved = json.loads(conf.write_metrics(tmp_path, metrics).read_text())
    assert saved["cases"] == 100 and saved["git_commit"] == "def" and saved["config_git_commit"] == "abc"
    assert set(saved["versions"]) == {"v1", "v2", "v3"} and saved["undercall_cases"] == int(table["undercall"].sum())
    versions["v1"]["bands"]["low"]["note"] = "sub-007"
    with pytest.raises(ValueError, match="case IDs"):
        conf.build_metrics(table, {}, "def", versions, {}, {})
