"""Vision B operating point (cross-fitted threshold) on synthetic scores. No RSNA data, no GPU."""

import json

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from strokecopilot.vision.haemorrhage import operating_point as op  # noqa: E402
from strokecopilot.vision.haemorrhage.aggregate import Calibrator  # noqa: E402

CALIBRATOR = Calibrator(method="topk", k=2, temperature=1.3, bias=-0.8, threshold=0.278, kind="platt")


def _scans(n_patients=400, seed=0, separation=2.0):
    """Synthetic scans: 1-3 per patient, a bleed in about 40% of patients, overlapping scores."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        bleeder = rng.random() < 0.4
        for s in range(int(rng.integers(1, 4))):
            label = bleeder and (s == 0 or rng.random() < 0.5)
            z = rng.normal(separation if label else 0.0, 1.0)
            rows.append({"series_id": f"ID_s{p}_{s}", "patient_id": f"ID_p{p}", "label": int(label), "score": float(1 / (1 + np.exp(-(z - 1))))})
    return pd.DataFrame(rows)


def _slices(scans, seed=1):
    """Slices whose top-2 mean per scan is close to the scan score."""
    rng = np.random.default_rng(seed)
    rows = []
    for scan in scans.itertuples():
        for i in range(6):
            prob = scan.score if i < 2 else scan.score * rng.uniform(0.1, 0.9)
            rows.append({"series_id": scan.series_id, "patient_id": scan.patient_id, "image_id": f"ID_i{scan.series_id}_{i}",
                         "any": int(scan.label and i < 2), "prob": prob})
    return pd.DataFrame(rows)


SCANS = _scans()


# --- halves ---------------------------------------------------------------------------------------


def test_patient_halves_never_split_a_patient_and_are_fixed_by_the_seed():
    halves = op.patient_halves(SCANS["patient_id"], SCANS["label"], seed=7)
    per_patient = pd.Series(halves).groupby(SCANS["patient_id"].to_numpy()).nunique()
    assert (per_patient == 1).all() and set(np.unique(halves)) == {0, 1}
    assert np.array_equal(halves, op.patient_halves(SCANS["patient_id"], SCANS["label"], seed=7))
    assert not np.array_equal(halves, op.patient_halves(SCANS["patient_id"], SCANS["label"], seed=8))


def test_patient_halves_balance_bleeding_patients():
    halves = op.patient_halves(SCANS["patient_id"], SCANS["label"], seed=3)
    per_patient = pd.DataFrame({"p": SCANS["patient_id"], "h": halves, "l": SCANS["label"]}).groupby("p").agg(h=("h", "first"), l=("l", "max"))
    for _, group in per_patient.groupby("l"):
        sizes = group["h"].value_counts()
        assert abs(sizes.get(0, 0) - sizes.get(1, 0)) <= 1


# --- cross-fitting --------------------------------------------------------------------------------


def test_cross_fit_measures_each_half_with_the_other_halfs_threshold():
    report, called = op.cross_fit(SCANS["score"], SCANS["label"], SCANS["patient_id"], target=0.95, seed=11)
    halves = op.patient_halves(SCANS["patient_id"], SCANS["label"], seed=11)
    for fold in report["folds"]:
        measured = halves == fold["measured_on"]
        chosen = ~measured
        assert fold["sensitivity_where_chosen"] >= 0.95
        expected = SCANS["score"].to_numpy()[measured] >= fold["threshold_raw"] - 1e-9
        assert np.array_equal(called[measured], expected)
        positives = int(SCANS["label"].to_numpy()[chosen].sum())
        assert fold["scans_chosen_on"] == int(chosen.sum()) and positives > 0
    pooled = report["pooled"]
    assert pooled["tp"] + pooled["fn"] == int(SCANS["label"].sum()) and pooled["bleeds"] + pooled["no_bleeds"] == len(SCANS)
    assert pooled["tp"] == sum(f["tp"] for f in report["folds"]) and pooled["fp"] == sum(f["fp"] for f in report["folds"])
    assert pooled["sensitivity_ci95"][0] <= pooled["sensitivity"] <= pooled["sensitivity_ci95"][1]


def test_cross_fit_on_separable_scores_is_perfect():
    scans = _scans(separation=50.0)
    report, _ = op.cross_fit(scans["score"], scans["label"], scans["patient_id"])
    assert report["pooled"]["sensitivity"] == 1.0 and report["pooled"]["specificity"] == 1.0


def test_same_split_estimate_is_more_optimistic_than_cross_fitted_on_average():
    gaps = []
    for seed in range(10):
        scans = _scans(n_patients=150, seed=seed)
        same = op.full_split_threshold(scans["score"].to_numpy(), scans["label"].to_numpy(), 0.95)
        report, _ = op.cross_fit(scans["score"], scans["label"], scans["patient_id"], 0.95, seed)
        gaps.append(same["same_split_sensitivity"] - report["pooled"]["sensitivity"])
    assert np.mean(gaps) > 0  # choosing and measuring on the same scans flatters the sensitivity


def test_wilson_known_values():
    low, high = op.wilson(95, 100)
    assert low == pytest.approx(0.8882, abs=1e-4) and high == pytest.approx(0.9785, abs=1e-4)
    assert op.wilson(0, 0) == (None, None)
    rates = op.rates({"tp": 95, "fn": 5, "tn": 80, "fp": 20})
    assert rates["missed_bleeds_per_100_bleeds"] == 5.0 and rates["false_alarms_per_100_no_bleed"] == 20.0


def test_patient_bootstrap_brackets_the_estimate_and_is_reproducible():
    report, called = op.cross_fit(SCANS["score"], SCANS["label"], SCANS["patient_id"])
    a = op.patient_bootstrap(called, SCANS["label"], SCANS["patient_id"], n=400, seed=5)
    b = op.patient_bootstrap(called, SCANS["label"], SCANS["patient_id"], n=400, seed=5)
    assert a == b
    for key in ("sensitivity", "specificity"):
        low, high = a[f"{key}_ci95"]
        assert low <= report["pooled"][key] <= high and high - low < 0.2


def test_repeat_and_sweep_report_spread_and_trade_off():
    spread = op.repeat(SCANS["score"], SCANS["label"], SCANS["patient_id"], 0.95, seeds=range(5))
    assert spread["seeds"] == 5 and spread["sensitivity"]["min"] <= spread["sensitivity"]["median"] <= spread["sensitivity"]["max"]
    rows = op.sweep(SCANS["score"], SCANS["label"], SCANS["patient_id"], targets=(0.9, 0.95, 0.99), seeds=range(3), calibrator=CALIBRATOR)
    raw = [r["threshold_raw"] for r in rows]
    assert raw == sorted(raw, reverse=True)  # higher sensitivity needs a lower threshold
    spec = [r["cross_fitted"]["specificity"] for r in rows]
    assert spec[0] >= spec[-1]
    for r in rows:
        assert r["threshold_calibrated"] == pytest.approx(float(CALIBRATOR.apply(np.array([r["threshold_raw"]]))[0]), abs=1e-5)


# --- the proposal rule ----------------------------------------------------------------------------


def _row(target, median_sens, low, spec=0.9):
    return {"target": target, "threshold_raw": 1 - target, "threshold_calibrated": 1 - target,
            "cross_fitted": {"sensitivity": median_sens, "sensitivity_ci95": [low, 1.0], "specificity": spec, "specificity_ci95": [0.8, 0.95],
                             "missed_bleeds_per_100_bleeds": 0, "false_alarms_per_100_no_bleed": 0},
            "median_over_seeds": {"sensitivity": median_sens, "specificity": spec}, "seeds_below_target": 0}


def test_propose_picks_the_lowest_passing_target_and_a_strict_alternative():
    rows = [_row(0.95, 0.944, 0.92), _row(0.96, 0.953, 0.935), _row(0.97, 0.963, 0.951), _row(0.98, 0.975, 0.96)]
    proposal = op.propose(rows, 0.95, candidates=(0.95, 0.96, 0.97, 0.98))
    assert proposal["proposed"]["rule_target"] == 0.96 and proposal["strict_alternative"]["rule_target"] == 0.97
    none = op.propose([_row(0.95, 0.90, 0.88)], 0.95, candidates=(0.95,))
    assert none["proposed"] is None and none["strict_alternative"] is None


# --- the whole run --------------------------------------------------------------------------------


def test_run_gives_numbers_only_and_checks_calibration():
    slices = _slices(SCANS)
    metrics = op.run(slices, CALIBRATOR, seeds=range(4), targets=(0.9, 0.95, 0.97, 0.99), bootstrap=200)
    assert metrics["scans"] == len(SCANS) and metrics["patients"] == SCANS["patient_id"].nunique()
    assert metrics["bleed_scans"] == int(SCANS["label"].sum()) and 0.5 < metrics["scan_auc"] <= 1
    assert metrics["main"]["pooled"]["bleeds"] == metrics["bleed_scans"] and metrics["repeats"]["seeds"] == 4
    assert metrics["exp6_threshold"]["threshold_calibrated"] == 0.278
    text = json.dumps(metrics)
    assert "ID_" not in text
    with pytest.raises(ValueError):
        op.run(slices, Calibrator(method="topk", k=2, temperature=-1.0, bias=0.0, threshold=0.3, kind="platt"))


def test_scan_table_refuses_a_scan_with_two_patients():
    slices = _slices(SCANS.head(3))
    slices.loc[slices.index[0], "patient_id"] = "ID_other"
    with pytest.raises(ValueError):
        op.scan_table(slices, "topk", 2)


def test_tradeoff_figure_is_written(tmp_path):
    metrics = op.run(_slices(SCANS), CALIBRATOR, seeds=range(3), targets=(0.9, 0.95, 0.97, 0.99), bootstrap=100)
    op.tradeoff_figure(metrics, str(tmp_path / "tradeoff.png"))
    assert (tmp_path / "tradeoff.png").stat().st_size > 0


def test_reasoning_text_is_filled_with_numbers_and_no_ids():
    metrics = op.run(_slices(SCANS), CALIBRATOR, seeds=range(3), targets=(0.9, 0.95, 0.96, 0.97, 0.98, 0.99), bootstrap=100)
    text = op.reasoning_text(metrics, isles_fp_rate=0.031, isles_threshold=0.278)
    assert "ID_" not in text and "{" not in text
    assert "missed bleed" in text and "3.1%" in text and f"{metrics['scans']:,}" in text
    assert op.reasoning_text(metrics).count("ISLES") == 0


def test_one_wilson_for_both_vision_b_checks():
    from strokecopilot.vision.haemorrhage import isles_check as ic
    from strokecopilot.vision.haemorrhage import stats
    assert op.wilson is stats.wilson and ic.wilson is stats.wilson
