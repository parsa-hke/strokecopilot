"""Phase 5 analyses (DEC-35, DEC-36): exact Shapley values, the rule-engine game and the t-test statistics."""

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from strokecopilot.analysis import rule_shap, shapley, ttests
from strokecopilot.nlp.verifier import rules
from strokecopilot.pipeline import end_to_end


# ------------------------------------------------------------------------------------------- Shapley

def test_additive_game_gives_each_player_its_own_worth():
    worth = [3.0, -1.0, 0.5, 2.0]
    phi, v_all, v_none = shapley.shapley(4, lambda m: sum(w for i, w in enumerate(worth) if m >> i & 1))
    assert np.allclose(phi, worth) and v_all == sum(worth) and v_none == 0


def test_glove_game_known_values_and_efficiency():
    # player 0 holds a left glove, players 1 and 2 right gloves; a pair is worth 1
    phi, v_all, v_none = shapley.shapley(3, lambda m: float(m & 1 and m & 6 != 0))
    assert np.allclose(phi, [2 / 3, 1 / 6, 1 / 6])
    assert math.isclose(phi.sum(), v_all - v_none)


def test_vector_outputs_and_limits():
    phi, _, _ = shapley.shapley(2, lambda m: np.array([m & 1, 2 * (m >> 1 & 1)], dtype=float))
    assert np.allclose(phi, [[1, 0], [0, 2]])
    with pytest.raises(ValueError):
        shapley.coalition_values(17, lambda m: 0.0)
    with pytest.raises(ValueError):
        shapley.shapley_from_values([0.0, 1.0, 2.0])


# ---------------------------------------------------------------------------------- Rule-engine game

def test_players_cover_every_rule_input_once():
    rule_shap.check_players()
    assert set(rule_shap.VISION) <= set(rule_shap.PLAYERS)


def test_masking_keeps_derived_fields_only_with_both_volumes():
    case = rules.Case(core=10.0, penumbra=60.0, ratio=6.0, mismatch=50.0, wake_up=True, minutes=600.0)
    kept = rule_shap.masked(case, {"core", "tmax6", "time"})
    assert (kept.ratio, kept.mismatch, kept.wake_up) == (6.0, 50.0, True)
    gone = rule_shap.masked(case, {"core"})
    assert gone.ratio is None and gone.mismatch is None and gone.penumbra is None and gone.wake_up is False
    assert all(getattr(rule_shap.masked(case, set()), f.name) in (None, False) for f in dataclasses.fields(case))


def test_haemorrhage_fixture_is_decided_by_the_bleed():
    case = next(c for c in end_to_end.load_cases("fixtures") if c["id"] == "fx03_haemorrhage_present")
    r = rule_shap.explain_case(rules.case_from(case["vision"], case["facts"]))
    assert r["decisions"] == {"thrombolysis": "no", "thrombectomy": "no"}
    for d in rule_shap.DECISIONS:
        phi = r["players"][d]
        assert math.isclose(sum(phi.values()), 1 - r["baseline_same"][d], abs_tol=1e-9)  # efficiency
        if not r["baseline_same"][d]:
            assert max(phi, key=phi.get) == "bleed"
        v = r["vision_vs_note"][d]
        assert math.isclose(v["vision"] + v["note"], 1 - r["baseline_same"][d], abs_tol=1e-9)


def test_run_and_summary_on_a_few_scenarios():
    from strokecopilot.nlp.verifier import scenarios
    rows, summary = rule_shap.run(scenarios.generate()[:4])
    assert len(rows) == 4 and summary["thrombectomy"]["all"]["cases"] == 4
    assert set(summary["thrombolysis"]["all"]["mean_phi"]) == set(rule_shap.PLAYERS)


# ------------------------------------------------------------------------------------------- t-tests

def test_paired_matches_scipy_and_reports_its_parts():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 30)
    b = a - rng.normal(0.3, 0.5, 30)
    a_nan = np.append(a, np.nan)
    row = ttests.paired(a_nan, np.append(b, 0.0), "x")
    ref = stats.ttest_rel(a, b)
    assert row["n"] == 30 and row["dropped"] == 1 and row["df"] == 29
    assert math.isclose(row["t"], ref.statistic) and math.isclose(row["p"], ref.pvalue)
    d = a - b
    assert math.isclose(row["effect_size"], d.mean() / d.std(ddof=1))
    lo, hi = row["ci"]
    assert lo < d.mean() < hi and math.isclose(hi - d.mean(), d.mean() - lo)
    assert row["rank_test"]["name"] == "Wilcoxon signed-rank" and 0 <= row["rank_test"]["p"] <= 1


def test_welch_matches_scipy_and_its_df_formula():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 12), rng.normal(1, 3, 20)
    row = ttests.welch(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert math.isclose(row["t"], ref.statistic) and math.isclose(row["p"], ref.pvalue)
    va, vb = a.var(ddof=1) / 12, b.var(ddof=1) / 20
    assert math.isclose(row["df"], (va + vb) ** 2 / (va ** 2 / 11 + vb ** 2 / 19))
    assert row["effect_size"] < 0 and row["n"] == [12, 20]


def test_holm_worked_example_and_nan():
    assert np.allclose(ttests.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    out = ttests.holm([0.01, float("nan"), 0.04])
    assert np.isnan(out[1]) and np.allclose(out[[0, 2]], [0.02, 0.04])


def test_bootstrap_agreement_and_case_rate():
    row = {"ci": [-0.5, -0.1]}
    assert ttests.agrees_with_bootstrap(row, [-0.4, -0.05]) is True
    assert not ttests.agrees_with_bootstrap(row, [-0.4, 0.02])
    assert ttests.case_rate({"hallucinated": 2, "claims": 8}) == 0.25
    assert math.isnan(ttests.case_rate({"hallucinated": 0, "claims": 0}))


def test_figure_readme_line_is_added_once_and_replaced(tmp_path):
    from strokecopilot.analysis import figures
    (tmp_path / "README.md").write_text("# Figures\n\n- **fig_other**: kept\n- **fig_rule_engine_shap**: old\n")
    figures.update_readme(tmp_path, "fig_rule_engine_shap")
    figures.update_readme(tmp_path, "fig_rule_engine_shap")
    text = (tmp_path / "README.md").read_text()
    assert "- **fig_other**: kept" in text and "old" not in text
    assert text.count("fig_rule_engine_shap") == 1


def test_ttest_plan_stops_on_a_mismatch_with_the_reported_bootstrap():
    from strokecopilot.analysis import ttest_plan
    boot = {"difference": -0.149, "ci95": [-0.166, -0.132]}
    ttest_plan._check(boot, dict(boot), "T3a")
    with pytest.raises(ttest_plan.Mismatch):
        ttest_plan._check(boot, {"difference": -0.150, "ci95": [-0.166, -0.132]}, "T3a")


def test_case_rates_leave_format_failures_out():
    rates = ttests.case_rates([{"case_id": "x", "format_failure": True, "traces": []}], "hallucination_rate")
    assert len(rates) == 1 and math.isnan(rates[0])
    assert ttests.case_rates([], "hallucination_rate") == []


def test_forest_labels():
    from strokecopilot.analysis import figures
    assert figures._forest_label({"id": "T5a", "label": "qwen3.5-27b minus Sonnet"}) == "Qwen3.5-27B"
    assert figures._forest_label({"id": "T3a", "label": "claude-sonnet-5-5: with minus without"}) == "Sonnet 5.5"
    assert figures._forest_label({"id": "T6b", "label": "x"}) == "held-out routes (51)"


def test_linear_shap_of_a_logistic_mapping_adds_up():
    from strokecopilot.analysis import confidence_shap as cs
    mapping = {"features": ["a", "b"], "fill": [0.0, 1.0], "mean": [0.5, 1.0], "std": [2.0, 0.5],
               "weights": [1.5, -0.4], "intercept": -0.3}
    X = np.array([[1.0, 1.2], [np.nan, 0.4], [3.0, 2.0]])
    phi, base, logit = cs.shap_values(mapping, X)
    assert np.allclose(phi.sum(axis=1) + base, logit)
    assert np.allclose(phi.mean(axis=0), 0)  # centred on the cases explained
    s = cs.summarise(mapping, phi, np.array([True, False, False]))
    assert math.isclose(sum(v["share_of_attribution"] for v in s["signals"].values()), 1.0)


def test_retrieval_pairs_and_digit_share():
    from strokecopilot.analysis import retrieval_shap as rs
    p = rs.pairs()
    assert len(p) == 12 and all(claim for _, claim in p)
    share, tokens = rs.digit_share(["core ", "< ", "70 ", "mL", ""], [0.1, 0.0, 0.3, 0.1, 0.5])
    assert math.isclose(share, 0.6) and math.isclose(tokens, 0.25)


def test_surrogate_design_and_validation_on_synthetic_claims():
    from strokecopilot.analysis import surrogate_flags as sf
    rng = np.random.default_rng(0)
    rows = []
    for g in range(60):
        for _ in range(10):
            kind = rng.choice(["rule", "vision"])
            rows.append({"group": f"c{g}", "model": "m", "case type": "scenario", "field": "rationale", "kind": kind,
                         "sources": 1, "cites a passage": 0, "cites imaging": 1, "cites the note": 0, "derived": 0,
                         "words": 8, "numbers": 1, "flagged": int(kind == "rule" and rng.random() < 0.9)})
    X, y, groups, cols, owner = sf.design(rows)
    assert X.shape == (600, len(cols)) and set(owner.values()) >= {"kind", "model", "numbers"}
    v = sf.validate(X, y, groups, n_boot=200)
    assert v["auc"] > 0.9 and v["ci95"][0] > 0.5 and v["cases"] == 60
    e = sf.explain(X, y, cols, owner)
    assert next(iter(e["share_of_attribution"])) == "kind"


def test_pixel_helpers():
    from strokecopilot.analysis import visionb_pixel_shap as vp
    assert vp._pool(np.ones((320, 320))).shape == (10, 10)
    rows = [{"shap": {"brain": 0.6, "bone": 0.1, "other": 0.1, "outside": 0.2},
             "shap_positive": {"brain": 0.5, "bone": 0.2, "other": 0.1, "outside": 0.2},
             "gradcam": {"brain": 0.8, "bone": 0.1, "other": 0.05, "outside": 0.05}, "spearman_shap_gradcam": 0.4}] * 3
    s = vp.summarise(rows)
    assert s["slabs"] == 3 and s["slabs_brain_over_bone"] == {"shap": 3, "gradcam": 3}
    assert math.isclose(s["median_share"]["gradcam"]["brain"], 0.8)


def test_reasoner_shap_versions_hide_each_group_and_keep_the_full_case():
    from strokecopilot.analysis import reasoner_shap as rs
    pairs = rs.scenarios_with_specs()
    chosen = rs.choose(pairs)
    assert len(chosen) == 24 and len({s.id for s, _ in chosen}) == 24
    assert [s.id for s, _ in rs.choose(pairs)] == [s.id for s, _ in chosen]  # seeded
    s, spec = chosen[0]
    vs = rs.versions(s, spec)
    assert len(vs) == 64
    values, full = rs.rule_values(vs)
    assert full == {d: s.expected[d]["value"] for d in rs.DECISIONS} and values[-1].tolist() == [1.0, 1.0]
    empty = vs[0]
    assert empty["vision"]["core_ml"] is None and empty["vision"]["occlusion_site"] == "unknown"
    assert empty["vision"]["haemorrhage"] == s.vision["haemorrhage"]  # the bleed stays fixed
    for key in ("age", "nihss", "premorbid_mrs", "inr", "anticoagulant"):
        assert empty["facts"].get(key).type == "not_recorded"
    assert empty["facts"].get("onset_to_imaging") is None and empty["facts"].get("lkw_to_imaging") is None
    phi = rs.shapley_table(values)
    for k, d in enumerate(rs.DECISIONS):
        assert math.isclose(sum(phi[d].values()), values[-1, k] - values[0, k], abs_tol=1e-9)


def test_reasoner_shap_model_values_need_every_answer():
    from strokecopilot.analysis import reasoner_shap as rs
    rec = {"thrombolysis": {"recommend": True}, "thrombectomy": {"recommend": False}}
    other = {"thrombolysis": {"recommend": False}, "thrombectomy": {"recommend": False}}
    values, full = rs.model_values([other] * 63 + [rec])
    assert full == {"thrombolysis": True, "thrombectomy": False} and values[0].tolist() == [0.0, 1.0]
    assert rs.model_values([rec, None] + [rec] * 62) == (None, None)


def test_visiona_masking_keeps_the_crop_region_and_fills_as_dec35_says():
    from strokecopilot.analysis import visiona_channel_shap as vs
    rng = np.random.default_rng(0)
    img = np.zeros((6, 4, 5, 6), dtype=np.float32)
    img[:, 1:3, 1:4, 1:5] = rng.uniform(1, 9, size=(6, 2, 3, 4))
    region = np.any(img != 0, axis=0)
    x, r = vs.masked(img, 0, ncct_mean=40.0)  # every input absent
    assert np.array_equal(r, region) and np.array_equal(np.any(x != 0, axis=0) | (x[4] == 0) & region, region)
    assert np.all(x[0][region] == 40.0) and np.all(x[0][~region] == 0)
    assert np.allclose(x[1][region], img[1][region].mean()) and np.ptp(x[2][region]) == 0
    assert not x[4].any() and not x[5].any()
    full, _ = vs.masked(img, vs.FULL, ncct_mean=40.0)
    assert np.array_equal(full, img)
    assert vs.wrong_side(19.0, 25.0) == 1 and vs.wrong_side(25.0, 29.0) == 0
    assert vs.dice(np.zeros(3), np.zeros(3)) == 1.0


def test_visiona_summary_is_exact_on_an_additive_game():
    from strokecopilot.analysis import visiona_channel_shap as vs
    worth = [5.0, 3.0, 0.0, 1.0, 10.0]
    rows = {}
    for case, ref in (("a", 30.0), ("b", 2.0)):
        for m in range(vs.FULL + 1):
            ml = sum(w for i, w in enumerate(worth) if m >> i & 1)
            rows[(case, m)] = {"case": case, "mask": m, "pred_ml": ml, "ref_ml": ref, "dice": 0.5, "wrong_side": 0}
    s = vs.summarise(rows, recorded_ml={"a": 19.0, "b": 19.0})
    assert s["complete"] == 2 and s["full_coalition_reproduces_recorded"]["max_ml_difference"] == 0.0
    assert np.allclose(list(s["pred_ml"]["mean_abs"].values()), worth)
    assert s["dice"]["share"] is None  # a flat output has nothing to attribute


def test_window_fill_and_summary():
    import pandas as pd

    from strokecopilot.analysis import visionb_window_shap as ws
    img = np.full((2, 4, 4, 3), 200, dtype=np.uint8)
    out = ws.fill_image(img, 0b101, [10.4, 20.6, 30.0])  # brain and bone present
    assert out[..., 0].min() == 200 and np.all(out[..., 1] == 21) and out[..., 2].min() == 200
    assert img[..., 1].min() == 200  # the input is not changed
    rng = np.random.default_rng(0)
    y = rng.random(40) < 0.4
    # additive in the brain window only: brain carries everything
    values = {m: np.where(y, 0.9, 0.05) * (m & 1) + 0.02 for m in range(ws.FULL + 1)}
    s = ws.summarise(values, pd.DataFrame({"label": y}), 0.5)
    assert s["groups"]["bleed"]["mean_abs_phi"]["brain"] > 0.8 and s["groups"]["bleed"]["mean_abs_phi"]["bone"] == 0
    assert s["coalitions"]["1"]["auc"] == 1.0 and s["coalitions"]["0"]["auc"] == 0.5
    assert math.isclose(sum(s["auc_shapley"].values()), 0.5)
