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
    assert ttests.agrees_with_bootstrap(row, [-0.4, -0.05])
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
