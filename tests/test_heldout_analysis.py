"""DEC-31 D3 (pipeline/heldout_analysis.py) on the five fixtures and hand-made answers: no runs, no real cases."""

import json

import numpy as np
import pytest

from strokecopilot.config import FIXTURES_DIR
from strokecopilot.contract.validate import validate_vision_output
from strokecopilot.nlp.notes.facts import fixture_note
from strokecopilot.nlp.reasoner import run as rr
from strokecopilot.pipeline import heldout_analysis as ha
from strokecopilot.vision.perfusion import confidence as conf

STEMS = rr.fixture_stems()
FX = {s: json.loads((FIXTURES_DIR / f"{s}.json").read_text()) for s in STEMS}
FACTS = {s: fixture_note(s)[1] for s in STEMS}
TRUTH = {s: ha.decide(FX[s], FACTS[s]) for s in STEMS}
ALIASES = ha.aliases_for(STEMS, "D")
FX02, FX01 = "fx02_thrombectomy_large_mismatch", "fx01_thrombolysis_clear"


def answer(lysis, ectomy):
    return {"thrombolysis": lysis, "thrombectomy": ectomy}


def trace(field, flagged):
    return {"field": field, "subtype": 3 if flagged else None}


def test_fixture_truths_are_what_the_rules_say():
    assert TRUTH[FX02] == answer(False, True)
    assert TRUTH[FX01]["thrombectomy"] is None  # M2 dominance unknown: undetermined, excluded


def test_correctness_counts_undetermined_and_missing_answers():
    cell = {s: {"answer": dict(TRUTH[s]) if TRUTH[s]["thrombectomy"] is not None else answer(True, False), "traces": []}
            for s in STEMS}
    cell[FX02] = {"answer": answer(True, True), "traces": []}               # thrombolysis wrong
    cell["fx05_borderline_onset"] = {"answer": None, "traces": None}       # format failure
    out = ha.correctness(cell, TRUTH)
    assert out["cases"] == 5 and out["no_answer"] == 1
    assert out["thrombolysis"] == {**out["thrombolysis"], "correct": 3, "scored": 4, "undetermined_truth": 0}
    assert out["thrombectomy"]["undetermined_truth"] == 1 and out["thrombectomy"]["scored"] == 3
    assert out["thrombectomy"]["correct"] == 3
    assert out["pooled"]["correct"] == 6 and out["pooled"]["scored"] == 7
    assert out["pooled"]["ci95"] is None  # fewer than 10 cases: no interval


def test_correctness_interval_with_enough_cases():
    truth = {f"c{i}": answer(True, False) for i in range(20)}
    cell = {c: {"answer": answer(i % 4 != 0, False)} for i, c in enumerate(sorted(truth))}
    out = ha.correctness(cell, truth)
    lo, hi = out["thrombolysis"]["ci95"]
    assert out["thrombolysis"]["rate"] == 0.75 and lo < 0.75 < hi


def test_error_propagation_separates_vision_and_reasoning_errors():
    predicted = {s: json.loads(json.dumps(FX[s])) for s in STEMS}
    predicted[FX02]["haemorrhage"] = {"present": True, "confidence": 0.4}  # a false bleed flag on the predicted contract
    cell = {s: {"answer": dict(TRUTH[s]) if None not in TRUTH[s].values() else answer(True, False), "traces": []}
            for s in STEMS}
    # FX02: the model follows the false bleed flag and refuses thrombectomy (vision error, flagged in its own field)
    cell[FX02] = {"answer": answer(False, False), "traces": [trace("thrombectomy.rule_ids", True)]}
    # fx04: the contract is right but the model refuses thrombectomy (reasoning error); only the other decision's
    # own field is flagged, so it is not caught
    fx04 = "fx04_large_core_no_mismatch"
    cell[fx04] = {"answer": answer(False, False), "traces": [trace("thrombolysis.blocking_factors", True)]}
    # fx05: the model wrongly gives thrombolysis; a shared rationale claim is flagged: caught, not by its own fields
    fx05 = "fx05_borderline_onset"
    cell[fx05] = {"answer": answer(True, True), "traces": [trace("rationale", True), trace("thrombolysis.rule_ids", False)]}
    out = ha.error_propagation(cell, predicted, FX, FACTS, ALIASES)
    assert out["wrong_decisions"] == 3
    assert out["table"]["vision"] == {"wrong": 1, "caught": 1, "missed": 0, "caught_own_fields": 1}
    assert out["table"]["reasoning"] == {"wrong": 2, "caught": 1, "missed": 1, "caught_own_fields": 0}
    assert out["vision_error_fields"] == {"bleed flag": 1}
    assert out["not_scored"] == {"undetermined truth": 1}
    vision = [r for r in out["examples"] if r["cause"] == "vision"][0]
    assert vision["alias"] == ALIASES[FX02] and vision["rules_on_predicted"] is False
    ha.check_no_ids(out, STEMS)


def test_contract_differences_name_cutoffs_crossed():
    truth, pred = dict(FX[FX02]), dict(FX[FX02])
    pred = {**pred, "core_ml": 40.0, "occlusion_site": "unknown"}
    assert ha.contract_differences(pred, truth) == ["core across 21/31 mL", "occlusion site"]
    assert ha.contract_differences({**pred, "core_ml": None}, truth) == ["occlusion site"]


def test_perturb_rederives_mismatch_and_the_near_empty_warning():
    up = ha.perturb(FX[FX02], 1.2)
    assert up["core_ml"] == 21.6 and up["penumbra_ml"] == FX[FX02]["penumbra_ml"]
    assert up["mismatch_volume_ml"] == round(FX[FX02]["penumbra_ml"] - 21.6, 1)
    validate_vision_output(up)
    tiny = {**FX[FX01], "core_ml": 1.1, "warnings": []}
    tiny.update({k: v for k, v in ha.volumes.contract_volumes(1.1, tiny["penumbra_ml"]).items()})
    down = ha.perturb(tiny, 0.8)
    assert down["core_ml"] == 0.9 and down["warnings"] == [conf.NEAR_EMPTY_WARNING] and down["mismatch_ratio"] is None
    assert ha.perturb(down, 1.2)["warnings"] == []  # 0.9 x 1.2 = 1.08 -> 1.1 mL, above the 1 mL cut-off
    assert FX[FX02]["core_ml"] == 18.0  # the original is untouched


def test_rule_sensitivity_counts_changes():
    out = ha.rule_sensitivity(FX, FACTS)
    assert out["cases"] == 5 and out["x0.8"]["decisions"] == 10
    for key in ("x0.8", "x1.2"):
        r = out[key]
        assert r["changed"] == r["yes_no_flips"] + r["to_undetermined"] + r["from_undetermined"]
    # by hand: which fixture decisions change with the core scaled
    expected = sum(ha.decide(ha.perturb(FX[s], 1.2), FACTS[s])[d] != TRUTH[s][d] for s in STEMS for d in ha.DECISIONS)
    assert out["x1.2"]["changed"] == expected


def test_perturbed_cases_and_model_sensitivity():
    cases = [{"id": s, "vision": FX[s], "note": "", "facts": FACTS[s], "expected": TRUTH[s]} for s in STEMS[:2]]
    extra = ha.perturbed_cases(cases)
    assert len(extra) == 4 and extra[0]["id"] == f"{STEMS[0]}__core_x0.8" and extra[0]["base_id"] == STEMS[0]
    assert extra[1]["vision"]["core_ml"] == round(FX[STEMS[0]]["core_ml"] * 1.2, 1)
    base = {s: {"answer": answer(True, False)} for s in STEMS[:2]}
    pert = {e["id"]: {"answer": answer(True, False)} for e in extra}
    pert[f"{STEMS[1]}__core_x1.2"] = {"answer": answer(False, False)}
    out = ha.model_sensitivity(base, pert)
    assert out["x0.8"] == {"decisions_compared": 4, "changed": 0, "changed_share": 0.0}
    assert out["x1.2"]["changed"] == 1


def test_vision_summary():
    rows = [{"dice": 0.2, "pred_ml": 10.0, "ref_ml": 30.0, "abs_error_ml": 20.0, "crosses_21": True, "crosses_31": False,
             "crosses_51": False, "crosses_70": False},
            {"dice": 0.4, "pred_ml": 50.0, "ref_ml": 45.0, "abs_error_ml": 5.0, "crosses_21": False, "crosses_31": False,
             "crosses_51": False, "crosses_70": False}]
    out = ha.vision_summary(rows, flags=1, scans=2)
    assert out["dice"]["mean"] == 0.3 and out["abs_volume_error_ml"]["mean"] == 12.5
    assert out["wrong_side"] == {"21": 1, "31": 0, "51": 0, "70": 0} and out["signed_volume_error_ml_median"] == -7.5
    assert out["bleed_flags"]["rate"] == 0.5


def test_load_cell_reads_newest_scoring(tmp_path):
    cell = tmp_path / "cell"
    (cell / "outputs").mkdir(parents=True)
    (cell / "traces").mkdir()
    (cell / "traces_rescore").mkdir()
    (cell / "metrics.json").write_text(json.dumps({"per_case": [{"case": "x"}, {"case": "y"}]}))
    (cell / "metrics_rescore.json").write_text(json.dumps({"per_case": [{"case": "x"}, {"case": "y"}]}))
    (cell / "outputs" / "x.json").write_text(json.dumps({d: {"recommend": True} for d in ha.DECISIONS}))
    (cell / "traces" / "x.json").write_text(json.dumps({"traces": [trace("rationale", False)]}))
    (cell / "traces_rescore" / "x.json").write_text(json.dumps({"traces": [trace("rationale", True)]}))
    out = ha.load_cell(cell)
    assert out["x"]["answer"] == answer(True, True) and out["x"]["traces"][0]["subtype"] == 3
    assert out["y"] == {"answer": None, "traces": None}


def test_ids_never_reach_a_summary():
    with pytest.raises(ValueError, match="1 case IDs"):
        ha.check_no_ids({"note": f"see {STEMS[0]}"}, STEMS)
    assert ha.aliases_for(["b", "a"], "H") == {"a": "H-01", "b": "H-02"}
    assert np.isfinite(len(ha.check_no_ids({"ok": 1}, STEMS)))


def test_scenario_sensitivity_is_consistent_and_finds_core_sensitive_decisions():
    out = ha.scenario_sensitivity()
    assert out["cases"] == 150
    for key in ("x0.8", "x1.2"):
        r = out[key]
        assert r["decisions"] == 300 and sum(r["changes"].values()) == r["changed"]
    # the scenarios carry cores near the cut-offs (e.g. 28 and 45 mL, DEFUSE-3 near-miss), so something must change
    assert out["x0.8"]["changed"] + out["x1.2"]["changed"] > 0


def test_figures_draw_from_chart_data(tmp_path):
    pytest.importorskip("matplotlib")
    from strokecopilot.pipeline import figures

    rate = {"value": 0.1, "ci95": [0.08, 0.12], "numerator": 1, "denominator": 10}
    models = list(figures.LABELS)
    c = {"c9": {"rows": [{"model": m, "hallucination_rate": rate, "content_hallucination_rate": rate,
                          "critical_hallucination_rate": rate} for m in models]},
         "rulebook": {"rows": [{"model": m, "with": 0.04, "without": 0.18} for m in models[:4]]},
         "human": {"claims": 100, "kappa": 0.27, "flags_confirmed": [18, 54],
                   "by_verdict": [{"verdict": "citation_error", "n": 21, "agreement": 0.19}, {"verdict": "clean", "n": 46, "agreement": 0.93}]},
         "small": {"rows": [{"model": m, "held_out_rate": rate, "thrombectomy": {"yes_caught": 3, "yes": 10}} for m in models[4:]]},
         "vision_b": {"reliability": [{"mean_prob": 0.1, "observed": 0.1, "count": 5}], "subtypes": [{"subtype": "subdural", "caught": 9, "bleeds": 10}],
                      "scan_ece": 0.01, "scans": 100, "scan_auc": 0.98, "sensitivity": 0.96, "specificity": 0.88},
         "vision_a": {"cases": 129, "wrong_side": {"21": 30, "31": 28}, "dice": {"mean": 0.22}},
         "scenario_sensitivity": {"x0.8": {"changed": 9, "decisions": 300, "changed_share": 0.03, "yes_no_flips": 4},
                                  "x1.2": {"changed": 10, "decisions": 300, "changed_share": 0.033, "yes_no_flips": 0}}}
    names = figures.development_figures(c, tmp_path)
    figures.write_readme(tmp_path, names)
    assert len(names) == 7 and all((tmp_path / f"{n}.png").exists() and (tmp_path / f"{n}.svg").exists() for n in names)
    assert "fig_c9_hallucination_by_model" in (tmp_path / "README.md").read_text()
