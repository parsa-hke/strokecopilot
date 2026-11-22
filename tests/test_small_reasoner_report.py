"""DEC-30's report (small_reasoner_report.py): the seen/held-out split of the evaluation scenarios, and the report on
fake finished cells. No network, no data."""

import json

import pytest

pytest.importorskip("numpy")

from strokecopilot.nlp.verifier import metrics as vm  # noqa: E402
from strokecopilot.nlp.verifier import scenarios as sc  # noqa: E402
from strokecopilot.pipeline import small_reasoner_report as srr  # noqa: E402

HELD_OUT = ["Beyond 4.5 h and eligible for thrombectomy", "LVO 4.5-24 h without thrombectomy", "INR above 1.7",
            "No occlusion is identified"]  # DEC-30 as amended, recorded in EXP-43's meta.json


def test_the_split_is_dec30s_held_out_routes():
    assert sc.held_out_routes() == HELD_OUT
    routes = srr.scenario_routes()
    assert len(routes) == 150
    assert sum(r["group"] == "held_out" for r in routes.values()) == 51
    for r in routes.values():
        assert (r["group"] == "held_out") == any(x in HELD_OUT for x in r["routes"])


def _cell(root, model, arm, bad_by_case, wrong=()):
    """A finished cell with bad_by_case[c] flagged claims of 10 for each case, and one decision per case, wrong for
    the cases in `wrong`."""
    folder = root / model / arm
    (folder / "traces").mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"rulebook": True, "paid_run": {"model": model, "arm": arm}}))
    rows, records = [], []
    for c, bad in bad_by_case.items():
        traces = [{"claim_id": f"r[{i}]", "subtype": 4 if i < bad else None, "critical": i < bad} for i in range(10)]
        (folder / "traces" / f"{c}.json").write_text(json.dumps({"claims": [], "traces": traces}))
        ok = c not in wrong
        rows.append({"case": c, "error": None, "omissions": [], "uncertainty_omissions": [],
                     "decisions": {"thrombolysis": [ok, True, ok]}})
        records.append({"case_id": c, "traces": traces, "omissions": 0, "uncertainty_omissions": 0,
                        "decisions_correct": int(ok), "decisions_determined": 1})
    (folder / "metrics.json").write_text(json.dumps({"summary": vm.summarise(records, n_boot=50), "per_case": rows}))


def test_report_splits_seen_and_held_out_and_pairs_tuned_with_base(tmp_path):
    pytest.importorskip("matplotlib")
    routes = srr.scenario_routes()
    held = {c for c, r in routes.items() if r["group"] == "held_out"}
    fx, real = [f"fx0{i}" for i in range(1, 6)], [f"case-{i:02d}" for i in range(20)]
    # base: 3 of 10 everywhere; tuned: 1 on seen routes, 4 on held-out ones, 2 on fixtures and real cases, and the
    # held-out decisions wrong; Sonnet (EXP-37): 2 everywhere.
    plan = {("small", srr.BASE): lambda c: 3,
            ("small", srr.TUNED): lambda c: 4 if c in held else (1 if c in routes else 2),
            ("paid", "claude-sonnet-5-5"): lambda c: 2}
    for (root, model), bad in plan.items():
        for arm, ids in (("fixtures_rulebook", fx), ("scenarios_rulebook", sorted(routes)), ("real_predicted", real)):
            _cell(tmp_path / root, model, arm, {c: bad(c) for c in ids}, wrong=held if model == srr.TUNED else ())
    rep = srr.report(tmp_path / "small", tmp_path / "paid", tmp_path / "out", comparators=("claude-sonnet-5-5",))
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"report.json", "report.md", "small_reasoner.png"}
    assert rep["group_sizes"] == {"held_out": 51, "seen": 99}
    d = rep["tuned_minus_base"]
    assert d["held_out"]["hallucination_rate"]["cases"] == 51 and d["seen"]["hallucination_rate"]["cases"] == 99
    assert d["held_out"]["hallucination_rate"]["difference"] == pytest.approx(0.1)
    assert d["seen"]["hallucination_rate"]["difference"] == pytest.approx(-0.2)
    assert d["held_out"]["recommendation_correctness"]["difference"] == pytest.approx(-1.0)
    assert d["all"]["hallucination_rate"]["cases"] == 175
    assert rep["tuned_minus"]["claude-sonnet-5-5"]["seen"]["hallucination_rate"]["difference"] == pytest.approx(-0.1)
    held_dec = rep["decisions"][srr.TUNED]["held_out"]["thrombolysis"]
    assert held_dec == {"yes_caught": 0, "yes": 51, "no_kept": 0, "no": 0}  # the fake expects yes, the tuned one says no
    assert rep["decisions"][srr.BASE]["all"]["thrombolysis"]["yes_caught"] == 175
    inr = rep["per_route"]["INR above 1.7"]
    assert inr["held_out"] and inr["scenarios"] == 9 and inr[srr.TUNED]["recommendation_correctness"]["numerator"] == 0
    text = (tmp_path / "out" / "report.md").read_text()
    assert "Scenarios, held-out routes" in text and "Gemma 3 4B tuned" in text
    # Without the balanced cell the report is EXP-48's: no new key, model or section.
    assert set(rep) == {"small_root", "paid_root", "models", "held_out_routes", "group_sizes", "summary", "tuned_minus_base",
                        "tuned_minus", "base_minus", "per_route", "decisions"}
    assert rep["models"] == [srr.BASE, srr.TUNED, "claude-sonnet-5-5"] and set(inr) == {"held_out", "scenarios", srr.BASE, srr.TUNED}
    assert "balanced" not in text.lower() and "## 7" not in text and text.rstrip().splitlines()[-1].startswith("| No occlusion")


def _three_cells(tmp_path):
    """Fake cells for the base, tuned and balanced models and Sonnet. Flagged claims of 10: base 3 everywhere; tuned 1
    on seen routes, 4 on held-out routes, 2 elsewhere; balanced 2 on scenarios, 1 elsewhere; Sonnet 2. Each case
    expects thrombolysis "yes": the tuned model misses it on held-out routes, the balanced one on five seen scenarios."""
    routes = srr.scenario_routes()
    held = {c for c, r in routes.items() if r["group"] == "held_out"}
    missed = set(sorted(c for c in routes if c not in held)[:5])
    fx, real = [f"fx0{i}" for i in range(1, 6)], [f"case-{i:02d}" for i in range(20)]
    plan = {("small", srr.BASE): (lambda c: 3, ()),
            ("small", srr.TUNED): (lambda c: 4 if c in held else (1 if c in routes else 2), held),
            ("balanced", srr.BALANCED): (lambda c: 2 if c in routes else 1, missed),
            ("paid", "claude-sonnet-5-5"): (lambda c: 2, ())}
    for (root, model), (bad, wrong) in plan.items():
        for arm, ids in (("fixtures_rulebook", fx), ("scenarios_rulebook", sorted(routes)), ("real_predicted", real)):
            _cell(tmp_path / root, model, arm, {c: bad(c) for c in ids}, wrong=wrong)
    return held, missed


def test_report_adds_the_balanced_model_beside_tuned_and_base(tmp_path):
    pytest.importorskip("matplotlib")
    held, missed = _three_cells(tmp_path)
    metas = {}
    for model, exp, yes in ((srr.TUNED, "EXP-43", 246), (srr.BALANCED, "EXP-91", 594)):
        metas[model] = tmp_path / f"{exp}.json"
        metas[model].write_text(json.dumps({"experiment": exp, "n": 1200, "balanced": exp == "EXP-91",
                                            "thrombolysis": {"yes": 300, "no": 400, "undetermined": 500},
                                            "thrombectomy": {"yes": yes, "no": 1200 - yes}}))
    rep = srr.report(tmp_path / "small", tmp_path / "paid", tmp_path / "out", comparators=("claude-sonnet-5-5",),
                     balanced_root=tmp_path / "balanced", training_meta=metas)
    assert rep["models"] == [srr.BASE, srr.TUNED, srr.BALANCED, "claude-sonnet-5-5"]
    assert {g: rep["summary"][srr.BALANCED][g]["hallucination_rate"]["cases"] for g in srr.GROUPS} == \
        {"held_out": 51, "seen": 99, "fixtures": 5, "real": 20, "all": 175}
    vs_tuned, vs_base = rep["balanced_minus_tuned"], rep["balanced_minus_base"]
    assert set(vs_tuned) == set(vs_base) == set(srr.GROUPS) and set(vs_tuned["all"]) == set(srr.PAIRED_METRICS)
    assert vs_tuned["held_out"]["hallucination_rate"]["difference"] == pytest.approx(-0.2)  # 2 against 4 of 10
    assert vs_tuned["seen"]["hallucination_rate"]["difference"] == pytest.approx(0.1)  # 2 against 1
    assert vs_tuned["real"]["hallucination_rate"]["difference"] == pytest.approx(-0.1) and vs_tuned["real"]["hallucination_rate"]["cases"] == 20
    assert vs_base["all"]["hallucination_rate"]["cases"] == 175
    assert vs_base["fixtures"]["hallucination_rate"]["difference"] == pytest.approx(-0.2)  # 1 against 3
    assert vs_tuned["held_out"]["recommendation_correctness"]["difference"] == pytest.approx(1.0)  # tuned misses all 51
    assert vs_tuned["seen"]["recommendation_correctness"]["difference"] == pytest.approx(-5 / 99)
    assert rep["balanced_minus"]["claude-sonnet-5-5"]["seen"]["hallucination_rate"]["difference"] == pytest.approx(0.0)
    dec = rep["decisions"]
    assert dec[srr.BALANCED]["held_out"]["thrombolysis"] == {"yes_caught": 51, "yes": 51, "no_kept": 0, "no": 0}
    assert dec[srr.BALANCED]["seen"]["thrombolysis"]["yes_caught"] == 94 and dec[srr.BALANCED]["all"]["thrombolysis"]["yes_caught"] == 170
    assert dec[srr.TUNED]["held_out"]["thrombolysis"]["yes_caught"] == 0 and dec[srr.BASE]["all"]["thrombolysis"]["yes_caught"] == 175
    inr = rep["per_route"]["INR above 1.7"]
    assert inr[srr.BALANCED]["recommendation_correctness"]["numerator"] == 9 and inr[srr.TUNED]["recommendation_correctness"]["numerator"] == 0
    assert rep["training_answers"][srr.BALANCED]["thrombectomy"] == {"true": 594, "false": 606}
    assert rep["training_answers"][srr.TUNED] == {"experiment": "EXP-43", "n": 1200, "balanced": False,
                                                  "thrombolysis": {"true": 300, "false": 900}, "thrombectomy": {"true": 246, "false": 954}}
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"report.json", "report.md", "small_reasoner.png"}
    text = (tmp_path / "out" / "report.md").read_text()
    assert "## 7. Tuned on the decision-balanced set (post-freeze addition, report only)" in text
    assert "| Gemma 3 4B tuned, balanced set | EXP-91 | 1200 | 300 / 1200 (25.0%) | 594 / 1200 (49.5%) |" in text
    assert "| Scenarios, held-out routes | Gemma 3 4B tuned, balanced set | 51 / 51 | 0 / 0 | 0 / 0 | 0 / 0 |" in text  # 7.4
    assert "| Gemma 3 4B tuned, balanced set | Scenarios, held-out routes | 51 / 51 | 0 / 0 | 0 / 0 | 0 / 0 |" in text  # section 4
    assert sum(line.startswith("| Gemma 3 4B tuned, balanced set |") for line in text.splitlines()) == 1 + 3 + 1  # sections 3, 4, 7.1


def test_the_first_six_sections_do_not_change_when_the_balanced_model_is_added(tmp_path):
    """Sections 1, 2, 5 and 6 are word for word the report without the third cell; 3 and 4 only gain its rows."""
    pytest.importorskip("matplotlib")
    _three_cells(tmp_path)
    kw = {"comparators": ("claude-sonnet-5-5",)}
    two = srr.report(tmp_path / "small", tmp_path / "paid", tmp_path / "two", **kw)
    three = srr.report(tmp_path / "small", tmp_path / "paid", tmp_path / "three", balanced_root=tmp_path / "balanced", **kw)
    for key in ("tuned_minus_base", "tuned_minus", "base_minus", "group_sizes", "held_out_routes"):
        assert two[key] == three[key]
    for m in two["models"]:
        assert two["summary"][m] == three["summary"][m] and two["decisions"][m] == three["decisions"][m]
    assert "training_answers" not in three  # only when the meta files are given
    before = (tmp_path / "two" / "report.md").read_text().splitlines()
    after = (tmp_path / "three" / "report.md").read_text().split("\n## 7.")[0].splitlines()
    name = srr.DISPLAY[srr.BALANCED]
    assert [l for l in after if name not in l] == before
