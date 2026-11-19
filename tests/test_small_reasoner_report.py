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
                     "decisions": {"thrombolysis": [True, True, ok]}})
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
    inr = rep["per_route"]["INR above 1.7"]
    assert inr["held_out"] and inr["scenarios"] == 9 and inr[srr.TUNED]["recommendation_correctness"]["numerator"] == 0
    text = (tmp_path / "out" / "report.md").read_text()
    assert "Scenarios, held-out routes" in text and "Gemma 3 4B tuned" in text
