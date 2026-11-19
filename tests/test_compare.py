"""Tests for pipeline/compare.py: the flat table per run folder, the pivot (models x source x arm) and the figure."""

import csv
import json
from pathlib import Path

import pytest

from strokecopilot.pipeline import compare
from strokecopilot.pipeline.compare import METRICS, build_table, format_pivot, load_run, pivot_table, plot_table


@pytest.fixture
def exp12_folder(tmp_path):
    """Minimal EXP-12-style run folder."""
    folder = tmp_path / "EXP-12_test"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "experiment": "EXP-12",
        "mode": "fixtures",
        "reasoner": {"model": "claude-sonnet-5-5"},
    }))
    (folder / "metrics.json").write_text(json.dumps({
        "summary": {
            "hallucination_rate": {"value": 0.235, "numerator": 87, "denominator": 371, "ci95": None},
            "critical_hallucination_rate": {"value": None, "numerator": 0, "denominator": 0, "ci95": None},
            "omission_rate": {"value": None, "numerator": 0, "denominator": 0, "ci95": None},
        },
        "per_fixture": [],
    }))
    return folder


def test_load_run(exp12_folder):
    run = load_run(exp12_folder)
    assert run["experiment"] == "EXP-12"
    assert run["model"] == "claude-sonnet-5-5"
    assert run["source"] == "fixtures"
    assert run["arm"] == "with_rulebook"
    assert run["metrics_file"] == "metrics.json" and run["run_folder"] == str(exp12_folder)


def test_build_table_one_row(exp12_folder):
    table = build_table([exp12_folder])
    assert len(table) == 1
    assert table[0]["experiment"] == "EXP-12"
    assert table[0]["hallucination_rate"] == 0.235
    assert table[0]["critical_hallucination_rate"] is None
    assert table[0]["omission_rate"] is None


def test_build_table_two_rows(exp12_folder, tmp_path):
    folder2 = tmp_path / "EXP-13_test"
    folder2.mkdir()
    (folder2 / "config.json").write_text(json.dumps({
        "experiment": "EXP-13", "mode": "scenarios",
        "reasoner": {"model": "claude-sonnet-5-5"},
    }))
    (folder2 / "metrics.json").write_text(json.dumps({
        "summary": {
            "hallucination_rate": {"value": 0.10, "numerator": 10, "denominator": 100, "ci95": [0.05, 0.15]},
            "critical_hallucination_rate": {"value": 0.02, "numerator": 2, "denominator": 100, "ci95": None},
            "omission_rate": {"value": 0.01, "numerator": 1, "denominator": 100, "ci95": None},
        },
        "per_fixture": [],
    }))
    table = build_table([exp12_folder, folder2])
    assert len(table) == 2
    assert table[1]["experiment"] == "EXP-13"
    assert table[1]["hallucination_rate"] == 0.10


def test_plot_table_writes_file(exp12_folder, tmp_path):
    table = build_table([exp12_folder])
    out = tmp_path / "figure.png"
    plot_table(table, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_current_runs_read_the_rescore_and_the_source(tmp_path):
    """A run folder as pipeline/end_to_end.py writes it today: config `source`, per_case, and a rescore beside metrics.json."""
    folder = tmp_path / "EXP-X"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"experiment": "EXP-X", "source": "scenarios", "reasoner": {"model": "m"}}))
    stale = {"summary": {"hallucination_rate": {"value": 0.235, "ci95": None}}, "per_case": []}
    fresh = {"summary": {"hallucination_rate": {"value": 0.054, "ci95": [0.03, 0.08]},
                         "critical_hallucination_rate": {"value": 0.009, "ci95": None},
                         "omission_rate": {"value": 0.0, "ci95": None}}, "per_case": [{"case": "s1"}]}
    (folder / "metrics.json").write_text(json.dumps(stale))
    (folder / "metrics_rescore.json").write_text(json.dumps(fresh))
    (folder / "metrics_phase3_judge_v3.json").write_text(json.dumps(fresh))
    run = load_run(folder)
    assert run["source"] == "scenarios" and run["metrics_file"] == "metrics_rescore.json" and run["per_case"] == [{"case": "s1"}]
    table = build_table([folder, (folder, "metrics.json"), (folder, "metrics_phase3_judge_v3.json")])
    assert [r["hallucination_rate"] for r in table] == [0.054, 0.235, 0.054]
    assert table[0]["critical_hallucination_rate"] == 0.009 and table[1]["metrics_file"] == "metrics.json"


def test_unmeasured_rates_are_not_plotted_as_zero(exp12_folder, tmp_path):
    table = build_table([exp12_folder])
    assert table[0]["critical_hallucination_rate"] is None
    plot_table(table, tmp_path / "f.png")  # draws n/a, does not fail on None


def _run(tmp_path, name, model, arm, value, ci=None, source="scenarios"):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"experiment": name, "source": source, "arm": arm,
                                                    "reasoner": {"model": model}}))
    (folder / "metrics.json").write_text(json.dumps({"summary": {
        "hallucination_rate": {"value": value, "ci95": ci},
        "critical_hallucination_rate": {"value": None, "ci95": None},
        "omission_rate": {"value": 0.0, "ci95": None}}, "per_case": []}))
    return folder


def test_pivot_is_models_by_source_and_arm(tmp_path):
    folders = [_run(tmp_path, "E1", "m-a", "with_rulebook", 0.05, [0.03, 0.08]),
               _run(tmp_path, "E2", "m-a", "without_rulebook", 0.12),
               _run(tmp_path, "E3", "m-b", "with_rulebook", 0.07),
               _run(tmp_path, "E4", "m-b", "without_rulebook", 0.20)]
    table = build_table(folders)
    assert len(table) == 4  # flat: one row per run, what plot_table draws
    pivot = pivot_table(table)
    assert pivot["columns"] == [("scenarios", "with_rulebook"), ("scenarios", "without_rulebook")]
    assert [r["model"] for r in pivot["rows"]] == ["m-a", "m-b"]
    cell = pivot["rows"][0]["cells"][("scenarios", "with_rulebook")]
    assert cell["hallucination_rate"] == 0.05 and cell["hallucination_rate_ci"] == [0.03, 0.08] and cell["experiment"] == "E1"
    assert pivot["rows"][1]["cells"][("scenarios", "without_rulebook")]["hallucination_rate"] == 0.20
    text = format_pivot(pivot)
    assert len(text) == 2 and len(text[0]["cells"]) == 2 * len(METRICS)
    assert text[0]["cells"][("scenarios", "with_rulebook", "hallucination_rate")] == "0.050 [0.030, 0.080]"
    assert text[0]["cells"][("scenarios", "without_rulebook", "hallucination_rate")] == "0.120"
    assert text[0]["cells"][("scenarios", "with_rulebook", "critical_hallucination_rate")] == "n/a"


def test_pivot_marks_a_missing_cell_and_refuses_duplicates(tmp_path):
    folders = [_run(tmp_path, "E1", "m-a", "with_rulebook", 0.05), _run(tmp_path, "E2", "m-b", "without_rulebook", 0.1)]
    text = format_pivot(pivot_table(build_table(folders)))
    assert text[0]["cells"][("scenarios", "without_rulebook", "hallucination_rate")] == "not run"
    again = _run(tmp_path, "E9", "m-a", "with_rulebook", 0.06)
    with pytest.raises(ValueError, match="E1.*E9"):
        pivot_table(build_table(folders + [again]))


# --- C9 (17 Nov 2026): paid-run arms, per-case records, paired comparisons, the report -------------------------

def _cell(root, model, arm, cases, bad=0, fail=(), rulebook=True, source="scenarios", vision=None, repeat_of=None):
    """A finished paid-run cell: config, metrics.json per_case rows and traces with `bad` flagged claims per case
    (of 10), critical when flagged; cases in `fail` are format failures."""
    folder = root / model.replace("/", "_") / arm
    (folder / "traces").mkdir(parents=True)
    paid = {"model": model, "arm": arm, "source": source, "rulebook": rulebook}
    if vision:
        paid["vision"] = vision
    if repeat_of:
        paid["repeat_of"] = repeat_of
    (folder / "config.json").write_text(json.dumps({"source": source, "rulebook": rulebook, "paid_run": paid,
                                                    "reasoner": {"model": model}}))
    rows = []
    for c in cases:
        if c in fail:
            rows.append({"case": c, "error": "format failure: x"})
            continue
        traces = [{"claim_id": f"r[{i}]", "subtype": 4 if i < bad else None, "critical": i < bad} for i in range(10)]
        (folder / "traces" / f"{c}.json").write_text(json.dumps({"claims": [], "traces": traces}))
        rows.append({"case": c, "error": None, "claims": 10, "hallucinated": bad, "omissions": [],
                     "uncertainty_omissions": [], "decisions": {"thrombolysis": [True, True, True]}})
    records = [{"case_id": r["case"], "format_failure": bool(r["error"]), "traces": [] if r["error"] else
                json.loads((folder / "traces" / f"{r['case']}.json").read_text())["traces"],
                "omissions": 0, "uncertainty_omissions": 0, "decisions_correct": 1, "decisions_determined": 1}
               for r in rows]
    from strokecopilot.nlp.verifier import metrics as vm
    (folder / "metrics.json").write_text(json.dumps({"summary": vm.summarise(records, n_boot=50), "per_case": rows}))
    return folder


def test_paid_run_arms_are_told_apart():
    paid = lambda **kw: {"rulebook": kw.pop("rulebook", True), "paid_run": {"source": "real", **kw}}
    assert compare._arm_label(paid(vision="predicted")) == "with_rulebook/predicted_vision"
    assert compare._arm_label(paid(vision="groundtruth")) == "with_rulebook/groundtruth_vision"
    assert compare._arm_label(paid(vision="predicted", rulebook=False)) == "without_rulebook/predicted_vision"
    assert compare._arm_label(paid(vision="predicted", arm="real_predicted_repeat2", repeat_of="real_predicted")) == \
        "with_rulebook/predicted_vision/repeat2"
    assert compare._arm_label({"with_rulebook": False}) == "without_rulebook" and compare._arm_label({}) == "with_rulebook"


def test_case_records_rebuild_the_saved_summary_and_paired_compares_matched_cases(tmp_path):
    cases = [f"scenario-{i:03d}" for i in range(12)]
    a = _cell(tmp_path, "m", "scenarios_rulebook", cases, bad=1, fail={"scenario-000"})
    b = _cell(tmp_path, "m", "scenarios_norulebook", cases[:10], bad=3, rulebook=False)
    records = compare.case_records(a)
    assert len(records) == 12 and records[0]["format_failure"] and not records[1]["format_failure"]
    saved = json.loads((a / "metrics.json").read_text())["summary"]
    from strokecopilot.nlp.verifier import metrics as vm
    rebuilt = vm.summarise(records, n_boot=50)
    assert all(rebuilt[k]["numerator"] == saved[k]["numerator"] for k in ("hallucination_rate", "format_failure_rate"))
    d = compare.paired(a, b)  # only the 10 shared cases, one of them a format failure on side a
    assert d["cases"] == 10 and d["cases_a"] == 12 and d["cases_b"] == 10
    assert d["a"] == pytest.approx(0.1) and d["b"] == pytest.approx(0.3) and d["difference"] == pytest.approx(-0.2)
    assert d["ci95"] is not None and d["ci_a"] is not None
    with pytest.raises(ValueError, match="twice"):
        compare.paired([a, a], b)


def test_c9_report_writes_tables_figures_and_comparisons(tmp_path):
    pytest.importorskip("matplotlib")
    from strokecopilot.pipeline import paid_runs as pr
    fx, sc, real = [f"fx0{i}" for i in range(1, 6)], [f"scenario-{i:03d}" for i in range(12)], [f"case-{i:02d}" for i in range(10)]
    for model, bad in ((pr.SONNET, 1), ("gemma-3-27b", 3)):
        for arm, spec in pr.arms_for(model).items():
            ids = {"fixtures": fx, "scenarios": sc[:6] if spec.get("half") else sc, "real": real}[spec["source"]]
            _cell(tmp_path / "paid", model, arm, ids, bad=bad + (2 if not spec["rulebook"] else 0),
                  rulebook=spec["rulebook"], source=spec["source"], vision=spec.get("vision"), repeat_of=spec.get("repeat_of"))
    report = compare.c9_report(tmp_path / "paid", tmp_path / "out", models=[pr.SONNET, "gemma-3-27b"])
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"table.csv", "report.json", "report.md", "c9_rates.png", "c9_vs_sonnet.png"}
    assert report["headline"][pr.SONNET]["hallucination_rate"]["value"] == pytest.approx(0.1)
    assert report["rulebook"]["gemma-3-27b"]["difference"] == pytest.approx(-0.2) and report["rulebook"]["gemma-3-27b"]["cases"] == 21
    assert report["vs_sonnet"]["gemma-3-27b"]["hallucination_rate"]["difference"] == pytest.approx(0.2)
    assert set(report["sonnet_repeats"]) == {"real_predicted", "real_predicted_repeat2", "real_predicted_repeat3"}
    rows = list(csv.DictReader((tmp_path / "out" / "table.csv").open()))
    assert len(rows) == 9 + 7 and "Gemma 3 27B" in (tmp_path / "out" / "report.md").read_text()


def test_case_records_read_the_traces_that_belong_to_the_scoring(tmp_path):
    cell = _cell(tmp_path, "m", "scenarios_rulebook", ["scenario-001"], bad=3)
    assert sum(t["subtype"] is not None for t in compare.case_records(cell)[0]["traces"]) == 3
    (cell / "traces_rescore").mkdir()
    traces = [{"claim_id": f"r[{i}]", "subtype": 4 if i < 1 else None, "critical": i < 1} for i in range(10)]
    (cell / "traces_rescore" / "scenario-001.json").write_text(json.dumps({"claims": [], "traces": traces}))
    (cell / "metrics_rescore.json").write_text((cell / "metrics.json").read_text())
    assert sum(t["subtype"] is not None for t in compare.case_records(cell)[0]["traces"]) == 1  # the rescore's own
    assert sum(t["subtype"] is not None for t in compare.case_records(cell, "metrics.json")[0]["traces"]) == 3
    from strokecopilot.pipeline import end_to_end as e2e
    assert [e2e.traces_dir_for(n) for n in ("metrics.json", "metrics_rescore.json", "metrics_p20.json")] == \
        ["traces", "traces_rescore", "traces_p20"]
