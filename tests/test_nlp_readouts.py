"""The 902 report's post hoc read-outs (EXP-88 to EXP-90): known answers on made-up rows, no data needed."""

import pytest

from strokecopilot.analysis import nlp_readouts as nr


def _rows():
    """Two models, two strata each. Model a: clean 0 of 4 (size 900), flagged 2 of 4 (size 100). Model b: clean 1 of
    4 (size 600), flagged 4 of 4 (size 400)."""
    rows, i = [], 0
    spec = {("a", "clean"): (4, 0, 900), ("a", "content_error"): (4, 2, 100),
            ("b", "clean"): (4, 1, 600), ("b", "content_error"): (4, 4, 400)}
    for (model, verdict), (n, errors, size) in spec.items():
        for j in range(n):
            i += 1
            error = j < errors
            rows.append({"sample_id": f"S{i:03d}", "model": model, "verdict": verdict, "stratum_size": size,
                         "label_a": "error" if error else "no_error",
                         "label_b": "error" if error and j % 2 == 0 else "no_error",
                         "consensus": "error" if error else "no_error",
                         "note_a": "claim is true but not cited" if error and j % 2 else "", "note_b": "",
                         "rescored": verdict})
    return rows


def test_stratified_rate_weights_each_stratum_by_its_size():
    strata = nr.strata_counts(_rows())
    a = nr.stratified_rate([s for (m, _), s in strata.items() if m == "a"], n_boot=2000)
    b = nr.stratified_rate([s for (m, _), s in strata.items() if m == "b"], n_boot=2000)
    assert a["value"] == pytest.approx((900 * 0 + 100 * 0.5) / 1000)
    assert b["value"] == pytest.approx((600 * 0.25 + 400 * 1.0) / 1000)
    assert a["claims"] == 1000 and a["sampled"] == 8
    for r in (a, b):
        assert r["bootstrap_ci95"][0] <= r["value"] <= r["bootstrap_ci95"][1]


def test_jeffreys_interval_keeps_the_uncertainty_of_a_zero_count_stratum():
    """Model a's clean stratum shows 0 of 4: the bootstrap cannot move it, the Jeffreys interval can."""
    strata = nr.strata_counts(_rows())
    a = nr.stratified_rate([s for (m, _), s in strata.items() if m == "a"], n_boot=4000)
    assert a["bootstrap_ci95"][1] <= 0.1 + 1e-9  # only the small flagged stratum varies
    assert a["jeffreys_ci95"][1] > 0.3  # 0 of 4 on 900 claims leaves a lot of room


def test_human_check_reports_rates_differences_precision_and_kappa():
    out = nr.human_check(_rows(), n_boot=2000)
    assert out["human_error_rate"]["pooled"]["value"] == pytest.approx((50 + 150 + 400) / 2000)
    assert "_boot" not in out["human_error_rate"]["pooled"]
    assert out["by_rescored_verdict"] == {"clean": {"human_errors": 1, "claims": 8},
                                          "content_error": {"human_errors": 6, "claims": 8}}
    k = out["kappa"]
    assert k["claims"] == 16 and set(k["by_stratum"]) == {"clean", "content_error"}
    assert k["by_stratum"]["clean"]["disagreements"] == 0
    d = out["disagreements"]
    assert d["n"] == sum(r["label_a"] != r["label_b"] for r in _rows())
    assert 0 < d["note_names_a_citation"] <= d["n"]


def test_minus_sonnet_is_reported_when_sonnet_is_in_the_sample():
    rows = _rows()
    for r in rows:
        r["model"] = "claude-sonnet-5-5" if r["model"] == "a" else "gemma-3-27b"
    out = nr.human_check(rows, n_boot=2000)
    diff = out["minus_sonnet"]["gemma-3-27b"]
    assert diff["value"] == pytest.approx(0.55 - 0.05)
    assert diff["bootstrap_ci95"][0] <= diff["value"] <= diff["bootstrap_ci95"][1]


def _record(case, flags, failure=False):
    """flags: one entry per claim; None clean, 'content' false content, 'citation' citation only."""
    traces = [{"subtype": None if f is None else (2 if f == "content" else 6), "content_false": f == "content"}
              for f in flags]
    return {"case_id": case, "format_failure": failure, "traces": [] if failure else traces}


def test_a_format_failure_counts_against_the_answered_share():
    records = [_record("c1", [None, None]), _record("c2", [None, "content"]), _record("c3", ["citation"]),
               _record("c4", [], failure=True)]
    out = nr.answered_without_false_content(records, n_boot=500)
    assert (out["numerator"], out["denominator"], out["valid_answers"]) == (2, 4, 3)
    assert out["value"] == 0.5


def test_subgroup_differences_finds_the_largest_gap_and_skips_small_groups():
    records = {"m": [_record(f"y{i}", ["content", None]) for i in range(12)]       # young: 50%
               + [_record(f"o{i}", [None, None, None, "content"]) for i in range(12)]  # old: 25%
               + [_record(f"s{i}", ["content"]) for i in range(3)]                   # too small a group
               + [_record("f1", [], failure=True)]}
    groups = {**{f"y{i}": {"age band": "under 80"} for i in range(12)},
              **{f"o{i}": {"age band": "80 and over"} for i in range(12)},
              **{f"s{i}": {"age band": "not recorded"} for i in range(3)}, "f1": {"age band": "under 80"}}
    out = nr.subgroup_differences(records, groups, n_boot=500)
    assert out["comparisons"] == 1
    big = out["largest"]
    assert {big["a"], big["b"]} == {"80 and over", "under 80"}
    assert abs(big["difference"]) == pytest.approx(0.25)
    assert out["interval_excludes_zero"] == 1  # every case in a group is identical, so the interval has no width


@pytest.mark.parametrize("error, cause", [
    ("format failure: recommendation schema violation:\n  <root>: Additional properties are not allowed ('$defs', "
     "'additionalProperties' were unexpected)", "the output schema echoed back"),
    ("format failure: the reply is a JSON list, not an object", "a list, not an object"),
    ("no valid JSON: model returned no content (finish_reason 'stop')", "no answer returned"),
    ("format failure: recommendation schema violation:\n  thrombolysis: 'recommend' is a required property",
     "another schema violation"),
])
def test_failure_cause_reads_the_recorded_error(error, cause):
    assert nr.failure_cause(error) == cause


def test_format_failure_causes_counts_only_format_failures():
    rows = [{"case": "a"}, {"case": "b", "error": "format failure: the reply is a JSON list, not an object"},
            {"case": "c", "error": "verifier failed: timeout"}]
    out = nr.format_failure_causes(rows)
    assert out == {"cases": 3, "format_failures": 1, "causes": {"a list, not an object": 1}}
