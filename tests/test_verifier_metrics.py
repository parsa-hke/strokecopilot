"""Hallucination metrics: pooled rates, format failures, bootstrap intervals and paired differences."""

import pytest

pytest.importorskip("numpy")

from strokecopilot.nlp.verifier import metrics as vm  # noqa: E402


def _trace(subtype=None, critical=None, incomplete=False):
    return {"subtype": subtype, "critical": critical, "incomplete": incomplete}


def _case(i, subtypes, critical=None, **extra):
    return {"case_id": f"c{i}", "traces": [_trace(s, critical) for s in subtypes], "omissions": 0, "uncertainty_omissions": 0,
            "decisions_correct": 2, "decisions_determined": 2, **extra}


def test_rates_pool_claims_and_skip_format_failures():
    cases = [_case(0, [None, None, 1, 7]), _case(1, [None, 6]), {"case_id": "c2", "format_failure": True, "traces": []}]
    out = vm.summarise(cases)
    assert out["claims"] == 6
    assert out["hallucination_rate"]["value"] == pytest.approx(3 / 6)
    assert out["content_hallucination_rate"]["value"] == pytest.approx(1 / 6)
    assert out["source_attribution_accuracy"]["value"] == pytest.approx(3 / 5)
    assert out["format_failure_rate"]["value"] == pytest.approx(1 / 3)
    assert out["mean_per_case_hallucination_rate"] == pytest.approx((2 / 4 + 1 / 2) / 2)
    assert out["subtype_breakdown"]["wrong_number"]["count"] == 1 and out["hallucination_rate"]["ci95"] is None
    assert out["critical_hallucination_rate"]["value"] is None  # criticality not determined yet


def test_unknown_rulebook_fields_give_none():
    cases = [{"case_id": "a", "traces": [_trace()], "omissions": None}]
    out = vm.summarise(cases)
    assert out["omission_rate"]["value"] is None and out["recommendation_correctness"]["value"] is None


def test_bootstrap_interval_with_enough_cases():
    cases = [_case(i, [None] * 9 + [1 if i % 2 else None], critical=False) for i in range(20)]
    out = vm.summarise(cases, n_boot=2000)
    lo, hi = out["hallucination_rate"]["ci95"]
    assert lo <= out["hallucination_rate"]["value"] <= hi and lo < hi
    assert out["critical_hallucination_rate"]["value"] == 0.0


def test_paired_difference():
    a = [_case(i, [1, None]) for i in range(12)]
    b = [_case(i, [None, None]) for i in range(12)]
    out = vm.paired_difference(a, b, n_boot=1000)
    assert out["difference"] == pytest.approx(0.5) and out["ci95"] == [pytest.approx(0.5), pytest.approx(0.5)]
    with pytest.raises(ValueError):
        vm.paired_difference(a, b[:5])
