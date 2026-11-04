"""The rulebook in code, the scenario set, omissions and criticality. docs/HALLUCINATION_METRIC.md §5."""

import json
import random
import re
from pathlib import Path

import pytest

from strokecopilot.config import FIXTURES_DIR
from strokecopilot.contract.validate import validate_vision_output
from strokecopilot.nlp.corpus import ingest
from strokecopilot.nlp.notes.facts import fixture_note
from strokecopilot.nlp.notes.render import render
from strokecopilot.nlp.verifier import rules, scenarios as sc

CORPUS = ingest.load()
EXPECTED = json.loads((Path(FIXTURES_DIR) / "expected_rules.json").read_text())["decisions"]
WORD = {True: "yes", False: "no", None: "undetermined"}


def _fixture(stem):
    note, sheet = fixture_note(stem)
    return json.loads((Path(FIXTURES_DIR) / f"{stem}.json").read_text()), sheet


@pytest.fixture(scope="module")
def scenario_set():
    return sc.generate()


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_rules_reproduce_the_fixture_expectations(stem):
    vision, facts = _fixture(stem)
    verdicts = rules.decide(rules.case_from(vision, facts))
    assert {d: v.value for d, v in verdicts.items()} == {d: WORD[e] for d, e in EXPECTED[stem].items()}


# The rule code checked against the corpus: each threshold the code uses must be stated in the passage it cites.
THRESHOLDS = {
    "AHA2026-4.6.1-within-4.5h": ["4.5"], "AHA2026-4.6.3-wake-up-perfusion": ["9", "4.5"], "AHA2026-T3-extend": ["1.2", "10", "70"],
    "AHA2026-T3-trace3": ["1.8", "15", "70"], "AHA2026-4.6.3-lvo-4.5-24h": ["4.5", "24"], "AHA2026-T8-coagulopathy": ["1.7"],
    "AHA2026-T8-trauma-surgery": ["14"], "AHA2026-4.7.2-0-6h-aspects-3-10": ["6", "3 to 10", "0 or 1"],
    "AHA2026-4.7.2-0-6h-mrs-2": ["6", "2"], "AHA2026-4.7.2-0-6h-mrs-3-4": ["3 or 4"], "AHA2026-4.7.2-0-6h-aspects-0-2": ["0 to 2", "80"],
    "AHA2026-4.7.2-6-24h-aspects-6-10": ["6 to 24", "6 or more", "0 or 1"], "AHA2026-4.7.2-6-24h-aspects-3-5": ["3 to 5", "80"],
    "AHA2026-4.7.2-m2-dominant": ["6"], "AHA2026-4.7.3-basilar": ["24", "10", "6"], "AHA2026-4.7.3-basilar-nihss-6-9": ["6 to 9"],
    "DAWN-inclusion-mismatch-groups": ["80", "10", "21", "31", "20", "51"], "DAWN-inclusion-clinical": ["6 to 24", "0 or 1"],
    "DEFUSE3-inclusion-imaging": ["70", "1.8", "15"], "DEFUSE3-inclusion-clinical": ["18 to 90", "6 to 16", "0 to 2"],
}


@pytest.mark.parametrize("pid, numbers", sorted(THRESHOLDS.items()))
def test_rule_thresholds_are_stated_in_their_passages(pid, numbers):
    text = CORPUS[pid]["text"]
    for n in numbers:
        assert re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", text), (pid, n)


def test_every_rule_id_the_code_uses_exists_and_fits_its_decision(scenario_set):
    for s in scenario_set:
        for decision, verdict in s.expected.items():
            for pid in verdict["rule_ids"]:
                assert pid in CORPUS, pid
                assert CORPUS[pid]["treatment"] in (decision, "both"), (pid, decision)


def test_scenarios_are_valid_deterministic_and_cover_every_route(scenario_set):
    assert len(scenario_set) == sc.N and len({s.id for s in scenario_set}) == sc.N
    for s in scenario_set:
        validate_vision_output(s.vision)
        s.facts.check_against(s.note)
        assert s.vision["case_id"] == s.id and "scenario" in s.id
    assert all(c >= sc.MIN_PER_ROUTE for c in sc.routes(scenario_set).values())
    assert sc.manifest(sc.generate())["sha256"] == sc.manifest(scenario_set)["sha256"]


def test_scenario_notes_never_state_the_expected_answer(scenario_set):
    for s in scenario_set:
        assert "thrombolysis" not in s.note.lower() and "thrombectomy" not in s.note.lower()


def _rec(ivt, evt, evt_ids=("AHA2026-4.7.2-6-24h-aspects-6-10",), uncertainties=()):
    return {"thrombolysis": {"recommend": ivt, "rule_ids": [], "blocking_factors": []},
            "thrombectomy": {"recommend": evt, "rule_ids": list(evt_ids), "blocking_factors": []},
            "rationale": [], "uncertainties": list(uncertainties)}


def test_omissions():
    vision, facts = _fixture("fx03_haemorrhage_present")
    c = rules.case_from(vision, facts)
    found, _ = rules.omissions(_rec(True, True), c, rules.decide(c))
    assert "haemorrhage present, thrombolysis recommended" in found and "haemorrhage present, thrombectomy recommended" in found
    vision, facts = _fixture("fx02_thrombectomy_large_mismatch")  # 9 h, eligible for thrombectomy
    c = rules.case_from(vision, facts)
    assert rules.omissions(_rec(False, True), c, rules.decide(c))[0] == []
    vision, facts = _fixture("fx04_large_core_no_mismatch")  # core 110 mL
    c = rules.case_from(vision, facts)
    assert rules.omissions(_rec(False, True, ["DEFUSE3-inclusion-imaging"]), c, rules.decide(c))[0] == ["core 70 mL or more, thrombectomy recommended on DEFUSE-3 alone"]


def test_uncertainty_omissions():
    vision, facts = _fixture("fx01_thrombolysis_clear")  # M2: dominance unknown
    c = rules.case_from(vision, facts)
    verdicts = rules.decide(c)
    assert rules.omissions(_rec(True, False, uncertainties=["Whether the M2 branch is dominant is not reported."]), c, verdicts)[1] == []
    assert rules.omissions(_rec(True, False), c, verdicts)[1] == ["thrombectomy: m2_dominance not named in uncertainties"]


@pytest.mark.parametrize("stem, subject, value, unit, critical", [
    ("fx05_borderline_onset", "core_ml", "25.0", "mL", True),  # DAWN (core under 21 mL, age 80 or over) no longer met
    ("fx02_thrombectomy_large_mismatch", "aspects", "4", None, True),  # 6-24 h ASPECTS 3-5: undetermined
    ("fx02_thrombectomy_large_mismatch", "nihss", "18", None, False),  # still eligible
    ("fx02_thrombectomy_large_mismatch", "lkw_to_imaging", "3", "h", True),  # within 4.5 h: thrombolysis yes
    ("fx03_haemorrhage_present", "haemorrhage.present", "false", None, True),
])
def test_criticality_by_substitution(stem, subject, value, unit, critical):
    vision, facts = _fixture(stem)
    trace = {"subtype": 1, "substitution": {"subject": subject, "value": value, "unit": unit}}
    assert rules.criticality(trace, {}, vision, facts) is critical


def test_criticality_of_rule_errors_and_citations():
    vision, facts = _fixture("fx02_thrombectomy_large_mismatch")
    deciding = {"subtype": 3}
    assert rules.criticality(deciding, {"declared_sources": [{"type": "passage", "id": "AHA2026-4.7.2-6-24h-aspects-6-10"}]}, vision, facts) is True
    assert rules.criticality(deciding, {"declared_sources": [{"type": "passage", "id": "AHA2026-4.8-aspirin-48h"}]}, vision, facts) is False
    assert rules.criticality({"subtype": 6}, {}, vision, facts) is False
    vision, facts = _fixture("fx01_thrombolysis_clear")  # thrombectomy undetermined, thrombolysis yes: still decidable
    assert rules.criticality({"subtype": 1, "substitution": {"subject": "onset_to_imaging", "value": "5", "unit": "h"}}, {}, vision, facts) is True


def test_render_spans_are_verbatim_and_not_recorded_is_stated():
    note, sheet = render({"age": 70, "sex": "female", "onset": "wake_up", "minutes": 600, "nihss": 9, "mrs": 1,
                          "anticoagulant": False, "antiplatelet": False, "glucose": 6.0, "not_recorded": ["inr"]}, random.Random(0))
    sheet.check_against(note)
    assert sheet.get("inr").type == "not_recorded" and sheet.get("wake_up_stroke").value is True
    c = rules.case_from({"haemorrhage": {"present": False}}, sheet)
    assert c.wake_up and c.sleep_mid_minutes == 600 - 480 / 2 and c.time_basis == "lkw_to_imaging"
