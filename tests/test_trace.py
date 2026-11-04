"""Tracing on fixture fx02: the clean output traces, and one seeded error of each sub-type is caught as that sub-type."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from verifier_cases import (  # noqa: E402
    CLEAN_EXTRACTION, CLEAN_REC, CORPUS, R_CORE, R_EVT, R_IVT, FakeJudge, fake_extractor, operand, passage_source, vision_source,
)

from strokecopilot.config import FIXTURES_DIR  # noqa: E402
from strokecopilot.nlp.notes import facts as nf  # noqa: E402
from strokecopilot.nlp.verifier import claims as cl  # noqa: E402
from strokecopilot.nlp.verifier import trace as tr  # noqa: E402

STEMS = sorted(p.stem for p in (Path(FIXTURES_DIR) / "notes").glob("*.txt"))


def _inputs(stem="fx02_thrombectomy_large_mismatch", judge=None):
    note, sheet = nf.fixture_note(stem)
    vision = json.load(open(Path(FIXTURES_DIR) / f"{stem}.json"))
    return tr.Inputs(vision=vision, note=note, facts=sheet, corpus=CORPUS, judge=judge or FakeJudge())


def _run(rec=CLEAN_REC, extraction=CLEAN_EXTRACTION, inputs=None):
    claims = cl.extract(rec, fake_extractor(extraction))
    traces = tr.trace_all(claims, inputs or _inputs())
    return {t.claim_id: t for t in traces}


def test_every_fixture_has_a_valid_fact_sheet():
    assert len(STEMS) == 5
    for stem in STEMS:
        note, sheet = nf.fixture_note(stem)
        assert sheet.get("nihss") is not None and all(f.span in note for f in sheet)


def test_fact_sheet_rejects_unknown_keys():
    with pytest.raises(ValueError):
        nf.FactSheet([nf.Fact(key="shoe_size", type="number", value=9, span="x")])


@pytest.mark.parametrize("claimed, unit, true, true_unit, ok", [
    ("9", "h", 540, "min", True), ("9", "h", 550, "min", True), ("4", "h", 280, "min", False),
    ("18.0", "mL", 18.0, "mL", True), ("18", "mL", 18.4, "mL", True), ("18.0", "mL", 18.4, "mL", False),
    ("5.3", None, 5.28, None, True), ("96", "%", 0.96, "fraction", True), ("9", "mL", 540, "min", False),
])
def test_number_matching_follows_stated_precision(claimed, unit, true, true_unit, ok):
    assert tr.number_matches(claimed, unit, true, true_unit) is ok


def test_clean_output_traces():
    traces = _run()
    bad = {k: (t.subtype_name, t.reason) for k, t in traces.items() if not t.traceable}
    assert bad == {}
    assert traces["rationale[1].1"].supported_passages == ["TEST-IVT-4.5H"]
    assert not any(t.incomplete for t in traces.values())


def _extraction_with(entry_field, entry, index, **changes):
    extraction = copy.deepcopy(CLEAN_EXTRACTION)
    items = [c for c in extraction["claims"] if c["field"] == entry_field and c["entry"] == entry]
    items[index].update(changes)
    return extraction


def test_1_wrong_number():
    rec = copy.deepcopy(CLEAN_REC)
    rec["rationale"][0]["claim"] = rec["rationale"][0]["claim"].replace("18.0", "28.0")
    traces = _run(rec, _extraction_with("rationale", 0, 0, value="28.0"))
    t = traces["rationale[0].0"]
    assert t.subtype == 1 and t.substitution == {"subject": "core_ml", "value": "28.0", "unit": "mL"}


def test_2_invented_finding_and_null_fields():
    rec = copy.deepcopy(CLEAN_REC)
    rec["rationale"].append({"claim": "INR is 1.9.", "sources": [{"type": "note", "quote": "INR 1.9"}], "derived": False})
    extraction = copy.deepcopy(CLEAN_EXTRACTION)
    extraction["claims"].append({"field": "rationale", "entry": 5, "text": "INR is 1.9.", "kind": "patient", "subject": "inr",
                                 "value": "1.9", "unit": None, "applies_to_patient": False, "comparison": None})
    traces = _run(rec, extraction)
    assert traces["rationale[5].0"].subtype == 2 and traces["rationale[0].0"].traceable
    # fx03 has a null ASPECTS: a value is an invented finding, 0 is a wrong number, "unknown" is supported.
    inputs = _inputs("fx03_haemorrhage_present")
    assert tr.check_vision("aspects", "7", None, inputs.vision)[0] == 2
    assert tr.check_vision("aspects", "0", None, inputs.vision)[0] == 1
    assert tr.check_absence("aspects", inputs)[0] is None
    assert tr.check_vision("large_vessel_occlusion", "true", None, inputs.vision)[0] == 2


def test_3_wrong_rule_and_missing_passage():
    extraction = _extraction_with("rationale", 4, 0, text="Thrombectomy is recommended up to 48 h.")
    assert _run(extraction=extraction)["rationale[4].0"].subtype == 3
    rec = copy.deepcopy(CLEAN_REC)
    rec["rationale"][4]["sources"] = [passage_source("TEST-DOES-NOT-EXIST")]
    assert _run(rec)["rationale[4].0"].subtype == 3


def test_4_misapplied_rule():
    table = copy.deepcopy(FakeJudge().table)
    table[R_EVT] = {"TEST-EVT-6-24": ("supported", False)}
    traces = _run(inputs=_inputs(judge=FakeJudge(table)))
    assert traces["rationale[4].0"].subtype == 4 and traces["rationale[4].0"].content_false


def test_5_faulty_derivation():
    extraction = _extraction_with("rationale", 1, 1, comparison={"left": operand("lkw_to_imaging", "9", "h"), "op": "<", "right": operand(None, "4.5", "h", True)})
    t = _run(extraction=extraction)["rationale[1].1"]
    assert t.subtype == 5 and "540 < 270" in t.reason


def test_6_uncited_rule():
    rec = copy.deepcopy(CLEAN_REC)
    rec["rationale"][4]["sources"] = [vision_source("occlusion_site")]
    extraction = copy.deepcopy(CLEAN_EXTRACTION)
    extraction["claims"].append({"field": "rationale", "entry": 4, "text": "The occlusion is in M1.", "kind": "vision",
                                 "subject": "occlusion_site", "value": "M1", "unit": None, "applies_to_patient": False, "comparison": None})
    inputs = _inputs()
    inputs.retriever = lambda text, k: [("TEST-EVT-6-24", 0.8)]  # uncited, but the retriever finds the rule
    traces = _run(rec, extraction, inputs)
    assert traces["rationale[4].0"].subtype == 6 and not traces["rationale[4].0"].content_false
    assert traces["rationale[4].1"].traceable


def test_7_misattributed_source():
    rec = copy.deepcopy(CLEAN_REC)
    rec["rationale"][2]["sources"] = [vision_source("occlusion_site"), vision_source("core_ml")]  # core_ml cited for NIHSS
    traces = _run(rec)
    assert traces["rationale[2].1"].subtype == 7 and traces["rationale[2].0"].traceable


def test_incomplete_sources_stay_clean():
    rec = copy.deepcopy(CLEAN_REC)
    rec["rationale"][3]["sources"] = [passage_source("TEST-DEFUSE3")]  # the vision field is not declared
    t = _run(rec)["rationale[3].0"]
    assert t.traceable and t.incomplete


def test_absence_claims():
    inputs = _inputs()
    assert tr.check_absence("mass_effect", inputs)[0] is None
    assert tr.check_absence("core_ml", inputs)[0] == 2
    assert tr.check_absence("inr", inputs)[0] == 2  # the note is silent on INR


def test_judge_sees_cited_and_retrieved_passages():
    judge = FakeJudge()
    inputs = _inputs(judge=judge)
    inputs.retriever = lambda text, k: [("TEST-DEFUSE3", 0.9), ("TEST-IVT-4.5H", 0.1)]
    inputs.tau = 0.5
    tr.check_rule(R_EVT, ["TEST-EVT-6-24"], True, inputs)
    assert judge.calls[-1] == (R_EVT, ["TEST-EVT-6-24", "TEST-DEFUSE3"], True)


def test_judge_wrapper_uses_the_schema():
    seen = {}

    def llm(system, user, schema):
        seen.update(system=system, user=json.loads(user), schema=schema)
        return {"verdicts": [{"id": "TEST-EVT-6-24", "verdict": "supported", "scope_fits": True, "reason": "x"},
                             {"id": "NOT-ASKED", "verdict": "supported", "scope_fits": True, "reason": "x"}]}

    judge = tr.Judge(llm)
    out = judge(R_EVT, [{"id": "TEST-EVT-6-24", **CORPUS["TEST-EVT-6-24"]}], {"triage_note": "n"}, True)
    assert out == {"TEST-EVT-6-24": ("supported", True)}
    assert seen["schema"] is tr.JUDGE_SCHEMA and seen["user"]["applies_to_patient"] is True and "scope" in seen["system"].lower()


def test_comparison_rule_part_is_checked_in_code():
    # The comparison's application is the arithmetic; its rule part is only whether the threshold is in a passage.
    judge = FakeJudge()
    traces = _run(inputs=_inputs(judge=judge))
    assert R_IVT not in [text for text, _, _ in judge.calls] and R_CORE not in [text for text, _, _ in judge.calls]
    assert traces["rationale[1].1"].supported_passages == ["TEST-IVT-4.5H"]


def test_wrong_threshold_is_a_wrong_rule_and_a_wrong_operator_a_faulty_derivation():
    wrong = _extraction_with("rationale", 1, 1, comparison={"left": operand("lkw_to_imaging", "9", "h"), "op": ">", "right": operand(None, "6", "h", True)})
    assert _run(extraction=wrong)["rationale[1].1"].subtype == 3
    flipped = _extraction_with("rationale", 1, 1, comparison={"left": operand("lkw_to_imaging", "9", "h"), "op": "<=", "right": operand(None, "4.5", "h", True)})
    assert _run(extraction=flipped)["rationale[1].1"].subtype == 5


def test_threshold_found_only_by_retrieval_is_uncited():
    rec = copy.deepcopy(CLEAN_REC)
    rec["rationale"][3]["sources"] = [vision_source("core_ml")]  # the DEFUSE-3 passage is not cited
    inputs = _inputs()
    inputs.retriever = lambda text, k: [("TEST-DEFUSE3", 0.8), ("TEST-IVT-4.5H", 0.7)]
    assert _run(rec, inputs=inputs)["rationale[3].0"].subtype == 6


def test_judge_v2_scope_covers_rules_that_do_not_apply():
    system = tr.Judge(lambda *a: {"verdicts": []}).system
    assert tr.JUDGE_PROMPT == "judge_v2" and "does not apply" in system and "{{" not in system
