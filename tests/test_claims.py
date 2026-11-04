"""Claim extraction: regex numbers, citation claims and the LLM path with a fake model."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from verifier_cases import CLEAN_EXTRACTION, CLEAN_REC, fake_extractor  # noqa: E402

from strokecopilot.contract.validate import validate_recommendation  # noqa: E402
from strokecopilot.nlp.verifier import claims as cl  # noqa: E402


def test_clean_rec_is_valid():
    validate_recommendation(CLEAN_REC)


def test_numbers_keep_precision_and_skip_ids_and_sections():
    found = cl.numbers("Core 18.0 mL and NIHSS 16 per AHA2026-4.7.2-rec2 (§4.6.1), within 4.5 h; COR 2a; 9h.")
    assert [(v, u) for v, u, _ in found] == [("18.0", "mL"), ("16", None), ("4.5", "h"), ("9", "h")]
    assert cl.numbers("Section 4.6.1 applies to 2 trials.") == []
    assert cl.numbers("ASPECTS 6-10 and mRS 0–1 within 6 - 24 h") == []


def test_regex_claims_cover_what_the_llm_missed():
    claims = cl.extract(CLEAN_REC, fake_extractor(CLEAN_EXTRACTION))
    assert [c for c in claims if c.origin == "regex"] == []  # every number is covered
    missing = copy.deepcopy(CLEAN_EXTRACTION)
    missing["claims"] = [c for c in missing["claims"] if c["subject"] != "penumbra_ml"]
    regex = [c for c in cl.extract(CLEAN_REC, fake_extractor(missing)) if c.origin == "regex"]
    assert [(c.value, c.unit, c.field, c.entry) for c in regex] == [("95.0", "mL", "rationale", 0)]
    assert regex[0].declared_sources == CLEAN_REC["rationale"][0]["sources"]


def test_extract_points_back_to_entries():
    llm = fake_extractor(CLEAN_EXTRACTION)
    claims = cl.extract(CLEAN_REC, llm)
    assert len(claims) == len(CLEAN_EXTRACTION["claims"]) + 2  # plus one citation claim per rule_ids entry
    by_id = {c.id: c for c in claims}
    assert by_id["rationale[0].1"].subject == "penumbra_ml" and by_id["rationale[1].1"].derived
    assert by_id["thrombectomy.rule_ids[0]"].kind == "citation" and by_id["thrombectomy.rule_ids[0]"].value == "TEST-EVT-6-24"
    blocking = by_id["thrombolysis.blocking_factors[0].0"]
    # blocking factors rest on both decisions' rule_ids, own decision first
    assert blocking.declared_sources == [{"type": "passage", "id": "TEST-EXT-IVT"}, {"type": "passage", "id": "TEST-EVT-6-24"}]
    system, user, schema = llm.calls[0]
    assert "{{" not in system and "`core_ml`" in system and "`lkw_to_imaging`" in system and "`mass_effect`" in system
    assert "18.0" in user and "16" in user and schema is cl.EXTRACTOR_SCHEMA
    assert "Pre-stroke mRS" not in user  # the note never reaches the extractor


def test_unknown_entries_and_subjects():
    reply = {"claims": [
        dict(CLEAN_EXTRACTION["claims"][0], entry=99),
        dict(CLEAN_EXTRACTION["claims"][0], subject="pupil_size", kind="patient"),
    ]}
    claims = [c for c in cl.extract(CLEAN_REC, fake_extractor(reply)) if c.origin == "llm"]
    assert len(claims) == 1 and claims[0].subject == "other:pupil_size"


def test_schema_requires_every_field():
    item = cl.EXTRACTOR_SCHEMA["properties"]["claims"]["items"]
    assert set(item["required"]) == set(item["properties"]) and item["additionalProperties"] is False
