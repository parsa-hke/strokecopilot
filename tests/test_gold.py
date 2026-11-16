"""Gold answers for the fine-tuned reasoner (Item 4): every scenario's gold passes the code checks. No network."""

import json
import re

import pytest

from strokecopilot.nlp.corpus import ingest
from strokecopilot.nlp.reasoner import gold
from strokecopilot.nlp.verifier import rules
from strokecopilot.nlp.verifier import scenarios as sc
from strokecopilot.pipeline import end_to_end as e2e

CORPUS = ingest.load()


@pytest.fixture(scope="module")
def scenarios():
    return sc.generate()


def test_hm_writes_times_as_the_note_does():
    assert gold.hm(360) == "6 h 0 min" and gold.hm(280) == "4 h 40 min" and gold.hm(1100) == "18 h 20 min"


def test_every_scenario_gets_a_gold_answer_that_passes_the_checks(scenarios):
    for s in scenarios:
        reply = gold.gold_answer(s.vision, s.facts, CORPUS)
        assert gold.check(s.vision, s.facts, s.note, reply, CORPUS) == [], s.id


def test_gold_is_deterministic_and_leaves_its_inputs_alone(scenarios):
    s = scenarios[0]
    before = json.dumps(s.vision, sort_keys=True)
    assert gold.gold_answer(s.vision, s.facts, CORPUS) == gold.gold_answer(s.vision, s.facts, CORPUS)
    assert json.dumps(s.vision, sort_keys=True) == before


def test_gold_follows_prompt_v3(scenarios):
    for s in scenarios:
        reply = gold.gold_answer(s.vision, s.facts, CORPUS)
        verdicts = s.expected
        texts = [e["claim"] for e in reply["rationale"]]
        assert len(texts) == len(set(texts)), s.id  # a fact is stated once
        for decision in ("thrombolysis", "thrombectomy"):
            assert reply[decision]["recommend"] == (verdicts[decision]["value"] == rules.YES)
            for pid in reply[decision]["rule_ids"]:  # own treatment only (v3)
                assert CORPUS[pid]["treatment"] in (decision, "both") and not pid.startswith("[")
        for entry in reply["rationale"]:
            for source in entry["sources"]:
                if source["type"] == "vision":
                    assert source["field"] in s.vision or source["field"].startswith("haemorrhage.")
        for u in reply["uncertainties"]:  # each names a real passage in square brackets
            ids = re.findall(r"\[([^\]]+)\]", u)
            assert ids and all(i in CORPUS for i in ids), u
        for text in texts:  # imaging numbers are copied as vision_output.json writes them
            for field, unit in (("core_ml", " mL"), ("mismatch_volume_ml", " mL"), ("penumbra_ml", " mL")):
                label = {"core_ml": "Core volume", "mismatch_volume_ml": "Mismatch volume", "penumbra_ml": "Hypoperfused volume"}[field]
                if text.startswith(label + " "):
                    assert text.startswith(f"{label} {json.dumps(s.vision[field])}{unit} "), text


def test_fixture_gold_matches_the_expected_decisions():
    expected = json.load(open("fixtures/expected_rules.json"))["decisions"]
    for case in e2e.load_cases("fixtures"):
        reply = gold.gold_answer(case["vision"], case["facts"], CORPUS)
        assert gold.check(case["vision"], case["facts"], case["note"], reply, CORPUS) == []
        for decision, want in expected[case["id"]].items():
            if want is not None:
                assert reply[decision]["recommend"] == want, (case["id"], decision)


def test_fx02_gold_reads_as_designed():
    case = {c["id"]: c for c in e2e.load_cases("fixtures")}["fx02_thrombectomy_large_mismatch"]
    reply = gold.gold_answer(case["vision"], case["facts"], CORPUS)
    assert reply["thrombectomy"] == {"recommend": True, "rule_ids": ["AHA2026-4.7.2-6-24h-aspects-6-10"], "blocking_factors": []}
    assert "AHA2026-4.7.2-6-24h-aspects-6-10" not in reply["thrombolysis"]["rule_ids"]  # v3: own treatment only
    assert reply["thrombolysis"]["blocking_factors"] == ["Time from last known well to imaging 9 h 0 min is beyond 4.5 hours.",
                                                         "The patient meets the criteria for thrombectomy."]
    entry = next(e for e in reply["rationale"] if e["claim"] == "ASPECTS 8 is 6 or more.")
    assert entry["sources"] == [{"type": "vision", "field": "aspects"},
                                {"type": "passage", "id": "AHA2026-4.7.2-6-24h-aspects-6-10"}] and entry["derived"]


def test_check_catches_a_broken_gold(scenarios):
    s = next(x for x in scenarios if x.expected["thrombolysis"]["value"] == rules.YES)
    reply = gold.gold_answer(s.vision, s.facts, CORPUS)
    reply["thrombolysis"]["recommend"] = False
    reply["rationale"][0]["sources"].append({"type": "note", "quote": "not in the note"})
    reply["thrombectomy"]["rule_ids"].append("AHA2026-T8-ich")
    problems = gold.check(s.vision, s.facts, s.note, reply, CORPUS)
    assert any("does not match the verdict" in p for p in problems)
    assert any("not verbatim" in p for p in problems)
    assert any("is about thrombolysis" in p for p in problems)
