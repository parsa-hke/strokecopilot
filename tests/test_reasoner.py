"""Reasoner prompt, API schema, output assembly and the fixture run, with fake models. No network."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from verifier_cases import CLEAN_REC  # noqa: E402

from strokecopilot.config import FIXTURES_DIR  # noqa: E402
from strokecopilot.nlp.corpus import ingest  # noqa: E402
from strokecopilot.nlp.reasoner import prompt as rp  # noqa: E402
from strokecopilot.nlp.reasoner import run as rr  # noqa: E402
from strokecopilot.nlp.reasoner.providers import RefusalError, cost_usd  # noqa: E402

REPLY = {k: v for k, v in CLEAN_REC.items() if k not in rp.RUN_FIELDS}
FX02 = "fx02_thrombectomy_large_mismatch"


def _vision(stem=FX02):
    return json.loads((Path(FIXTURES_DIR) / f"{stem}.json").read_text())


class FakeLLM:
    def __init__(self, reply=REPLY, raise_=None):
        self.reply, self.raise_, self.calls = reply, raise_, []
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    def describe(self):
        return {"provider": "fake", "model": "fake-model"}

    def __call__(self, system, user, schema):
        self.calls.append((system, user, schema))
        self.usage["requests"] += 1
        self.usage["input_tokens"] += 1000
        self.usage["output_tokens"] += 100
        if self.raise_:
            raise self.raise_
        return copy.deepcopy(self.reply)


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_system_prompt_holds_the_whole_rulebook():
    system = rp.system_prompt()
    assert "{{" not in system
    assert all(f"[{p.id}]" in system for p in ingest.read())
    assert "null` means unknown" in system.replace("**", "")


def test_user_message_is_the_case_only():
    note, _ = __import__("strokecopilot.nlp.notes.facts", fromlist=["fixture_note"]).fixture_note(FX02)
    msg = json.loads(rp.user_message(_vision(), note))
    assert set(msg) == {"vision", "note"} and msg["note"] == note
    assert not set(rp.VISION_HIDDEN) & set(msg["vision"]) and msg["vision"]["core_ml"] == 18.0


def test_api_schema_is_api_safe():
    schema = rp.api_schema()
    assert not set(rp.RUN_FIELDS) & set(schema["properties"]) and not set(rp.RUN_FIELDS) & set(schema["required"])
    for node in _walk(schema):
        assert not set(node) & {"minLength", "minItems", "pattern", "oneOf", "$schema", "$id"}
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False


def test_recommend_assembles_and_validates():
    llm = FakeLLM()
    rec, record = rr.recommend(_vision(), "note", llm)
    assert record["error"] is None and rec["model"] == "fake-model" and rec["prompt_version"] == "reasoner_v1"
    assert rec["case_id"] == "fixture-fx02" and rec["thrombectomy"]["recommend"] is True
    assert llm.calls[0][2] == rp.api_schema()


def test_format_failure_and_refusal_are_recorded_not_repaired():
    bad = copy.deepcopy(REPLY)
    bad["rationale"][0]["sources"] = []
    rec, record = rr.recommend(_vision(), "note", FakeLLM(bad))
    assert rec is None and record["error"].startswith("format failure") and record["reply"] == bad
    rec, record = rr.recommend(_vision(), "note", FakeLLM(raise_=RefusalError("no")))
    assert rec is None and record["error"].startswith("refusal")


def test_run_fixtures_writes_the_run_folder(tmp_path):
    llm = FakeLLM()
    metrics = rr.run_fixtures(llm, tmp_path, stems=[FX02, "fx03_haemorrhage_present"], repeats=2, log=lambda *_: None)
    assert metrics["calls"] == 4 and metrics["format_failures"] == 0
    # the fake always says thrombolysis no, thrombectomy yes: 2 of 2 right for fx02, 1 of 2 for fx03, per repeat
    assert metrics["decisions_agreeing"] == 6 and metrics["decisions_judged"] == 8
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir())[0] == f"{FX02}_r1.json"
    call = json.loads((tmp_path / "calls" / f"{FX02}_r2.json").read_text())
    assert call["usage"]["requests"] == 1 and call["reply"]["thrombectomy"]["recommend"] is True
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["corpus_sha256"] == ingest.version() and config["repeats"] == 2
    assert (tmp_path / "system_prompt.md").read_text(encoding="utf-8") == rp.system_prompt()


def test_expected_decisions_cover_every_fixture():
    expected = json.loads(rr.EXPECTED.read_text())["decisions"]
    assert set(expected) == set(rr.fixture_stems())
    assert all(set(v) == {"thrombolysis", "thrombectomy"} for v in expected.values())


def test_cost_estimate():
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(3.0)
    assert cost_usd("unknown-model", usage) is None


def test_a_one_object_list_is_unwrapped_and_counted():
    """Parsa, 16 Nov 2026 (EXP-37): Qwen through OpenRouter wraps answers in a list, [ {...} ]. A list holding exactly
    one object is scored as that object, for every model, and the call record says so."""
    rec, record = rr.recommend(_vision(), "note", FakeLLM([copy.deepcopy(REPLY)]))
    assert rec is not None and record["error"] is None and record["unwrapped_list"] is True
    assert rec["thrombolysis"] == REPLY["thrombolysis"]
    _, plain = rr.recommend(_vision(), "note", FakeLLM())
    assert "unwrapped_list" not in plain


def test_a_reply_that_is_not_an_object_is_a_format_failure():
    """Anything else that is not an object (an empty list, two objects, text, a number) is a format failure."""
    for reply in ([], [copy.deepcopy(REPLY), copy.deepcopy(REPLY)], ["text"], "text", 3):
        rec, record = rr.recommend(_vision(), "note", FakeLLM(reply))
        assert rec is None and record["error"].startswith("format failure: the reply is a JSON")
        assert record["reply"] == reply  # the raw reply is kept for the calls/ record


def test_bracketed_passage_ids_are_stripped_and_counted():
    """Parsa, 16 Nov 2026 (EXP-37): MedGemma copied the rulebook's [ID] display into its citations. One pair of
    brackets is removed from rule_ids and passage sources, for every model, and counted; uncertainties are untouched."""
    reply = copy.deepcopy(REPLY)
    pid = reply["thrombolysis"]["rule_ids"][0] if reply["thrombolysis"]["rule_ids"] else ingest.read()[0].id
    reply["thrombolysis"]["rule_ids"] = [f"[{pid}]", pid]
    reply["rationale"] = [{"claim": "x", "sources": [{"type": "passage", "id": f"[{pid}]"}, {"type": "note", "quote": "[a]"}],
                           "derived": False}]
    reply["uncertainties"] = [f"ASPECTS is unknown; it matters for [{pid}]."]
    rec, record = rr.recommend(_vision(), "note", FakeLLM(reply))
    assert rec["thrombolysis"]["rule_ids"] == [pid, pid] and rec["rationale"][0]["sources"][0]["id"] == pid
    assert rec["rationale"][0]["sources"][1]["quote"] == "[a]" and rec["uncertainties"] == reply["uncertainties"]
    assert record["stripped_ids"] == 2
    assert rr._bare_id("[[x]]") == "[[x]]" and rr._bare_id("[]") == "[]" and rr._bare_id("A") == "A"
    _, plain = rr.recommend(_vision(), "note", FakeLLM())
    assert "stripped_ids" not in plain
