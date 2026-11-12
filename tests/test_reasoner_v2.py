"""Reasoner prompt v2 (Task 2, from the EXP-12 findings): it builds, keeps v1, and works with a fake model. No network."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_reasoner import FX02, FakeLLM, _vision  # noqa: E402

from strokecopilot.config import FIXTURES_DIR, PROMPTS_DIR  # noqa: E402
from strokecopilot.nlp.corpus import ingest  # noqa: E402
from strokecopilot.nlp.notes.facts import fixture_note  # noqa: E402
from strokecopilot.nlp.reasoner import prompt as rp  # noqa: E402
from strokecopilot.nlp.reasoner import run as rr  # noqa: E402

V2 = "reasoner_v2"


def _template(version):
    return (Path(PROMPTS_DIR) / f"{version}.md").read_text(encoding="utf-8")


def test_v2_holds_the_whole_rulebook():
    system = rp.system_prompt(V2)
    assert "{{" not in system
    assert all(f"[{p.id}]" in system for p in ingest.read())
    assert "null` means unknown" in system.replace("**", "")


def test_v1_is_kept_and_v2_only_adds():
    """v1 stays as it was, so the v1 against v2 comparison is fair; v2 keeps every v1 input definition."""
    assert rp.REASONER_PROMPT == "reasoner_v1"
    v1, v2 = _template("reasoner_v1"), _template(V2)
    assert v1 != v2
    inputs_v1 = v1.split("## How to decide")[0]
    assert inputs_v1.strip() in v2


def test_v2_states_each_targeted_fix():
    """One phrase per change in the change note, so a later edit cannot drop a fix unnoticed."""
    text = _template(V2).replace("**", "")
    for phrase in (
        "they are not a source",                          # 7: instructions are not claims
        "Check a rule's conditions before you use it",   # 6: rules applied when conditions don't hold
        "A blocking rule sets a limit",                   # 6b: a failed limit is still cited, not dropped
        "include the passages that set the limits",     # 2: rule_ids cover blocking factors
        "Cite the passage that states the rule or threshold you use",  # 1: right fact, wrong citation
        "Do not add any detail that is not in the quoted words",       # 5: invented note details
        "Copy every number exactly",                      # 8: numbers quoted exactly
        "the ID of the rule it affects in square brackets",            # 3: uncertainties carry an ID
        "Conclusions about the answer",                  # 4: no conclusions as claims
        "Inferences beyond the inputs",                  # 5: no predictions or summaries
    ):
        assert phrase in text, phrase


def test_v2_example_cannot_leak_a_fixture_answer():
    """The worked example uses invented IDs and numbers that appear in no rulebook passage, fixture or note."""
    example = _template(V2).split("## Example")[1].split("## Rulebook")[0]
    ids = set(re.findall(r"PASSAGE-[A-Z]+", _template(V2)))
    assert ids and not ids & {p.id for p in ingest.read()}
    numbers = set(re.findall(r"\d+(?:\.\d+)?", example))
    for stem in rr.fixture_stems():
        vision = json.loads((Path(FIXTURES_DIR) / f"{stem}.json").read_text())
        note, _ = fixture_note(stem)
        seen = {str(v) for v in vision.values() if isinstance(v, (int, float))} | set(re.findall(r"\d+(?:\.\d+)?", note))
        assert not numbers & seen, (stem, numbers & seen)


def test_v2_fake_reply_assembles_and_validates():
    llm = FakeLLM()
    rec, record = rr.recommend(_vision(), "note", llm, version=V2)
    assert record["error"] is None and rec["prompt_version"] == V2
    assert llm.calls[0][0] == rp.system_prompt(V2) and llm.calls[0][2] == rp.api_schema()


def test_v2_uncertainty_with_a_bracketed_id_is_valid_output():
    reply = json.loads(json.dumps(FakeLLM().reply))
    pid = ingest.read()[0].id
    reply["uncertainties"] = [f"ASPECTS is not reported; it is needed for [{pid}]."]
    rec, record = rr.recommend(_vision(), "note", FakeLLM(reply), version=V2)
    assert record["error"] is None and rec["uncertainties"] == reply["uncertainties"]


def test_v2_run_folder_records_the_version(tmp_path):
    metrics = rr.run_fixtures(FakeLLM(), tmp_path, stems=[FX02], version=V2, log=lambda *_: None)
    assert metrics["calls"] == 1 and metrics["format_failures"] == 0
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["prompt_version"] == V2
    assert (tmp_path / "system_prompt.md").read_text(encoding="utf-8") == rp.system_prompt(V2)
