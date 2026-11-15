"""Reasoner prompt v3 (EXP-39): v2 plus one rule_ids sentence, from the EXP-38 flag check. No network."""

from pathlib import Path

from strokecopilot.config import PROMPTS_DIR
from strokecopilot.nlp.reasoner import prompt as rp

V3_SENTENCE = ("List only passages about this decision's treatment. When a reason against this treatment is that the "
               "patient meets the other treatment's criteria, list those passages under the other decision, not this one. "
               "Your blocking factors are checked against the passages listed under both decisions.")
V2_SENTENCE = "Your blocking factors are checked only against the passages listed here, under either treatment."


def _read(name):
    return (Path(PROMPTS_DIR) / f"{name}.md").read_text(encoding="utf-8")


def test_v3_is_v2_with_only_the_rule_ids_sentence_changed():
    """So EXP-39 against EXP-38 measures this one change."""
    v2, v3 = _read("reasoner_v2"), _read("reasoner_v3")
    assert v2.count(V2_SENTENCE) == 1 and V2_SENTENCE not in v3
    assert v3 == v2.replace(V2_SENTENCE, V3_SENTENCE)


def test_v3_without_rulebook_is_dec29s_prompt_unchanged():
    """Without a rulebook rule_ids is only OWN-KNOWLEDGE, so the clash v3 fixes cannot arise there (DEC-29)."""
    assert _read("reasoner_v3_norulebook") == _read("reasoner_v2_norulebook")
    assert rp.system_prompt("reasoner_v3", rulebook=False) == rp.system_prompt("reasoner_v2", rulebook=False)


def test_v3_builds_with_the_rulebook():
    system = rp.system_prompt("reasoner_v3")
    assert "{{" not in system and V3_SENTENCE in system.replace("**", "")
