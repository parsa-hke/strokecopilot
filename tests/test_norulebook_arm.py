"""The without-rulebook arm (DEC-29): the no-rulebook prompt, and the verifier scoring rule content only. Fake judges
and fake models; no network."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).parent))
from test_batched import PickyJudge, Staged  # noqa: E402
from test_end_to_end import Fake, retriever  # noqa: E402
from verifier_cases import CORPUS as BASE, FakeJudge, passage_source, vision_source  # noqa: E402

from strokecopilot.nlp.reasoner import prompt as rp  # noqa: E402
from strokecopilot.nlp.verifier import claims as cl  # noqa: E402
from strokecopilot.nlp.verifier import trace as tr  # noqa: E402
from strokecopilot.pipeline import batched  # noqa: E402
from strokecopilot.pipeline import end_to_end as e2e  # noqa: E402

RULE = "DEFUSE-3 required an ischaemic core under 70 mL."
# The test corpus with the trial recorded on its DEFUSE-3 passage, as the real corpus records it (lexical candidates).
CORPUS = {**BASE, "TEST-DEFUSE3": {**BASE["TEST-DEFUSE3"], "source": "DEFUSE3"}}
OWN = passage_source(tr.OWN_KNOWLEDGE)


def _inputs(rulebook, judge):
    return tr.Inputs(vision={"core_ml": 18.0}, note="", facts=[], corpus=CORPUS, judge=judge, retriever=None,
                     rulebook=rulebook)


def _claim(text=RULE, sources=(OWN,), kind="rule", field="rationale", subject=None):
    return cl.Claim(id=f"{field}[0].0", field=field, entry=0, text=text, kind=kind, applies_to_patient=False,
                    declared_sources=list(sources), subject=subject)


def test_no_rulebook_prompt_is_v2_without_the_rulebook():
    with_book = rp.system_prompt("reasoner_v2")
    without = rp.system_prompt("reasoner_v2", rulebook=False)
    assert "OWN-KNOWLEDGE" in without and "{{RULEBOOK}}" not in without and "[DEFUSE3-" not in without
    assert "[DEFUSE3-" in with_book or "DEFUSE3-" in with_book
    # The input descriptions are word for word the same: only the rule source differs.
    inputs = lambda prompt: prompt.split("## Your inputs")[1].split("3. **")[0]
    assert inputs(with_book) == inputs(without)
    with pytest.raises(ValueError, match="no without-rulebook variant"):
        rp.system_prompt("reasoner_v1", rulebook=False)


def test_the_placeholder_is_a_missing_passage_with_the_rulebook_and_ignored_without_it():
    judge = FakeJudge({RULE: {"TEST-DEFUSE3": ("supported", None)}})
    with_book = tr.trace_claim(_claim(), _inputs(True, judge))
    assert with_book.subtype == 3 and "do not exist" in with_book.reason and judge.calls == []
    without = tr.trace_claim(_claim(), _inputs(False, judge))  # named trial: a lexical candidate
    assert without.subtype is None and without.supported_passages == ["TEST-DEFUSE3"]
    assert judge.calls == [(RULE, ["TEST-DEFUSE3"], False)]
    wrong = "DEFUSE-3 required an ischaemic core under 100 mL."
    judge = FakeJudge({wrong: {"TEST-DEFUSE3": ("contradicted", None)}})
    assert tr.trace_claim(_claim(wrong), _inputs(False, judge)).subtype == 3  # wrong content still counts


def test_without_the_rulebook_passage_citations_are_not_charged_but_imaging_ones_are():
    judge = FakeJudge({RULE: {"TEST-DEFUSE3": ("supported", None)}})
    claims = [_claim()]
    with_book = tr.trace_all(claims, _inputs(True, judge))
    assert with_book[0].subtype == 3
    traces = tr.trace_all(claims, _inputs(False, judge))
    assert traces[0].subtype is None  # in the rulebook arm a supported rule citing no real passage would be 6
    # An imaging source that supports nothing in the entry is still charged.
    wrong_field = [_claim(sources=(OWN, vision_source("penumbra_ml")))]
    assert tr.trace_all(wrong_field, _inputs(False, judge))[0].subtype == 7


def test_a_batched_no_rulebook_run_matches_a_live_one(tmp_path):
    cases = e2e.load_cases("fixtures")
    live_seen, batch_seen, server = [], [], {"batches": {}, "ended": set()}
    live = e2e.run_cases(cases, Fake("reasoner"), Fake("extractor"), PickyJudge(live_seen), retriever, tmp_path / "live",
                         log=lambda *_: None, version="reasoner_v2", rulebook=False)
    two = batched.run_batched(cases, Fake("reasoner"), Fake("extractor"), Staged(PickyJudge(batch_seen), "claude-haiku-5-5", server),
                              retriever, tmp_path / "batch", log=lambda *_: None, source="fixtures", version="reasoner_v2",
                              rulebook=False)
    assert sorted(batch_seen) == sorted(live_seen) and two["summary"] == live["summary"]
    for name in ("live", "batch"):
        config = json.loads((tmp_path / name / "config.json").read_text())
        assert config["rulebook"] is False and config["prompt_version"] == "reasoner_v2"
    calls = json.loads((tmp_path / "batch" / "calls" / f"{cases[0]['id']}.json").read_text())
    assert calls["error"] is None
