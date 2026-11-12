"""The three verifier fixes set aside from EXP-12 (Phase 3): rules named under uncertainties with no passage found,
judge prompt v3 for negative, conditional and reason-giving applications, and restatements of the output's own
decision dropped as claims. Fake judges only; no network."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from verifier_cases import CLEAN_REC, CORPUS, FakeJudge  # noqa: E402

from strokecopilot.nlp.corpus import ingest  # noqa: E402
from strokecopilot.nlp.verifier import claims as cl  # noqa: E402
from strokecopilot.nlp.verifier import trace as tr  # noqa: E402


def _inputs(judge, corpus=CORPUS, decision_passages=()):
    return tr.Inputs(vision={}, note="", facts=[], corpus=corpus, judge=judge, retriever=None,
                     decision_passages=list(decision_passages))


def _rule(text, field="uncertainties", applies=True, sources=(), origin="llm"):
    return cl.Claim(id=f"{field}[0].0", field=field, entry=0, text=text, kind="rule", applies_to_patient=applies,
                    declared_sources=list(sources), origin=origin)


# --- fix 1: candidates for rules named under uncertainties ------------------------------------------------------

def test_lexical_candidates_on_the_real_corpus():
    corpus = ingest.load()
    got = tr.lexical_candidates("The rulebook requires absence of mass effect only for ASPECTS 3 to 5 or 0 to 2", corpus)
    assert set(got) == {"AHA2026-4.7.2-6-24h-aspects-3-5", "AHA2026-4.7.2-0-6h-aspects-0-2"}
    trials = tr.lexical_candidates("The DEFUSE-3 and DAWN criteria are not met", corpus)
    assert trials and all(pid.startswith(("DEFUSE3-", "DAWN-")) for pid in trials)
    assert {"DEFUSE3-inclusion-imaging", "DAWN-inclusion-mismatch-groups"} <= set(trials)
    assert tr.lexical_candidates("Table 3 criteria require a mismatch ratio above 1.8", corpus) == ["AHA2026-T3-extend", "AHA2026-T3-trace3"]
    # A time range is not an ASPECTS range, and a claim with no name gets nothing.
    assert tr.lexical_candidates("Thrombectomy 6 to 24 hours after onset with NIHSS 6 or more", corpus) == []
    assert tr.lexical_candidates("Blood pressure must be lowered first", corpus) == []


def test_an_uncertainty_rule_is_judged_against_its_outputs_decision_passages():
    text = "Mass effect does not change the late-window thrombectomy recommendation"  # names no trial, table or range
    judge = FakeJudge({text: {"TEST-EVT-6-24": ("supported", True)}})
    claim = _rule(text)
    before = tr.trace_claim(claim, _inputs(judge))
    assert before.subtype == 3 and before.reason == "No passage is a candidate."
    after = tr.trace_claim(claim, _inputs(judge, decision_passages=["TEST-EXT-IVT", "TEST-EVT-6-24"]))
    assert after.subtype is None and after.supported_passages == ["TEST-EVT-6-24"]
    assert judge.calls[-1][1] == ["TEST-EXT-IVT", "TEST-EVT-6-24"]


def test_decision_passages_are_candidates_only_for_uncertainties():
    judge = FakeJudge({})
    tr.trace_claim(_rule("Some rule", field="rationale"), _inputs(judge, decision_passages=["TEST-EVT-6-24"]))
    assert judge.calls == []  # a rationale entry citing nothing still has no candidate


def test_a_false_rule_stays_a_wrong_rule_with_more_candidates():
    judge = FakeJudge({})  # every candidate: not addressed
    trace = tr.trace_claim(_rule("DEFUSE-3 enrolled patients up to 30 hours"), _inputs(judge, decision_passages=["TEST-EVT-6-24"]))
    assert trace.subtype == 3 and "TEST-DEFUSE3" not in judge.calls[-1][1]  # synthetic corpus: no DEFUSE3 source tag
    corpus = {**CORPUS, "TEST-DEFUSE3": {**CORPUS["TEST-DEFUSE3"], "source": "DEFUSE3"}}
    trace = tr.trace_claim(_rule("DEFUSE-3 enrolled patients up to 30 hours"), _inputs(FakeJudge({}), corpus))
    assert trace.subtype == 3


def test_decision_passages_helper():
    assert tr.decision_passages(CLEAN_REC) == ["TEST-EXT-IVT", "TEST-EVT-6-24"]


# --- fix 2: judge prompt v3 --------------------------------------------------------------------------------------

def test_judge_v3_is_the_default_and_keeps_v2s_rules():
    v3 = tr.Judge(lambda *a: {"verdicts": []})
    v2 = tr.Judge(lambda *a: {"verdicts": []}, "judge_v2")
    assert tr.JUDGE_PROMPT == "judge_v3" and v3.version == "judge_v3"  # DEC-23, 12 Nov 2026 (amends DEC-20)
    for phrase in ("not needed, or already met", "Criteria that are not met", "Conditional and counterfactual",
                   "A reason for a decision", "an unknown is not a failed criterion"):
        assert phrase in v3.system and phrase not in v2.system
    assert v2.system.split("## Scope")[0] == v3.system.split("## Scope")[0]  # only the scope section changes


def test_the_judge_version_is_part_of_the_cache_key(tmp_path):
    from strokecopilot.nlp.verifier import run as vr
    cache = vr.JsonCache(tmp_path / "c.json")
    v2 = vr.CachedJudge(None, cache, offline=True, version="judge_v2")
    v3 = vr.CachedJudge(None, cache, offline=True, version="judge_v3")
    assert v2.key("claim", "P", {}, True) != v3.key("claim", "P", {}, True)


# --- fix 3: restatements of the output's own decision ------------------------------------------------------------

@pytest.mark.parametrize("text,dropped", [
    ("The rulebook does not support thrombolysis for this patient", True),
    ("Thrombolysis is not recommended.", True),
    ("Thrombectomy is recommended for this patient", True),
    ("IV thrombolysis was not indicated", True),
    ("Thrombectomy is not recommended", False),  # contradicts the output's own decision: still a claim
    ("Thrombolysis is not recommended because the patient is eligible for thrombectomy", False),  # gives a reason
    ("Thrombectomy is recommended within 6 to 24 hours with ASPECTS 6 to 10", False),  # states a rule
])
def test_restatements(text, dropped):
    rec = copy.deepcopy(CLEAN_REC)  # thrombolysis no, thrombectomy yes
    claim = _rule(text, field="rationale")
    assert (cl.restatement_of(claim, rec) is not None) == dropped


def test_only_extracted_rule_claims_can_be_restatements():
    rec = copy.deepcopy(CLEAN_REC)
    assert cl.restatement_of(_rule("Thrombolysis is not recommended", origin="regex"), rec) is None
    claim = cl.Claim(id="x", field="rationale", entry=0, text="Thrombolysis is not recommended", kind="patient")
    assert cl.restatement_of(claim, rec) is None


def test_extract_and_refresh_leave_restatements_out():
    from strokecopilot.nlp.verifier import run as vr
    rec = copy.deepcopy(CLEAN_REC)
    rec["rationale"].append({"claim": "The rulebook does not support thrombolysis for this patient.", "sources": [], "derived": False})
    last = len(rec["rationale"]) - 1
    reply = {"claims": [{"field": "rationale", "entry": last, "text": "The rulebook does not support thrombolysis for this patient",
                         "kind": "rule", "subject": None, "value": None, "unit": None, "applies_to_patient": True, "comparison": None},
                        {"field": "rationale", "entry": 0, "text": "Thrombectomy is recommended in the 6-24 h window",
                         "kind": "rule", "subject": None, "value": None, "unit": None, "applies_to_patient": True, "comparison": None}]}
    claims = cl.extract(rec, lambda system, user, schema: reply)
    texts = [c.text for c in claims if c.origin == "llm"]
    assert texts == ["Thrombectomy is recommended in the 6-24 h window"]
    saved = [c.to_dict() for c in cl.extract(rec, lambda *a: {"claims": []})] + [
        cl.Claim(id=f"rationale[{last}].0", field="rationale", entry=last, text=reply["claims"][0]["text"], kind="rule").to_dict()]
    assert all(c.text != reply["claims"][0]["text"] for c in vr.refresh_claims(rec, saved))


def test_metric_doc_lists_the_restatement_rule():
    doc = (Path(__file__).parents[1] / "docs" / "HALLUCINATION_METRIC.md").read_text()
    assert "restatement of the output's own decision" in doc


# --- the EXP-23 recheck, with a stub judge on copies of small fake runs -----------------------------------------

def test_recheck_runs_both_prompts_and_writes_a_comparison(tmp_path):
    pytest.importorskip("numpy")
    from strokecopilot.nlp.verifier import recheck as rc
    from strokecopilot.nlp.verifier import run as vr
    from strokecopilot.pipeline import end_to_end as e2e
    from test_end_to_end import Fake, retriever

    exp12 = tmp_path / "exp12"
    e2e.run_fixtures(Fake("reasoner"), Fake("extractor"), Fake("judge"), retriever, exp12, log=lambda *_: None)
    exp11 = tmp_path / "exp11"
    vr.run_seeded(Fake("extractor"), {rc.JUDGE_NAME: Fake("judge")}, retriever, exp11, stems=["fx02_thrombectomy_large_mismatch"],
                  log=lambda *_: None)
    made = []
    out = rc.recheck(exp11, exp12, tmp_path / "out", lambda: made.append(Fake("judge")) or made[-1], retriever=retriever,
                     log=lambda *_: None)
    assert set(out["by_prompt"]) == {"judge_v2", "judge_v3"} and len(made) == 4
    other = out["by_prompt"]["judge_v2" if tr.JUDGE_PROMPT == "judge_v3" else "judge_v3"]  # not the runs' own prompt
    assert other["exp11"]["judge_calls"] > 0 and other["exp12"]["judge_calls"] > 0  # a new prompt version re-judges
    assert out["by_prompt"][tr.JUDGE_PROMPT]["exp12"]["judge_calls"] == 0  # the runs' own prompt: all cached
    assert (exp12 / "flagged_phase3_judge_v3.json").exists() and (exp11 / "metrics_phase3_judge_v2.json").exists()
    assert json.loads((tmp_path / "out" / "metrics.json").read_text())["tau"] == 0.6
