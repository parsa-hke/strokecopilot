"""Verifier run plumbing with fake models: caches, resume, the τ replay and scoring. No network."""

import json

import pytest

from strokecopilot.nlp.verifier import run as vr
from strokecopilot.nlp.verifier import seeded_errors as se

FX02 = "fx02_thrombectomy_large_mismatch"


class FakeLLM:
    """Counts calls; the extractor makes one rule claim per rationale entry, the judge supports everything."""

    def __init__(self, role):
        self.role, self.calls, self.model = role, 0, f"fake-{role}"
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    def __call__(self, system, user, schema):
        self.calls += 1
        self.usage["requests"] += 1
        payload = json.loads(user[user.index("["):] if self.role == "extractor" else user)
        if self.role == "extractor":
            return {"claims": [{"field": e["field"], "entry": e["entry"], "text": e["text"], "kind": "rule", "subject": None,
                                "value": None, "unit": None, "applies_to_patient": False, "comparison": None}
                               for e in payload if e["field"] == "rationale"]}
        return {"verdicts": [{"id": p["id"], "verdict": "supported", "scope_fits": True, "reason": "fake"} for p in payload["passages"]]}


def retriever(text, k):
    return [("AHA2026-T8-ich", 0.9), ("AHA2026-4.6.1-within-4.5h", 0.6), ("DAWN-inclusion-clinical", 0.3)][:k]


def _run(tmp_path, extractor=None, judge=None):
    extractor, judge = extractor or FakeLLM("extractor"), judge or FakeLLM("judge")
    metrics = vr.run_seeded(extractor, {"fake": judge}, retriever, tmp_path, stems=[FX02], log=lambda *_: None)
    return metrics, extractor, judge


def test_run_writes_traces_and_scores_every_tau(tmp_path):
    metrics, extractor, judge = _run(tmp_path)
    assert metrics["outputs"] == 15  # gold plus 14 seeds
    # the extractor sees only the text: the 2 source-swap seeds and the rule-ID seed reuse the gold extraction
    assert extractor.calls == 12 and metrics["extractor"]["calls"] == 12
    fake = metrics["judges"]["fake"]
    assert fake["failures"] == [] and set(fake["by_tau"]) == {str(t) for t in vr.TAU_GRID}
    assert (tmp_path / "traces" / "fake" / f"{FX02[:4]}-gold.json").exists()
    # the judge sees each unchanged claim once, not once per seeded copy
    assert 0 < judge.calls < 15 * 5


def test_second_run_resumes_from_the_caches(tmp_path):
    first, _, _ = _run(tmp_path)
    second, extractor, judge = _run(tmp_path)
    assert extractor.calls == 0 and judge.calls == 0
    assert second["judges"]["fake"]["by_tau"] == first["judges"]["fake"]["by_tau"]


def test_higher_tau_drops_retrieved_candidates(tmp_path):
    metrics, _, _ = _run(tmp_path)
    by = metrics["judges"]["fake"]["by_tau"]
    # the fake judge supports every candidate, so uncited rules are found only through retrieval
    assert by["-1.0"]["recall"]["uncited_rule"]["n"] == 2
    # the retriever's scores are 0.9, 0.6 and 0.3: at τ 0.85 only one retrieved passage is still a candidate
    assert by["-1.0"] != by["0.85"] or by["-1.0"]["recall"] == by["0.85"]["recall"]


def test_offline_judge_and_retriever_refuse_to_call(tmp_path):
    judge = vr.CachedJudge(None, vr.JsonCache(tmp_path / "j.json"), offline=True)
    with pytest.raises(KeyError):
        judge("claim", [{"id": "X", "text": "t"}], {}, False)
    ret = vr.CachedRetriever(None, vr.JsonCache(tmp_path / "r.json"), offline=True)
    with pytest.raises(KeyError):
        ret("claim", 5)


def test_json_cache_persists(tmp_path):
    cache = vr.JsonCache(tmp_path / "c.json")
    cache.put("a", [1, 2])
    assert vr.JsonCache(tmp_path / "c.json").get("a") == [1, 2]


def test_cached_judge_only_sends_unjudged_passages(tmp_path):
    seen = []

    def judge(text, passages, case, applied):
        seen.append([p["id"] for p in passages])
        return {p["id"]: ("supported", True) for p in passages}

    cj = vr.CachedJudge(judge, vr.JsonCache(tmp_path / "j.json"))
    cj("c", [{"id": "A", "text": ""}], {"x": 1}, True)
    out = cj("c", [{"id": "A", "text": ""}, {"id": "B", "text": ""}], {"x": 1}, True)
    assert seen == [["A"], ["B"]] and out == {"A": ("supported", True), "B": ("supported", True)}


def test_judge_cache_key_includes_the_prompt_version(tmp_path):
    cache = vr.JsonCache(tmp_path / "j.json")
    a = vr.CachedJudge(None, cache, version="judge_v1").key("c", "P", {}, True)
    b = vr.CachedJudge(None, cache, version="judge_v2").key("c", "P", {}, True)
    assert a != b
