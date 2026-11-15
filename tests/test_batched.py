"""Two-pass paid runs through the Message Batches API (DEC-24, amended): the batch client with a fake Anthropic client,
and the two-pass runner with the end-to-end fake models. No network."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).parent))
from test_end_to_end import Fake, retriever  # noqa: E402

from strokecopilot.nlp.reasoner import providers  # noqa: E402
from strokecopilot.nlp.reasoner.providers import ClaudeBatch, RefusalError  # noqa: E402
from strokecopilot.pipeline import batched  # noqa: E402
from strokecopilot.pipeline import end_to_end as e2e  # noqa: E402


class FakeBatches:
    """messages.batches with create / retrieve / results; replies keyed by custom ID."""

    def __init__(self, replies, polls_before_end=2):
        self.replies, self.polls, self.created = replies, polls_before_end, []

    def create(self, requests):
        self.created.append(requests)
        return NS(id=f"batch_{len(self.created)}")

    def retrieve(self, batch_id):
        self.polls -= 1
        return NS(processing_status="ended" if self.polls <= 0 else "in_progress")

    def results(self, batch_id):
        usage = NS(input_tokens=10, output_tokens=100, cache_read_input_tokens=5, cache_creation_input_tokens=0)
        for req in self.created[-1]:
            kind, payload = self.replies[req["custom_id"]]
            if kind == "errored":
                yield NS(custom_id=req["custom_id"], result=NS(type="errored"))
                continue
            message = NS(stop_reason=kind, usage=usage, content=[NS(type="text", text=payload)])
            yield NS(custom_id=req["custom_id"], result=NS(type="succeeded", message=message))


def _client(replies):
    return NS(messages=NS(batches=FakeBatches(replies)))


def test_claude_batch_sends_the_claudejson_request_and_sorts_results():
    client = _client({"a": ("end_turn", '{"x": 1}'), "b": ("refusal", ""), "c": ("errored", None), "d": ("max_tokens", "{")})
    batch = ClaudeBatch("claude-sonnet-5-5", client=client, sleep=lambda s: None)
    reqs = {k: ("system text", f"user {k}", {"type": "object"}) for k in "abcd"}
    out = batch.run(reqs, log=lambda *_: None)
    assert out["a"] == {"reply": {"x": 1}} and out["b"]["refusal"] and "errored" in out["c"]["error"] and "max_tokens" in out["d"]["error"]
    sent = client.messages.batches.created[0]
    assert {r["custom_id"] for r in sent} == set("abcd")
    params = sent[0]["params"]
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"} and params["output_config"]["effort"] == "medium"
    assert params["output_config"]["format"]["type"] == "json_schema" and batch.batch_ids == ["batch_1"]
    assert batch.usage["requests"] == 3 and batch.usage["output_tokens"] == 300  # the errored one has no usage
    assert batch.run({}) == {} and ClaudeBatch.PRICE_FACTOR == 0.5


def test_prefilled_answers_from_gathered_results():
    key = batched.request_key("s", "u")
    llm = batched.Prefilled({key: {"reply": {"ok": True}}}, NS(model="m", usage={}))
    assert llm("s", "u", {}) == {"ok": True} and llm.price_factor == 1.0
    with pytest.raises(RuntimeError, match="No answer"):
        llm("s", "other", {})
    refused = batched.Prefilled({key: {"error": "declined", "refusal": True}}, NS(model="m", usage={}))
    with pytest.raises(RefusalError):
        refused("s", "u", {})


def test_two_pass_run_matches_a_direct_run(tmp_path):
    cases = e2e.load_cases("fixtures")
    direct = e2e.run_cases(cases, Fake("reasoner"), Fake("extractor"), Fake("judge"), retriever, tmp_path / "direct", log=lambda *_: None)
    two = batched.run_batched(cases, Fake("reasoner"), Fake("extractor"), Fake("judge"), retriever, tmp_path / "two",
                              log=lambda *_: None, source="fixtures")
    assert two["summary"]["claims"] == direct["summary"]["claims"]
    assert two["summary"]["hallucination_rate"] == direct["summary"]["hallucination_rate"]
    config = json.loads((tmp_path / "two" / "config.json").read_text())
    assert config["batched"] == {"reasoner": False, "extractor": False, "batch_ids": []}


class FakeBatch:
    """A batched llm built on a Fake: run() answers every request, counts batches, and prices like a batch."""

    PRICE_FACTOR = 0.5

    def __init__(self, role, model):
        self.inner, self.model, self.batches, self.batch_ids = Fake(role), model, [], []
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    def describe(self):
        return {"model": self.model, "batch": True}

    def run(self, requests, log=print):
        self.batches.append(len(requests))
        self.batch_ids.append(f"batch_{len(self.batches)}")
        self.usage["requests"] += len(requests)
        self.usage["output_tokens"] += 1_000_000 * len(requests)
        return {k: {"reply": self.inner(*req)} for k, req in requests.items()}

    def __call__(self, *a):
        raise AssertionError("a batched llm is never called directly")


def test_batched_run_sends_one_batch_per_pass_and_halves_the_cost(tmp_path):
    cases = e2e.load_cases("fixtures")
    reasoner, extractor = FakeBatch("reasoner", "claude-sonnet-5-5"), FakeBatch("extractor", "claude-sonnet-5-5")
    metrics = batched.run_batched(cases, reasoner, extractor, Fake("judge"), retriever, tmp_path, log=lambda *_: None, source="fixtures")
    assert reasoner.batches == [5] and extractor.batches == [5] and metrics["summary"]["cases"] == 5
    full = providers.cost_usd("claude-sonnet-5-5", extractor.usage)
    assert metrics["cost_usd_estimate"]["extractor"] == pytest.approx(full * 0.5)
    again = batched.run_batched(cases, FakeBatch("reasoner", "claude-sonnet-5-5"), extractor, Fake("judge"), retriever, tmp_path,
                                log=lambda *_: None, source="fixtures")
    assert extractor.batches == [5]  # every extraction was cached: no second batch
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["batched"]["reasoner"] and config["batched"]["extractor"] and again["summary"]["cases"] == 5
