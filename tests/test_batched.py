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
    assert config["batched"] == {"reasoner": False, "extractor": False, "judge": False, "reasoner_workers": 1, "batch_ids": []}


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


class Crashing(Fake):
    """A direct reasoner whose server goes away after `ok` calls: the error is not one gather() records, so the run stops."""

    def __init__(self, ok):
        super().__init__("reasoner")
        self.ok = ok

    def __call__(self, *req):
        if self.calls >= self.ok:
            raise ConnectionError("server gone")
        return super().__call__(*req)


class Direct:
    """A direct llm that records each user message; it fails one request and the server goes away after `crash_at`."""

    def __init__(self, model, calls, crash_at=None):
        self.model, self.calls, self.crash_at = model, calls, crash_at

    def __call__(self, system, user, schema):
        if len(self.calls) == self.crash_at:
            raise ConnectionError("server gone")  # not an error gather() records: the run stops
        self.calls.append(user)
        if user == "u c":
            raise RuntimeError("bad")  # a format failure, recorded and saved like an answer
        return {"u": user}


def test_gather_saves_direct_answers_and_resumes_after_a_crash(tmp_path):
    path, calls = tmp_path / batched.ANSWERS_FILE, []
    reqs = {k: ("s", f"u {k}", {}) for k in "abcde"}
    with pytest.raises(ConnectionError):
        batched.gather(reqs, Direct("m", calls, crash_at=3), log=lambda *_: None, answers_path=path)
    assert calls == ["u a", "u b", "u c"] and len(path.read_text().splitlines()) == 3
    path.write_text(path.read_text() + '{"key": "d", "mod')  # a line cut off mid-write is skipped
    logs = []
    out = batched.gather(reqs, Direct("m", calls), log=logs.append, answers_path=path)
    assert calls == ["u a", "u b", "u c", "u d", "u e"]  # only the rest was asked
    assert out["a"] == {"reply": {"u": "u a"}} and out["c"] == {"error": "bad", "refusal": False} and set(out) == set(reqs)
    assert "3 of 5" in logs[0]
    assert batched.gather(reqs, Direct("m", calls), log=lambda *_: None, answers_path=path) == out and len(calls) == 5
    with pytest.raises(RuntimeError, match="not 'other'"):  # another model's answers are never mixed in
        batched.gather(reqs, Direct("other", calls), log=lambda *_: None, answers_path=path)


def test_gather_leaves_a_batch_llm_unchanged(tmp_path):
    path = tmp_path / batched.ANSWERS_FILE
    batch = FakeBatch("extractor", "claude-sonnet-5-5")
    batch.run = lambda requests, log=print: {k: {"reply": {"k": k}} for k in requests}
    assert batched.gather({"a": ("s", "u", {})}, batch, answers_path=path) == {"a": {"reply": {"k": "a"}}}
    assert not path.exists()


def test_run_batched_resumes_a_crashed_direct_reasoner_run(tmp_path):
    cases = e2e.load_cases("fixtures")
    with pytest.raises(ConnectionError):
        batched.run_batched(cases, Crashing(ok=2), Fake("extractor"), Fake("judge"), retriever, tmp_path, log=lambda *_: None,
                            source="fixtures")
    assert len((tmp_path / batched.ANSWERS_FILE).read_text().splitlines()) == 2
    reasoner = Fake("reasoner")
    metrics = batched.run_batched(cases, reasoner, Fake("extractor"), Fake("judge"), retriever, tmp_path, log=lambda *_: None,
                                  source="fixtures")
    assert reasoner.calls == 3 and metrics["summary"]["cases"] == 5
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["resumed_reasoner_answers"] == 2


class Slow:
    """A direct llm that sleeps, records how many calls overlap, and fails on request "boom"."""

    model = "slow-model"

    def __init__(self, pause=0.05):
        import threading

        self.pause, self.lock, self.active, self.peak, self.calls = pause, threading.Lock(), 0, 0, 0
        self.usage = {"requests": 0}

    def __call__(self, system, user, schema):
        import time

        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.pause)
            if user == "boom":
                raise ConnectionError("server unreachable")
            if user == "refuse":
                raise RefusalError("declined")
            return {"echo": user}
        finally:
            with self.lock:
                self.active -= 1


def test_gather_runs_direct_requests_concurrently_and_saves_each(tmp_path):
    reqs = {f"k{i}": ("s", f"u{i}", {}) for i in range(12)}
    reqs["kr"] = ("s", "refuse", {})
    llm, path = Slow(), tmp_path / "answers.jsonl"
    out = batched.gather(reqs, llm, log=lambda *_: None, answers_path=path, workers=4)
    assert set(out) == set(reqs) and out["k3"] == {"reply": {"echo": "u3"}} and out["kr"]["refusal"]
    assert 1 < llm.peak <= 4
    assert len(path.read_text().splitlines()) == 13
    again = Slow()
    assert batched.gather(reqs, again, log=lambda *_: None, answers_path=path, workers=4) == out and again.calls == 0
    serial = Slow()
    batched.gather({k: reqs[k] for k in ("k0", "k1")}, serial, log=lambda *_: None)
    assert serial.peak == 1  # workers defaults to 1: one at a time, as before


def test_gather_stops_on_a_server_error_and_keeps_finished_answers(tmp_path):
    reqs = {f"k{i}": ("s", f"u{i}", {}) for i in range(3)}
    reqs["kb"] = ("s", "boom", {})
    path = tmp_path / "answers.jsonl"
    with pytest.raises(ConnectionError):
        batched.gather(reqs, Slow(), log=lambda *_: None, answers_path=path, workers=2)
    saved = batched.load_answers(path, "slow-model")
    assert "kb" not in saved and all("reply" in r for r in saved.values())


# ---------------------------------------------------------------------------- The batched judge and staged batches

import hashlib  # noqa: E402

from strokecopilot.nlp.verifier import run as vr  # noqa: E402
from strokecopilot.nlp.verifier import trace as tr  # noqa: E402


class PickyJudge(Fake):
    """A judge whose verdict depends only on the request, so a live and a batched run must agree claim by claim. It
    records every request it is sent."""

    def __init__(self, seen):
        super().__init__("judge")
        self.seen = seen

    def __call__(self, system, user, schema):
        self.seen.append((system, user, json.dumps(schema, sort_keys=True)))
        reply = super().__call__(system, user, schema)
        for v in reply["verdicts"]:
            mark = int(hashlib.sha256((v["id"] + user).encode()).hexdigest(), 16) % 4
            v["verdict"], v["scope_fits"] = ("not_addressed" if mark == 0 else "supported"), mark != 1
        return reply


class Staged:
    """A batch llm with submit() and collect(), like ClaudeBatch. `server` stands for Anthropic's side and outlives the
    object, so a new object per call is a new laptop session."""

    PRICE_FACTOR = 0.5

    def __init__(self, inner, model, server):
        self.inner, self.model, self.server, self.batch_ids = inner, model, server, []
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    def describe(self):
        return {"model": self.model, "batch": True}

    def submit(self, requests, log=print):
        batch_id = f"b{len(self.server['batches']) + 1}"
        self.server["batches"][batch_id] = dict(requests)
        self.batch_ids.append(batch_id)
        return batch_id

    def collect(self, batch_id, keys, log=print, wait=True):
        if batch_id not in self.server["ended"]:
            if not wait:
                return None
            self.server["ended"].add(batch_id)
        requests = self.server["batches"][batch_id]
        self.usage["requests"] += len(requests)
        self.usage["output_tokens"] += 1_000_000 * len(requests)
        return {k: {"reply": self.inner(*req)} for k, req in requests.items()}

    def run(self, requests, log=print):
        return self.collect(self.submit(requests), list(requests))

    def __call__(self, *a):
        raise AssertionError("a batched llm is never called directly")


def test_batched_judge_sends_the_live_requests_and_gives_the_live_result(tmp_path):
    cases = e2e.load_cases("fixtures")
    live_seen, batch_seen, server = [], [], {"batches": {}, "ended": set()}
    live = e2e.run_cases(cases, Fake("reasoner"), Fake("extractor"), PickyJudge(live_seen), retriever, tmp_path / "live",
                         log=lambda *_: None)
    judge = Staged(PickyJudge(batch_seen), "claude-haiku-5-5", server)
    two = batched.run_batched(cases, Fake("reasoner"), Fake("extractor"), judge, retriever, tmp_path / "batch",
                              log=lambda *_: None, source="fixtures")
    assert live_seen and sorted(batch_seen) == sorted(live_seen)  # byte for byte: system prompt, user message, schema
    assert len(batch_seen) == len(set(batch_seen))  # nothing asked twice
    assert two["summary"] == live["summary"]
    for name in ("flagged.json",):
        assert json.loads((tmp_path / "batch" / name).read_text()) == json.loads((tmp_path / "live" / name).read_text())
    for case in cases:  # every claim's trace is the same
        a = json.loads((tmp_path / "live" / "traces" / f"{case['id']}.json").read_text())
        b = json.loads((tmp_path / "batch" / "traces" / f"{case['id']}.json").read_text())
        assert a == b
    assert json.loads((tmp_path / "batch" / "config.json").read_text())["batched"]["judge"] is True
    assert two["judge_calls"] == live["judge_calls"]


def test_collecting_judge_leaves_out_passages_already_on_their_way(tmp_path):
    passages = {pid: {"id": pid, "text": f"text of {pid}"} for pid in ("p1", "p2", "p3")}
    calls = [("the same claim", ["p1", "p2"]), ("the same claim", ["p2", "p3"]), ("another claim", ["p1"])]
    live_seen = []
    live = vr.CachedJudge(tr.Judge(PickyJudge(live_seen)), vr.JsonCache())
    want = [live(text, [passages[p] for p in pids], {"x": 1}, True) for text, pids in calls]
    assert [len(json.loads(user)["passages"]) for _, user, _ in live_seen] == [2, 1, 1]  # p2 is not sent again

    results, source, cache, got = {}, PickyJudge([]), vr.JsonCache(), None
    for round_no in range(3):
        recorder = batched.Recording(results, source)
        collecting = batched.CollectingJudge(tr.Judge(recorder), cache, tr.JUDGE_PROMPT)
        got = []
        for text, pids in calls:
            try:
                got.append(collecting(text, [passages[p] for p in pids], {"x": 1}, True))
            except batched.Pending:
                got.append(None)
        if not recorder.wanted:
            break
        if round_no == 0:
            assert sorted(user for _, user, _ in recorder.wanted.values()) == sorted(user for _, user, _ in live_seen)
        results.update({key: {"reply": source(*req)} for key, req in recorder.wanted.items()})
    assert got == want and round_no == 1  # one round of requests, then everything is answered


def test_staged_run_survives_closed_sessions_without_submitting_twice(tmp_path):
    cases = e2e.load_cases("fixtures")
    seen, server = [], {"batches": {}, "ended": set()}

    def session():  # new objects every time: a new laptop session
        return batched.run_batched(
            cases, Staged(Fake("reasoner"), "claude-sonnet-5-5", server), Staged(Fake("extractor"), "claude-sonnet-5-5", server),
            Staged(PickyJudge(seen), "claude-haiku-5-5", server), retriever, tmp_path / "run", log=lambda *_: None,
            source="fixtures", wait=False)

    first = session()
    assert first["waiting"] == "reasoner" and first["requests"] == 5 and len(server["batches"]) == 1
    assert session()["waiting"] == "reasoner" and len(server["batches"]) == 1  # still running: nothing submitted again
    stages = []
    for _ in range(8):
        server["ended"].update(server["batches"])
        result = session()
        if "waiting" not in result:
            break
        stages.append(result["waiting"])
        assert session()["waiting"] == result["waiting"]  # asking again changes nothing
    assert stages[0] == "extractor" and set(stages[1:]) == {"judge"} and "summary" in result
    submitted = len(server["batches"])
    assert submitted == 2 + len(stages) - 1  # one batch per stage and judge round

    live_seen = []
    live = e2e.run_cases(cases, Fake("reasoner"), Fake("extractor"), PickyJudge(live_seen), retriever, tmp_path / "live",
                         log=lambda *_: None)
    assert result["summary"] == live["summary"] and sorted(seen) == sorted(live_seen)
    # Usage is added up over the sessions, so the cost covers the whole run at the batch price.
    judged = len(live_seen)
    assert result["usage"]["judge"]["requests"] == judged
    assert result["cost_usd_estimate"]["judge"] == pytest.approx(
        providers.cost_usd("claude-haiku-5-5", {"output_tokens": 1_000_000 * judged}) * 0.5)
    assert result["cost_usd_estimate"]["reasoner"] == pytest.approx(
        providers.cost_usd("claude-sonnet-5-5", {"output_tokens": 5_000_000}) * 0.5)
    ledger = json.loads((tmp_path / "run" / batched.LEDGER_FILE).read_text())
    assert all(entry["collected"] for entry in ledger["batches"].values()) and len(ledger["batches"]) == submitted
    assert session()["summary"] == live["summary"] and len(server["batches"]) == submitted  # a rerun costs nothing


def test_claude_batch_collects_in_a_later_session():
    client = _client({"a": ("end_turn", '{"x": 1}')})
    first = ClaudeBatch("claude-haiku-5-5", client=client, sleep=lambda s: None)
    batch_id = first.submit({"a": ("s", "u", {"type": "object"})}, log=lambda *_: None)
    assert first.collect(batch_id, ["a"], log=lambda *_: None, wait=False) is None  # still in progress: no waiting
    later = ClaudeBatch("claude-haiku-5-5", client=client, sleep=lambda s: None)
    assert later.collect(batch_id, ["a", "lost"], log=lambda *_: None, wait=False) == {
        "a": {"reply": {"x": 1}}, "lost": {"error": "no batch result", "refusal": False}}
    assert later.batch_ids == [batch_id] and later.usage["requests"] == 1
