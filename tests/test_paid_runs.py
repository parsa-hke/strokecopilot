"""The paid-run plan (DEC-24): cells, arms and the resumable step, with fake models. No network, no data."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).parent))
from test_batched import PickyJudge, Staged  # noqa: E402
from test_end_to_end import Fake, retriever  # noqa: E402

from strokecopilot.pipeline import end_to_end as e2e  # noqa: E402
from strokecopilot.pipeline import paid_runs as pr  # noqa: E402


def test_cells_follow_the_dec24_matrix():
    ready = pr.cells()
    assert len(ready) == 4 * 4 + 2  # four ready arms per model, plus Sonnet's two repeats
    everything = pr.cells(ready_only=False)
    assert len(everything) == 4 * 7 + 2
    assert [arm for model, arm in ready if model == pr.SONNET][-2:] == ["real_predicted_repeat2", "real_predicted_repeat3"]
    assert not any("repeat" in arm for model, arm in everything if model != pr.SONNET)
    assert pr.FALLBACK_PAIR == ("qwen2.5-32b", "baichuan-m2-32b")


def test_scenario_arms_have_150_and_75_cases():
    paths = pr.Paths(Path("x"), Path("x"), Path("x"), Path("x"))
    full = pr.arm_cases(pr.SONNET, "scenarios_rulebook", paths)
    half = pr.arm_cases(pr.SONNET, "scenarios_norulebook", paths)
    assert len(full) == 150 and len(half) == 75 and {c["id"] for c in half} < {c["id"] for c in full}
    assert [c["id"] for c in half] == [c["id"] for c in pr.half_scenarios(e2e.load_cases("scenarios"))]  # deterministic
    path = lambda c: (c["expected"]["thrombolysis"], c["expected"]["thrombectomy"])
    for p in {path(c) for c in full}:  # each decision path keeps about half
        n, k = sum(path(c) == p for c in full), sum(path(c) == p for c in half)
        assert abs(k - n / 2) <= 1
    assert len(pr.arm_cases(pr.SONNET, "fixtures_rulebook", paths)) == 5


def test_llms_use_the_settings_of_earlier_runs(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-not-a-real-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-not-a-real-key")
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    sonnet = pr.reasoner_llm(pr.SONNET)
    assert sonnet.describe() == {"provider": "anthropic", "model": "claude-sonnet-5-5", "effort": "high",
                                 "max_tokens": 20000, "batch": True}
    extractor, judge = pr.verifier_llms()
    assert extractor.describe() == {"provider": "anthropic", "model": "claude-sonnet-5-5", "effort": "medium",
                                    "max_tokens": 16000, "batch": True}
    assert judge.describe() == {"provider": "anthropic", "model": "claude-haiku-5-5", "effort": "medium",
                                "max_tokens": 8000, "batch": True}
    qwen = pr.reasoner_llm("qwen3.5-27b")
    assert qwen.describe()["provider"] == "openrouter" and qwen.model == "qwen/qwen3.5-27b"
    gemma = pr.reasoner_llm("gemma-3-27b", port=8003)
    assert gemma.describe()["base_url"] == "http://localhost:8003/v1" and gemma.describe()["quantization"] == "fp8"


def test_advance_moves_a_cell_stage_by_stage_and_status_reads_the_folder(tmp_path):
    paths = pr.Paths(tmp_path, tmp_path, tmp_path, tmp_path / "out")
    server, seen = {"batches": {}, "ended": set()}, []

    def step():
        return pr.advance(pr.SONNET, "fixtures_rulebook", paths, Staged(Fake("reasoner"), pr.SONNET, server),
                          Staged(Fake("extractor"), pr.SONNET, server), Staged(PickyJudge(seen), "claude-haiku-5-5", server),
                          retriever, log=lambda *_: None)

    assert [r["state"] for r in pr.status([pr.SONNET], paths)] == ["not started"] * 6
    assert step()["waiting"] == "reasoner"
    assert pr.status([pr.SONNET], paths)[0] == {"model": pr.SONNET, "arm": "fixtures_rulebook", "state": "waiting: reasoner batch"}
    for _ in range(8):
        server["ended"].update(server["batches"])
        result = step()
        if "done" in result:
            break
    assert result["done"]["cases"] == 5 and result["cost_usd_estimate"]["judge"] is not None
    assert pr.status([pr.SONNET], paths)[0]["state"] == "done"
    config = __import__("json").loads((pr.cell_dir(paths, pr.SONNET, "fixtures_rulebook") / "config.json").read_text())
    assert config["paid_run"] == {"model": pr.SONNET, "arm": "fixtures_rulebook", "cases": 5, "source": "fixtures", "rulebook": True}
    with pytest.raises(pr.NotReady, match="not defined yet"):
        pr.advance(pr.SONNET, "fixtures_norulebook", paths, None, None, None, retriever)


class FakeOps:
    """A pod that records what the driver does to it."""

    def __init__(self):
        self.events = []

    def tunnel(self, port):
        ops = self

        class T:
            def __enter__(self):
                ops.events.append(f"tunnel {port}")

            def __exit__(self, *exc):
                ops.events.append("tunnel closed")
                return False

        return T()

    def start(self, argv):
        self.events.append(f"start {argv[argv.index('--served-model-name') + 1]}")
        return "serve started"

    def stop(self):
        self.events.append("stop")
        return "serve stopped"

    def log_tail(self):
        return "last lines"

    def down(self, log=print):
        self.events.append("down")


class DirectFake(Fake):
    def __init__(self, model, crash=False):
        super().__init__("reasoner")
        self.model, self.crash = model, crash

    def __call__(self, system, user, schema):
        if self.crash:
            raise ConnectionError("server gone")
        return super().__call__(system, user, schema)


def _serve_setup(tmp_path, monkeypatch, models=("qwen2.5-32b", "baichuan-m2-32b")):
    from strokecopilot.compute import jobs

    monkeypatch.setitem(jobs.RUNS, "EXP-98", {"what": "test", "hours": 1.0, "est_hours": 0.5,
                                              "serve": {"models": models, "allow_gated": False, "allow_medgemma": False}})
    server = {"batches": {}, "ended": set()}
    verifier = (Staged(Fake("extractor"), pr.SONNET, server), Staged(PickyJudge([]), "claude-haiku-5-5", server))
    return pr.Paths(tmp_path, tmp_path, tmp_path, tmp_path / "out"), verifier


def test_pod_driver_serves_each_model_then_destroys_the_pod(tmp_path, monkeypatch):
    paths, verifier = _serve_setup(tmp_path, monkeypatch)
    ops, ready = FakeOps(), []
    results = pr.run_pod_models("EXP-98", paths, retriever, log=lambda *_: None, ops=ops, verifier=verifier,
                                llm_for=lambda model, port: DirectFake(model), ready=lambda port, model, log: ready.append(model),
                                only_arms=["fixtures_rulebook"])
    assert ops.events == ["tunnel 8000", "start qwen2.5-32b", "stop", "start baichuan-m2-32b", "stop", "tunnel closed", "down"]
    assert ready == ["qwen2.5-32b", "baichuan-m2-32b"]
    assert results[("qwen2.5-32b", "fixtures_rulebook")]["waiting"] == "extractor"  # answers saved, extraction submitted
    assert pr.reasoner_answered("qwen2.5-32b", "fixtures_rulebook", paths)
    assert (paths.out_root / "qwen2.5-32b" / "serve_log_tail.txt").read_text() == "last lines"
    again = FakeOps()  # a second run has nothing to ask: no server is started, the pod is still destroyed
    assert pr.run_pod_models("EXP-98", paths, retriever, log=lambda *_: None, ops=again, verifier=verifier,
                             llm_for=lambda model, port: DirectFake(model), ready=lambda *a, **k: None,
                             only_arms=["fixtures_rulebook"]) == {}
    assert again.events == ["tunnel 8000", "tunnel closed", "down"]


def test_pod_driver_destroys_the_pod_when_the_server_fails(tmp_path, monkeypatch):
    paths, verifier = _serve_setup(tmp_path, monkeypatch)
    ops = FakeOps()
    with pytest.raises(ConnectionError):
        pr.run_pod_models("EXP-98", paths, retriever, log=lambda *_: None, ops=ops, verifier=verifier,
                          llm_for=lambda model, port: DirectFake(model, crash=True), ready=lambda *a, **k: None,
                          only_arms=["fixtures_rulebook"])
    assert ops.events == ["tunnel 8000", "start qwen2.5-32b", "stop", "tunnel closed", "down"]
    assert not pr.reasoner_answered("qwen2.5-32b", "fixtures_rulebook", paths)


def test_pod_driver_refuses_gated_models_until_the_policy_allows_them(tmp_path, monkeypatch):
    from strokecopilot.nlp.reasoner import open_models as om

    paths, verifier = _serve_setup(tmp_path, monkeypatch, models=("gemma-3-27b",))
    ops = FakeOps()
    with pytest.raises(om.NotServable, match="gated"):
        pr.run_pod_models("EXP-98", paths, retriever, log=lambda *_: None, ops=ops, verifier=verifier,
                          llm_for=lambda model, port: DirectFake(model), ready=lambda *a, **k: None,
                          only_arms=["fixtures_rulebook"])
    assert ops.events == ["tunnel 8000", "tunnel closed", "down"]  # nothing was served, and the pod is not left up


def test_wait_ready_polls_until_the_model_is_listed():
    answers = iter([ConnectionError("refused"), {"data": []}, {"data": [{"id": "m"}]}])

    def fetch():
        item = next(answers)
        if isinstance(item, Exception):
            raise item
        return item

    naps = []
    pr.wait_ready(8000, "m", log=lambda *_: None, fetch=fetch, sleep=naps.append)
    assert naps == [15, 15]
    with pytest.raises(TimeoutError):
        pr.wait_ready(8000, "m", timeout_s=-1, log=lambda *_: None, fetch=lambda: {"data": []}, sleep=lambda s: None)


def test_smoke_reports_one_request_and_advance_all_skips_pod_models(tmp_path):
    out = pr.smoke("x", DirectFake("qwen/qwen3.5-27b"))
    assert out["valid"] and out["error"] is None and out["usage"]["requests"] == 1
    assert set(out["projected_cost_usd"]) == {195, 295}
    paths = pr.Paths(tmp_path, tmp_path, tmp_path, tmp_path / "out")
    server = {"batches": {}, "ended": set()}
    verifier = (Staged(Fake("extractor"), pr.SONNET, server), Staged(PickyJudge([]), "claude-haiku-5-5", server))
    rows = pr.advance_all(["gemma-3-27b"], paths, retriever, log=lambda *_: None, verifier=verifier,
                          llm_for=lambda model: DirectFake(model), only_arms=["fixtures_rulebook", "scenarios_rulebook"])
    assert [r["state"] for r in rows] == ["needs the pod", "needs the pod"]
