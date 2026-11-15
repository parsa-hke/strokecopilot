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
