"""Provider wrappers with stub clients: request shape, usage counting, refusals and secrets. No network."""

import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from strokecopilot.nlp.reasoner import providers
from strokecopilot.nlp.reasoner.providers import ClaudeJSON, RefusalError


def _client(stop_reason="end_turn", text='{"ok": true}', sent=None):
    class Messages:
        def create(self, **kwargs):
            if sent is not None:
                sent.update(kwargs)
            return NS(stop_reason=stop_reason, content=[NS(type="text", text=text)],
                      usage=NS(input_tokens=100, output_tokens=7, cache_read_input_tokens=90, cache_creation_input_tokens=0))

    return NS(messages=Messages())


def test_request_shape_and_usage():
    sent = {}
    llm = ClaudeJSON("claude-haiku-5-5", effort="low", client=_client(sent=sent))
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    assert llm("system text", "user text", schema) == {"ok": True}
    assert sent["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": schema}}
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"} and "temperature" not in sent
    assert llm.usage["requests"] == 1 and llm.usage["cache_read_input_tokens"] == 90
    assert llm.describe()["model"] == "claude-haiku-5-5"


def test_refusal_and_truncation_raise():
    with pytest.raises(RefusalError):
        ClaudeJSON("claude-opus-5-5", client=_client(stop_reason="refusal"))("s", "u", {})
    with pytest.raises(RuntimeError, match="max_tokens"):
        ClaudeJSON("claude-opus-5-5", client=_client(stop_reason="max_tokens"))("s", "u", {})


def _fake_colab(monkeypatch, values):
    """Stand-in google.colab.userdata: get() returns the stored value or raises, as Colab does for a missing secret."""
    def get(name):
        if name not in values:
            raise KeyError(name)
        return values[name]

    google, colab = ModuleType("google"), ModuleType("google.colab")
    colab.userdata = NS(get=get)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def test_secret_reads_the_environment_when_no_notebook_store(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails, as on a laptop
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)
    monkeypatch.setenv("FAKE_PROVIDER_KEY", "from-env")
    assert providers.secret("FAKE_PROVIDER_KEY") == "from-env"
    monkeypatch.delenv("FAKE_PROVIDER_KEY")
    assert providers.secret("FAKE_PROVIDER_KEY") is None


def test_secret_prefers_colab_and_falls_through_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)
    monkeypatch.setenv("FAKE_PROVIDER_KEY", "from-env")
    _fake_colab(monkeypatch, {"FAKE_PROVIDER_KEY": "from-colab"})
    assert providers.secret("FAKE_PROVIDER_KEY") == "from-colab"
    _fake_colab(monkeypatch, {})  # Colab present but the secret is not set: fall through to the environment
    assert providers.secret("FAKE_PROVIDER_KEY") == "from-env"
    _fake_colab(monkeypatch, {"FAKE_PROVIDER_KEY": ""})  # set but empty: also fall through
    assert providers.secret("FAKE_PROVIDER_KEY") == "from-env"


def test_api_key_is_the_anthropic_secret(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-not-a-real-key")
    assert providers.api_key() == "fake-not-a-real-key"


# ---------- VLLMJSON: open models on a vLLM server (DEC-24). A stub client stands in for the server. ----------

QWEN = "Qwen/Qwen3.5-27B"  # any model name works: the stub never checks it
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


def _vllm_client(content='{"ok": true}', finish_reason="stop", refusal=None, sent=None, cached=40, reasoning=0,
                 thinking=None):
    """A stand-in for openai.OpenAI(...) pointed at vLLM: chat.completions.create returns one canned choice."""
    class Completions:
        def create(self, **kwargs):
            if sent is not None:
                sent.update(kwargs)
            usage = NS(prompt_tokens=100, completion_tokens=7, prompt_tokens_details=NS(cached_tokens=cached),
                       completion_tokens_details=NS(reasoning_tokens=reasoning))
            message = NS(content=content, refusal=refusal)
            if thinking is not None:
                message.reasoning_content = thinking  # vLLM's reasoning parser puts the thinking here
            return NS(choices=[NS(message=message, finish_reason=finish_reason)], usage=usage)

    return NS(chat=NS(completions=Completions()))


def test_vllm_request_shape_and_usage():
    sent = {}
    llm = providers.VLLMJSON(QWEN, quantization="awq-4bit", client=_vllm_client(sent=sent, reasoning=5))
    assert llm("system text", "user text", SCHEMA) == {"ok": True}
    assert sent["model"] == QWEN and sent["temperature"] == 0.0 and sent["max_tokens"] == providers.VLLM_MAX_TOKENS == 8000
    assert sent["messages"] == [{"role": "system", "content": "system text"}, {"role": "user", "content": "user text"}]
    assert sent["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "response", "schema": SCHEMA, "strict": True}}
    # Same usage keys as ClaudeJSON; cached prompt tokens are split out of input_tokens.
    assert llm.usage["requests"] == 1 and llm.usage["input_tokens"] == 60 and llm.usage["cache_read_input_tokens"] == 40
    assert llm.usage["output_tokens"] == 7 and llm.usage["reasoning_tokens"] == 5
    llm("s", "u", SCHEMA)
    assert llm.usage["requests"] == 2 and llm.usage["input_tokens"] == 120
    assert llm.describe()["provider"] == "vllm" and llm.describe()["quantization"] == "awq-4bit"
    assert providers.cost_usd(llm.model, llm.usage) is None  # GPU cost comes from the pod's billing, not tokens
    assert llm.describe()["billing"] == "gpu-time (pod), not token-priced"  # so a None cost never reads as free


def test_vllm_records_base_url_only_for_a_local_server():
    """A tunnelled server's URL is kept; a pod's public address is not stored in run configs."""
    for url in ("http://localhost:8000/v1", "http://127.0.0.1:8001/v1"):
        assert providers.VLLMJSON(QWEN, base_url=url, client=_vllm_client()).describe()["base_url"] == url
    for url in ("http://203.0.113.7:41234/v1", "https://example.invalid/v1", "http://localhost.example.invalid/v1",
                "https://localhost:8000/v1", ""):
        assert providers.VLLMJSON(QWEN, base_url=url, client=_vllm_client()).describe()["base_url"] == "remote"


def test_vllm_extra_body_is_sent_and_recorded():
    """vLLM-only options, such as turning Qwen3.5's thinking off, reach the request and the run's config."""
    sent, off = {}, {"chat_template_kwargs": {"enable_thinking": False}}
    llm = providers.VLLMJSON(QWEN, extra_body=off, client=_vllm_client(sent=sent))
    llm("s", "u", SCHEMA)
    assert sent["extra_body"] == off and llm.describe()["extra_body"] == off
    sent.clear()
    providers.VLLMJSON(QWEN, client=_vllm_client(sent=sent))("s", "u", SCHEMA)
    assert "extra_body" not in sent  # nothing extra unless asked


def test_vllm_merge_system_for_templates_without_a_system_role():
    sent = {}
    providers.VLLMJSON(QWEN, merge_system=True, client=_vllm_client(sent=sent))("rules", "case", SCHEMA)
    assert sent["messages"] == [{"role": "user", "content": "rules\n\ncase"}]


def test_vllm_failures_raise_what_the_pipeline_catches():
    with pytest.raises(RefusalError):
        providers.VLLMJSON(QWEN, client=_vllm_client(refusal="cannot help"))("s", "u", SCHEMA)
    with pytest.raises(RuntimeError, match="max_tokens") as error:
        providers.VLLMJSON(QWEN, client=_vllm_client(content='{"ok": tr', finish_reason="length"))("s", "u", SCHEMA)
    assert '{"ok": tr' in str(error.value)  # the partial answer is kept
    with pytest.raises(RuntimeError, match="no content"):
        providers.VLLMJSON(QWEN, client=_vllm_client(content="   "))("s", "u", SCHEMA)
    with pytest.raises(RuntimeError, match="no valid JSON") as error:
        providers.VLLMJSON(QWEN, client=_vllm_client(content='{"ok": tru'))("s", "u", SCHEMA)
    assert '{"ok": tru' in str(error.value)  # the start of the raw reply is kept for the run folder


def test_vllm_failures_report_the_split_out_thinking():
    """When the server's reasoning parser separates the thinking, its length and start go in the error."""
    long_thought = "Let me weigh the core volume. " * 40
    with pytest.raises(RuntimeError, match="max_tokens") as error:
        providers.VLLMJSON(QWEN, client=_vllm_client(content=None, finish_reason="length",
                                                     thinking=long_thought))("s", "u", SCHEMA)
    message = str(error.value)
    assert f"thinking {len(long_thought)} chars" in message and long_thought[:200] in message
    assert long_thought[:201] not in message  # only the start
    with pytest.raises(RuntimeError, match="no valid JSON") as error:
        providers.VLLMJSON(QWEN, client=_vllm_client(content="nope", thinking="short"))("s", "u", SCHEMA)
    assert "thinking 5 chars" in str(error.value)
    with pytest.raises(RuntimeError) as error:
        providers.VLLMJSON(QWEN, client=_vllm_client(content="nope"))("s", "u", SCHEMA)
    assert "thinking" not in str(error.value)  # no reasoning field, no note


def test_vllm_is_called_directly_by_the_batched_pipeline():
    """batched.gather treats an llm with run() as a batch; VLLMJSON must be called one request at a time."""
    from strokecopilot.pipeline import batched

    llm = providers.VLLMJSON(QWEN, client=_vllm_client())
    assert not hasattr(llm, "run")
    assert batched.gather({"a": ("s", "u", SCHEMA)}, llm, log=lambda *_: None) == {"a": {"reply": {"ok": True}}}
    bad = providers.VLLMJSON(QWEN, client=_vllm_client(finish_reason="length"))
    assert batched.gather({"a": ("s", "u", SCHEMA)}, bad, log=lambda *_: None)["a"]["refusal"] is False


def test_vllm_reasoner_run_end_to_end_on_a_fixture(tmp_path, monkeypatch):
    """The real reasoner code, with a stub server that returns a valid recommendation, writes a normal run folder."""
    import json
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parent))  # removed again after the test
    from test_reasoner import FX02, FakeLLM, _vision

    from strokecopilot.nlp.reasoner import run as rr

    reply = json.dumps(FakeLLM().reply)
    llm = providers.VLLMJSON(QWEN, client=_vllm_client(content=reply))
    rec, record = rr.recommend(_vision(), "note", llm)
    assert record["error"] is None and rec["model"] == QWEN
    metrics = rr.run_fixtures(providers.VLLMJSON(QWEN, client=_vllm_client(content=reply)), tmp_path, stems=[FX02],
                              log=lambda *_: None)
    assert metrics["calls"] == 1 and metrics["format_failures"] == 0 and metrics["cost_usd_estimate"] is None
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model"]["provider"] == "vllm" and config["model"]["temperature"] == 0.0


def test_vllm_builds_its_client_from_a_secret_or_no_key(monkeypatch):
    """Without a stub client, VLLMJSON builds openai.OpenAI(base_url, api_key). A fake openai module records the call."""
    made = []
    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: made.append(kwargs) or NS()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)

    providers.VLLMJSON(QWEN, base_url="http://localhost:8000/v1")
    assert made[-1]["base_url"] == "http://localhost:8000/v1" and made[-1]["api_key"] == "EMPTY"  # local vLLM: no key
    assert made[-1]["timeout"] == 1800 and made[-1]["max_retries"] == 1  # one long answer may take minutes

    monkeypatch.setenv("FAKE_VLLM_KEY", "fake-not-a-real-key")
    providers.VLLMJSON(QWEN, base_url="https://example.invalid/v1", key_name="FAKE_VLLM_KEY")
    assert made[-1]["api_key"] == "fake-not-a-real-key"


def test_every_key_name_the_providers_read_is_scanned_for():
    """scripts/pod refuses to start if any of these keys' values is in the repo tree (compute/keyscan)."""
    from strokecopilot.compute import keyscan

    for name in ("ANTHROPIC_API_KEY", providers.VLLM_KEY_NAME, "HF_TOKEN", "OPENAI_API_KEY"):
        assert name in keyscan.ENV_KEYS
