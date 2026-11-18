"""The open-model table (DEC-24) and the vLLM serve command for the pod. Pure functions: nothing is served."""

import dataclasses

import pytest

from strokecopilot.nlp.reasoner import open_models as om


def test_table_holds_the_dec24_models_and_the_fallback_pair():
    ids = {m.hf_id for m in om.MODELS.values()}
    assert ids == {"Qwen/Qwen3.5-27B", "google/medgemma-27b-text-it", "google/gemma-3-27b-it",
                   "baichuan-inc/Baichuan-M2-32B", "Qwen/Qwen2.5-32B-Instruct", "google/gemma-3-4b-it"}
    for key, m in om.MODELS.items():
        assert m.name == key and m.sources
        assert m.checkpoint is not None or "checkpoint" in m.verify  # nothing missing silently
    assert om.MODELS["medgemma-27b-text"].needs_terms_clearance and om.MODELS["medgemma-27b-text"].gated


def test_serve_command_binds_to_localhost_for_an_ssh_tunnel():
    argv = om.serve_command("qwen2.5-32b", port=8001)
    assert argv[:3] == ["vllm", "serve", "Qwen/Qwen2.5-32B-Instruct-AWQ"]
    opts = dict(zip(argv[3::2], argv[4::2]))
    assert opts == {"--served-model-name": "qwen2.5-32b", "--max-model-len": "16384", "--host": "127.0.0.1",
                    "--port": "8001", "--structured-outputs-config.backend": "xgrammar"}
    assert "--quantization" not in argv  # read from the checkpoint
    baichuan = om.serve_command("baichuan-m2-32b", kv_cache_dtype="fp8")
    assert baichuan[baichuan.index("--reasoning-parser") + 1] == "qwen3"
    assert baichuan[baichuan.index("--kv-cache-dtype") + 1] == "fp8"


def test_serve_command_refuses_unverified_entries():
    with pytest.raises(om.NotServable, match="verify"):
        om.serve_command("qwen3.5-27b")  # served through OpenRouter instead (DEC-24 amended)
    unchecked = dataclasses.replace(om.MODELS["qwen2.5-32b"], max_model_len=None)
    with pytest.raises(om.NotServable, match="max_model_len"):
        om.serve_command(unchecked)
    flagged = dataclasses.replace(om.MODELS["qwen2.5-32b"], verify=("merge_system",))
    with pytest.raises(om.NotServable, match="merge_system"):
        om.serve_command(flagged)


def test_serve_command_refuses_gated_and_medgemma_without_explicit_flags():
    # Filled-in copies, so only the policy refusals are left.
    gemma = dataclasses.replace(om.MODELS["gemma-3-27b"], checkpoint="example/gemma-4bit", quantization="awq-4bit", verify=())
    with pytest.raises(om.NotServable, match="gated"):
        om.serve_command(gemma)
    assert om.serve_command(gemma, allow_gated=True)[2] == "example/gemma-4bit"
    med = dataclasses.replace(om.MODELS["medgemma-27b-text"], checkpoint="example/medgemma-4bit", quantization="awq-4bit",
                              verify=())
    with pytest.raises(om.NotServable, match="HAI-DEF"):
        om.serve_command(med, allow_gated=True)
    with pytest.raises(om.NotServable, match="gated"):
        om.serve_command(med, allow_medgemma=True)
    assert om.serve_command(med, allow_gated=True, allow_medgemma=True)[2] == "example/medgemma-4bit"


def test_gemma_pair_is_served_identically_at_online_fp8():
    gemma = om.serve_command("gemma-3-27b", allow_gated=True)
    med = om.serve_command("medgemma-27b-text", allow_gated=True, allow_medgemma=True)
    assert gemma[2] == "google/gemma-3-27b-it" and med[2] == "google/medgemma-27b-text-it"
    assert gemma[gemma.index("--quantization") + 1] == "fp8" == med[med.index("--quantization") + 1]
    strip = lambda argv: [a for a in argv[3:] if a not in ("gemma-3-27b", "medgemma-27b-text")]
    assert strip(gemma) == strip(med)  # same settings apart from the model
    with pytest.raises(om.NotServable, match="gated"):
        om.serve_command("gemma-3-27b")
    with pytest.raises(om.NotServable, match="HAI-DEF"):
        om.serve_command("medgemma-27b-text", allow_gated=True)


class _Client:
    def __init__(self):
        from types import SimpleNamespace as NS

        self.sent = []
        reply = NS(choices=[NS(finish_reason="stop", message=NS(content='{"ok": true}', refusal=None))],
                   usage=NS(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=None,
                            completion_tokens_details=None))
        self.chat = NS(completions=NS(create=lambda **kw: (self.sent.append(kw), reply)[1]))


def test_api_llm_routes_qwen_through_openrouter_without_storing_a_key():
    from strokecopilot.nlp.reasoner import providers

    client = _Client()
    llm = om.api_llm("qwen3.5-27b", client=client)
    assert llm("system", "user", {"type": "object"}) == {"ok": True}
    sent = client.sent[0]
    assert sent["model"] == "qwen/qwen3.5-27b" and sent["temperature"] == 0.0 and sent["max_tokens"] == 20000
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["extra_body"]["provider"] == {"order": ["alibaba"], "allow_fallbacks": False, "require_parameters": True,
                                              "data_collection": "deny"}
    assert "JSON" in __import__("strokecopilot.nlp.reasoner.prompt", fromlist=["x"]).system_prompt("reasoner_v2")
    d = llm.describe()
    assert d["provider"] == "openrouter" and d["base_url"] == om.OPENROUTER_URL and d["billing"] == "per token (API)"
    assert "key" not in str(d).lower().replace("key_name", "")
    assert providers.cost_usd(llm.model, llm.usage) == pytest.approx((100 * 0.195 + 20 * 1.56) / 1e6)


def test_keyscan_watches_the_openrouter_key():
    from strokecopilot.compute import keyscan

    assert "OPENROUTER_API_KEY" in keyscan.ENV_KEYS


class _FlakyClient(_Client):
    """Answers with no choices `empty` times (OpenRouter's upstream error in a normal reply), then normally."""

    def __init__(self, empty):
        from types import SimpleNamespace as NS

        super().__init__()
        good = self.chat.completions.create
        bad = NS(choices=None, usage=None, model_extra={"error": {"code": 502, "message": "upstream error"}})
        self.calls = 0

        def create(**kw):
            self.calls += 1
            return bad if self.calls <= empty else good(**kw)

        self.chat = NS(completions=NS(create=create))


def test_an_answerless_reply_is_asked_again_and_never_scored():
    from strokecopilot.nlp.reasoner import providers
    from strokecopilot.pipeline import batched

    client = _FlakyClient(empty=2)
    llm = om.api_llm("qwen3.5-27b", client=client)
    llm.retry_waits_s = (0, 0, 0)
    assert llm("system", "user", {"type": "object"}) == {"ok": True}
    assert client.calls == 3 and llm.usage["provider_errors"] == 2 and llm.usage["requests"] == 1

    always = om.api_llm("qwen3.5-27b", client=_FlakyClient(empty=99))
    always.retry_waits_s = (0, 0, 0)
    with pytest.raises(providers.ProviderError, match="no answer 4 times; last error: .*upstream error"):
        always("system", "user", {"type": "object"})
    assert not isinstance(providers.ProviderError("x"), RuntimeError)  # so a run stops instead of scoring it
    with pytest.raises(providers.ProviderError):  # batched.gather stops too; nothing is saved as a format failure
        batched.gather({"k": ("s", "u", {})}, always, log=lambda *_: None, workers=2)


def test_qwen_route_merges_the_system_prompt_and_a_dropped_prompt_stops_the_run():
    """EXP-37: OpenRouter's Qwen route dropped the system message. The route now merges it into the user message, and
    a reply billed far fewer prompt tokens than the text sent raises ProviderError instead of being scored."""
    from types import SimpleNamespace as NS

    from strokecopilot.nlp.reasoner import providers

    client = _Client()
    llm = om.api_llm("qwen3.5-27b", client=client)
    assert llm("SYSTEM TEXT", "user", {"type": "object"}) == {"ok": True}  # 100 tokens billed for 15 characters
    messages = client.sent[0]["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user" and messages[0]["content"].startswith("SYSTEM TEXT")
    assert llm.describe()["merge_system"] is True
    short = NS(choices=[NS(finish_reason="stop", message=NS(content='{"ok": true}', refusal=None))],
               usage=NS(prompt_tokens=95, completion_tokens=5, prompt_tokens_details=None, completion_tokens_details=None))
    dropped = om.api_llm("qwen3.5-27b", client=NS(chat=NS(completions=NS(create=lambda **kw: short))))
    with pytest.raises(providers.ProviderError, match="dropped part of the prompt"):
        dropped("x" * 11000, "user", {"type": "object"})


def test_the_small_pair_is_served_identically_but_for_the_weights(tmp_path):
    base, tuned = (om.serve_command(m, port=8000, allow_gated=True) for m in om.SMALL_MODELS)
    assert base[2] == "google/gemma-3-4b-it" and tuned[2] == om.MODELS["gemma-3-4b-tuned"].checkpoint
    strip = lambda argv: [a for a in argv[3:] if not a.startswith("gemma-3-4b")]
    assert strip(base) == strip(tuned)  # only the checkpoint and the served name differ
    assert base[base.index("--dtype") + 1] == "bfloat16" and "--quantization" not in base
    with pytest.raises(om.NotServable):
        om.serve_command("gemma-3-4b")  # gated: needs the token
    assert om.serve_command("gemma-3-4b-tuned")  # local: no token


def test_fill_from_base_copies_what_the_merge_did_not_write(tmp_path):
    base, tuned = tmp_path / "base", tmp_path / "tuned"
    base.mkdir(), tuned.mkdir()
    for name in ("config.json", "model-00001-of-00002.safetensors", "model.safetensors.index.json", "tokenizer.json",
                 "preprocessor_config.json", "chat_template.json"):
        (base / name).write_text("base")
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (tuned / name).write_text("tuned")
    assert om.fill_from_base(tuned, base) == ["chat_template.json", "preprocessor_config.json"]
    assert (tuned / "config.json").read_text() == "tuned" and (tuned / "tokenizer.json").read_text() == "tuned"
    assert not (tuned / "model-00001-of-00002.safetensors").exists()
