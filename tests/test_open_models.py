"""The open-model table (DEC-24) and the vLLM serve command for the pod. Pure functions: nothing is served."""

import dataclasses

import pytest

from strokecopilot.nlp.reasoner import open_models as om


def test_table_holds_the_dec24_models_and_the_fallback_pair():
    ids = {m.hf_id for m in om.MODELS.values()}
    assert ids == {"Qwen/Qwen3.5-27B", "google/medgemma-27b-text-it", "google/gemma-3-27b-it",
                   "baichuan-inc/Baichuan-M2-32B", "Qwen/Qwen2.5-32B-Instruct"}
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
