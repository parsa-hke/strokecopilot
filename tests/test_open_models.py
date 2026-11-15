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
    for key in ("qwen3.5-27b", "gemma-3-27b", "medgemma-27b-text"):
        with pytest.raises(om.NotServable, match="verify"):
            om.serve_command(key, allow_gated=True, allow_medgemma=True)
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
