"""The open reasoner models of DEC-24 and how each is served (DEC-21; hosting amended in DEC-24, 15 Nov 2026).

Hosting under the amendment:
- Qwen3.5-27B runs through the OpenRouter API at full precision: API_ROUTES and api_llm(). Its pod entry in MODELS
  stays unverified (no official 4-bit checkpoint fits 24 GB).
- Gemma 3 27B and MedGemma 27B text run together on one rented 48 GB GPU, both from their BF16 weights with vLLM's
  online 8-bit quantization (`--quantization fp8`), so the medical-training pair is served identically. Fit and speed
  are unmeasured until the one-request GPU dummy.
- The fallback pair (Baichuan-M2-32B, Qwen2.5-32B-Instruct) keeps its official 4-bit checkpoints.

MODELS holds Qwen3.5-27B, MedGemma 27B text and Gemma 3 27B, and the fallback pair Baichuan-M2-32B and Qwen2.5-32B.
Every Hugging Face id here was checked against its model card on 15 Nov 2026. A field that could not be checked is
None and named in the entry's `verify`, and serve_command() refuses that entry until someone fills it from the card.

serve_command() returns the argv for `vllm serve`, bound to 127.0.0.1, so the server is reachable only through an SSH
tunnel: the pipeline runs on the Mac, reaches the server at http://localhost:<port>/v1, and every API key stays on the
Mac (DEC-21). It is not wired into compute/jobs.py or pod.py yet: that waits for a one-request GPU dummy per model.

Two refusals are policy, not data, and each has an explicit flag that is off by default:
- allow_gated: a gated repository needs a Hugging Face token on the pod, which DEC-21 currently forbids (Parsa's
  decision pending).
- allow_medgemma: MedGemma needs Parsa's reading of the HAI-DEF terms §3.2 (DEC-24).

max_model_len 16384 holds the reasoner's system prompt (about 6k to 7k tokens on these tokenizers) plus
providers.VLLM_MAX_TOKENS (8000). Whether weights plus a 16k KV cache fit 24 GB is unmeasured: a 32B model at 4-bit is
about 18 to 19 GB, and its 16-bit KV cache (64 layers, 8 KV heads of 128) is about 4 GiB at 16k tokens, so
kv_cache_dtype="fp8" may be needed. The GPU dummy decides.
"""

from dataclasses import dataclass, field

SERVE_FIELDS = ("hf_id", "checkpoint", "quantization", "max_model_len", "gated", "merge_system")


@dataclass(frozen=True)
class OpenModel:
    name: str  # --served-model-name, and the `model` VLLMJSON sends
    hf_id: str | None  # the model card
    checkpoint: str | None  # the 4-bit repository vLLM loads
    quantization: str | None  # recorded in VLLMJSON(quantization=...); vLLM reads the method from the checkpoint
    max_model_len: int | None
    gated: bool | None  # the repository asks users to accept terms before download (needs an HF token)
    merge_system: bool | None  # True if the chat template has no system role (VLLMJSON(merge_system=True))
    thinks_by_default: bool | None
    thinking_off: dict | None  # VLLMJSON extra_body that turns thinking off; whether to use it is Parsa's call
    reasoning_parser: str | None  # vllm serve --reasoning-parser, so the thinking never reaches the parsed JSON
    needs_terms_clearance: bool = False  # MedGemma: HAI-DEF terms §3.2
    verify: tuple = ()  # fields still to check against the model card
    notes: str = ""
    sources: tuple = field(default=(), compare=False)

    def unverified(self):
        """Fields serve_command() needs that are missing or still to verify."""
        return sorted({f for f in SERVE_FIELDS if getattr(self, f) is None} | set(self.verify))


MODELS = {
    "qwen3.5-27b": OpenModel(
        name="qwen3.5-27b", hf_id="Qwen/Qwen3.5-27B", checkpoint=None, quantization=None, max_model_len=16384,
        gated=False, merge_system=False, thinks_by_default=True,
        thinking_off={"chat_template_kwargs": {"enable_thinking": False}}, reasoning_parser="qwen3",
        verify=("checkpoint", "quantization"),
        notes="The official Qwen/Qwen3.5-27B-GPTQ-Int4 exists but is about 30 GB (attention layers kept in 16-bit, "
              "per a Kaitchup comparison, Mar 2026), so it does not fit 24 GB. A community 4-bit build (for example "
              "QuantTrio/Qwen3.5-27B-AWQ) is unverified: size, quality and vLLM support to be checked. The card "
              "recommends a context of 128k or more to keep thinking quality.",
        sources=("https://huggingface.co/Qwen/Qwen3.5-27B", "https://huggingface.co/Qwen/Qwen3.5-27B-GPTQ-Int4")),
    "medgemma-27b-text": OpenModel(
        name="medgemma-27b-text", hf_id="google/medgemma-27b-text-it", checkpoint="google/medgemma-27b-text-it",
        quantization="fp8", max_model_len=16384, gated=True, merge_system=False, thinks_by_default=None,
        thinking_off=None, reasoning_parser=None, needs_terms_clearance=True,
        notes="Gated behind the Health AI Developer Foundations terms. No official 4-bit checkpoint on the card, so it "
              "is served from the BF16 weights with vLLM's online FP8 on a 48 GB GPU, like Gemma 3 27B (DEC-24 "
              "amended); about 27 GB of weights, fit unmeasured. The card's examples pass a system message. Whether it "
              "thinks by default is not stated on the card.",
        sources=("https://huggingface.co/google/medgemma-27b-text-it",)),
    "gemma-3-27b": OpenModel(
        name="gemma-3-27b", hf_id="google/gemma-3-27b-it", checkpoint="google/gemma-3-27b-it", quantization="fp8",
        max_model_len=16384, gated=True, merge_system=False, thinks_by_default=False, thinking_off=None,
        reasoning_parser=None,
        notes="Gated (Gemma terms). Served from the BF16 weights with vLLM's online FP8 on a 48 GB GPU, the same setting "
              "as MedGemma (DEC-24 amended); about 27 GB of weights, fit unmeasured. "
              "google/gemma-3-27b-it-qat-q4_0-unquantized holds BF16 weights trained for Q4_0, not a checkpoint vLLM "
              "loads at 4-bit as is. The card's examples pass a system message.",
        sources=("https://huggingface.co/google/gemma-3-27b-it",
                 "https://huggingface.co/google/gemma-3-27b-it-qat-q4_0-unquantized")),
    "baichuan-m2-32b": OpenModel(
        name="baichuan-m2-32b", hf_id="baichuan-inc/Baichuan-M2-32B", checkpoint="baichuan-inc/Baichuan-M2-32B-GPTQ-Int4",
        quantization="gptq-int4", max_model_len=16384, gated=False, merge_system=False, thinks_by_default=None,
        thinking_off={"chat_template_kwargs": {"thinking_mode": "off"}}, reasoning_parser="qwen3",
        notes="Fallback pair (DEC-24). Official GPTQ 4-bit; its card serves it with --reasoning-parser qwen3 and "
              "suggests an fp8 KV cache. The chat template renders a leading system message; thinking_mode 'on' or "
              "'off' is optional and, when not passed, the template adds nothing (the model decides).",
        sources=("https://huggingface.co/baichuan-inc/Baichuan-M2-32B",
                 "https://huggingface.co/baichuan-inc/Baichuan-M2-32B-GPTQ-Int4")),
    "qwen2.5-32b": OpenModel(
        name="qwen2.5-32b", hf_id="Qwen/Qwen2.5-32B-Instruct", checkpoint="Qwen/Qwen2.5-32B-Instruct-AWQ",
        quantization="awq-4bit", max_model_len=16384, gated=False, merge_system=False, thinks_by_default=False,
        thinking_off=None, reasoning_parser=None,
        notes="Fallback pair (DEC-24): Baichuan-M2-32B's base. DEC-24 names Qwen2.5-32B; the instruct model is used "
              "because the base model is not tuned for chat (Parsa to confirm). Native context 32,768.",
        sources=("https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ",)),
}


class NotServable(ValueError):
    """serve_command() refused: unverified fields, or a policy flag not given."""


def serve_command(model, port=8000, allow_gated=False, allow_medgemma=False, structured_output_backend="xgrammar",
                  kv_cache_dtype=None):
    """argv for `vllm serve` on the pod, reachable only on 127.0.0.1 (an SSH tunnel). `model` is a MODELS key or entry.

    A pre-quantized checkpoint needs no --quantization (vLLM reads its quantization_config); an entry served from
    16-bit weights with quantization "fp8" gets --quantization fp8 (vLLM's online 8-bit, DEC-24 amended)."""
    entry = MODELS[model] if isinstance(model, str) else model
    missing = entry.unverified()
    if missing:
        raise NotServable(f"{entry.name}: verify {', '.join(missing)} against the model card first.")
    if entry.needs_terms_clearance and not allow_medgemma:
        raise NotServable(f"{entry.name}: needs Parsa's clearance of the HAI-DEF terms (DEC-24); pass allow_medgemma.")
    if entry.gated and not allow_gated:
        raise NotServable(f"{entry.name}: gated, so the pod would need a Hugging Face token, which DEC-21 forbids for "
                          "now; pass allow_gated once Parsa decides.")
    argv = ["vllm", "serve", entry.checkpoint, "--served-model-name", entry.name,
            "--max-model-len", str(entry.max_model_len), "--host", "127.0.0.1", "--port", str(int(port)),
            "--structured-outputs-config.backend", structured_output_backend]
    if entry.quantization == "fp8":
        argv += ["--quantization", "fp8"]
    if entry.reasoning_parser:
        argv += ["--reasoning-parser", entry.reasoning_parser]
    if kv_cache_dtype:
        argv += ["--kv-cache-dtype", kv_cache_dtype]
    return argv


# ---------------------------------------------------------------------------- Models served through an API

OPENROUTER_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ApiRoute:
    name: str  # the comparison's name for the model (as in MODELS)
    provider: str
    base_url: str
    model_id: str  # the id the API expects
    key_name: str  # read with providers.secret(): an environment variable on the Mac, never in the repo
    max_tokens: int
    extra_body: dict
    merge_system: bool = False  # True: the system prompt goes at the top of the user message (VLLMJSON)
    notes: str = ""
    sources: tuple = field(default=(), compare=False)


API_ROUTES = {
    "qwen3.5-27b": ApiRoute(
        name="qwen3.5-27b", provider="openrouter", base_url=OPENROUTER_URL, model_id="qwen/qwen3.5-27b",
        key_name="OPENROUTER_API_KEY", max_tokens=20000,
        # OpenRouter provider routing, pinned to Alibaba (Qwen's developer), with no fallback: it honours every request
        # parameter (the json_schema, temperature) and does not keep prompts. Checked 16 Nov 2026: of the six
        # providers, the only one listed at 16-bit (Novita) lacks structured outputs, the other four run fp8, and
        # Alibaba lists no precision (most likely the original weights). Alibaba refuses JSON output unless a message
        # says "json"; the reasoner prompt does ("Return one JSON object").
        extra_body={"provider": {"order": ["alibaba"], "allow_fallbacks": False, "require_parameters": True,
                                 "data_collection": "deny"}},
        # EXP-37 (16 Nov 2026): this route dropped the system message (a probe billed 95 prompt tokens for an
        # 11k-character system prompt; EXP-36's smoke the day before billed 7,758), so 295 answers were written without
        # the instructions or the rulebook and were set aside. The same text at the top of the user message arrives.
        merge_system=True,
        notes="DEC-24 amended (15 Nov 2026): through OpenRouter, pinned to Alibaba, instead of a 4-bit build on the pod. "
              "List price about $0.20 in / $1.56 out per million tokens (15 Nov 2026; providers vary). The model "
              "thinks by default; whether to turn that off is a run-plan choice. max_tokens matches the Claude "
              "reasoner's 20000, since the API context (262k) is not the limit a pod's is.",
        sources=("https://openrouter.ai/qwen/qwen3.5-27b", "https://openrouter.ai/qwen/qwen3.5-27b/providers")),
}


def api_llm(name, client=None, **overrides):
    """A VLLMJSON-style llm for a model in API_ROUTES: the same call signature and JSON schema enforcement as the pod
    models, through the API's OpenAI-compatible endpoint. The key is read from route.key_name at call time (never
    passed in, never stored); `client` is for tests."""
    from strokecopilot.nlp.reasoner.providers import VLLMJSON

    route = API_ROUTES[name]
    settings = {"base_url": route.base_url, "key_name": route.key_name, "max_tokens": route.max_tokens,
                "extra_body": route.extra_body, "provider": route.provider, "billing": "per token (API)",
                "merge_system": route.merge_system, **overrides}
    return VLLMJSON(route.model_id, client=client, **settings)
