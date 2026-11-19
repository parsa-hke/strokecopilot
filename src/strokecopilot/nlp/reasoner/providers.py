"""Thin, identically-shaped LLM wrappers (plan §8.3): Claude (direct and batched) and open models served by vLLM.

Every wrapper is a callable `llm(system, user, schema) -> dict`: one request whose reply is JSON
matching `schema`. The verifier (claim extractor and judge) and, later, the reasoner use this
shape, so tests pass a plain function instead.

Keys come from Colab or Kaggle Secrets, or the environment, never from code (secret()): ANTHROPIC_API_KEY for
Claude; for an open model on a vLLM server, none for a local server, else the secret named by VLLMJSON's key_name
(VLLM_API_KEY by convention, the variable `vllm serve` reads). compute/keyscan.ENV_KEYS lists every such name. Current Claude models reject `temperature` (docs/HALLUCINATION_METRIC.md §6), so none is
sent. Server-side refusal fallbacks are left off on purpose: a fallback would swap the judge or
extractor model mid-run, and the verifier must record exactly which model produced each verdict.
A refusal is raised instead, and the caller records it.
"""

import json
import os
import threading
import time
from urllib.parse import urlparse

DEFAULT_MAX_TOKENS = 16000
# VLLMJSON's default: vLLM rejects a request whose prompt plus max_tokens exceeds the server's --max-model-len.
VLLM_MAX_TOKENS = 8000
VLLM_KEY_NAME = "VLLM_API_KEY"  # the conventional key_name for a vLLM server started with an API key
VLLM_TIMEOUT_S = 1800  # one long reasoner answer at 4-bit on one card can take minutes; the openai default is 600 s

# USD per million tokens (input, output) on the Claude API, prompts up to 100K tokens. Cache reads: Opus 5.5 $0.20,
# Sonnet 5.5 $0.10 (0.05x input; it was $0.20 here until 12 Nov 2026, which overstated logged Sonnet costs, as Hana
# found), Haiku 5.5 $0.01. 5-minute cache writes are 1.25x input. Check before quoting a cost.
PRICES = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.10},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
    # OpenRouter list price for Qwen3.5-27B on 15 Nov 2026 (DEC-24 amended); the provider it routes to may differ.
    "qwen/qwen3.5-27b": {"input": 0.195, "output": 1.56, "cache_read": 0.195},
}


def cost_usd(model, usage):
    """Estimated cost of `usage` (a ClaudeJSON.usage dict), or None for an unknown model."""
    p = PRICES.get(model)
    if p is None:
        return None
    return (usage.get("input_tokens", 0) * p["input"] + usage.get("output_tokens", 0) * p["output"]
            + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
            + usage.get("cache_creation_input_tokens", 0) * p["input"] * 1.25) / 1e6


class ProviderError(Exception):
    """The provider sent back no answer at all (a reply with no choices, as OpenRouter does when its upstream provider
    fails). Not the model's reply, so it is never scored: not a RuntimeError, which run.py and batched.gather record as
    a format failure, so a run stops loudly instead, and resumes from its saved answers."""


# An answerless reply is asked again after these waits (seconds) before VLLMJSON raises ProviderError (EXP-37: one
# OpenRouter reply with no choices stopped the Qwen pass).
PROVIDER_RETRY_WAITS_S = (10, 30, 90)


# Text runs about 3 to 4.5 characters per token in these tokenizers. A prompt billed at fewer than one token per
# MIN_CHARS_PER_BILLED_TOKEN characters sent was cut by the provider: VLLMJSON stops rather than score the answer.
MIN_CHARS_PER_BILLED_TOKEN = 8


def _provider_error(response):
    """The error a provider put in an answerless reply, shortened; no request content and no key."""
    extra = getattr(response, "model_extra", None) or {}
    error = extra.get("error") if isinstance(extra, dict) else None
    return str(error if error is not None else getattr(response, "error", None))[:300]


class RefusalError(RuntimeError):
    """The model declined the request (stop_reason "refusal")."""


def secret(name):
    """The secret `name` from Colab Secrets, then Kaggle Secrets, then the environment; None if none has it.

    Keys never live in code. A source that is missing, not granted or empty is skipped, so the same notebook works on
    Colab, Kaggle or a laptop. The value is returned, never logged."""
    try:
        from google.colab import userdata

        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name)


def api_key():
    """ANTHROPIC_API_KEY from Colab Secrets, Kaggle Secrets or the environment."""
    return secret("ANTHROPIC_API_KEY")


class ClaudeJSON:
    """Structured-output call to one Claude model. Usage is summed in `self.usage`."""

    def __init__(self, model, effort="medium", max_tokens=DEFAULT_MAX_TOKENS, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key(), max_retries=4)
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "seconds": 0.0}
        self._usage_lock = threading.Lock()  # calls may overlap (the Space's concurrent judge calls, DEC-32)

    def describe(self):
        return {"provider": "anthropic", "model": self.model, "effort": self.effort, "max_tokens": self.max_tokens}

    def __call__(self, system, user, schema):
        started = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            # The system prompt is the same for every claim, so it is cached.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        with self._usage_lock:
            self.usage["requests"] += 1
            self.usage["seconds"] += time.perf_counter() - started
            for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                self.usage[name] += getattr(response.usage, name, 0) or 0
        if response.stop_reason == "refusal":
            raise RefusalError(f"{self.model} declined the request.")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"{self.model} hit max_tokens ({self.max_tokens}) before finishing the JSON.")
        text = next(block.text for block in response.content if block.type == "text")
        return json.loads(text)


class ClaudeBatch:
    """The same structured-output requests as ClaudeJSON, sent through the Message Batches API: half the price, same
    model, results within a day (usually under an hour). DEC-24, amended 12 Nov 2026.

    run({key: (system, user, schema)}) returns {key: {"reply": parsed JSON} or {"error": text, "refusal": bool}}. Keys
    become custom IDs, so they must match ^[A-Za-z0-9_-]{1,64}$ (the pipeline uses 16-hex request hashes)."""

    PRICE_FACTOR = 0.5

    def __init__(self, model, effort="medium", max_tokens=DEFAULT_MAX_TOKENS, client=None, poll_s=60, sleep=time.sleep):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key(), max_retries=4)
        self.client, self.model, self.effort, self.max_tokens = client, model, effort, max_tokens
        self.poll_s, self.sleep = poll_s, sleep
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "seconds": 0.0}
        self.batch_ids = []

    def describe(self):
        return {"provider": "anthropic", "model": self.model, "effort": self.effort, "max_tokens": self.max_tokens, "batch": True}

    def params(self, system, user, schema):
        return {"model": self.model, "max_tokens": self.max_tokens,
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user}],
                "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}}}

    def submit(self, requests, log=print):
        """Create the batch and return its ID. Nothing waits here: the batch runs at Anthropic, so the laptop can close
        and collect() can fetch the results in a later session (results are kept for 29 days)."""
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": key, "params": self.params(*req)} for key, req in requests.items()])
        self.batch_ids.append(batch.id)
        log(f"Batch {batch.id}: {len(requests)} requests to {self.model}.")
        return batch.id

    def collect(self, batch_id, keys, log=print, wait=True, timeout_h=24.0):
        """{key: result} for a submitted batch, or None if it has not ended and wait is False. With wait, polls every
        poll_s seconds for up to timeout_h hours. `keys` are the custom IDs sent; one without a result is an error."""
        started = time.perf_counter()
        if batch_id not in self.batch_ids:
            self.batch_ids.append(batch_id)  # submitted in an earlier session
        while True:
            status = self.client.messages.batches.retrieve(batch_id)
            if status.processing_status == "ended":
                break
            if not wait:
                log(f"Batch {batch_id} is still {status.processing_status}.")
                return None
            if time.perf_counter() - started > timeout_h * 3600:
                raise RuntimeError(f"Batch {batch_id} still {status.processing_status} after {timeout_h} h.")
            self.sleep(self.poll_s)
        out = {}
        for item in self.client.messages.batches.results(batch_id):
            result = item.result
            if result.type != "succeeded":
                out[item.custom_id] = {"error": f"batch result {result.type}", "refusal": False}
                continue
            message = result.message
            self.usage["requests"] += 1
            for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                self.usage[name] += getattr(message.usage, name, 0) or 0
            if message.stop_reason == "refusal":
                out[item.custom_id] = {"error": f"{self.model} declined the request.", "refusal": True}
            elif message.stop_reason == "max_tokens":
                out[item.custom_id] = {"error": f"{self.model} hit max_tokens ({self.max_tokens}) before finishing the JSON.", "refusal": False}
            else:
                text = next(block.text for block in message.content if block.type == "text")
                try:
                    out[item.custom_id] = {"reply": json.loads(text)}
                except json.JSONDecodeError as e:
                    out[item.custom_id] = {"error": f"no valid JSON: {e}", "refusal": False}
        self.usage["seconds"] += time.perf_counter() - started
        for key in set(keys) - set(out):
            out[key] = {"error": "no batch result", "refusal": False}
        log(f"Batch {batch_id} ended: {sum('reply' in v for v in out.values())} of {len(keys)} succeeded.")
        return out

    def run(self, requests, log=print, timeout_h=24.0):
        """submit() then collect() in one blocking call."""
        if not requests:
            return {}
        return self.collect(self.submit(requests, log), list(requests), log, True, timeout_h)


class VLLMJSON:
    """Structured-output call to an open model served by vLLM's OpenAI-compatible server (DEC-24, DEC-21).

    One class serves every open model in the comparison (Qwen3.5-27B, MedGemma 27B text, Gemma 3 27B, and the fallback
    pair): only `model` and `base_url` change. It has the same call signature and attributes as ClaudeJSON, so
    run.py, end_to_end.py and batched.py use it unchanged. It has no run() method on purpose: batched.gather treats any
    llm with run() as a batch, and vLLM answers one request at a time.

    - Temperature 0 by default (vLLM accepts it, unlike current Claude models).
    - The schema goes in response_format as json_schema, which vLLM enforces with constrained decoding, so every model
      gets the same schema (prompt.api_schema()) and the comparison stays fair.
    - Only the final answer is parsed. If vLLM runs a reasoning parser, the model's thinking arrives separately and is
      counted (usage["reasoning_tokens"]), not parsed.
    - Usage keys match ClaudeJSON: input_tokens excludes cached prompt tokens, which go in cache_read_input_tokens.
    - cost_usd() returns None for these models: GPU time is billed per hour by the pod, not per token, and the pod's
      own log records it (compute/pod.py). describe() says so ("billing"), so a None cost never reads as free.
    - A local vLLM server needs no key. For a server that does, pass key_name (VLLM_KEY_NAME by convention) and the
      key comes from secret().
    - describe() records base_url only for a server on localhost or 127.0.0.1 (an SSH tunnel); any other address is
      recorded as "remote", so a pod's public IP and port never land in a run's config.json.
    - The client waits up to VLLM_TIMEOUT_S per request and retries once, not the openai defaults (600 s, 2 retries).
    - merge_system puts the system prompt at the top of the user message, for a chat template without a system role.
    - extra_body passes vLLM-only request options and is recorded in describe(). For example, Qwen3.5 thinks before
      answering by default; extra_body={"chat_template_kwargs": {"enable_thinking": False}} turns that off.
    - Context: vLLM rejects a request whose prompt plus max_tokens exceeds the server's --max-model-len. The reasoner's
      system prompt is about 10.6k tokens on Claude's tokenizer (EXP-8) and about 6k to 7k on the Qwen and Gemma
      tokenizers; answers ran about 3.1k to 3.4k (EXP-8, EXP-9). So the default is VLLM_MAX_TOKENS (8000): prompt plus
      answer budget stays near 15k, which a 16k context holds on a 24 GB card at 4-bit. A thinking model spends its
      reasoning from the same budget. The 20000 used for Claude (run.REASONER_MAX_TOKENS) would need about 27k.
    - A reply whose billed prompt tokens are far below the text sent (under one per MIN_CHARS_PER_BILLED_TOKEN
      characters) raises ProviderError: the provider dropped part of the prompt, as OpenRouter's Qwen route dropped the
      system message in EXP-37.
    - A reply with no choices (an API provider's upstream error, sent as a normal reply) is asked again after each of
      retry_waits_s, counted in usage["provider_errors"], and then raises ProviderError, which stops the run.
    - Errors from the server itself (unreachable, or a 4xx such as a request over --max-model-len or a schema it
      rejects) are not caught here, as in ClaudeJSON: a run stops loudly instead of recording every case as a format
      failure.
    - A cut-off answer (finish_reason "length") or one that does not parse raises with the start of the reply and, if
      the server split out the model's thinking, its length and start, so calls/ shows where the budget went.
    """

    def __init__(self, model, base_url="http://localhost:8000/v1", key_name=None, max_tokens=VLLM_MAX_TOKENS,
                 temperature=0.0, quantization=None, merge_system=False, extra_body=None, client=None,
                 provider="vllm", billing="gpu-time (pod), not token-priced"):
        if client is None:
            import openai

            key = secret(key_name) if key_name else None
            client = openai.OpenAI(base_url=base_url, api_key=key or "EMPTY", timeout=VLLM_TIMEOUT_S, max_retries=1)
        self.client = client
        self.model, self.base_url, self.max_tokens, self.temperature = model, base_url, max_tokens, temperature
        self.quantization, self.merge_system, self.extra_body = quantization, merge_system, extra_body
        self.provider, self.billing = provider, billing  # an OpenAI-compatible API (open_models.api_llm) says so
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0, "reasoning_tokens": 0, "seconds": 0.0, "provider_errors": 0}
        self.retry_waits_s = PROVIDER_RETRY_WAITS_S
        self._lock = threading.Lock()  # batched.gather may call from several threads (workers > 1)

    def describe(self):
        return {"provider": self.provider, "model": self.model, "base_url": recorded_url(self.base_url),
                "temperature": self.temperature, "max_tokens": self.max_tokens, "quantization": self.quantization,
                "merge_system": self.merge_system, "extra_body": self.extra_body,
                "billing": self.billing}

    def messages(self, system, user):
        if self.merge_system:
            return [{"role": "user", "content": f"{system}\n\n{user}"}]
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def __call__(self, system, user, schema):
        started = time.perf_counter()
        request = {
            "model": self.model,
            "messages": self.messages(system, user),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "response", "schema": schema, "strict": True}},
        }
        if self.extra_body:
            request["extra_body"] = self.extra_body
        for attempt in range(len(self.retry_waits_s) + 1):
            response = self.client.chat.completions.create(**request)
            if getattr(response, "choices", None):
                break
            with self._lock:
                self.usage["provider_errors"] += 1
            if attempt == len(self.retry_waits_s):
                raise ProviderError(f"{self.model}: the provider sent no answer {attempt + 1} times; last error: "
                                    f"{_provider_error(response)}")
            time.sleep(self.retry_waits_s[attempt])
        usage = getattr(response, "usage", None)
        billed = getattr(usage, "prompt_tokens", None) if usage is not None else None
        sent = len(system) + len(user)
        if billed is not None and billed * MIN_CHARS_PER_BILLED_TOKEN < sent:
            # Far fewer prompt tokens billed than the text sent: the provider dropped part of the prompt (EXP-37).
            raise ProviderError(f"{self.model}: {billed} prompt tokens billed for {sent} characters sent; the provider "
                                "dropped part of the prompt. Nothing is scored; the run stops.")
        with self._lock:
            self.usage["requests"] += 1
            self.usage["seconds"] += time.perf_counter() - started
            if usage is not None:
                prompt = getattr(usage, "prompt_tokens", 0) or 0
                cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                self.usage["input_tokens"] += prompt - cached
                self.usage["cache_read_input_tokens"] += cached
                self.usage["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                details = getattr(usage, "completion_tokens_details", None)
                self.usage["reasoning_tokens"] += getattr(details, "reasoning_tokens", 0) or 0
        choice = response.choices[0]
        message = choice.message
        if getattr(message, "refusal", None):
            raise RefusalError(f"{self.model} declined the request: {message.refusal}")
        text = message.content or ""
        thinking = _reasoning_note(message)
        if choice.finish_reason == "length":
            raise RuntimeError(f"{self.model} hit max_tokens ({self.max_tokens}) before finishing the JSON; "
                               f"reply began {text[:500]!r}{thinking}")
        if not text.strip():
            raise RuntimeError(f"{self.model} returned no content (finish_reason {choice.finish_reason!r}){thinking}.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # A RuntimeError, so run.py records it as "no valid JSON" with the start of the raw reply kept in calls/.
            raise RuntimeError(f"no valid JSON from {self.model}: {e}; reply began {text[:500]!r}{thinking}") from e


def recorded_url(base_url):
    """base_url as a run config may store it: unchanged on localhost or 127.0.0.1 (an SSH tunnel), else "remote"."""
    try:
        parsed = urlparse(base_url or "")
        host = parsed.hostname
    except ValueError:
        return "remote"
    if parsed.scheme == "https" and host == "openrouter.ai":
        return base_url  # a public API address, not a pod's (DEC-24 amended)
    return base_url if parsed.scheme == "http" and host in ("localhost", "127.0.0.1") else "remote"


def _reasoning_note(message):
    """'; thinking N chars, began ...' when vLLM's reasoning parser split the model's thinking out, else ''."""
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if not isinstance(reasoning, str) or not reasoning:
        return ""
    return f"; thinking {len(reasoning)} chars, began {reasoning[:200]!r}"
