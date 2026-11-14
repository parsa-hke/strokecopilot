"""Thin, identically-shaped LLM wrappers (plan §8.3): Claude (direct and batched) and open models served by vLLM.

Every wrapper is a callable `llm(system, user, schema) -> dict`: one request whose reply is JSON
matching `schema`. The verifier (claim extractor and judge) and, later, the reasoner use this
shape, so tests pass a plain function instead.

The key comes from Colab or Kaggle Secrets, or the ANTHROPIC_API_KEY environment variable, never
from code. Current Claude models reject `temperature` (docs/HALLUCINATION_METRIC.md §6), so none is
sent. Server-side refusal fallbacks are left off on purpose: a fallback would swap the judge or
extractor model mid-run, and the verifier must record exactly which model produced each verdict.
A refusal is raised instead, and the caller records it.
"""

import json
import os
import time

DEFAULT_MAX_TOKENS = 16000

# USD per million tokens (input, output) on the Claude API, prompts up to 100K tokens. Cache reads: Opus 5.5 $0.20,
# Sonnet 5.5 $0.10 (0.05x input; it was $0.20 here until 12 Nov 2026, which overstated logged Sonnet costs, as Hana
# found), Haiku 5.5 $0.01. 5-minute cache writes are 1.25x input. Check before quoting a cost.
PRICES = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.10},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
}


def cost_usd(model, usage):
    """Estimated cost of `usage` (a ClaudeJSON.usage dict), or None for an unknown model."""
    p = PRICES.get(model)
    if p is None:
        return None
    return (usage.get("input_tokens", 0) * p["input"] + usage.get("output_tokens", 0) * p["output"]
            + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
            + usage.get("cache_creation_input_tokens", 0) * p["input"] * 1.25) / 1e6


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

    def run(self, requests, log=print, timeout_h=24.0):
        if not requests:
            return {}
        started = time.perf_counter()
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": key, "params": self.params(*req)} for key, req in requests.items()])
        self.batch_ids.append(batch.id)
        log(f"Batch {batch.id}: {len(requests)} requests to {self.model}.")
        while True:
            status = self.client.messages.batches.retrieve(batch.id)
            if status.processing_status == "ended":
                break
            if time.perf_counter() - started > timeout_h * 3600:
                raise RuntimeError(f"Batch {batch.id} still {status.processing_status} after {timeout_h} h.")
            self.sleep(self.poll_s)
        out = {}
        for item in self.client.messages.batches.results(batch.id):
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
        missing = set(requests) - set(out)
        for key in missing:
            out[key] = {"error": "no batch result", "refusal": False}
        log(f"Batch {batch.id} ended: {sum('reply' in v for v in out.values())} of {len(requests)} succeeded.")
        return out


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
      own log records it (compute/pod.py).
    - A local vLLM server needs no key. For a server that does, pass key_name and the key comes from secret().
    - merge_system puts the system prompt at the top of the user message, for a chat template without a system role.
    - extra_body passes vLLM-only request options and is recorded in describe(). For example, Qwen3.5 thinks before
      answering by default; extra_body={"chat_template_kwargs": {"enable_thinking": False}} turns that off.
    - Context: the server's --max-model-len must hold the prompt plus max_tokens. The reasoner's system prompt is about
      10.6k tokens (EXP-8) and its answers about 3.1k to 3.4k (EXP-8, EXP-9), so max_tokens around 8000 leaves room
      on a 24 GB card; the 20000 used for Claude (run.REASONER_MAX_TOKENS) needs a context of about 31k.
    - Errors from the server itself (unreachable, or a 4xx such as a schema it rejects) are not caught here, as in
      ClaudeJSON: a run stops loudly instead of recording every case as a format failure.
    """

    def __init__(self, model, base_url="http://localhost:8000/v1", key_name=None, max_tokens=DEFAULT_MAX_TOKENS,
                 temperature=0.0, quantization=None, merge_system=False, extra_body=None, client=None):
        if client is None:
            import openai

            key = secret(key_name) if key_name else None
            client = openai.OpenAI(base_url=base_url, api_key=key or "EMPTY", max_retries=2)
        self.client = client
        self.model, self.base_url, self.max_tokens, self.temperature = model, base_url, max_tokens, temperature
        self.quantization, self.merge_system, self.extra_body = quantization, merge_system, extra_body
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0, "reasoning_tokens": 0, "seconds": 0.0}

    def describe(self):
        return {"provider": "vllm", "model": self.model, "base_url": self.base_url, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "quantization": self.quantization, "merge_system": self.merge_system,
                "extra_body": self.extra_body}

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
        response = self.client.chat.completions.create(**request)
        self.usage["requests"] += 1
        self.usage["seconds"] += time.perf_counter() - started
        usage = getattr(response, "usage", None)
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
        if choice.finish_reason == "length":
            raise RuntimeError(f"{self.model} hit max_tokens ({self.max_tokens}) before finishing the JSON.")
        text = message.content
        if not text or not text.strip():
            raise RuntimeError(f"{self.model} returned no content (finish_reason {choice.finish_reason!r}).")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # A RuntimeError, so run.py records it as "no valid JSON" with the start of the raw reply kept in calls/.
            raise RuntimeError(f"no valid JSON from {self.model}: {e}; reply began {text[:500]!r}") from e
