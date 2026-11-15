"""Thin, identically-shaped LLM wrappers (plan §8.3). Phase 2 has Claude only.

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


def api_key():
    """ANTHROPIC_API_KEY from Colab Secrets, Kaggle Secrets or the environment."""
    try:
        from google.colab import userdata

        return userdata.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


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
