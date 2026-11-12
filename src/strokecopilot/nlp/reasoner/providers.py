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
