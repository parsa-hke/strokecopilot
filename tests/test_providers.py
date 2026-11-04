"""ClaudeJSON with a stub client: request shape, usage counting and refusals. No network."""

from types import SimpleNamespace as NS

import pytest

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
