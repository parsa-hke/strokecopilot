"""Versioned reasoner prompt (prompts/reasoner_vN.md) and the schema sent to the API. Plan §8.3, DEC-19.

The system prompt holds the instructions and the whole rulebook, so it is identical for every case and is cached.
The user message holds only the case: the vision JSON and the triage note.
"""

import copy
import json
from pathlib import Path

from strokecopilot.config import PROMPTS_DIR
from strokecopilot.contract.validate import load_schema
from strokecopilot.nlp.corpus import ingest

REASONER_PROMPT = "reasoner_v1"
# Filled in by run.py, not by the model.
RUN_FIELDS = ("case_id", "model", "prompt_version")
# Bookkeeping fields of vision_output.json that say nothing about the patient.
VISION_HIDDEN = ("schema_version", "case_id", "model_version")
# Structured outputs accept a subset of JSON Schema: no string or array length limits, no patterns.
_UNSUPPORTED = ("minLength", "maxLength", "minItems", "maxItems", "pattern", "$schema", "$id")


def system_prompt(version=REASONER_PROMPT, rulebook_path=ingest.PASSAGES):
    template = (Path(PROMPTS_DIR) / f"{version}.md").read_text(encoding="utf-8")
    if "{{RULEBOOK}}" not in template:
        raise ValueError(f"{version} has no {{{{RULEBOOK}}}} slot")
    return template.replace("{{RULEBOOK}}", ingest.prompt_block(rulebook_path))


def user_message(vision, note):
    shown = {k: v for k, v in vision.items() if k not in VISION_HIDDEN}
    return json.dumps({"vision": shown, "note": note}, ensure_ascii=False, indent=1)


def _clean(node):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in _UNSUPPORTED:
                continue
            out["anyOf" if key == "oneOf" else key] = _clean(value)
        return out
    if isinstance(node, list):
        return [_clean(v) for v in node]
    return node


def api_schema():
    """schemas/recommendation.json without the run fields and without constraints the API rejects.

    The full schema, constraints included, is checked on the assembled output by validate_recommendation."""
    schema = copy.deepcopy(load_schema("recommendation"))
    for name in RUN_FIELDS:
        schema["properties"].pop(name)
        schema["required"].remove(name)
    return _clean(schema)
