"""Validate JSON objects against schemas/ (plan §6.2).

The pipeline must call validate_vision_output() before anything reaches the reasoner; an
invalid object raises ContractError and is never passed on.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Union

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from strokecopilot.config import SCHEMAS_DIR
from strokecopilot.contract.schema import VisionOutput


class ContractError(ValueError):
    """An object does not satisfy a schema in schemas/."""


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    """Load schemas/<name>.json, e.g. load_schema("vision_output")."""
    with open(SCHEMAS_DIR / f"{name}.json", encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    return schema


def _check_json_schema(obj: Any, name: str) -> None:
    validator = Draft202012Validator(load_schema(name))
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path))
    if errors:
        lines = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors]
        raise ContractError(f"{name} schema violation:\n  " + "\n  ".join(lines))


def validate_vision_output(obj: Any) -> VisionOutput:
    """Check obj against vision_output.json, then the rounding/derived-field rules."""
    _check_json_schema(obj, "vision_output")
    try:
        return VisionOutput.model_validate(obj)
    except ValidationError as e:
        raise ContractError(f"vision_output rule violation:\n{e}") from e


def validate_recommendation(obj: Any) -> dict:
    """Check obj against recommendation.json and return it unchanged."""
    _check_json_schema(obj, "recommendation")
    return obj


def load_vision_output(path: Union[str, Path]) -> VisionOutput:
    """Read and validate a vision_output JSON file."""
    with open(path, encoding="utf-8") as f:
        return validate_vision_output(json.load(f))
