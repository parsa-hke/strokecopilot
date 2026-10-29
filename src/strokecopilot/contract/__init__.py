"""The vision -> NLP contract (plan §6): schema mirror, builder, validator."""

from strokecopilot.contract.schema import VisionOutput, derive_mismatch
from strokecopilot.contract.validate import (
    ContractError,
    load_vision_output,
    validate_recommendation,
    validate_vision_output,
)

__all__ = [
    "ContractError",
    "VisionOutput",
    "derive_mismatch",
    "load_vision_output",
    "validate_recommendation",
    "validate_vision_output",
]
