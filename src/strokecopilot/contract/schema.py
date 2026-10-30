"""Pydantic mirror of schemas/vision_output.json (plan §6.1-6.2).

The JSON Schema is the contract; this model mirrors it field for field and adds the rules JSON
Schema cannot express: volumes rounded to one decimal, and mismatch fields derived from the
two volumes. tests/test_contract.py checks that the two stay in sync.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.1"
OcclusionSite = Literal["ICA", "M1", "M2", "basilar", "none", "unknown"]

# Below this core volume the ratio is numerically meaningless and is reported as null.
MIN_CORE_FOR_RATIO_ML = 1.0
VOLUME_DECIMALS = 1
RATIO_DECIMALS = 2


def derive_mismatch(
    core_ml: Optional[float], penumbra_ml: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """Return (mismatch_ratio, mismatch_volume_ml) from the already-rounded volumes.

    ratio = penumbra / core, null if core < 1 mL; volume = penumbra - core.
    Either volume null -> both null.
    """
    if core_ml is None or penumbra_ml is None:
        return None, None
    volume = round(penumbra_ml - core_ml, VOLUME_DECIMALS)
    if core_ml < MIN_CORE_FOR_RATIO_ML:
        return None, volume
    return round(penumbra_ml / core_ml, RATIO_DECIMALS), volume


class _Strict(BaseModel):
    # strict: no silent coercion ("true" -> True, null -> 0); forbid: no undeclared fields.
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Haemorrhage(_Strict):
    present: bool
    confidence: float = Field(ge=0, le=1)


class Confidence(_Strict):
    core: float = Field(ge=0, le=1)
    penumbra: float = Field(ge=0, le=1)


class VisionOutput(_Strict):
    schema_version: Literal["0.1"]
    case_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    haemorrhage: Haemorrhage
    core_ml: Optional[float] = Field(ge=0)
    penumbra_ml: Optional[float] = Field(ge=0)
    mismatch_ratio: Optional[float] = Field(ge=0)
    mismatch_volume_ml: Optional[float]
    occlusion_site: OcclusionSite
    aspects: Optional[int] = Field(ge=0, le=10)
    confidence: Confidence
    warnings: list[str]

    @model_validator(mode="after")
    def _check_rounding_and_derived(self) -> "VisionOutput":
        for name in ("core_ml", "penumbra_ml", "mismatch_volume_ml"):
            value = getattr(self, name)
            if value is not None and round(value, VOLUME_DECIMALS) != value:
                raise ValueError(f"{name}={value} is not rounded to {VOLUME_DECIMALS} decimal")
        expected_ratio, expected_volume = derive_mismatch(self.core_ml, self.penumbra_ml)
        if self.mismatch_ratio != expected_ratio:
            raise ValueError(
                f"mismatch_ratio={self.mismatch_ratio}, expected {expected_ratio} "
                f"from core_ml={self.core_ml}, penumbra_ml={self.penumbra_ml}"
            )
        if self.mismatch_volume_ml != expected_volume:
            raise ValueError(
                f"mismatch_volume_ml={self.mismatch_volume_ml}, expected {expected_volume} "
                f"from core_ml={self.core_ml}, penumbra_ml={self.penumbra_ml}"
            )
        if any(not w for w in self.warnings):
            raise ValueError("warnings must be non-empty strings")
        return self
