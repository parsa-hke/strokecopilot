"""Contract tests (plan §6): fixtures validate, derived fields are correct, nulls are handled."""

import copy
import json

import pytest
from pydantic import ValidationError

from strokecopilot.config import FIXTURE_NOTES_DIR, FIXTURES_DIR
from strokecopilot.contract import (
    ContractError,
    VisionOutput,
    derive_mismatch,
    validate_recommendation,
    validate_vision_output,
)
from strokecopilot.contract.validate import load_schema

FIXTURE_STEMS = [
    "fx01_thrombolysis_clear",
    "fx02_thrombectomy_large_mismatch",
    "fx03_haemorrhage_present",
    "fx04_large_core_no_mismatch",
    "fx05_borderline_onset",
]


def _load_fixture(stem):
    with open(FIXTURES_DIR / f"{stem}.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def fx02():
    return _load_fixture("fx02_thrombectomy_large_mismatch")


# --- fixtures --------------------------------------------------------------------------------


def test_exactly_the_five_fixtures_exist():
    assert sorted(p.stem for p in FIXTURES_DIR.glob("fx*.json")) == FIXTURE_STEMS


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_fixture_validates(stem):
    out = validate_vision_output(_load_fixture(stem))
    assert out.case_id == f"fixture-{stem[:4]}"


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_fixture_has_note(stem):
    note = (FIXTURE_NOTES_DIR / f"{stem}.txt").read_text(encoding="utf-8")
    assert "NIHSS" in note


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_note_does_not_reveal_treatment_or_imaging(stem):
    # Plan §8.3: notes carry baseline clinical facts only.
    note = (FIXTURE_NOTES_DIR / f"{stem}.txt").read_text(encoding="utf-8").lower()
    for term in ("groin", "recanal", "tici", "thrombectomy", "thrombolysis", "penumbra", "aspects"):
        assert term not in note


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_fixture_does_not_use_real_case_ids(stem):
    # Fixtures are synthetic; a real ISLES'24 ID here could leak a held-out case.
    assert not _load_fixture(stem)["case_id"].startswith("sub-stroke")


# --- derived fields ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "core, penumbra, ratio, volume",
    [
        (18.0, 95.0, 5.28, 77.0),
        (110.0, 120.0, 1.09, 10.0),
        (15.0, 29.5, 1.97, 14.5),
        (1.0, 20.0, 20.0, 19.0),  # exactly 1 mL: ratio is computed
        (0.6, 22.4, None, 21.8),  # core < 1 mL: ratio null, volume still computed
        (0.0, 12.0, None, 12.0),  # zero core is 0.0, not null
        (30.0, 20.0, 0.67, -10.0),  # penumbra < core is representable
        (None, 40.0, None, None),
        (12.0, None, None, None),
        (None, None, None, None),
    ],
)
def test_derive_mismatch(core, penumbra, ratio, volume):
    assert derive_mismatch(core, penumbra) == (ratio, volume)


def test_wrong_mismatch_ratio_rejected(fx02):
    fx02["mismatch_ratio"] = 5.3
    with pytest.raises(ContractError, match="mismatch_ratio"):
        validate_vision_output(fx02)


def test_wrong_mismatch_volume_rejected(fx02):
    fx02["mismatch_volume_ml"] = 76.0
    with pytest.raises(ContractError, match="mismatch_volume_ml"):
        validate_vision_output(fx02)


def test_volume_with_two_decimals_rejected(fx02):
    fx02["core_ml"] = 18.04
    with pytest.raises(ContractError, match="rounded"):
        validate_vision_output(fx02)


# --- null handling -------------------------------------------------------------------------------


def test_all_volumes_null_is_valid(fx02):
    for key in ("core_ml", "penumbra_ml", "mismatch_ratio", "mismatch_volume_ml"):
        fx02[key] = None
    out = validate_vision_output(fx02)
    assert out.core_ml is None and out.mismatch_ratio is None


def test_null_core_with_non_null_ratio_rejected(fx02):
    fx02["core_ml"] = None
    with pytest.raises(ContractError, match="mismatch_ratio"):
        validate_vision_output(fx02)


def test_null_is_not_coerced_to_zero(fx02):
    fx02["penumbra_ml"] = None
    fx02["mismatch_ratio"] = None
    fx02["mismatch_volume_ml"] = None
    out = validate_vision_output(fx02)
    assert out.penumbra_ml is None
    assert out.core_ml == 18.0


def test_null_aspects_is_valid_and_stays_null(fx02):
    fx02["aspects"] = None
    assert validate_vision_output(fx02).aspects is None


def test_fixtures_cover_null_and_scored_aspects():
    values = [_load_fixture(stem)["aspects"] for stem in FIXTURE_STEMS]
    assert None in values and any(v is not None for v in values)


def test_zero_core_stays_zero(fx02):
    fx02.update(core_ml=0.0, mismatch_ratio=None, mismatch_volume_ml=95.0)
    assert validate_vision_output(fx02).core_ml == 0.0


@pytest.mark.parametrize("key", ["core_ml", "mismatch_ratio", "occlusion_site", "aspects", "warnings"])
def test_missing_field_rejected(fx02, key):
    # null must be explicit; a missing key is not "not computed".
    del fx02[key]
    with pytest.raises(ContractError, match="required"):
        validate_vision_output(fx02)


# --- schema-level rejections -------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, value",
    [
        (("schema_version",), "0.2"),
        (("occlusion_site",), "M3"),
        (("core_ml",), -1.0),
        (("aspects",), 11),
        (("aspects",), -1),
        (("aspects",), 7.5),
        (("haemorrhage", "present"), "true"),
        (("haemorrhage", "confidence"), 1.2),
        (("confidence", "penumbra"), -0.1),
        (("warnings",), [""]),
    ],
)
def test_invalid_value_rejected(fx02, path, value):
    target = fx02
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ContractError):
        validate_vision_output(fx02)


def test_extra_field_rejected(fx02):
    fx02["mass_effect"] = True
    with pytest.raises(ContractError, match="mass_effect"):
        validate_vision_output(fx02)


# --- the pydantic mirror matches the JSON Schema ------------------------------------------------


def _object_shape(schema):
    """{property: sub-shape or None} for every nested object in a JSON Schema."""
    return {
        name: _object_shape(sub) if sub.get("type") == "object" else None
        for name, sub in schema["properties"].items()
    }


def _model_shape(model):
    from pydantic import BaseModel

    shape = {}
    for name, field in model.model_fields.items():
        ann = field.annotation
        is_model = isinstance(ann, type) and issubclass(ann, BaseModel)
        shape[name] = _model_shape(ann) if is_model else None
    return shape


def test_pydantic_mirror_matches_json_schema_fields():
    assert _model_shape(VisionOutput) == _object_shape(load_schema("vision_output"))


def test_pydantic_mirror_matches_json_schema_required():
    schema = load_schema("vision_output")
    assert set(schema["required"]) == set(VisionOutput.model_fields)
    assert all(f.is_required() for f in VisionOutput.model_fields.values())


def test_pydantic_mirror_matches_occlusion_enum():
    schema_enum = load_schema("vision_output")["properties"]["occlusion_site"]["enum"]
    model_enum = VisionOutput.model_json_schema()["properties"]["occlusion_site"]["enum"]
    assert model_enum == schema_enum


def test_pydantic_rejects_what_json_schema_rejects(fx02):
    fx02["occlusion_site"] = "M3"
    with pytest.raises(ValidationError):
        VisionOutput.model_validate(fx02)


# --- recommendation schema -----------------------------------------------------------------------


@pytest.fixture
def recommendation():
    return {
        "case_id": "fixture-fx02",
        "model": "example-model",
        "prompt_version": "reasoner_v1",
        "thrombolysis": {
            "recommend": False,
            "rule_ids": ["AHA2026-4.6.3-rec2"],
            "blocking_factors": ["last known well 9 h ago, beyond 4.5 h"],
        },
        "thrombectomy": {"recommend": True, "rule_ids": ["DEFUSE3-inclusion-2"], "blocking_factors": []},
        "rationale": [
            {"claim": "Infarct core volume is 18.0 mL.", "sources": [{"type": "vision", "field": "core_ml"}], "derived": False},
            {"claim": "The patient was last known well 9 h ago.", "sources": [{"type": "note", "quote": "Time from last known well to imaging: 9 h 0 min."}], "derived": False},
            {"claim": "DEFUSE-3 requires core volume under 70 mL.", "sources": [{"type": "passage", "id": "DEFUSE3-inclusion-2"}], "derived": False},
            {
                "claim": "The core of 18.0 mL is within the DEFUSE-3 limit.",
                "sources": [{"type": "vision", "field": "core_ml"}, {"type": "passage", "id": "DEFUSE3-inclusion-2"}],
                "derived": True,
            },
        ],
        "uncertainties": ["onset time is last-known-well, not witnessed"],
    }


def test_recommendation_example_validates(recommendation):
    assert validate_recommendation(recommendation) is recommendation


@pytest.mark.parametrize(
    "source",
    [
        {"type": "vision", "field": "core volume"},
        {"type": "vision", "field": "case_id"},
        {"type": "passage"},
        {"type": "note"},
        {"type": "guess"},
        None,
    ],
)
def test_recommendation_bad_source_rejected(recommendation, source):
    recommendation["rationale"][0]["sources"] = [source]
    with pytest.raises(ContractError):
        validate_recommendation(recommendation)


def test_recommendation_claim_needs_a_source(recommendation):
    recommendation["rationale"][0]["sources"] = []
    with pytest.raises(ContractError):
        validate_recommendation(recommendation)


def test_recommendation_claim_needs_derived_flag(recommendation):
    del recommendation["rationale"][0]["derived"]
    with pytest.raises(ContractError, match="derived"):
        validate_recommendation(recommendation)


def test_recommendation_bad_prompt_version_rejected(recommendation):
    recommendation["prompt_version"] = "v1"
    with pytest.raises(ContractError):
        validate_recommendation(recommendation)


def _leaf_paths(schema, prefix=""):
    for name, sub in schema["properties"].items():
        if sub.get("type") == "object":
            yield from _leaf_paths(sub, f"{prefix}{name}.")
        else:
            yield f"{prefix}{name}"


def test_recommendation_vision_fields_cover_vision_output():
    metadata = {"schema_version", "case_id", "model_version"}
    vision_fields = set(_leaf_paths(load_schema("vision_output"))) - metadata
    enum = load_schema("recommendation")["$defs"]["vision_source"]["properties"]["field"]["enum"]
    assert set(enum) == vision_fields


def test_validators_do_not_mutate_input(fx02):
    before = copy.deepcopy(fx02)
    validate_vision_output(fx02)
    assert fx02 == before
