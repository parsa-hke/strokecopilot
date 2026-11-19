"""The Space's bank (demo/bank.py): preset inputs, aliasing, writing and the leak scan. No run cells needed."""

import json

import pytest

from strokecopilot.demo import bank, builder
from strokecopilot.nlp.verifier import scenarios as sc
from strokecopilot.pipeline import end_to_end as e2e


def test_every_preset_maps_to_valid_builder_inputs():
    tags = {s.id: s.tags for s in sc.generate()}
    cases = e2e.load_cases("scenarios") + e2e.load_cases("fixtures")
    inputs = [bank.preset_inputs(c, tags.get(c["id"])) for c in cases]
    assert all(i is not None for i in inputs)
    s = cases[0]
    assert inputs[0]["core_ml"] == round(s["vision"]["core_ml"] * 2) / 2 and inputs[0]["site"] == s["vision"]["occlusion_site"]


def test_aliasing_replaces_the_id_everywhere():
    obj = {"vision": {"case_id": "sub-stroke9901"}, "rec": {"case_id": "sub-stroke9901"}, "text": "x sub-stroke9901 y"}
    out = bank._alias_everything(obj, "sub-stroke9901", "D-01")
    assert "sub-stroke9901" not in json.dumps(out) and out["vision"]["case_id"] == "D-01"


def test_write_and_scan(tmp_path):
    row = {"id": "scenario-001", "kind": "scenario", "title": "t", "rules": {}, "models": [], "inputs": None}
    out = bank.write([(row, {"id": "scenario-001", "arms": {}})], tmp_path / "b")
    assert json.loads((out / "index.json").read_text())["cases"][0]["id"] == "scenario-001"
    assert bank.scan(out, ["sub-stroke9901"]) == []
    (out / "cases" / "leak.json").write_text('{"note": "sub-stroke9901", "k": "sk-ant-api03-abcdefghijkl"}')
    assert len(bank.scan(out, ["sub-stroke9901"])) == 2
    with pytest.raises(FileExistsError):
        bank.write([], out)


def test_builder_bounds_cover_every_preset_feature():
    for s in sc.generate():
        f = builder.features(s.vision, s.facts, {d: v for d, v in s.expected.items()})
        assert f["site"] in builder.CHOICES["site"]
