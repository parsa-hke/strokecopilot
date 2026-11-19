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


def test_space_pyproject_drops_the_split_from_the_wheel():
    from strokecopilot.demo import space_build as sb

    text = '[tool.hatch.build.targets.wheel.force-include]\n"schemas" = "strokecopilot/schemas"\n"splits" = "strokecopilot/splits"\n'
    out = sb.space_pyproject(text)
    assert '"splits"' not in out and '"schemas"' in out


def test_space_tree_check_catches_every_leak(tmp_path):
    from strokecopilot.demo import space_build as sb

    tree = tmp_path / "space"
    (tree / "static").mkdir(parents=True)
    (tree / "static" / "app.js").write_text("const ok = 1; // @decorator.style is not an address")
    assert sb.check(tree, ["sub-stroke9901"]) == []
    (tree / "pkg" / "splits").mkdir(parents=True)
    (tree / "pkg" / "splits" / "folds.json").write_text("{}")
    (tree / "static" / "x.js").write_text("// contact someone@example.org")
    (tree / "static" / "y.json").write_text('{"case": "sub-stroke9901"}')
    (tree / "static" / "z.exe").write_bytes(b"MZ")
    problems = "; ".join(sb.check(tree, ["sub-stroke9901"]))
    for expected in ("split folder", "split file", "e-mail", "case ID", "file type not allowed"):
        assert expected in problems


def test_story_slice_is_where_the_mask_is_largest():
    import numpy as np

    from strokecopilot.demo import images

    a = np.zeros((4, 4, 6), bool)
    a[:2, :2, 1] = True
    b = np.zeros_like(a)
    b[:, :, 4] = True
    assert images.pick_slice(a) == 1 and images.pick_slice(a, b) == 4


def test_heldout_cases_are_aliased_with_their_own_kind(tmp_path, monkeypatch):
    fixture = e2e.load_cases("fixtures")[0]

    def fake_real_case(case_id, vision_dir, notes_dir):
        return {**fixture, "id": case_id, "vision": {**fixture["vision"], "case_id": case_id}, "note": f"Note for {case_id}."}

    monkeypatch.setattr(e2e, "real_case", fake_real_case)
    monkeypatch.setattr(bank, "_arm_view", lambda case, folder: {"available": True, "seen": case["id"], "folder": folder.name})
    run = tmp_path / "EXP-65"
    for arm in bank.HELDOUT_ARMS.values():
        (run / "claude-sonnet-5-5" / arm).mkdir(parents=True)
        (run / "claude-sonnet-5-5" / arm / "config.json").write_text("{}")
    ids = ["sub-stroke9902", "sub-stroke9901"]
    out = bank.heldout_cases(run, {"predicted": "p", "groundtruth": "g"}, "n", ids)
    assert [r["id"] for r, _ in out] == ["H-01", "H-02"]
    row, file = out[0]
    assert row["kind"] == file["kind"] == "heldout" and row["title"] == "Held-out case H-01"
    assert row["models"] == ["claude-sonnet-5-5"]
    assert file["arms"]["predicted"]["answers"]["claude-sonnet-5-5"]["folder"] == "heldout_predicted"
    assert file["arms"]["groundtruth"]["answers"]["claude-sonnet-5-5"]["folder"] == "heldout_groundtruth"
    assert not any(i in json.dumps(out) for i in ids)


def test_heldout_chart_data_keeps_numbers_and_aliases_only():
    from strokecopilot.demo import charts

    rate = {"value": 0.05, "ci95": [0.02, 0.08], "numerator": 5, "denominator": 100}
    corr = {"cases": 20, "no_answer": 1, "pooled": {"correct": 22, "scored": 23, "rate": 0.9565, "ci95": [0.87, 1.0]}}
    model = {"predicted": {"correctness": corr, "error_propagation": {"table": {"vision": {"wrong": 1, "caught": 0}, "reasoning": {"wrong": 0, "caught": 0}},
                                                                       "examples": [{"alias": "H-07", "decision": "thrombolysis", "cause": "vision", "caught": False,
                                                                                     "contract_differences": ["bleed flag"]}]}},
             "groundtruth": {"correctness": corr},
             "c9": {"heldout": {"hallucination_rate": rate, "format_failure_rate": {**rate, "numerator": 1}}, "development": {"hallucination_rate": rate}}}
    sens = {"decisions": 40, "changed": 0}
    report = {"experiment": "EXP-66", "cases": 20, "truth": {"thrombolysis": {"undetermined": 17}, "thrombectomy": {"no": 20}},
              "rules_on_predicted_correct": {"pooled": corr["pooled"]}, "models": {"claude-sonnet-5-5": model},
              "rule_sensitivity": {"x0.8": sens, "x1.2": sens, "cases": 20}, "sonnet_sensitivity": None,
              "vision": {"dice": {"mean": 0.18}}}
    out = charts.heldout(report)
    assert [r["model"] for r in out["rows"]] == ["claude-sonnet-5-5"]
    r = out["rows"][0]
    assert r["predicted"]["correct"] == 22 and r["format_failures"] == 1 and r["errors"]["vision"]["wrong"] == 1
    assert r["examples"] == [{"alias": "H-07", "decision": "thrombolysis", "cause": "vision", "caught": False}]
    assert set(out["rule_sensitivity"]) == {"x0.8", "x1.2"} and out["source"].startswith("EXP-66")


def test_space_readme_metadata_passes_hugging_face_limits():
    from strokecopilot.demo import space_build

    head = (space_build.SPACE_SRC / "README.md").read_text().split("---")[1]
    meta = dict(line.split(": ", 1) for line in head.strip().splitlines())
    assert meta["sdk"] == "docker" and meta["app_port"] == "7860"
    assert len(meta["short_description"]) <= 60  # Hugging Face refuses the upload otherwise
