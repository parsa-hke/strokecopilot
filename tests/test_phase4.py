"""Phase 4's held-out commands (pipeline/phase4.py) on fixtures posing as held-out cases: no real held-out data."""

import json

import pytest

from strokecopilot.config import FIXTURES_DIR
from strokecopilot.nlp.reasoner import run as rr
from strokecopilot.pipeline import phase4

STEMS = rr.fixture_stems()


def test_refuses_without_the_flag(monkeypatch):
    monkeypatch.delenv("ALLOW_HELDOUT", raising=False)
    with pytest.raises(phase4.Phase4Error, match="ALLOW_HELDOUT=1"):
        phase4.heldout_cases()


def test_check_counts_and_never_prints_ids(tmp_path):
    dirs = {k: tmp_path / k for k in ("predicted", "groundtruth", "notes")}
    for d in dirs.values():
        d.mkdir()
    for s in STEMS:
        obj = json.loads((FIXTURES_DIR / f"{s}.json").read_text())
        obj["case_id"] = s
        (dirs["groundtruth"] / f"{s}.json").write_text(json.dumps(obj))
        obj = {**obj, "haemorrhage": {"present": True, "confidence": 0.5}}  # predicted: a false bleed flag on every case
        (dirs["predicted"] / f"{s}.json").write_text(json.dumps(obj))
        for ext in (".txt", ".facts.json"):
            (dirs["notes"] / f"{s}{ext}").write_text((FIXTURES_DIR / "notes" / f"{s}{ext}").read_text())
    out = phase4.check(dirs, STEMS)
    assert out["valid"] == {"predicted": 5, "groundtruth": 5, "notes": 5}
    assert out["rules_on_predicted"]["thrombolysis"] == {"no": 5}
    assert out["rules_on_groundtruth"]["thrombectomy"].get("yes", 0) >= 3
    assert out["rules_on_predicted_vs_truth"]["pooled"]["scored"] > 0
    assert not any(s in json.dumps(out) for s in STEMS)


def fake_dirs(tmp_path):
    dirs = {k: tmp_path / k for k in ("predicted", "groundtruth", "notes")}
    for d in dirs.values():
        d.mkdir(exist_ok=True)
    for s in STEMS:
        obj = json.loads((FIXTURES_DIR / f"{s}.json").read_text())
        obj["case_id"] = s
        for arm in ("predicted", "groundtruth"):
            (dirs[arm] / f"{s}.json").write_text(json.dumps(obj))
        for ext in (".txt", ".facts.json"):
            (dirs["notes"] / f"{s}{ext}").write_text((FIXTURES_DIR / "notes" / f"{s}{ext}").read_text())
    return dirs


class FakeLLM:
    """reasoner: gold answers; extractor: one rule claim per rationale entry; judge: supports everything."""

    def __init__(self, role, model):
        self.role, self.model = role, model
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    def __call__(self, system, user, schema):
        from strokecopilot.nlp.corpus import ingest
        from strokecopilot.nlp.reasoner import gold

        self.usage["requests"] += 1
        if self.role == "reasoner":
            from strokecopilot.nlp.notes.facts import fixture_note

            stem = next(st for st in STEMS if fixture_note(st)[0] == json.loads(user)["note"])
            vision = json.loads((FIXTURES_DIR / f"{stem}.json").read_text())
            return gold.gold_answer(vision, fixture_note(stem)[1], ingest.load())
        if self.role == "extractor":
            items = json.loads(user[user.index("["):])
            return {"claims": [{"field": e["field"], "entry": e["entry"], "text": e["text"], "kind": "rule", "subject": None, "value": None,
                                "unit": None, "applies_to_patient": False, "comparison": None} for e in items if e["field"] == "rationale"]}
        return {"verdicts": [{"id": p["id"], "verdict": "supported", "scope_fits": True, "reason": "fake"} for p in json.loads(user)["passages"]]}


def test_cells_and_sensitivity_arms(tmp_path):
    assert len(phase4.cells()) == 6 * 2 + 2
    dirs = fake_dirs(tmp_path)
    base = phase4.arm_cases(phase4.pr.SONNET, "heldout_predicted", dirs, STEMS)
    up = phase4.arm_cases(phase4.pr.SONNET, "heldout_core_x1.2", dirs, STEMS)
    assert [c["id"] for c in up] == [f"{s}__core_x1.2" for s in STEMS]
    assert up[1]["vision"]["core_ml"] == round(base[1]["vision"]["core_ml"] * 1.2, 1)
    with pytest.raises(KeyError):
        phase4.arm_cases("gemma-3-27b", "heldout_core_x1.2", dirs, STEMS)  # only Sonnet has the sensitivity arms


def test_colab_pass_import_and_advance(tmp_path):
    pytest.importorskip("sentence_transformers")
    dirs = fake_dirs(tmp_path)
    colab, mac = tmp_path / "drive_run", tmp_path / "mac_run"
    model = "gemma-3-4b"
    done = phase4.reasoner_pass([model], dirs, STEMS, colab, lambda m: FakeLLM("reasoner", m), workers=2, log=lambda *_: None)
    assert done == {(model, "heldout_predicted"): 5, (model, "heldout_groundtruth"): 5}
    assert phase4.import_answers(colab, mac, [model])[(model, "heldout_predicted")] == 5
    assert phase4.import_answers(colab, mac, [model])[(model, "heldout_predicted")] == 0  # nothing new the second time
    assert phase4.answered(model, "heldout_predicted", dirs, STEMS, mac) == (5, 5)
    rows = phase4.advance_all([model], dirs, STEMS, mac, wait=True, log=lambda *_: None,
                              llm_for=lambda m: FakeLLM("reasoner", m),
                              verifier=(FakeLLM("extractor", "claude-sonnet-5-5"), FakeLLM("judge", "claude-haiku-5-5")))
    assert [r["state"] for r in rows] == ["done", "done"]
    assert (phase4.cell_dir(mac, model, "heldout_groundtruth") / "metrics.json").exists()
    assert {r["state"] for r in phase4.status([model], dirs, STEMS, mac)} == {"done"}


def test_open_models_wait_for_colab(tmp_path):
    pytest.importorskip("sentence_transformers")
    dirs = fake_dirs(tmp_path)
    rows = phase4.advance_all(["medgemma-27b-text"], dirs, STEMS, tmp_path / "run", llm_for=lambda m: None,
                              verifier=(None, None))
    assert all(r["state"].startswith("needs Colab (0/5") for r in rows)


def test_analyse_on_finished_cells_only(tmp_path):
    pytest.importorskip("sentence_transformers")
    dirs = fake_dirs(tmp_path)
    run = tmp_path / "run"
    model = "gemma-3-4b"
    phase4.reasoner_pass([model], dirs, STEMS, run, lambda m: FakeLLM("reasoner", m), workers=2, log=lambda *_: None)
    phase4.advance_all([model], dirs, STEMS, run, wait=True, log=lambda *_: None, llm_for=lambda m: FakeLLM("reasoner", m),
                       verifier=(FakeLLM("extractor", "claude-sonnet-5-5"), FakeLLM("judge", "claude-haiku-5-5")))
    rep = phase4.analyse(dirs, STEMS, run, tmp_path / "no_drive_runs", models=[model, "medgemma-27b-text"],
                         development_cells={})
    assert rep["not_finished"] == ["medgemma-27b-text"] and list(rep["models"]) == [model]
    assert rep["models"][model]["predicted"]["correctness"]["cases"] == 5
    assert "vision" not in rep and rep["scenario_sensitivity"]["cases"] == 150
    assert not any(s in json.dumps(rep) for s in STEMS)
