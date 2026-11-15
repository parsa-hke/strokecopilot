"""Fixture-mode end to end with fake models: every stage runs, outputs are saved and a hallucination rate comes out."""

import json

import pytest

pytest.importorskip("numpy")

from strokecopilot.nlp.notes.facts import fixture_note  # noqa: E402
from strokecopilot.nlp.reasoner import run as rr  # noqa: E402
from strokecopilot.nlp.verifier import seeded_errors as se  # noqa: E402
from strokecopilot.pipeline import end_to_end as e2e  # noqa: E402

RUN_FIELDS = ("case_id", "model", "prompt_version")


class Fake:
    """reasoner: the gold output for the fixture whose note it is given; extractor: one rule claim per rationale
    entry; judge: supports everything."""

    def __init__(self, role, fail_on=None):
        self.role, self.fail_on, self.model, self.calls = role, fail_on, f"fake-{role}", 0
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        self.notes = {fixture_note(s)[0]: s for s in rr.fixture_stems()}

    def describe(self):
        return {"model": self.model}

    def __call__(self, system, user, schema):
        self.calls += 1
        self.usage["requests"] += 1
        if self.role == "reasoner":
            stem = self.notes.get(json.loads(user)["note"], "fx03_haemorrhage_present")  # a scenario gets fx03's gold
            if stem == self.fail_on:
                return {"oops": True}  # fails the schema: a format failure
            return {k: v for k, v in se.load_gold(stem).items() if k not in RUN_FIELDS}
        if self.role == "extractor":
            items = json.loads(user[user.index("["):])
            return {"claims": [{"field": e["field"], "entry": e["entry"], "text": e["text"], "kind": "rule", "subject": None,
                                "value": None, "unit": None, "applies_to_patient": False, "comparison": None}
                               for e in items if e["field"] == "rationale"]}
        return {"verdicts": [{"id": p["id"], "verdict": "supported", "scope_fits": True, "reason": "fake"}
                             for p in json.loads(user)["passages"]]}


def retriever(text, k):
    return [("AHA2026-T8-ich", 0.9), ("AHA2026-4.6.1-within-4.5h", 0.7)][:k]


def test_fixture_mode_runs_every_stage(tmp_path):
    metrics = e2e.run_fixtures(Fake("reasoner"), Fake("extractor"), Fake("judge"), retriever, tmp_path, log=lambda *_: None)
    s = metrics["summary"]
    assert s["cases"] == 5 and s["format_failure_rate"]["value"] == 0
    assert s["claims"] > 0 and 0 <= s["hallucination_rate"]["value"] <= 1
    assert s["recommendation_correctness"]["value"] == 1.0  # the gold decisions match the expected ones
    assert len(list((tmp_path / "outputs").iterdir())) == 5 and len(list((tmp_path / "traces").iterdir())) == 5
    assert json.loads((tmp_path / "config.json").read_text())["tau"] == 0.6


def test_a_reasoner_format_failure_is_counted_not_verified(tmp_path):
    metrics = e2e.run_fixtures(Fake("reasoner", fail_on="fx03_haemorrhage_present"), Fake("extractor"), Fake("judge"),
                               retriever, tmp_path, log=lambda *_: None)
    assert metrics["summary"]["format_failure_rate"]["value"] == pytest.approx(1 / 5)
    assert not (tmp_path / "traces" / "fx03_haemorrhage_present.json").exists()
    row = next(r for r in metrics["per_case"] if r["case"] == "fx03_haemorrhage_present")
    assert row["error"].startswith("format failure")


def test_rescore_replays_a_finished_run_without_models(tmp_path):
    first = e2e.run_fixtures(Fake("reasoner"), Fake("extractor"), Fake("judge"), retriever, tmp_path, log=lambda *_: None)
    again = e2e.rescore(tmp_path, retriever=retriever, log=lambda *_: None)
    assert again["summary"]["claims"] == first["summary"]["claims"]
    assert again["summary"]["hallucination_rate"] == first["summary"]["hallucination_rate"]
    flagged = json.loads((tmp_path / "flagged_rescore.json").read_text())
    assert len(flagged) == sum(r["hallucinated"] for r in again["per_case"]) and {"text", "reason", "kind"} <= set(flagged[0])


def test_criticality_and_omissions_reach_the_summary(tmp_path):
    metrics = e2e.run_fixtures(Fake("reasoner"), Fake("extractor"), Fake("judge"), retriever, tmp_path, log=lambda *_: None)
    s = metrics["summary"]
    assert s["omission_rate"]["value"] == 0.0  # the gold outputs ignore no hard rule
    assert s["critical_hallucination_rate"]["denominator"] > 0  # criticality is decided for some claims now


def test_scenarios_run_through_the_same_chain(tmp_path):
    cases = e2e.load_cases("scenarios")[:4]
    metrics = e2e.run_cases(cases, Fake("reasoner"), Fake("extractor"), Fake("judge"), retriever, tmp_path,
                            log=lambda *_: None, source="scenarios")
    assert metrics["summary"]["cases"] == 4
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["source"] == "scenarios" and config["scenarios"]["n"] == 150
    again = e2e.rescore(tmp_path, retriever=retriever, log=lambda *_: None)
    assert again["source"] == "scenarios" and again["summary"]["cases"] == 4


def test_real_cases_load_with_rule_expectations_and_refuse_heldout(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    from strokecopilot.data.splits import HeldoutError

    monkeypatch.delenv("ALLOW_HELDOUT", raising=False)
    stem = "fx02_thrombectomy_large_mismatch"
    vision_dir, notes_dir = tmp_path / "vision", tmp_path / "notes"
    vision_dir.mkdir(), notes_dir.mkdir()
    folds = {"heldout": ["sub-stroke9990"], "folds": {str(k): [f"sub-stroke99{k}{j}" for j in range(4)] for k in range(5)}}
    ids = [c for v in folds["folds"].values() for c in v] + ["sub-stroke9990"]
    for case in ids:  # a fixture's contract and note under placeholder IDs
        vision = json.loads((Path(e2e.FIXTURES_DIR) / f"{stem}.json").read_text())
        vision["case_id"] = case
        (vision_dir / f"{case}.json").write_text(json.dumps(vision))
        for ext in (".txt", ".facts.json"):
            shutil.copy(Path(e2e.FIXTURES_DIR) / "notes" / f"{stem}{ext}", notes_dir / f"{case}{ext}")
    with pytest.raises(HeldoutError):
        e2e.sample_real_ids(vision_dir, n=10, folds=folds)
    (vision_dir / "sub-stroke9990.json").unlink()
    sample = e2e.sample_real_ids(vision_dir, n=10, folds=folds)
    assert len(sample) == 10 and sample == e2e.sample_real_ids(vision_dir, n=10, folds=folds)
    assert all(sum(c in sample for c in v) == 2 for v in folds["folds"].values())
    cases = e2e.load_cases("real", ids=sample, vision_dir=vision_dir, notes_dir=notes_dir, folds=folds)
    fixture = next(c for c in e2e.load_cases("fixtures") if c["id"] == stem)
    assert len(cases) == 10 and cases[0]["expected"] == fixture["expected"]
    with pytest.raises(HeldoutError):
        e2e.load_cases("real", ids=["sub-stroke9990"], vision_dir=vision_dir, notes_dir=notes_dir, folds=folds)
    with pytest.raises(ValueError, match="need ids"):
        e2e.load_cases("real", vision_dir=vision_dir, notes_dir=notes_dir)
