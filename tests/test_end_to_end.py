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
            stem = self.notes[json.loads(user)["note"]]
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
    row = next(r for r in metrics["per_fixture"] if r["fixture"] == "fx03_haemorrhage_present")
    assert row["error"].startswith("format failure")
