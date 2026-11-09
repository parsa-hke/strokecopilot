"""Tests for nlp/verifier/adjudicate.py."""

import csv
import json
from pathlib import Path

import pytest

from strokecopilot.nlp.verifier.adjudicate import (
    blind_and_export,
    cohen_kappa,
    collect_flagged,
    sample_claims,
    summarise_agreement,
)


@pytest.fixture
def two_run_folders(tmp_path):
    """Two fake run folders with flagged claims in each."""
    for exp, model, claims in [
        ("EXP-A", "model-1", [
            {"fixture": "fx01", "flagged": [{"claim_id": "rationale[0].0", "subtype": "wrong_rule", "reason": "r"}]},
            {"fixture": "fx02", "flagged": [{"claim_id": "rationale[1].0", "subtype": "uncited_rule", "reason": "r"}]},
        ]),
        ("EXP-B", "model-2", [
            {"fixture": "fx01", "flagged": [{"claim_id": "rationale[2].0", "subtype": "wrong_rule", "reason": "r"}]},
        ]),
    ]:
        folder = tmp_path / exp
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({
            "experiment": exp, "mode": "fixtures", "reasoner": {"model": model},
        }))
        (folder / "metrics.json").write_text(json.dumps({"per_fixture": claims, "summary": {}}))
    return [tmp_path / "EXP-A", tmp_path / "EXP-B"]


def test_collect_flagged(two_run_folders):
    out = collect_flagged(two_run_folders)
    assert len(out) == 3
    assert {e["model"] for e in out} == {"model-1", "model-2"}
    assert all(e["subtype"] is not None for e in out)


def test_sample_claims_caps_per_model(two_run_folders):
    sampled = sample_claims(collect_flagged(two_run_folders), n_per_model=1, seed=42)
    by_model = {}
    for e in sampled:
        by_model[e["model"]] = by_model.get(e["model"], 0) + 1
    assert all(v <= 1 for v in by_model.values())


def test_blind_and_export_has_no_model(tmp_path, two_run_folders):
    sampled = sample_claims(collect_flagged(two_run_folders), n_per_model=5, seed=42)
    out = tmp_path / "sheet.csv"
    blind_and_export(sampled, out, seed=42)
    text = out.read_text().lower()
    assert "model-1" not in text
    assert "model-2" not in text
    assert "exp-a" not in text
    assert "exp-b" not in text
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == len(sampled)
    assert "label_A" in rows[0]
    assert "label_B" in rows[0]


def test_cohen_kappa_perfect():
    assert cohen_kappa(["real", "verifier", "real"], ["real", "verifier", "real"]) == 1.0


def test_cohen_kappa_disagree():
    k = cohen_kappa(["real", "real", "real", "real"], ["verifier", "verifier", "verifier", "verifier"])
    assert k < 0.0  # total disagreement → kappa negative


def test_summarise_agreement(two_run_folders):
    k = summarise_agreement(
        labels_a=["real_reasoner_error", "verifier_mistake", "real_reasoner_error"],
        labels_b=["real_reasoner_error", "real_reasoner_error", "real_reasoner_error"],
        subtypes=["wrong_rule", "uncited_rule", "wrong_rule"],
    )
    assert k["n"] == 3
    assert k["kappa"] is not None
    assert 0.0 <= k["agreement_with_verifier_A"] <= 1.0