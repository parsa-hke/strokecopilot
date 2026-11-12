"""Tests for nlp/verifier/adjudicate.py: collection from run folders, stratified sampling, the blind sheet and its key,
case inputs per alias, Cohen's kappa against known values, and agreement per sub-type and per model. No network."""

import csv
import json

import pytest

from strokecopilot.nlp.verifier import adjudicate as adj

STEMS = ("fx01_thrombolysis_clear", "fx02_thrombectomy_large_mismatch")


def _flag(case, claim_id, subtype, text):
    return {"case": case, "claim_id": claim_id, "subtype": subtype, "critical": False, "reason": "verifier reason",
            "kind": "rule", "text": text, "sources": ["AHA2026-4.6.1-within-4.5h"]}


@pytest.fixture
def two_run_folders(tmp_path):
    """Two run folders laid out as pipeline/end_to_end.py writes them (config.json with source, flagged files)."""
    runs = [("EXP-A", "model-1", [_flag(STEMS[0], "rationale[0].0", "wrong_rule", "Claim one"),
                                  _flag(STEMS[1], "thrombolysis.blocking_factors[0].1", "uncited_rule", "Claim two")]),
            ("EXP-B", "model-2", [_flag(STEMS[0], "rationale[2].0", "wrong_rule", "Claim three")])]
    folders = []
    for exp, model, flagged in runs:
        folder = tmp_path / exp
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({"experiment": exp, "source": "fixtures", "reasoner": {"model": model}}))
        (folder / "flagged.json").write_text(json.dumps([_flag(STEMS[0], "rationale[9].0", "wrong_rule", "stale")]))
        (folder / "flagged_rescore.json").write_text(json.dumps(flagged))
        folders.append(folder)
    return folders


def test_collect_flagged_reads_the_newest_scoring(two_run_folders):
    out = adj.collect_flagged(two_run_folders)
    assert len(out) == 3 and {e["model"] for e in out} == {"model-1", "model-2"}
    assert all(e["flagged_file"] == "flagged_rescore.json" and e["text"] != "stale" for e in out)
    explicit = adj.collect_flagged(two_run_folders[:1], flagged_name="flagged.json")
    assert [e["text"] for e in explicit] == ["stale"]


def test_collect_flagged_refuses_held_out_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(adj.splits, "check_ids", lambda ids: (_ for _ in ()).throw(adj.splits.HeldoutError("held out")))
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"experiment": "X"}))
    (folder / "flagged.json").write_text(json.dumps([_flag("some-case", "rationale[0].0", "wrong_rule", "t")]))
    with pytest.raises(adj.splits.HeldoutError):
        adj.collect_flagged([folder])


def test_sample_claims_caps_per_model_is_deterministic_and_leaves_input_alone(two_run_folders):
    flagged = adj.collect_flagged(two_run_folders)
    before = list(flagged)
    one = adj.sample_claims(flagged, n_per_model=1, seed=42)
    assert sorted(e["model"] for e in one) == ["model-1", "model-2"]
    assert adj.sample_claims(flagged, n_per_model=1, seed=42) == one and flagged == before


def test_the_sheet_is_blind_and_the_key_unblinds_it(tmp_path, two_run_folders):
    sampled = adj.sample_claims(adj.collect_flagged(two_run_folders), n_per_model=5, seed=42)
    sheet, key, aliases = adj.blind_and_export(sampled, tmp_path / "sheet.csv", tmp_path / "private" / "key.csv", seed=42)
    text = sheet.read_text().lower()
    for hidden in ("model-1", "model-2", "exp-a", "exp-b", "wrong_rule", "uncited_rule", "verifier reason", *STEMS):
        assert hidden not in text, hidden
    rows = list(csv.DictReader(sheet.open()))
    assert len(rows) == len(sampled) and {r["claim"] for r in rows} == {"Claim one", "Claim two", "Claim three"}
    assert {r["field"] for r in rows} == {"rationale", "thrombolysis.blocking_factors"}
    assert all(r["case_alias"].startswith("C") for r in rows) and len(aliases) == 3  # one alias per run and case
    keyed = {r["sample_id"]: r for r in csv.DictReader(key.open())}
    assert set(keyed) == {r["sample_id"] for r in rows} and {r["model"] for r in keyed.values()} == {"model-1", "model-2"}


def test_case_inputs_are_written_per_alias_without_ids(tmp_path, two_run_folders):
    sampled = adj.sample_claims(adj.collect_flagged(two_run_folders), n_per_model=5)
    _, _, aliases = adj.blind_and_export(sampled, tmp_path / "s.csv", tmp_path / "k.csv")
    out = adj.export_case_inputs(aliases, tmp_path / "cases")
    files = sorted(p.name for p in out.iterdir())
    assert files == sorted(f"{a}.md" for a in aliases.values())
    body = (out / files[0]).read_text()
    assert "## Triage note" in body and '"case_id"' not in body and "fixture-" not in body


@pytest.mark.parametrize("a, b, expected", [
    (["r", "v", "r"], ["r", "v", "r"], 1.0),                       # perfect agreement, two categories
    (["r", "v", "r", "v"], ["v", "r", "v", "r"], -1.0),            # complete disagreement with balanced use
    (list("aabba"), list("aabaa"), 0.5454545),                     # po 0.8, pe 0.56
    (["r", "r", "r", "r"], ["v", "v", "v", "v"], 0.0),             # each labeller constant: po 0, pe 0
])
def test_cohen_kappa_known_values(a, b, expected):
    assert adj.cohen_kappa(a, b) == pytest.approx(expected, abs=1e-6)


def test_cohen_kappa_undefined_and_bad_input():
    assert adj.cohen_kappa(["r", "r"], ["r", "r"]) is None  # one category throughout: pe = 1
    with pytest.raises(ValueError):
        adj.cohen_kappa(["r"], ["r", "v"])
    with pytest.raises(ValueError):
        adj.cohen_kappa([], [])


def test_agreement_per_subtype_and_per_model(tmp_path, two_run_folders):
    sampled = adj.sample_claims(adj.collect_flagged(two_run_folders), n_per_model=5)
    sheet, key, _ = adj.blind_and_export(sampled, tmp_path / "s.csv", tmp_path / "k.csv")
    rows = list(csv.DictReader(sheet.open()))
    for r in rows:
        r["label_A"], r["label_B"] = "real_reasoner_error", ("verifier_mistake" if r["claim"] == "Claim two" else "real_reasoner_error")
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=adj.SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = adj.summarise_agreement(adj.read_labels(sheet, key))
    assert summary["n"] == 3 and summary["agreement_with_verifier_A"]["overall"] == 1.0
    assert summary["agreement_with_verifier_B"]["by_subtype"]["uncited_rule"] == {"n": 1, "agreement": 0.0}
    assert set(summary["by_model"]) == {"model-1", "model-2"}


def test_read_labels_rejects_unknown_labels(tmp_path, two_run_folders):
    sampled = adj.sample_claims(adj.collect_flagged(two_run_folders), n_per_model=5)
    sheet, key, _ = adj.blind_and_export(sampled, tmp_path / "s.csv", tmp_path / "k.csv")
    rows = list(csv.DictReader(sheet.open()))
    rows[0]["label_A"], rows[0]["label_B"] = "maybe", "real_reasoner_error"
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=adj.SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="maybe"):
        adj.read_labels(sheet, key)
