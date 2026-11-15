"""Tests for nlp/verifier/adjudicate.py: claims from traces joined to the flagged file by claim ID, stratified sampling
of flagged and clean claims, the blind sheet and its key, case inputs per alias, Cohen's kappa against known values,
labels and consensus, and agreement per verdict, sub-type and model. No network."""

import csv
import json

import pytest

from strokecopilot.nlp.verifier import adjudicate as adj

STEMS = ("fx01_thrombolysis_clear", "fx02_thrombectomy_large_mismatch")


def _claim(claim_id, text):
    field = claim_id.split("[", 1)[0]
    return {"id": claim_id, "field": field, "entry": 0, "text": text, "kind": "rule", "origin": "llm",
            "declared_sources": [{"type": "passage", "id": "AHA2026-4.6.1-within-4.5h"}]}


def _flag(case, claim_id, subtype, text):
    """As pipeline/end_to_end.flagged_claims writes it; the text is the extractor's wording, cut to 240 characters."""
    return {"case": case, "claim_id": claim_id, "subtype": subtype, "critical": False, "reason": "verifier reason",
            "kind": "rule", "text": text, "sources": ["AHA2026-4.6.1-within-4.5h"]}


def _run(tmp_path, name, model, claims_by_case, flagged, flagged_name="flagged.json"):
    folder = tmp_path / name
    (folder / "traces").mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"experiment": name, "source": "fixtures", "reasoner": {"model": model}}))
    for case, claims in claims_by_case.items():
        (folder / "traces" / f"{case}.json").write_text(json.dumps({"claims": claims, "traces": []}))
    if flagged is not None:
        (folder / flagged_name).write_text(json.dumps(flagged))
    return folder


@pytest.fixture
def two_run_folders(tmp_path):
    """Two run folders laid out as pipeline/end_to_end.py writes them: config.json, traces/<case>.json, flagged.json.
    Run A: 4 claims, 2 flagged (one content, one citation). Run B: 3 claims, 1 flagged."""
    a = _run(tmp_path, "EXP-A", "model-1",
             {STEMS[0]: [_claim("rationale[0].0", "Claim one"), _claim("rationale[0].1", "Claim one b")],
              STEMS[1]: [_claim("thrombolysis.blocking_factors[0].0", "Claim two"), _claim("uncertainties[0].0", "Claim u")]},
             [_flag(STEMS[0], "rationale[0].0", "wrong_rule", "Claim one"),
              _flag(STEMS[1], "thrombolysis.blocking_factors[0].0", "uncited_rule", "Claim two")])
    b = _run(tmp_path, "EXP-B", "model-2",
             {STEMS[0]: [_claim("rationale[2].0", "Claim three"), _claim("rationale[2].1", "Claim four"),
                         _claim("rationale[3].0", "Claim five")]},
             [_flag(STEMS[0], "rationale[2].0", "wrong_rule", "Claim three")])
    return [a, b]


def _label_sheet(sheet, labels):
    """labels: {claim text: (label_A, label_B, consensus)}; rows not named stay unlabelled."""
    rows = list(csv.DictReader(sheet.open()))
    for r in rows:
        r["label_A"], r["label_B"], r["consensus"] = labels.get(r["claim"], ("", "", ""))
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=adj.SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _labelled(tmp_path, folders, labels):
    sampled = adj.sample_claims(adj.collect_all_claims(folders), n_total=100)
    sheet, key, _ = adj.blind_and_export(sampled, tmp_path / "s.csv", tmp_path / "k.csv")
    _label_sheet(sheet, labels)
    return adj.read_labels(sheet, key)


# ---- collection ----

def test_collect_all_claims_reads_every_claim_with_its_verdict(two_run_folders):
    out = adj.collect_all_claims(two_run_folders)
    assert len(out) == 7 and {e["model"] for e in out} == {"model-1", "model-2"}
    verdicts = {e["text"]: e["verdict"] for e in out}
    assert verdicts == {"Claim one": "content_error", "Claim one b": "clean", "Claim two": "citation_error",
                        "Claim u": "clean", "Claim three": "content_error", "Claim four": "clean", "Claim five": "clean"}
    first = next(e for e in out if e["text"] == "Claim one")
    assert first["source"] == "fixtures" and first["flagged_file"] == "flagged.json" and first["subtype"] == "wrong_rule"
    assert first["field"] == "rationale" and first["sources"] == ["AHA2026-4.6.1-within-4.5h"]


def test_join_is_by_claim_id_not_by_text(tmp_path):
    """The flagged file's text is reworded and cut; the claim ID is what joins. A clean claim with the flagged text
    elsewhere must stay clean."""
    folder = _run(tmp_path, "EXP-R", "m", {STEMS[0]: [_claim("rationale[0].0", "Core is 18 mL"),
                                                     _claim("rationale[0].1", "Penumbra is 95 mL"),
                                                     _claim("rationale[1].0", "Penumbra 95")]},
                  [_flag(STEMS[0], "rationale[0].1", "wrong_number", "Penumbra 95")])
    out = {e["claim_id"]: e for e in adj.collect_all_claims([folder])}
    assert out["rationale[0].1"]["verdict"] == "content_error" and out["rationale[0].1"]["text"] == "Penumbra is 95 mL"
    assert out["rationale[1].0"]["verdict"] == "clean" and out["rationale[0].0"]["verdict"] == "clean"


def test_a_flagged_claim_id_not_in_the_traces_raises(tmp_path):
    folder = _run(tmp_path, "EXP-M", "m", {STEMS[0]: [_claim("rationale[0].0", "t")]},
                  [_flag(STEMS[0], "rationale[7].0", "wrong_rule", "t")])
    with pytest.raises(ValueError, match="not among the claims"):
        adj.collect_all_claims([folder])


def test_missing_flagged_file_raises_unless_allowed(tmp_path):
    folder = _run(tmp_path, "EXP-N", "m", {STEMS[0]: [_claim("rationale[0].0", "t")]}, None)
    with pytest.raises(FileNotFoundError, match="every claim"):
        adj.collect_all_claims([folder])
    with pytest.raises(FileNotFoundError):
        adj.collect_all_claims([folder], flagged_name="flagged_phase3_judge_v3.json")
    out = adj.collect_all_claims([folder], allow_missing_flagged=True)
    assert [e["verdict"] for e in out] == ["clean"] and out[0]["flagged_file"] is None


def test_the_newest_scoring_is_read_by_default_or_the_file_named(two_run_folders):
    folder = two_run_folders[0]
    (folder / "flagged_rescore.json").write_text(json.dumps([_flag(STEMS[1], "uncertainties[0].0", "invented_finding", "u")]))
    newest = adj.collect_all_claims([folder], claims="saved")
    assert {e["text"] for e in newest if e["verdict"] != "clean"} == {"Claim u"}
    assert all(e["flagged_file"] == "flagged_rescore.json" for e in newest)
    named = adj.collect_all_claims([folder], flagged_name="flagged.json")
    assert {e["text"] for e in named if e["verdict"] != "clean"} == {"Claim one", "Claim two"}


def test_a_rescore_uses_the_claim_set_its_metrics_counted(two_run_folders, monkeypatch):
    """A rescore drops restatements and rebuilds regex claims, so its claims can differ from traces/. The kit takes the
    set whose count matches metrics_rescore.json, and refuses when none does."""
    from strokecopilot.nlp.verifier import claims as cl
    from strokecopilot.nlp.verifier import run as vr

    folder = two_run_folders[0]
    (folder / "outputs").mkdir()
    for stem in STEMS:
        (folder / "outputs" / f"{stem}.json").write_text("{}")
    monkeypatch.setattr(vr, "refresh_claims", lambda rec, saved: [cl.Claim(**c) for c in saved[:1]])
    (folder / "flagged_rescore.json").write_text(json.dumps([_flag(STEMS[0], "rationale[0].0", "wrong_rule", "x")]))
    (folder / "metrics_rescore.json").write_text(json.dumps({"per_case": [{"case": STEMS[0], "claims": 1},
                                                                          {"case": STEMS[1], "claims": 2}]}))
    out = adj.collect_all_claims([folder])
    assert [e["text"] for e in out] == ["Claim one", "Claim two", "Claim u"]  # refreshed for one case, saved for the other
    (folder / "metrics_rescore.json").write_text(json.dumps({"per_case": [{"case": STEMS[0], "claims": 5}]}))
    with pytest.raises(ValueError, match="older code"):
        adj.collect_all_claims([folder])


def test_collect_refuses_held_out_cases(two_run_folders, monkeypatch):
    def refuse(ids):
        raise adj.splits.HeldoutError("held out")
    monkeypatch.setattr(adj.splits, "check_ids", refuse)
    with pytest.raises(adj.splits.HeldoutError):
        adj.collect_all_claims(two_run_folders)


def test_unknown_subtype_raises():
    assert adj.verdict_of(None) == "clean" and adj.verdict_of("misattributed_source") == "citation_error"
    with pytest.raises(ValueError):
        adj.verdict_of("made_up")


# ---- sampling ----

def _many(models=("m1", "m2"), clean=200, content=6, citation=3):
    out = []
    for model in models:
        for verdict, n in (("clean", clean), ("content_error", content), ("citation_error", citation)):
            out += [{"model": model, "verdict": verdict, "claim_id": f"rationale[{i}].0", "case": "c", "run_folder": model}
                    for i in range(n)]
    return out


def test_sample_size_and_strata():
    claims = _many()
    sampled = adj.sample_claims(claims, seed=1)
    assert len(sampled) == 100
    strata = adj.strata(sampled)
    # small strata give everything; the clean strata share the rest equally
    assert strata[("m1", "content_error")] == {"size": 6, "sampled": 6}
    assert strata[("m2", "citation_error")] == {"size": 3, "sampled": 3}
    assert strata[("m1", "clean")]["sampled"] == strata[("m2", "clean")]["sampled"] == 41
    assert sum(s["sampled"] for s in strata.values()) == 100
    clean = next(e for e in sampled if e["verdict"] == "clean")
    assert clean["weight"] == pytest.approx(200 / 41) and clean["stratum_size"] == 200


def test_sample_is_deterministic_and_leaves_input_alone():
    claims = _many()
    before = [dict(e) for e in claims]
    assert adj.sample_claims(claims, seed=3) == adj.sample_claims(claims, seed=3)
    assert claims == before
    assert len(adj.sample_claims(claims[:5], n_total=100)) == 5  # never more than there are


# ---- sheet, key, case inputs ----

def test_the_sheet_is_blind_and_the_key_unblinds_it(tmp_path, two_run_folders):
    sampled = adj.sample_claims(adj.collect_all_claims(two_run_folders), n_total=100, seed=42)
    sheet, key, aliases = adj.blind_and_export(sampled, tmp_path / "sheet.csv", tmp_path / "private" / "key.csv", seed=42)
    text = sheet.read_text().lower()
    for hidden in ("model-1", "model-2", "exp-a", "exp-b", "wrong_rule", "uncited_rule", "content_error", "verifier reason",
                   *STEMS):
        assert hidden not in text, hidden
    rows = list(csv.DictReader(sheet.open()))
    assert len(rows) == len(sampled) == 7
    assert {r["field"] for r in rows} == {"rationale", "thrombolysis.blocking_factors", "uncertainties"}
    assert all(r["case_alias"].startswith("C") for r in rows) and len(aliases) == 3  # one alias per run and case
    keyed = {r["sample_id"]: r for r in csv.DictReader(key.open())}
    assert set(keyed) == {r["sample_id"] for r in rows} and {r["model"] for r in keyed.values()} == {"model-1", "model-2"}
    assert {r["verdict"] for r in keyed.values()} == set(adj.VERDICTS) and all(r["weight"] for r in keyed.values())


def test_case_inputs_are_written_per_alias_without_ids(tmp_path, two_run_folders):
    sampled = adj.sample_claims(adj.collect_all_claims(two_run_folders))
    _, _, aliases = adj.blind_and_export(sampled, tmp_path / "s.csv", tmp_path / "k.csv")
    out = adj.export_case_inputs(aliases, tmp_path / "cases")
    files = sorted(p.name for p in out.iterdir())
    assert files == sorted(f"{a}.md" for a in aliases.values())
    body = (out / files[0]).read_text()
    assert "## Triage note" in body and '"case_id"' not in body and "fixture-" not in body


# ---- kappa ----

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


# ---- labels, consensus, agreement ----

E, N = "error", "no_error"


def test_clean_claims_labelled_no_error_agree_with_the_verifier(tmp_path, two_run_folders):
    rows = _labelled(tmp_path, two_run_folders, {"Claim one b": (N, N, ""), "Claim four": (N, N, ""),
                                                 "Claim one": (E, E, "")})
    s = adj.summarise_agreement(rows)
    assert s["agreement_with_verifier"]["overall"]["agreement"] == 1.0
    assert s["agreement_with_verifier"]["by_verdict"]["clean"] == {"n": 2, "agreement": 1.0, "agreement_weighted": 1.0}


def test_kappa_uses_every_row_both_labelled(tmp_path, two_run_folders):
    """An unresolved disagreement has no consensus, so it is left out of agreement with the verifier, but it lowers
    kappa: kappa is computed on every row both labellers filled, not only on rows with a consensus."""
    agree = {"Claim one": (E, E, ""), "Claim two": (E, E, ""), "Claim three": (E, E, ""),
             "Claim one b": (N, N, ""), "Claim four": (N, N, ""), "Claim five": (N, N, "")}
    full = adj.summarise_agreement(_labelled(tmp_path, two_run_folders, agree))
    assert full["kappa"] == 1.0 and full["unresolved_disagreements"] == 0
    split = adj.summarise_agreement(_labelled(tmp_path, two_run_folders, {**agree, "Claim u": (E, N, "")}))
    assert split["n_labelled_by_both"] == 7 and split["n_with_consensus"] == 6
    assert split["unresolved_disagreements"] == 1 and split["kappa"] < 1.0
    expected = adj.cohen_kappa([E, E, E, N, N, N, E], [E, E, E, N, N, N, N])
    assert split["kappa"] == pytest.approx(expected)


def test_consensus_column_else_the_shared_label(tmp_path, two_run_folders):
    rows = _labelled(tmp_path, two_run_folders, {"Claim one": (E, N, N),       # resolved by discussion: no error
                                                 "Claim two": (E, E, ""),      # agreed: error
                                                 "Claim u": (E, N, "")})       # unresolved
    by_text = {r["claim_id"]: adj.consensus_of(r) for r in rows}
    assert by_text["rationale[0].0"] == N and by_text["thrombolysis.blocking_factors[0].0"] == E
    assert by_text["uncertainties[0].0"] is None
    s = adj.summarise_agreement(rows)
    assert s["n_with_consensus"] == 2 and s["unresolved_disagreements"] == 1
    assert s["agreement_with_verifier"]["overall"]["agreement"] == 0.5  # the verifier flagged Claim one: disagrees


def test_agreement_per_subtype_and_per_model(tmp_path, two_run_folders):
    rows = _labelled(tmp_path, two_run_folders, {"Claim one": (E, E, ""), "Claim two": (N, N, ""),
                                                 "Claim three": (E, E, ""), "Claim four": (E, E, "")})
    s = adj.summarise_agreement(rows)["agreement_with_verifier"]
    assert s["by_subtype"]["wrong_rule"]["n"] == 2 and s["by_subtype"]["wrong_rule"]["agreement"] == 1.0
    assert s["by_subtype"]["uncited_rule"]["agreement"] == 0.0  # flagged, humans say no error
    assert s["by_subtype"]["clean"]["agreement"] == 0.0          # a miss: clean, humans say error
    assert s["by_model"]["model-1"]["n"] == 2 and s["by_model"]["model-1"]["agreement"] == 0.5
    assert s["by_model"]["model-2"]["agreement"] == 0.5 and s["by_model"]["model-2"]["kappa"] is None


def test_read_labels_validates_and_maps_old_labels(tmp_path, two_run_folders):
    rows = _labelled(tmp_path, two_run_folders, {"Claim one": ("real_reasoner_error", "verifier_mistake", "")})
    first = next(r for r in rows if r["claim_id"] == "rationale[0].0" and r["model"] == "model-1")
    assert (first["label_A"], first["label_B"]) == (E, N)
    with pytest.raises(ValueError, match="maybe"):
        _labelled(tmp_path, two_run_folders, {"Claim one": ("maybe", E, "")})
    with pytest.raises(ValueError, match="consensus"):
        _labelled(tmp_path, two_run_folders, {"Claim one": (E, N, "unsure")})
