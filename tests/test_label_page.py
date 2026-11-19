"""The labelling webpage (label_page.py): blind, complete, and its downloads read back through adjudicate. No network."""

import csv
import json
import re

import pytest

from strokecopilot.nlp.verifier import adjudicate as adj
from strokecopilot.nlp.verifier import label_page as lp

ITEMS = [{"sample_id": "S001", "case_alias": "C001", "field": "rationale", "claim": "Core volume 12.0 mL is below 70 mL.",
          "context": "Core 12.0 mL is below the 70 mL limit.", "sources": [{"type": "vision", "field": "core_ml"},
                                                                           {"type": "passage", "id": "P-1"}]},
         {"sample_id": "S002", "case_alias": "C002", "field": "uncertainties", "claim": "ASPECTS is not reported.",
          "context": None, "sources": []}]
CASES = {"C001": {"vision": {"core_ml": 12.0, "haemorrhage": {"present": False, "confidence": 0.02}}, "note": "NIHSS 9."},
         "C002": {"vision": {"core_ml": 30.0, "aspects": None}, "note": "Age 71. </script><script>alert(1)</script>"}}
PASSAGES = [{"id": "P-1", "text": "Core below 70 mL."}]


def _data(page):
    raw = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
    return json.loads(raw.replace("<\\/", "</"))


def test_page_holds_the_sample_and_nothing_that_unblinds_it():
    page = lp.build_page(ITEMS, CASES, PASSAGES, "Lorna")
    data = _data(page)
    assert data["items"] == ITEMS and data["cases"] == CASES and data["labeller"] == "A" and data["name"] == "Lorna"
    assert data["partner"] == "Benny" and data["half"] == 1
    assert data["fields"] == adj.SHEET_FIELDS and data["sample"] == lp.sample_hash(ITEMS)
    assert page.count("</script>") == 2  # a note cannot close the data block early
    for word in ("model", "verdict", "subtype", "run_folder", "key.csv"):
        assert f'"{word}"' not in page
    benny = _data(lp.build_page(ITEMS, CASES, PASSAGES, "Benny"))
    assert benny["labeller"] == "B" and benny["partner"] == "Lorna" and benny["items"] == data["items"]
    assert _data(lp.build_page(ITEMS[:1], CASES, PASSAGES, "Hana"))["cases"] == {"C001": CASES["C001"]}  # only its cases
    with pytest.raises(ValueError, match="not a labeller"):
        lp.build_page(ITEMS, CASES, PASSAGES, "Parsa")


def test_pairs_are_mixed_and_each_name_has_one_role():
    assert lp.PAIRS == {1: ("Lorna", "Benny"), 2: ("Mohamed", "Hana")}
    assert [lp.role(n) for n in ("Lorna", "Benny", "Mohamed", "Hana")] == [(1, "A", "Benny"), (1, "B", "Lorna"),
                                                                          (2, "A", "Hana"), (2, "B", "Mohamed")]


def test_split_halves_is_even_within_every_stratum_and_fixed():
    key = [{"sample_id": f"S{i:03d}", "model": f"m{i % 4}", "verdict": ("clean", "content_error", "citation_error")[i % 3]}
           for i in range(1, 101)]
    halves = lp.split_halves(key)
    assert sorted(halves) == [k["sample_id"] for k in key] and list(halves.values()).count(1) == 50
    for m in range(4):
        for v in ("clean", "content_error", "citation_error"):
            ids = [k["sample_id"] for k in key if k["model"] == f"m{m}" and k["verdict"] == v]
            ones = sum(halves[i] == 1 for i in ids)
            assert abs(2 * ones - len(ids)) <= 1
    assert lp.split_halves(key) == halves and lp.split_halves(key, seed=1) != halves


def test_answer_entry_finds_the_context_of_each_claim_kind():
    rec = {"rationale": [{"claim": "R0"}], "uncertainties": ["U0"],
           "thrombolysis": {"blocking_factors": ["B0"], "rule_ids": ["P-1"]}}
    assert lp.answer_entry(rec, "rationale[0].1") == "R0" and lp.answer_entry(rec, "uncertainties[0].0") == "U0"
    assert lp.answer_entry(rec, "thrombolysis.blocking_factors[0].0") == "B0"
    assert lp.answer_entry(rec, "thrombolysis.rule_ids[0]") == "P-1" and lp.answer_entry(rec, "rationale[4].0") is None


def _download(path, column, labels):
    """A sheet as the page downloads it: every field, CRLF lines, only this labeller's columns filled."""
    note_col = "note_" + column[-1]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=adj.SHEET_FIELDS, lineterminator="\r\n")
        w.writeheader()
        for (sid, claim), (lab, note) in zip([("S001", ITEMS[0]["claim"]), ("S002", ITEMS[1]["claim"])], labels):
            w.writerow({"sample_id": sid, "case_alias": "C00" + sid[-1], "field": "rationale", "claim": claim,
                        "declared_sources": "core_ml", column: lab, note_col: note})


def test_pair_downloads_merge_combine_and_read_back_through_adjudicate(tmp_path):
    _download(tmp_path / "lorna.csv", "label_A", [("error", 'unsure, "70"'), ("no_error", "")])
    _download(tmp_path / "benny.csv", "label_B", [("error", ""), ("error", "x")])
    pair1 = lp.merge_sheets(tmp_path / "lorna.csv", tmp_path / "benny.csv", tmp_path / "pair1.csv")
    _download(tmp_path / "mohamed.csv", "label_A", [("no_error", ""), ("no_error", "")])
    _download(tmp_path / "hana.csv", "label_B", [("no_error", ""), ("no_error", "")])
    for name in ("mohamed", "hana"):  # pair 2 holds the other half: S003 and S004
        path = tmp_path / f"{name}.csv"
        path.write_text(path.read_text().replace("S001", "S003").replace("S002", "S004"))
    pair2 = lp.merge_sheets(tmp_path / "mohamed.csv", tmp_path / "hana.csv", tmp_path / "pair2.csv")
    merged = lp.combine_sheets([pair2, pair1], tmp_path / "merged.csv")
    with (tmp_path / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=adj.KEY_FIELDS)
        w.writeheader()
        for sid, verdict in (("S001", "content_error"), ("S002", "clean"), ("S003", "clean"), ("S004", "clean")):
            w.writerow({"sample_id": sid, "model": "m", "verdict": verdict, "weight": "1"})
    rows = adj.read_labels(merged, tmp_path / "key.csv")
    assert [r["sample_id"] for r in rows] == ["S001", "S002", "S003", "S004"]
    assert [(r["label_A"], r["label_B"]) for r in rows[:2]] == [("error", "error"), ("no_error", "error")]
    summary = adj.summarise_agreement(rows)
    assert summary["n_labelled_by_both"] == 4 and summary["unresolved_disagreements"] == 1
    by_pair = lp.kappa_by_pair(rows, {"S001": 1, "S002": 1, "S003": 2, "S004": 2})
    assert by_pair[1]["labellers"] == ["Lorna", "Benny"] and by_pair[1]["n"] == 2 and by_pair[2]["n"] == 2
    assert by_pair[2]["kappa"] is None  # both said no_error throughout: kappa is undefined, not 1
    assert next(csv.DictReader(merged.open()))["note_A"] == 'unsure, "70"'
    with pytest.raises(ValueError, match="both pairs"):
        lp.combine_sheets([pair1, pair1], tmp_path / "twice.csv")
    _download(tmp_path / "other.csv", "label_B", [("error", "")] * 2)
    (tmp_path / "other.csv").write_text((tmp_path / "other.csv").read_text().replace("S002", "S009"))
    with pytest.raises(ValueError, match="not the same sample"):
        lp.merge_sheets(tmp_path / "lorna.csv", tmp_path / "other.csv", tmp_path / "bad.csv")


def test_consensus_page_shows_both_labels_and_fills_the_consensus_column(tmp_path):
    prior = {"S001": [{"name": "Lorna", "letter": "A", "label": "error", "note": "not cited"},
                      {"name": "Benny", "letter": "B", "label": "no_error", "note": ""}]}
    page = lp.build_consensus_page(ITEMS[:1], CASES, PASSAGES, prior)
    data = _data(page)
    assert data["labeller"] == "consensus" and data["items"][0]["prior"] == prior["S001"] and data["cases"] == {"C001": CASES["C001"]}
    assert "['consensus', 'note_consensus']" in page and page.count("</script>") == 2
    for word in ("model", "verdict", "subtype", "run_folder"):
        assert f'"{word}"' not in page
    combined = tmp_path / "combined.csv"
    with combined.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=adj.SHEET_FIELDS)
        w.writeheader()
        w.writerow({"sample_id": "S001", "claim": "x", "label_A": "error", "label_B": "no_error"})
        w.writerow({"sample_id": "S002", "claim": "y", "label_A": "no_error", "label_B": "no_error"})
    mine = tmp_path / "mine.csv"
    with mine.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=adj.SHEET_FIELDS)
        w.writeheader()
        w.writerow({"sample_id": "S001", "consensus": "error", "note_consensus": "the guide: uncited is an error"})
    out = lp.merge_consensus(combined, mine, tmp_path / "final.csv")
    rows = {r["sample_id"]: r for r in csv.DictReader(out.open())}
    assert rows["S001"]["consensus"] == "error" and rows["S002"]["consensus"] == ""
    with mine.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=adj.SHEET_FIELDS).writerow({"sample_id": "S002", "consensus": "error"})
    with pytest.raises(ValueError, match="agreed"):
        lp.merge_consensus(combined, mine, tmp_path / "final2.csv")
