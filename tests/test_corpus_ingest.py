"""Rulebook parsing, the committed passages.jsonl, and the copy check. Plan §5.3, §8.3 (DEC-18)."""

import json

import pytest

from strokecopilot.nlp.corpus import ingest

GOOD = """# Rulebook
Preamble text that is ignored.

## Topic one

### AHA2026-4.6.1-within-4.5h
- source: AHA2026
- section: 4.6.1
- pointer: §4.6.1; COR 1
- page: e340
- treatment: thrombolysis
- must_have: yes
- status: checked

First line of text
continues here.

### DAWN-inclusion-groups
- source: DAWN
- section: Methods
- pointer: Methods
- page:
- treatment: thrombectomy
- must_have: no
- status: draft

Second passage.
"""


def test_parse_reads_fields_topic_and_text():
    a, b = ingest.parse(GOOD)
    assert a.id == "AHA2026-4.6.1-within-4.5h" and a.topic == "Topic one" and a.must_have and a.page == "e340"
    assert a.text == "First line of text continues here."
    assert b.status == "draft" and not b.must_have and ingest.unchecked([a, b]) == [b.id]


@pytest.mark.parametrize("change, message", [
    (("- source: DAWN", "- source: NEJM"), "unknown source"),
    (("### DAWN-inclusion-groups", "### DAWN inclusion"), "malformed ID"),
    (("### DAWN-inclusion-groups", "### DEFUSE3-inclusion-groups"), "does not start with its source"),
    (("- status: draft", "- status: done"), "unknown status"),
    (("- status: draft", "- status: checked"), "has no page"),
    (("- must_have: no", "- must_have: maybe"), "not yes or no"),
    (("- treatment: thrombectomy\n", ""), "lacks treatment"),
    (("Second passage.", "word " * 121), "limit is 120"),
    (("### DAWN-inclusion-groups", "### AHA2026-4.6.1-within-4.5h"), "duplicate ID"),
    (("Second passage.", ""), "has no text"),
])
def test_parse_rejects_problems(change, message):
    with pytest.raises(ValueError, match=message):
        ingest.parse(GOOD.replace(*change))


def test_build_load_and_version(tmp_path):
    src, out = tmp_path / "RULEBOOK.md", tmp_path / "passages.jsonl"
    src.write_text(GOOD)
    digest = ingest.build(src, out)
    assert digest == ingest.version(out)
    corpus = ingest.load(out)
    assert set(corpus["DAWN-inclusion-groups"]) == {"source", "section", "pointer", "treatment", "text"}
    assert ingest.must_have(out) == ["AHA2026-4.6.1-within-4.5h"]


def test_committed_passages_match_the_rulebook():
    assert ingest.PASSAGES.read_text(encoding="utf-8") == ingest.to_jsonl(ingest.read()), \
        "Run: python -m strokecopilot.nlp.corpus.ingest build"


def test_rulebook_covers_the_must_have_rules():
    passages = ingest.read()
    ids = {p.id for p in passages}
    assert len(ids) == len(passages)
    must = {p.id for p in passages if p.must_have}
    # plan §8.3: thrombolysis window, haemorrhage contraindication, DEFUSE-3 and DAWN inclusion criteria
    assert {"AHA2026-4.6.1-within-4.5h", "AHA2026-T8-ich"} <= must
    assert any(i.startswith("DEFUSE3-") for i in must) and any(i.startswith("DAWN-") for i in must)
    for p in passages:
        json.dumps(p.text)  # serialisable


def test_copy_check_flags_long_shared_runs_only():
    source = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho"
    copied = ingest.Passage("AHA2026-x-a", "AHA2026", "", "", "", "other", False, "draft", "",
                            "Intro words then alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu end.")
    own = ingest.Passage("AHA2026-x-b", "AHA2026", "", "", "", "other", False, "draft", "",
                         "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda other words")
    other_source = ingest.Passage("DAWN-x-c", "DAWN", "", "", "", "other", False, "draft", "", copied.text)
    flagged = ingest.copy_check([copied, own, other_source], {"AHA2026": source})
    assert flagged == {"AHA2026-x-a": 13}


def test_cli_prints_ids_and_counts_only(capsys, monkeypatch):
    monkeypatch.setattr(ingest, "pdf_text", lambda path: "secret source wording " * 10)
    ingest.main(["copy-check", "AHA2026=/nowhere.pdf"])
    out = capsys.readouterr().out
    assert "secret" not in out and "passages with AHA2026" in out
