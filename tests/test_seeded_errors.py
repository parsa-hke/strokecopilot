"""Gold outputs are clean in every way code can check, and every seed is one valid, labelled edit."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from strokecopilot.config import FIXTURES_DIR
from strokecopilot.contract.validate import validate_recommendation
from strokecopilot.nlp.corpus import ingest
from strokecopilot.nlp.notes import facts as nf
from strokecopilot.nlp.verifier import claims as cl
from strokecopilot.nlp.verifier import seeded_errors as se
from strokecopilot.nlp.verifier import trace as tr

CORPUS = ingest.load()
EXPECTED = json.loads((Path(FIXTURES_DIR) / "expected_rules.json").read_text())["decisions"]
STEMS = se.gold_stems()


def _inputs(stem):
    note, sheet = nf.fixture_note(stem)
    vision = json.loads((Path(FIXTURES_DIR) / f"{stem}.json").read_text())
    return tr.Inputs(vision=vision, note=note, facts=sheet, corpus=CORPUS, judge=None)


def _minutes(text):
    # "2 h 30 min" is one time; the regex extractor would read it as two numbers.
    return re.sub(r"(\d+) h (\d+) min", lambda m: f"{int(m.group(1)) * 60 + int(m.group(2))} min", text)


def test_gold_set_covers_every_fixture():
    assert STEMS == sorted(p.stem for p in Path(FIXTURES_DIR).glob("fx*.json"))


@pytest.mark.parametrize("stem", STEMS)
def test_gold_output_is_clean_by_code(stem):
    rec = se.load_gold(stem)
    validate_recommendation(rec)
    inputs = _inputs(stem)
    for decision in ("thrombolysis", "thrombectomy"):
        assert all(pid in CORPUS for pid in rec[decision]["rule_ids"])
        want = EXPECTED[stem][decision]
        assert want is None or rec[decision]["recommend"] == want
    for i, entry in enumerate(rec["rationale"]):
        for s in entry["sources"]:
            if s["type"] == "note":
                assert s["quote"] in inputs.note, (i, s["quote"])
            elif s["type"] == "passage":
                assert s["id"] in CORPUS, (i, s["id"])
        for value, unit, _ in cl.numbers(_minutes(entry["claim"])):
            claim = cl.Claim(id=f"rationale[{i}].n", field="rationale", entry=i, text=entry["claim"], kind="number",
                             value=value, unit=unit, origin="regex", declared_sources=entry["sources"])
            subtype, *_ = tr._number_claim(claim, inputs)
            assert subtype is None, (stem, i, value, unit)


def _number_flags(stem, rec):
    inputs, out = _inputs(stem), []
    for i, entry in enumerate(rec["rationale"]):
        for value, unit, _ in cl.numbers(_minutes(entry["claim"])):
            claim = cl.Claim(id=f"rationale[{i}].n", field="rationale", entry=i, text=entry["claim"], kind="number",
                             value=value, unit=unit, origin="regex", declared_sources=entry["sources"])
            if tr._number_claim(claim, inputs)[0] is not None:
                out.append(i)
    return out


def test_number_check_catches_a_planted_wrong_number():
    item = next(x for x in se.seeded_set() if x["id"] == "fx01-1a")
    assert _number_flags(item["fixture"], item["rec"]) == [0]


def test_seeds_are_valid_single_edits():
    seeds = se.load_seeds()
    assert len({s["id"] for s in seeds}) == len(seeds) == 70
    counts = Counter((s["fixture"], s["subtype"]) for s in seeds)
    assert set(counts.values()) == {2} and len(counts) == 35
    for item in se.seeded_set(seeds):
        validate_recommendation(item["rec"])
        gold = se.load_gold(item["fixture"])
        label = item["label"]
        if label["field"] == "rationale":
            assert 0 <= label["entry"] < len(item["rec"]["rationale"])
            changed = [i for i, e in enumerate(item["rec"]["rationale"]) if i >= len(gold["rationale"]) or e != gold["rationale"][i]]
            assert changed == [label["entry"]], item["id"]
        else:
            decision = label["field"].split(".")[0]
            assert item["rec"][decision]["rule_ids"][label["entry"]] not in CORPUS
        assert item["rec"]["thrombolysis"]["recommend"] == gold["thrombolysis"]["recommend"]  # decisions untouched


def test_misattribution_seeds_keep_the_claim_text():
    for item in se.seeded_set():
        if item["subtype"] == 7:
            gold = se.load_gold(item["fixture"])["rationale"][item["label"]["entry"]]
            entry = item["rec"]["rationale"][item["label"]["entry"]]
            assert entry["claim"] == gold["claim"] and entry["sources"] != gold["sources"]


def test_apply_rejects_bad_edits():
    gold = se.load_gold(STEMS[0])
    with pytest.raises(ValueError):
        se.apply(gold, {"id": "x", "edits": [{"op": "replace", "entry": 0, "old": "not there", "new": "y"}], "label": {"field": "rationale", "entry": 0}})
    with pytest.raises(ValueError):
        se.apply(gold, {"id": "x", "edits": [{"op": "set_entry", "entry": 0, "sources": gold["rationale"][0]["sources"]}], "label": {"field": "rationale", "entry": 0}})


def test_score_strict_lenient_and_false_flags():
    t = lambda cid, st=None: {"claim_id": cid, "subtype": st}
    seeds = [
        {"id": "a", "subtype": 1, "label": {"field": "rationale", "entry": 2}, "traces": [t("rationale[2].0", 1), t("rationale[0].0")]},
        {"id": "b", "subtype": 1, "label": {"field": "rationale", "entry": 2}, "traces": [t("rationale[2].0", 5), t("rationale[3].0", 2)]},
        {"id": "c", "subtype": 3, "label": {"field": "thrombolysis.rule_ids", "entry": 0}, "traces": [t("thrombolysis.rule_ids[0]", 3)]},
    ]
    clean = [{"fixture": "f", "traces": [t("rationale[0].0"), t("rationale[1].0", 7)]}, {"fixture": "g", "traces": [t("rationale[0].0")]}]
    out = se.score(seeds, clean)
    assert out["recall"]["wrong_number"] == {"subtype": 1, "n": 2, "strict": 0.5, "lenient": 1.0, "missed": ["b"]}
    assert out["recall"]["wrong_rule"]["strict"] == 1.0
    assert out["false_flag_rate_claims"] == pytest.approx(1 / 3) and out["clean_outputs_flagged"] == "1/2"
    assert out["flags_outside_label"] == 1 and out["meets_bar"] is False
