"""The clinician pack (Phase 5, task A): strata, seeded selection, instruments and manifest, on a fake bank."""

import hashlib
import json

import pytest

from strokecopilot.pipeline import clinician_pack as cp

VISION = {"haemorrhage": {"present": False, "confidence": 0.05}, "core_ml": 5.0, "penumbra_ml": 50.0,
          "mismatch_ratio": 10.0, "mismatch_volume_ml": 45.0, "occlusion_site": "M1", "aspects": None,
          "confidence": {"core": 0.7, "penumbra": 0.9}, "warnings": []}
RULE = {"value": "yes", "reason": "Within 4.5 h with a disabling deficit: COR 1.", "rule_ids": [], "deciding": [],
        "unknown": []}


def answer(recommend=(True, False), agrees=(True, True), flagged_field=None):
    rec = {"rationale": [{"claim": "Core is 5.0 mL.", "sources": [{"type": "vision", "field": "core_ml"}],
                          "derived": False}], "uncertainties": ["ASPECTS is not reported."]}
    for d, r in zip(cp.DECISIONS, recommend):
        rec[d] = {"recommend": r, "rule_ids": ["P1"], "blocking_factors": []}
    claims = [{"field": "rationale", "text": "Core is 5.0 mL", "flagged": False, "subtype": None}]
    if flagged_field:
        claims.append({"field": flagged_field, "text": "x", "flagged": True, "subtype": "wrong_rule",
                       "critical": False, "reason": "no passage says so"})
    decisions = {d: {"recommended": r, "rulebook": r if a else (not r), "agrees": a}
                 for d, r, a in zip(cp.DECISIONS, recommend, agrees)}
    return {"available": True, "recommendation": rec, "claims": claims, "decisions": decisions,
            "counts": {"claims": len(claims), "flagged": int(bool(flagged_field)), "critical": 0}}


def sim_case(alias, kind, sonnet, gemma=None):
    answers = {cp.SONNET: sonnet}
    if gemma:
        answers[cp.OTHER_MODEL] = gemma
    return alias, {"id": alias, "kind": kind,
                   "arms": {"rulebook": {"vision": VISION, "note": "Onset 2 h.", "answers": answers}}}


def dev_case(alias, pred_value, truth_value, sonnet, bleed=False):
    pred_vision = {**VISION, "haemorrhage": {"present": bleed, "confidence": 0.5 if bleed else 0.05}}

    def arm(v, value):
        return {"vision": v, "note": "Onset 1 h.", "rules": {d: {**RULE, "value": value} for d in cp.DECISIONS},
                "answers": {cp.SONNET: sonnet}}
    return alias, {"id": alias, "kind": "development",
                   "arms": {"predicted": arm(pred_vision, pred_value), "groundtruth": arm(VISION, truth_value)}}


def bank(tmp_path, extra=()):
    entries = [sim_case(f"fx0{i}", "fixture", answer()) for i in range(1, 4)]
    entries += [sim_case("scenario-001", "scenario", answer()),  # S1
                sim_case("scenario-002", "scenario", answer(flagged_field="rationale")),  # S2
                sim_case("scenario-003", "scenario", answer(agrees=(False, True)),  # S3
                         gemma=answer(agrees=(False, True), flagged_field="thrombolysis.blocking_factors")),  # M
                sim_case("scenario-004", "scenario", answer(),
                         gemma=answer(agrees=(False, True)))]  # Gemma wrong but not caught: not M
    entries += [dev_case("D-01", "yes", "yes", answer()),  # D1
                dev_case("D-02", "no", "no", answer(recommend=(False, False)), bleed=True),  # D2
                dev_case("D-03", "no", "yes", answer(recommend=(False, False)), bleed=True),  # V: missed
                dev_case("D-04", "no", "yes", answer(recommend=(False, False), flagged_field="rationale"),
                         bleed=True)]  # caught, so D2
    entries += list(extra)
    (tmp_path / "cases").mkdir(parents=True)
    rows = []
    for alias, case in entries:
        (tmp_path / "cases" / f"{alias}.json").write_text(json.dumps(case))
        rows.append({"id": alias, "kind": case["kind"], "rules": {d: RULE for d in cp.DECISIONS}})
    (tmp_path / "index.json").write_text(json.dumps({"banner": "", "cases": rows}))
    return tmp_path


PASSAGES = [{"id": "P1", "topic": "T", "pointer": "§1", "page": "e1", "text": "Rule one."},
            {"id": "P2", "topic": "T", "pointer": "§2", "page": "e2", "text": "Rule two."}]


def test_pools_follow_the_strata(tmp_path):
    p = cp.pools(cp.load_bank(bank(tmp_path)))
    assert p["S1"] == ["scenario-001", "scenario-004"]
    assert p["S2"] == ["scenario-002"]
    assert p["S3"] == ["scenario-003"]
    assert p["M"] == ["scenario-003"]
    assert p["V"] == ["D-03"]
    assert p["D1"] == ["D-01"] and p["D2"] == ["D-02", "D-04"]
    assert len(p["F"]) == 3


def test_selection_is_seeded_and_never_reuses_a_case(tmp_path):
    cases = cp.load_bank(bank(tmp_path))
    with pytest.raises(ValueError, match="stratum M"):  # scenario-003 is S3 and the only M case
        cp.select(cases)
    _, extra = sim_case("scenario-005", "scenario", answer(),
                        gemma=answer(agrees=(True, False), flagged_field="rationale"))
    cases["scenario-005"] = {**extra, "index": {"id": "scenario-005", "kind": "scenario"}}
    a, sizes = cp.select(cases, seed=1)
    b, _ = cp.select(cases, seed=1)
    assert a == b and sizes["M"] == 2
    assert [c["n"] for c in a] == list(range(1, 10))
    assert len({c["alias"] for c in a}) == 9
    assert {c["alias"] for c in a if c["stratum"] == "M"} == {"scenario-005"}
    assert all(c["model"] == (cp.OTHER_MODEL if c["stratum"] == "M" else cp.SONNET) for c in a)


def test_bank_with_a_heldout_case_is_refused(tmp_path):
    b = bank(tmp_path)
    index = json.loads((b / "index.json").read_text())
    index["cases"].append({"id": "H-01", "kind": "heldout"})
    (b / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="heldout"):
        cp.load_bank(b)


def test_write_builds_both_instruments_and_a_manifest(tmp_path):
    b = bank(tmp_path / "bank", extra=[sim_case("scenario-005", "scenario", answer(),
                                                gemma=answer(agrees=(True, False), flagged_field="rationale"))])
    pfile = tmp_path / "passages.jsonl"
    pfile.write_text("\n".join(json.dumps(p) for p in PASSAGES))
    out = tmp_path / "out"
    m = cp.write(b, out, seed=3, passages_path=pfile)
    # instrument 1: every passage, two items per fixture, every convention, the open questions
    assert m["items_instrument1"] == len(cp.ABOUT_YOU) + 2 + 2 * 3 + len(cp.CONVENTIONS) + len(cp.OPEN_QUESTIONS_1)
    md1 = (out / "instrument1_rulebook_review.md").read_text()
    md2 = (out / "instrument2_interview.md").read_text()
    for text in (md1, md2):
        assert cp.DISCLAIMER in text
    assert "Rule one." in md1 and "**C12.**" in md1 and "How may we quote you?" in md1 and "**P2.**" in md1
    assert md2.count("## Case ") == 9 and "Gemma 3 27B" in md2 and "How may we quote you?" in md2
    assert "rule not stated in the rulebook" in md2  # flags in plain words
    for name, sha in m["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
    assert json.loads((out / "selection.json").read_text())["seed"] == 3


def test_form_spec_carries_every_item_and_requires_only_consent(tmp_path):
    b = bank(tmp_path / "bank", extra=[sim_case("scenario-005", "scenario", answer(),
                                                gemma=answer(agrees=(True, False), flagged_field="rationale"))])
    cases = cp.load_bank(b)
    chosen, _ = cp.select(cases, seed=3)
    spec = cp.form_spec(PASSAGES, cases, chosen)
    f1, f2 = (f["items"] for f in spec["forms"])
    _, rows1 = cp.instrument1(PASSAGES, cases)
    _, rows2 = cp.instrument2(cases, chosen, PASSAGES)
    titles1 = [i["title"] for i in f1]
    titles2 = [i["title"] for i in f2]
    for r in rows1:
        assert sum(t.startswith(f"{r['item_id']}. ") for t in titles1) == 1, r["item_id"]
    for r in rows2:
        assert sum(t.startswith(f"{r['question_id']}. ") for t in titles2) == 1, r["question_id"]
    for items in (f1, f2):
        assert [i["title"] for i in items if i.get("required")] == [
            "Consent", "How may we quote you?", "May we thank you by name in the acknowledgements?"]
        assert all(i["choices"] for i in items if i["type"] in ("mc", "checkbox"))
    # the expected answers come only after every case, in plain text
    last_case = max(k for k, i in enumerate(f2) if i["type"] == "page" and i["title"].startswith("Case "))
    reveal = next(k for k, i in enumerate(f2) if i["title"] == "Our expected answers")
    assert reveal > last_case and all("**" not in i.get("help", "") for i in f2)
    script = cp.apps_script(spec)
    start = script.index("const SPEC = ") + len("const SPEC = ")
    assert json.loads(script[start:script.index(";\n\nfunction buildForms")]) == spec


def test_case_document_has_no_questions_or_expected_answers(tmp_path):
    b = bank(tmp_path / "bank", extra=[sim_case("scenario-005", "scenario", answer(),
                                                gemma=answer(agrees=(True, False), flagged_field="rationale"))])
    cases = cp.load_bank(b)
    chosen, _ = cp.select(cases, seed=3)
    doc = cp.case_document(cases, chosen, PASSAGES)
    assert doc.count("## Case ") == 9 and cp.DISCLAIMER in doc
    assert "expected **" not in doc and "Q1." not in doc and "Rule one." in doc
