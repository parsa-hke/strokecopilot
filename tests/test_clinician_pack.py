"""The clinician pack (Phase 5, task A): the review, its form and its manifest, on a fake bank."""

import hashlib
import json

import pytest

from strokecopilot.pipeline import clinician_pack as cp

VISION = {"haemorrhage": {"present": False, "confidence": 0.05}, "core_ml": 5.0, "penumbra_ml": 50.0,
          "mismatch_ratio": 10.0, "mismatch_volume_ml": 45.0, "occlusion_site": "M1", "aspects": None,
          "confidence": {"core": 0.7, "penumbra": 0.9}, "warnings": []}
RULE = {"value": "yes", "reason": "Within 4.5 h with a disabling deficit: COR 1.", "rule_ids": [], "deciding": [],
        "unknown": []}


def sim_case(alias, kind):
    return alias, {"id": alias, "kind": kind, "arms": {"rulebook": {"vision": VISION, "note": "Onset 2 h."}}}


def bank(tmp_path):
    entries = [sim_case(f"fx0{i}", "fixture") for i in range(1, 4)]
    entries += [sim_case("scenario-001", "scenario"), sim_case("D-01", "development")]
    (tmp_path / "cases").mkdir(parents=True)
    rows = []
    for alias, case in entries:
        (tmp_path / "cases" / f"{alias}.json").write_text(json.dumps(case))
        rows.append({"id": alias, "kind": case["kind"], "rules": {d: RULE for d in cp.DECISIONS}})
    (tmp_path / "index.json").write_text(json.dumps({"banner": "", "cases": rows}))
    return tmp_path


PASSAGES = [{"id": "P1", "topic": "T", "pointer": "§1", "page": "e1", "text": "Rule one."},
            {"id": "P2", "topic": "T", "pointer": "§2", "page": "e2", "text": "Rule two."}]


def test_bank_with_a_heldout_case_is_refused(tmp_path):
    b = bank(tmp_path)
    index = json.loads((b / "index.json").read_text())
    index["cases"].append({"id": "H-01", "kind": "heldout"})
    (b / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="heldout"):
        cp.load_bank(b)


def test_write_builds_the_review_and_a_manifest(tmp_path):
    b = bank(tmp_path / "bank")
    pfile = tmp_path / "passages.jsonl"
    pfile.write_text("\n".join(json.dumps(p) for p in PASSAGES))
    out = tmp_path / "out"
    m = cp.write(b, out, passages_path=pfile)
    # every passage, two items per fixture, every convention, the open questions
    assert m["items_instrument1"] == len(cp.ABOUT_YOU) + 2 + 2 * 3 + len(cp.CONVENTIONS) + len(cp.OPEN_QUESTIONS_1)
    md = (out / "instrument1_rulebook_review.md").read_text()
    assert cp.DISCLAIMER in md
    assert "Rule one." in md and "**C12.**" in md and "How may we quote you?" in md and "**P2.**" in md
    assert "scenario-001" not in md and "D-01" not in md  # only the fixtures are shown
    assert sorted(m["files"]) == ["build_forms.gs", "forms_spec.json", "instrument1_responses.csv",
                                  "instrument1_rulebook_review.md"]
    for name, sha in m["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha


def test_form_spec_carries_every_item_and_requires_only_consent(tmp_path):
    cases = cp.load_bank(bank(tmp_path / "bank"))
    spec = cp.form_spec(PASSAGES, cases)
    (form,) = spec["forms"]
    items = form["items"]
    _, rows = cp.instrument1(PASSAGES, cases)
    titles = [i["title"] for i in items]
    for r in rows:
        assert sum(t.startswith(f"{r['item_id']}. ") for t in titles) == 1, r["item_id"]
    assert [i["title"] for i in items if i.get("required")] == [
        "Consent", "How may we quote you?", "May we thank you by name in the acknowledgements?"]
    assert all(i["choices"] for i in items if i["type"] in ("mc", "checkbox"))
    script = cp.apps_script(spec)
    start = script.index("const SPEC = ") + len("const SPEC = ")
    assert json.loads(script[start:script.index(";\n\nfunction buildForms")]) == spec
