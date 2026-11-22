"""The replay helpers behind notebooks/62_replay_no_drive.ipynb: fetch, one_case and the printed lines. Fake models,
a made-up bundle folder, no network."""

import hashlib
import json

import pytest

pytest.importorskip("numpy")

from test_end_to_end import Fake, retriever  # noqa: E402

from strokecopilot.pipeline import end_to_end as e2e  # noqa: E402
from strokecopilot.pipeline import replay  # noqa: E402

FX02 = "fx02_thrombectomy_large_mismatch"


def _cell(tmp_path, fail_on=None):
    """An extracted bundle with one cell: the five fixtures answered and verified by fake models."""
    work = tmp_path / "work"
    cell = work / "fake-model" / "fixtures_rulebook"
    cell.mkdir(parents=True)
    metrics = e2e.run_fixtures(Fake("reasoner", fail_on=fail_on), Fake("extractor"), Fake("judge"), retriever, cell,
                               log=lambda *_: None)
    return work, cell, metrics


def test_one_case_scores_one_fixture_offline_and_leaves_the_cell_alone(tmp_path):
    work, cell, metrics = _cell(tmp_path)
    before = {p.relative_to(cell): p.read_bytes() for p in cell.rglob("*") if p.is_file()}
    one = replay.one_case(work, FX02, model="fake-model")
    row = next(r for r in metrics["per_case"] if r["case"] == FX02)
    assert one["vision"]["case_id"] == "fixture-fx02" and "NIHSS on arrival: 16" in one["note"]
    assert one["answer"] == json.loads((cell / "outputs" / f"{FX02}.json").read_text()) and one["error"] is None
    assert len(one["claims"]) == len(one["traces"]) == row["claims"] > 0
    assert len(one["flagged"]) == row["hallucinated"]
    assert one["expected"] == {"thrombolysis": False, "thrombectomy": True}
    assert replay.decision_lines(one) == ["thrombolysis: answered no, expected no, right",
                                          "thrombectomy: answered yes, expected yes, right"]
    assert {p.relative_to(cell): p.read_bytes() for p in cell.rglob("*") if p.is_file()} == before
    copied = work / "one_case" / "fake-model" / "fixtures_rulebook" / FX02
    assert [p.name for p in (copied / "outputs").iterdir()] == [f"{FX02}.json"]  # no other fixture is scored
    lines = replay.claim_lines(one)
    assert sum(line.startswith(("ok  ", "FLAG")) for line in lines) == len(one["claims"])
    assert sum(line.startswith("FLAG") for line in lines) == len(one["flagged"])
    assert replay.one_case(work, FX02, model="fake-model")["traces"] == one["traces"]  # a second call starts clean


def test_one_case_reports_a_format_failure_and_refuses_an_unknown_case(tmp_path):
    work, _, _ = _cell(tmp_path, fail_on=FX02)
    one = replay.one_case(work, FX02, model="fake-model")
    assert one["answer"] is None and one["error"].startswith("format failure")
    assert one["claims"] == [] and replay.claim_lines(one) == [] and replay.decision_lines(one) == []
    with pytest.raises(ValueError, match="No saved call"):
        replay.one_case(work, "fx99_not_a_fixture", model="fake-model")


def test_claim_lines_mark_a_flag_with_its_subtype_and_reason():
    result = {"claims": [{"id": "rationale[0].0", "text": "ASPECTS 8 is below the threshold of 6"},
                         {"id": "thrombectomy.rule_ids[0]", "text": "A passage about thrombectomy. " * 10}],
              "traces": [{"claim_id": "rationale[0].0", "kind": "comparison", "subtype": 4, "subtype_name": "faulty_derivation",
                          "critical": True, "reason": "With the true values, 8 < 6 is false."},
                         {"claim_id": "thrombectomy.rule_ids[0]", "kind": "citation", "subtype": None, "subtype_name": None,
                          "critical": False, "reason": "The passage exists."}]}
    lines = replay.claim_lines(result, width=40)
    assert lines == ["FLAG rationale[0].0            comparison  ASPECTS 8 is below the threshold of 6",
                     "     -> faulty_derivation (critical): With the true values, 8 < 6 is false.",
                     "ok   thrombectomy.rule_ids[0]  citation    A passage about thrombectomy. A passage "]


def test_fetch_prefers_a_local_copy_and_checks_the_sha256(tmp_path):
    bundle = tmp_path / "local" / replay.BUNDLE_NAME
    bundle.parent.mkdir()
    bundle.write_bytes(b"not a real bundle")
    sha = hashlib.sha256(b"not a real bundle").hexdigest()
    logged = []
    assert replay.fetch(local=[tmp_path / "missing.tar.gz", bundle], sha256=sha, log=logged.append) == bundle
    assert logged[0].startswith("Local copy") and "sha256 as recorded" in logged[0]
    dest = tmp_path / "downloaded.tar.gz"  # a file:// URL stands in for the release asset
    assert replay.fetch(dest, url=bundle.as_uri(), sha256=sha, log=logged.append) == dest
    assert dest.read_bytes() == b"not a real bundle" and logged[1].startswith("Downloaded downloaded.tar.gz")
    with pytest.raises(ValueError, match="sha256"):
        replay.fetch(tmp_path / "bad.tar.gz", url=bundle.as_uri(), sha256="0" * 64, log=logged.append)
    assert not (tmp_path / "bad.tar.gz").exists() and bundle.exists()  # a failed download is removed; a local copy is kept
    with pytest.raises(ValueError, match="sha256"):
        replay.fetch(local=[bundle], sha256="0" * 64, log=logged.append)
    assert bundle.exists()
    assert replay.asset_url() == f"{replay.RELEASE_URL}/download/{replay.RELEASE_TAG}/{replay.BUNDLE_NAME}"
