"""Training scenarios and the fine-tuning dataset (Item 4, DEC-30). No network."""

import json

import pytest

from strokecopilot.nlp.reasoner import train_data as td
from strokecopilot.nlp.verifier import scenarios as sc

TEST_SHA256 = "ea5b5977cd8e0bf78cb8929f3b1ab61f3ba512944fd21ffa94d30c65b5112e42"  # the 150 before grids became a parameter


@pytest.fixture(scope="module")
def test_set():
    return sc.generate()


@pytest.fixture(scope="module")
def training(test_set):
    return sc.generate_training(300, exclude=test_set)


def test_the_evaluation_scenarios_are_unchanged(test_set):
    assert sc.manifest(test_set)["sha256"] == TEST_SHA256


def test_training_grids_share_no_decision_value_with_the_test_grids():
    for key in ("minutes", "ages", "nihss", "sleep", "profiles"):
        assert not set(sc.TRAIN_GRIDS[key]) & set(sc.TEST_GRIDS[key]), key
    assert not (set(sc.TRAIN_GRIDS["aspects"]) - {None}) & (set(sc.TEST_GRIDS["aspects"]) - {None})
    assert sc.TRAIN_GRIDS["inr"] != sc.TEST_GRIDS["inr"]


def test_training_scenarios_skip_held_out_routes_and_repeat_no_case(test_set, training):
    held = sc.held_out_routes()
    assert held == ["Beyond 4.5 h and eligible for thrombectomy", "LVO 4.5-24 h without thrombectomy", "INR above 1.7",
                    "No occlusion is identified"]  # drawn by seed from the 14 non-core routes (DEC-30 as amended)
    assert not set(held) & set(sc.CORE_ROUTES)
    assert any(s.expected["thrombolysis"]["reason"].startswith(sc.CORE_ROUTES[0]) for s in training)
    assert any(s.expected["thrombolysis"]["reason"].startswith(sc.CORE_ROUTES[1]) for s in training)
    for s in training:
        assert not any(v["reason"].startswith(r) for v in s.expected.values() for r in held), s.id
    test_sigs = {sc.signature(s.vision, s.facts) for s in test_set}
    train_sigs = [sc.signature(s.vision, s.facts) for s in training]
    assert not test_sigs & set(train_sigs) and len(set(train_sigs)) == len(train_sigs)
    assert [s.id for s in training[:2]] == ["train-0001", "train-0002"]
    assert [s.id for s in sc.generate_training(20, exclude=test_set)] == [s.id for s in training[:20]]  # deterministic


def test_held_out_routes_can_be_given(test_set):
    chosen = ["INR above 1.7", "Basilar occlusion"]
    for s in sc.generate_training(60, held_out=chosen, exclude=test_set):
        assert not any(v["reason"].startswith(r) for v in s.expected.values() for r in chosen)


def test_examples_carry_the_evaluation_request_and_a_checked_gold_answer(training):
    items = td.examples(training[:40])
    assert len(items) == 40 and all(json.loads(x["answer"])["rationale"] is not None for x in items)
    train, val = td.split(items)
    assert len(val) == 4 and len(train) == 36 and not {x["id"] for x in train} & {x["id"] for x in val}
    assert '"vision"' in items[0]["user"] and '"note"' in items[0]["user"]


def test_build_writes_the_dataset(tmp_path):
    meta = td.build(tmp_path, 50)
    assert meta["n"] == 50 and meta["train"] + meta["val"] == 50 and meta["test_set_sha256"] == TEST_SHA256
    rows = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert len(rows) == meta["train"] and set(rows[0]) == {"id", "route", "user", "answer"}
    assert (tmp_path / "system_prompt.md").read_text().startswith("You are the reasoning step")


def test_rare_routes_are_not_crowded_out(test_set):
    """The route cap keeps a draw that adds to a short route even when its paired route is full (17 Nov 2026: a first
    cap rejected those and left 4 DAWN and 2 DEFUSE-3 scenarios in 1,200)."""
    counts = sc.routes(sc.generate_training(1200, exclude=test_set))
    assert counts["ASPECTS unknown; DAWN group met"] >= 20 and counts["ASPECTS unknown; DEFUSE-3"] >= 10
    assert counts["Within 4.5 h with a disabling deficit"] > 0 and counts["Acute intracranial haemorrhage"] > 0
    assert all(counts[r] == 0 for r in sc.held_out_routes())
