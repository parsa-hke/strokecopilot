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


# --- EXP-43 stays reproducible; the decision-balanced set (EXP-91; Parsa, 22 Nov 2026) --------------------------

EXP43_SHA256 = {"train.jsonl": "c18da3177263f5e1a0113edcd7ca71edebae33d3824f1dda1c7d7584f64afa8d",
                "val.jsonl": "18f7c8c423d0d468ea4b71edc582dc0eea2bb77d05b9003c25645a5bb2ba400c"}  # the files EXP-45 trained on
EXP91_SHA256 = {"train.jsonl": "99010f244d9373b0c4e4ce088269ab38449f6158818af156fe1241c034798963",
                "val.jsonl": "4d6d62a67ec987207c81a1515e76454379e962b9ee37d7061e4372a5a1ba40d8"}
HELD_OUT = ["Beyond 4.5 h and eligible for thrombectomy", "LVO 4.5-24 h without thrombectomy", "INR above 1.7",
            "No occlusion is identified"]


def test_the_default_build_is_exp43_byte_for_byte(tmp_path):
    """The balance option must not move the default: the same files and counts as runs/EXP-43_train_set (meta.json)."""
    meta = td.build(tmp_path, 1200)
    assert {name: td.file_sha256(tmp_path / name) for name in EXP43_SHA256} == EXP43_SHA256
    assert meta["thrombolysis"] == {"yes": 280, "no": 393, "undetermined": 527}
    assert meta["thrombectomy"] == {"yes": 246, "no": 570, "undetermined": 384}
    assert meta["held_out_routes"] == HELD_OUT and meta["seed"] == 20261117 and "balanced" not in meta
    assert meta["routes"]["ASPECTS unknown; DAWN group met"] == 36 and meta["routes"]["Relative contraindication"] == 335


def test_a_balanced_set_reaches_half_yes_when_the_caps_allow(test_set):
    """At 120 scenarios the caps are far away, so both treatments reach exactly half; every rule of the plain
    generator still holds."""
    make = lambda: sc.generate_training(120, seed=sc.BALANCED_SEED, exclude=test_set, balance=True, pool=20_000)
    balanced = make()  # no held_out given: DEC-30's four, not the routes the new seed would draw
    assert sc.held_out_routes(sc.BALANCED_SEED) != HELD_OUT == sc.held_out_routes()
    counts = sc.decision_balance(balanced)
    assert counts["thrombolysis"]["recommend_true"] == counts["thrombectomy"]["recommend_true"] == 60
    assert counts["thrombolysis"]["recommend_false"] == counts["thrombectomy"]["recommend_false"] == 60
    assert sum(counts["pairs"].values()) == 120
    for s in balanced:
        assert not any(v["reason"].startswith(r) for v in s.expected.values() for r in HELD_OUT), s.id
    signatures = [sc.signature(s.vision, s.facts) for s in balanced]
    assert len(set(signatures)) == 120 and not set(signatures) & {sc.signature(s.vision, s.facts) for s in test_set}
    assert sc.cap_violations(balanced) == []
    assert [s.id for s in balanced[:2]] == ["train-0001", "train-0002"] and balanced[0].vision["case_id"] == "train-0001"
    assert [sc.signature(s.vision, s.facts) for s in make()] == signatures  # deterministic
    plain = sc.decision_balance(sc.generate_training(120, exclude=test_set))
    assert plain["thrombectomy"]["recommend_true"] < 40  # the unbalanced generator is mostly "no"


def test_undetermined_is_trained_as_not_recommended(training):
    """Balance is counted on the answer the model learns to give: gold.py writes recommend = true only for "yes"."""
    corpus = td.ingest.load()
    for s in training[:60]:
        reply = td.gold.gold_answer(s.vision, s.facts, corpus)
        assert {d: reply[d]["recommend"] for d in ("thrombolysis", "thrombectomy")} == sc.recommended(s.expected)
    assert any(v["value"] == "undetermined" for s in training[:60] for v in s.expected.values())


def test_cap_violations_replays_the_generators_rules(training):
    assert sc.cap_violations(training) == []
    assert sc.cap_violations(training[:30] + training[:1]) == [training[0].id]  # a repeated case
    assert sc.cap_violations(training, per_path=5, per_undetermined=5)  # a tighter cap finds the paths over it


@pytest.fixture(scope="module")
def exp91(tmp_path_factory):
    out = tmp_path_factory.mktemp("exp91")
    return out, td.build_balanced(out)


def test_the_balanced_build_is_exp91(exp91):
    """The recorded set (runs/EXP-91_train_set_balanced): thrombectomy at half, thrombolysis as far as the caps allow.
    Only two thrombolysis "yes" routes can be trained with DEC-30's routes held out, and the route cap holds them to
    200 and 120."""
    out, meta = exp91
    assert {name: td.file_sha256(out / name) for name in EXP91_SHA256} == EXP91_SHA256 == meta["files_sha256"]
    assert (meta["n"], meta["train"], meta["val"], meta["seed"]) == (1200, 1080, 120, 20261122)
    assert meta["held_out_routes"] == HELD_OUT and all(meta["routes"][r] == 0 for r in HELD_OUT)
    assert meta["thrombolysis"] == {"yes": 320, "no": 394, "undetermined": 486}
    assert meta["thrombectomy"] == {"yes": 594, "no": 411, "undetermined": 195}
    assert meta["recommend"] == {"thrombolysis": {"true": 320, "false": 880}, "thrombectomy": {"true": 594, "false": 606}}
    for d in ("thrombolysis", "thrombectomy"):  # the two slices add up to the whole
        assert all(meta["recommend_train"][d][k] + meta["recommend_val"][d][k] == meta["recommend"][d][k] for k in ("true", "false"))
    assert meta["answer_pairs"] == {"both": 80, "thrombolysis_only": 240, "thrombectomy_only": 514, "neither": 366}
    assert meta["caps"] == {"per_path": 40, "per_undetermined": 20, "per_route": 120}  # EXP-43's, not relaxed
    assert meta["cap_violations"] == 0 and meta["largest_decision_path"] == 40
    assert meta["cases_shared_with"] == {"evaluation_scenarios": 0, "exp43_training_scenarios": 0}
    assert meta["gold_checks"] == {"examples": 1200, "passed": 1200}
    assert meta["test_set_sha256"] == TEST_SHA256 and meta["prompt_version"] == "reasoner_v3"
    assert meta["routes"]["Within 4.5 h with a disabling deficit"] == 200 and meta["routes"]["Extended window, not eligible"] == 120
    rows = [json.loads(l) for name in EXP91_SHA256 for l in (out / name).read_text().splitlines()]
    assert len({r["id"] for r in rows}) == 1200 and set(rows[0]) == {"id", "route", "user", "answer"}
    assert td.answer_counts(rows) == meta["recommend"]


def test_check_dataset_refuses_the_wrong_or_a_changed_set(exp91, tmp_path):
    import shutil

    from strokecopilot.nlp.reasoner import finetune as ft

    out, _ = exp91
    folder = tmp_path / "set"
    shutil.copytree(out, folder)
    meta = json.loads((folder / "meta.json").read_text())
    (folder / "meta.json").write_text(json.dumps({**meta, "experiment": "EXP-91"}))
    assert ft.check_dataset(folder, "EXP-91", TEST_SHA256)["train"] == 1080
    with pytest.raises(ValueError, match="not EXP-43"):
        ft.check_dataset(folder, "EXP-43", TEST_SHA256)
    with pytest.raises(ValueError, match="evaluation set"):
        ft.check_dataset(folder, "EXP-91", "0" * 64)
    with (folder / "val.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="val.jsonl"):
        ft.check_dataset(folder, "EXP-91", TEST_SHA256)
