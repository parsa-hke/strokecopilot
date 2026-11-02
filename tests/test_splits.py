"""The fixed split: no held-out leakage, disjoint folds, stable output (plan §5.5, DEC-5)."""

import json
from collections import Counter

import pytest

from strokecopilot.data import splits

CASES = [f"sub-stroke{i:04d}" for i in range(1, 190) if i % 5 != 0][:149]


@pytest.fixture
def inputs():
    volumes = {case: float((i * 37) % 200) for i, case in enumerate(CASES)}
    centres = {case: 1 if i % 3 else 2 for i, case in enumerate(CASES)}
    return volumes, centres


@pytest.fixture
def folds(inputs):
    return splits.make_splits(*inputs)


def _fold_cases(folds):
    return [case for group in folds["folds"].values() for case in group]


def test_sizes_and_coverage(folds):
    assert len(folds["heldout"]) == 20
    assert len(_fold_cases(folds)) == 129
    assert set(folds["heldout"]) | set(_fold_cases(folds)) == set(CASES)
    sizes = [len(group) for group in folds["folds"].values()]
    assert len(sizes) == 5 and max(sizes) - min(sizes) <= 1


def test_groups_are_disjoint(folds):
    assert not set(folds["heldout"]) & set(_fold_cases(folds))
    counts = Counter(_fold_cases(folds))
    assert max(counts.values()) == 1


def test_same_input_gives_same_split_whatever_the_order(inputs, folds):
    volumes, centres = inputs
    shuffled = dict(reversed(list(volumes.items())))
    assert splits.make_splits(shuffled, centres) == folds


def test_seed_changes_the_split(inputs, folds):
    assert splits.make_splits(*inputs, seed=1)["heldout"] != folds["heldout"]


def test_stratification_is_balanced(inputs, folds):
    volumes, centres = inputs
    strata = splits._strata(volumes, centres)
    assert sum(len(v) for v in strata.values()) == 149
    for cases in strata.values():
        expected = len(cases) * 20 / 149
        assert abs(len(set(cases) & set(folds["heldout"])) - expected) <= 1
        per_fold = [len(set(cases) & set(group)) for group in folds["folds"].values()]
        assert max(per_fold) - min(per_fold) <= 1


def test_centre_mismatch_is_refused(inputs):
    volumes, centres = inputs
    with pytest.raises(ValueError):
        splits.make_splits(volumes, {k: centres[k] for k in list(centres)[:-1]})


def test_validate_catches_duplicates():
    with pytest.raises(ValueError, match="more than once"):
        splits.validate_folds({"heldout": ["a"], "folds": {"0": ["a"]}})


def test_write_never_replaces(folds, tmp_path):
    path = tmp_path / "folds.json"
    splits.write_folds(folds, path)
    assert splits.load_folds(path) == folds
    with pytest.raises(FileExistsError):
        splits.write_folds(folds, path)
    assert json.loads(path.read_text()) == folds


def test_guard_blocks_heldout_without_flag(folds, monkeypatch):
    monkeypatch.delenv("ALLOW_HELDOUT", raising=False)
    with pytest.raises(splits.HeldoutError) as error:
        splits.check_ids([folds["heldout"][0]], folds)
    assert folds["heldout"][0] not in str(error.value)
    with pytest.raises(splits.HeldoutError):
        splits.heldout_ids(folds)
    splits.check_ids(_fold_cases(folds), folds)


def test_guard_opens_only_with_exact_flag(folds, monkeypatch):
    monkeypatch.setenv("ALLOW_HELDOUT", "true")
    with pytest.raises(splits.HeldoutError):
        splits.heldout_ids(folds)
    monkeypatch.setenv("ALLOW_HELDOUT", "1")
    assert splits.heldout_ids(folds) == folds["heldout"]
    splits.check_ids(folds["heldout"], folds)


def test_train_val(folds, monkeypatch):
    monkeypatch.delenv("ALLOW_HELDOUT", raising=False)
    train, val = splits.train_val_ids(2, folds)
    assert val == folds["folds"]["2"]
    assert not set(train) & set(val)
    assert not set(train + val) & set(folds["heldout"])
    assert splits.development_ids(folds) == sorted(train + val)
    with pytest.raises(KeyError):
        splits.train_val_ids(9, folds)
