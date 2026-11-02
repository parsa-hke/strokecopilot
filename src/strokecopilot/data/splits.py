"""Reads splits/folds.json and guards against held-out use before Phase 4 (plan §5.5, DEC-5).

The split is made once, stratified by centre and infarct volume tertile, because the two centres
differ in size and possibly in what they record. Error messages give counts, never IDs.
"""

import json
import os
import random
from collections import defaultdict

from strokecopilot.config import SPLITS_DIR

SEED = 20261102
N_HELDOUT = 20
N_FOLDS = 5
VERSION = 1
STRATIFIED_BY = "center+infarct_volume_tertile"
FOLDS_PATH = SPLITS_DIR / "folds.json"


class HeldoutError(RuntimeError):
    """A held-out case was requested before Phase 4."""


def _strata(volumes, centres):
    ranked = sorted(volumes, key=lambda case: (volumes[case], case))
    strata = defaultdict(list)
    for rank, case in enumerate(ranked):
        strata[(str(centres[case]), rank * 3 // len(ranked))].append(case)
    return dict(sorted(strata.items()))


def _heldout_quota(strata, n_heldout, n_cases):
    """Share n_heldout across strata in proportion to their size, by largest remainder."""
    exact = {key: len(cases) * n_heldout / n_cases for key, cases in strata.items()}
    quota = {key: int(value) for key, value in exact.items()}
    by_remainder = sorted(exact, key=lambda key: (quota[key] - exact[key], key))
    for key in by_remainder[: n_heldout - sum(quota.values())]:
        quota[key] += 1
    return quota


def make_splits(volumes, centres, n_heldout=N_HELDOUT, n_folds=N_FOLDS, seed=SEED):
    """Build the split from {case: infarct mL} and {case: centre}. The same input gives the same split."""
    if set(volumes) != set(centres):
        raise ValueError("volumes and centres must cover the same cases.")
    rng = random.Random(seed)
    strata = _strata(volumes, centres)
    quota = _heldout_quota(strata, n_heldout, len(volumes))

    heldout, folds = [], {str(k): [] for k in range(n_folds)}
    next_fold = 0
    for key, cases in strata.items():
        cases = sorted(cases)
        rng.shuffle(cases)
        heldout += cases[: quota[key]]
        for case in cases[quota[key] :]:
            folds[str(next_fold % n_folds)].append(case)
            next_fold += 1

    result = {
        "version": VERSION,
        "stratified_by": STRATIFIED_BY,
        "seed": seed,
        "heldout": sorted(heldout),
        "folds": {k: sorted(v) for k, v in folds.items()},
    }
    validate_folds(result, expected_cases=set(volumes))
    return result


def validate_folds(folds, expected_cases=None):
    groups = [folds["heldout"], *folds["folds"].values()]
    seen = [case for group in groups for case in group]
    if len(seen) != len(set(seen)):
        raise ValueError(f"{len(seen) - len(set(seen))} cases appear more than once in the split.")
    if expected_cases is not None and set(seen) != set(expected_cases):
        raise ValueError(f"The split covers {len(set(seen))} cases, expected {len(expected_cases)}.")


def write_folds(folds, path):
    """Write the split once. An existing file is never replaced."""
    with open(path, "x") as f:
        json.dump(folds, f, indent=2)
        f.write("\n")


def load_folds(path=FOLDS_PATH):
    with open(path) as f:
        folds = json.load(f)
    validate_folds(folds)
    return folds


def heldout_allowed():
    return os.environ.get("ALLOW_HELDOUT") == "1"


def check_ids(ids, folds=None):
    """Raise if any ID is held out, unless ALLOW_HELDOUT=1, which only Phase 4 notebooks set."""
    folds = folds or load_folds()
    if heldout_allowed():
        return
    found = set(ids) & set(folds["heldout"])
    if found:
        raise HeldoutError(f"{len(found)} held-out cases were requested before Phase 4.")


def heldout_ids(folds=None):
    folds = folds or load_folds()
    if not heldout_allowed():
        raise HeldoutError("Held-out cases are not available before Phase 4.")
    return list(folds["heldout"])


def development_ids(folds=None):
    folds = folds or load_folds()
    return sorted(case for group in folds["folds"].values() for case in group)


def train_val_ids(fold, folds=None):
    folds = folds or load_folds()
    key = str(fold)
    if key not in folds["folds"]:
        raise KeyError(f"No fold {fold}.")
    val = list(folds["folds"][key])
    train = sorted(case for k, group in folds["folds"].items() if k != key for case in group)
    check_ids(train + val, folds)
    return train, val
