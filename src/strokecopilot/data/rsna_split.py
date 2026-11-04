"""RSNA patient split: train / validation / test by a seeded hash of patient ID (plan §5.2, §8.2).

Every scan of a patient lands in the same split. Patients are ranked by a SHA-256 hash of the seed
and their ID, separately among patients with and without a bleed, and cut 80/10/10. Stratifying
this way keeps validation and test at the natural bleed rate: in a simulation at the real counts
(18,938 patients, 21,744 scans), a plain hash cut missed a 1.5-point tolerance in 35% of seeds.
The split depends only on the index, never on row order, so every Kaggle session rebuilds the same one. Nothing is saved: the split holds patient IDs,
which never leave Kaggle (docs/DATA_USE.md). `fingerprint` gives one hash to confirm the split is
unchanged between sessions.

The test split stays closed until the final Vision B model is chosen: `rows` raises for it unless
ALLOW_RSNA_TEST=1 is set. Messages and summaries give counts only.
"""

import hashlib
import os

import pandas as pd

SEED = 20261103
SHARES = (("train", 0.8), ("val", 0.1), ("test", 0.1))
SPLITS = tuple(name for name, _ in SHARES)
RATE_TOLERANCE_PP = 1.5
TEST_FLAG = "ALLOW_RSNA_TEST"


class TestSplitError(RuntimeError):
    """The RSNA test split was requested before it was opened."""


def _rank_key(patient_id, seed):
    return hashlib.sha256(f"{seed}:{patient_id}".encode()).hexdigest()


def assign(scans, seed=SEED):
    """{patient_id: split} from a scan table with patient_id and any (rsna.scan_table)."""
    bleed = scans.groupby("patient_id")["any"].max().astype(int)
    out = {}
    for _, group in bleed.groupby(bleed):
        ranked = sorted(group.index, key=lambda pid: _rank_key(pid, seed))
        start = 0
        for k, (name, share) in enumerate(SHARES):
            end = len(ranked) if k == len(SHARES) - 1 else start + round(share * len(ranked))
            out.update((pid, name) for pid in ranked[start:end])
            start = end
    return out


def add_split(table, assignment):
    """A copy of an index or scan table with a split column. Raises if a patient has no split."""
    split = table["patient_id"].map(assignment)
    if split.isna().any():
        raise ValueError(f"{int(split.isna().sum())} rows have a patient without a split.")
    return table.assign(split=split)


def summary(scans):
    """Counts and bleed rates per split, from a scan table with a split column. Numbers only."""
    def describe(frame):
        return {
            "patients": int(frame["patient_id"].nunique()),
            "scans": int(len(frame)),
            "slices": int(frame["slices"].sum()),
            "scan_bleed_rate": round(float(frame["any"].mean()), 4),
            "patient_bleed_rate": round(float(frame.groupby("patient_id")["any"].max().mean()), 4),
        }

    out = {"all": describe(scans)}
    out.update({name: describe(scans[scans["split"] == name]) for name in SPLITS})
    out["patient_share"] = {name: round(out[name]["patients"] / out["all"]["patients"], 4) for name in SPLITS}
    return out


def check(scans, tolerance_pp=RATE_TOLERANCE_PP):
    """Raise if a patient spans splits or if validation or test drifts from the overall scan bleed rate."""
    spans = scans.groupby("patient_id")["split"].nunique()
    if (spans > 1).any():
        raise ValueError(f"{int((spans > 1).sum())} patients have scans in more than one split.")
    report = summary(scans)
    overall = report["all"]["scan_bleed_rate"]
    for name in ("val", "test"):
        gap = abs(report[name]["scan_bleed_rate"] - overall) * 100
        if gap > tolerance_pp:
            raise ValueError(f"The {name} scan bleed rate is {gap:.2f} points from the overall rate, above {tolerance_pp}.")
    return report


def fingerprint(assignment):
    """One SHA-256 over every patient and split, to confirm two sessions built the same split."""
    lines = "\n".join(f"{pid},{name}" for pid, name in sorted(assignment.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def rows(table, name):
    """The rows of one split. The test split raises unless ALLOW_RSNA_TEST=1."""
    if name not in SPLITS:
        raise ValueError(f"Unknown split {name!r}; use one of {SPLITS}.")
    if name == "test" and os.environ.get(TEST_FLAG) != "1":
        raise TestSplitError(f"The RSNA test split is closed until the final Vision B model is chosen. Set {TEST_FLAG}=1 only then.")
    return table[table["split"] == name]
