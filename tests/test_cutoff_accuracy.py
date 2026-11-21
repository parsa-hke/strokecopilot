"""Vision A as a yes/no call at each decision cut-off (report-only analysis of saved volumes)."""

import csv

import numpy as np

from strokecopilot.analysis import cutoff_accuracy as ca
from strokecopilot.vision.perfusion.evaluate import crosses


def volumes():
    ref = np.array([0.0, 5.0, 20.9, 21.0, 30.0, 60.0, 120.0])
    return {"reference": ref, "nnunet": np.array([0.0, 25.0, 3.0, 21.0, 10.0, 80.0, 30.0]),
            "threshold": np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 25.0])}


def test_counts_at_21_ml_by_hand():
    row = ca.call(volumes()["nnunet"], volumes()["reference"], 21.0)
    # truly at or above 21 mL: 21, 30, 60, 120 -> predicted 21, 10, 80, 30: three caught, one missed
    assert (row["tp"], row["fn"]) == (3, 1)
    # truly below: 0, 5, 20.9 -> predicted 0, 25, 3: one over-call
    assert (row["tn"], row["fp"]) == (2, 1)
    assert row["sensitivity"] == 0.75 and abs(row["specificity"] - 2 / 3) < 1e-12
    assert row["sensitivity_ci95"][0] < 0.75 < row["sensitivity_ci95"][1]


def test_wrong_side_matches_evaluate_crosses():
    v = volumes()
    table = ca.cutoff_table(v)
    for cutoff in (21.0, 31.0, 51.0, 70.0):
        for method in ("nnunet", "threshold"):
            expected = sum(crosses(p, r, cutoff) for p, r in zip(v[method], v["reference"]))
            assert table[f"{cutoff:g}"][method]["wrong_side"] == expected


def test_always_under_catches_nothing_and_never_over_calls():
    table = ca.cutoff_table(volumes())
    for row in table.values():
        assert row["always_under"]["sensitivity"] == 0.0 and row["always_under"]["specificity"] == 1.0
        assert row["always_under"]["wrong_side"] == row["always_under"]["at_or_above"]


def test_empty_group_gives_none():
    row = ca.call(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 21.0)
    assert row["sensitivity"] is None and row["specificity"] == 1.0


def test_load_volumes_drops_the_case_column(tmp_path):
    path = tmp_path / "per_case.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "ref_ml", "nnunet_pred_ml", "threshold_pred_ml"])
        w.writerow(["made-up-1", "10.5", "0", "3.25"])
        w.writerow(["made-up-2", "80", "35.5", "20"])
    v = ca.load_volumes(path)
    assert set(v) == {"reference", "nnunet", "threshold"}
    assert v["reference"].tolist() == [10.5, 80.0] and v["nnunet"].tolist() == [0.0, 35.5]
    s = ca.summary(v)
    assert s["cases"] == 2 and abs(s["nnunet"]["abs_error_ml_mean"] - (10.5 + 44.5) / 2) < 1e-12
