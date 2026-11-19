"""Vision B's final RSNA test-split report (final_check.py) on synthetic slices: no data, no GPU."""

import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from strokecopilot.vision.haemorrhage import aggregate as agg  # noqa: E402
from strokecopilot.vision.haemorrhage import final_check as fc  # noqa: E402

SUBTYPE_P = {"intraparenchymal": 0.4, "subdural": 0.3, "subarachnoid": 0.2, "epidural": 0.1}


def fake_slices(n_patients=200, seed=0):
    """One or two scans per patient, 10 slices each; about 40% of scans bleed, on two slices, with one subtype.
    Bleeding slices score high, most others low, and a few non-bleeding scans have one hot slice (false alarms)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        for s in range(1 + int(rng.random() < 0.3)):
            sid, bleed = f"s{p}_{s}", rng.random() < 0.4
            subtype = rng.choice(list(SUBTYPE_P), p=list(SUBTYPE_P.values())) if bleed else None
            probs = rng.uniform(0.0, 0.2, 10)
            hot = [4, 5] if bleed else []
            if bleed:
                probs[hot] = rng.uniform(0.3, 0.98, 2)
            elif rng.random() < 0.15:
                probs[2] = rng.uniform(0.5, 0.9)
            for z in range(10):
                row = {"series_id": sid, "patient_id": f"p{p}", "prob": probs[z], "any": int(z in hot)}
                row.update({t: int(z in hot and t == subtype) for t in fc.SUBTYPES})
                rows.append(row)
    return pd.DataFrame(rows)


def identity(threshold=0.5, method="topk", k=2):
    return agg.Calibrator(method=method, k=k, threshold=threshold)


def test_weighted_auc_matches_rank_auc_with_ties():
    rng = np.random.default_rng(1)
    labels = rng.random(300) < 0.4
    scores = np.round(rng.random(300), 1)  # many ties
    assert fc._weighted_auc(labels, scores, np.ones(300)) == pytest.approx(agg.roc_auc(labels, scores))


def test_weighted_auc_equals_duplicated_rows():
    rng = np.random.default_rng(2)
    labels, scores = rng.random(80) < 0.5, rng.random(80)
    w = rng.integers(0, 4, 80)
    dup_labels, dup_scores = np.repeat(labels, w), np.repeat(scores, w)
    assert fc._weighted_auc(labels, scores, w.astype(float)) == pytest.approx(agg.roc_auc(dup_labels, dup_scores))


def test_report_counts_match_a_direct_count():
    slices = fake_slices()
    cal = identity(0.5)
    rep = fc.report(slices, cal, n=200)
    scans = fc.scan_frame(slices, cal)
    called = scans["prob"].to_numpy() >= 0.5
    labels = scans["label"].to_numpy().astype(bool)
    pre = rep["preregistered"]
    assert pre["sensitivity"] == pytest.approx(called[labels].mean(), abs=1e-4)
    assert pre["specificity"] == pytest.approx((~called[~labels]).mean(), abs=1e-4)
    assert rep["scans"] == len(scans) and rep["patients"] == slices["patient_id"].nunique()
    for s in fc.SUBTYPES:
        mask = scans[s].to_numpy() == 1
        r = pre["subtype_sensitivity"][s]
        assert r["bleeds"] == int(mask.sum()) and r["caught"] == int(called[mask].sum())
    assert sum(r["bleeds"] for r in pre["subtype_sensitivity"].values()) == int(labels.sum())  # one subtype each here
    for key in ("scan_auc", "sensitivity", "specificity"):
        lo, hi = pre[f"{key}_ci95"]
        assert lo <= pre[key] <= hi
    assert 0 <= pre["scan_ece"] <= 1


def test_threshold_is_applied_on_the_calibrated_scale():
    slices = fake_slices()
    low, high = fc.report(slices, identity(0.2), n=50), fc.report(slices, identity(0.8), n=50)
    assert low["preregistered"]["sensitivity"] >= high["preregistered"]["sensitivity"]
    assert low["preregistered"]["specificity"] <= high["preregistered"]["specificity"]
    shifted = agg.Calibrator(method="topk", k=2, temperature=1.0, bias=3.0, threshold=0.5)  # raises every probability
    assert fc.report(slices, shifted, n=50)["preregistered"]["sensitivity"] >= low["preregistered"]["sensitivity"] - 1e-9


def test_report_is_deterministic_and_holds_no_ids():
    slices = fake_slices()
    a, b = fc.report(slices, identity(), n=100), fc.report(slices, identity(), n=100)
    text = json.dumps(a, default=str)
    assert text == json.dumps(b, default=str)
    ids = set(slices["series_id"]) | set(slices["patient_id"])
    assert not any(f'"{i}"' in text for i in ids)
    assert fc.ids_in('{"x": "ID_abc"}', set()) and fc.ids_in('{"x": "p3"}', {"p3"}) and not fc.ids_in(text, set())


def test_missing_subtype_column_is_refused():
    with pytest.raises(ValueError, match="lacks"):
        fc.report(fake_slices(20).drop(columns=["epidural"]), identity(), n=10)


def test_scan_with_two_patients_is_refused():
    slices = fake_slices(20)
    slices.loc[slices.index[0], "patient_id"] = "other"
    with pytest.raises(ValueError, match="more than one patient"):
        fc.report(slices, identity(), n=10)


def test_validation_check_gates_the_test_split():
    rep = fc.report(fake_slices(), identity(0.3), n=100)
    pre = rep["preregistered"]
    assert fc.validation_check(rep, expected_auc=pre["scan_auc"], sensitivity_floor=0.0) == []
    problems = fc.validation_check(rep, expected_auc=pre["scan_auc"] + 0.05, sensitivity_floor=1.01)
    assert len(problems) == 2 and "AUC" in problems[0] and "sensitivity" in problems[1]


def test_summary_lines_name_every_subtype():
    lines = fc.summary_lines(fc.report(fake_slices(), identity(), n=50))
    assert all(any(s in line for line in lines) for s in fc.SUBTYPES)
