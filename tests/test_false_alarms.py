"""Item 2's pre-registered selection (false_alarms.py) on synthetic validation slices: no data, no GPU."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("torch")
pytest.importorskip("scipy")

from strokecopilot.vision.haemorrhage import false_alarms as fa  # noqa: E402

SEEDS = (11, 12, 13)


def fake_slices(n_patients=150, seed=0):
    """Two scans per patient, 12 slices each, bottom (skull base) to top. Brain fraction rises from 0.05 to 0.60 in
    steps of 0.05. A third of scans bleed: most bleeds sit mid-head on two neighbouring slices; subarachnoid ones on
    slices 3 and 4 (brain fraction 0.20 and 0.25: kept by a mild gate, dropped by the 0.70 one, whose cut is 0.42).
    Half the scans without a bleed have two hot skull-base slices (0 and 1, brain fraction 0.05 and 0.10): the
    false alarms a gate should remove."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        for s in range(2):
            sid, bleed = f"s{p}_{s}", rng.random() < 1 / 3
            subtype = rng.choice(["intraparenchymal", "subdural", "subarachnoid"], p=[0.5, 0.3, 0.2]) if bleed else None
            probs = rng.uniform(0.0, 0.15, 12)
            bf = np.linspace(0.05, 0.60, 12)
            hot = []
            if bleed and subtype == "subarachnoid":
                hot = [3, 4]
            elif bleed:
                i = int(rng.integers(5, 10))
                hot = [i, i + 1]
            elif rng.random() < 0.5:
                probs[[0, 1]] = rng.uniform(0.6, 0.95, 2)  # skull-base false alarm
            probs[hot] = rng.uniform(0.6, 0.95, len(hot))
            for z in range(12):
                rows.append({"series_id": sid, "patient_id": f"p{p}", "z_mm": 5.0 * z, "prob": probs[z],
                             "brain_fraction": bf[z], "head_px": 1000 + 100 * z, "any": int(z in hot),
                             **{t: int(bleed and subtype == t and z in hot) for t in fa.SUBTYPES}})
    return pd.DataFrame(rows)


def test_selection_finds_the_gain_and_respects_subtypes():
    slices = fake_slices()
    report = fa.select(slices, seeds=SEEDS, bootstrap=300, log=lambda *_: None)
    rows = report["candidates"]
    assert list(rows)[0] == "none" and len(rows) == 10
    assert report["scans"] == 300 and report["patients"] == 150
    # Gating the skull-base slice removes most false alarms ...
    assert rows["gate(0.25)"]["specificity"] > rows["none"]["specificity"] + 0.2 and rows["gate(0.25)"]["eligible"]
    # A gate that drops the subarachnoid bleeds' slices keeps overall sensitivity near the target (the threshold
    # falls to catch them) but loses that subtype: the subtype rule rules it out ...
    assert not rows["gate(0.40)"]["eligible"]
    assert any("subarachnoid" in r for r in rows["gate(0.40)"]["ineligible_because"])
    # ... and where the threshold falls far enough to hold every subtype, specificity collapses instead.
    assert rows["gate(0.70)"]["specificity"] < rows["none"]["specificity"]
    assert report["proposal"] in ("gate(0.25)", "both(0.25)")
    best = rows[report["proposal"]]
    assert best["eligible"] and best["specificity"] == max(r["specificity"] for r in rows.values() if r["eligible"])
    assert report["specificity_gain"]["gain"] > 0 and report["specificity_gain"]["ci95"][0] > 0
    assert report["meets_adoption_rule"] is True
    text = repr(report)
    assert "s1_0" not in text and "p1" not in text  # no scan or patient ID in the report


def test_none_is_proposed_when_nothing_beats_it():
    slices = fake_slices(n_patients=60, seed=1)
    slices["brain_fraction"] = 0.5  # the gate removes nothing; adjacency is no better
    report = fa.select(slices, seeds=SEEDS, ratios=(0.4,), bootstrap=100, log=lambda *_: None)
    assert len(report["candidates"]) == 4
    if report["proposal"] == "none":
        assert report["specificity_gain"] is None and report["meets_adoption_rule"] is False


def test_eligibility_margins():
    none = {"sensitivity": 0.96, "subtype_sensitivity": {s: 0.95 for s in fa.SUBTYPES}}
    ok = {"sensitivity": 0.952, "subtype_sensitivity": {**none["subtype_sensitivity"], "subdural": 0.925}}
    assert fa.eligible(ok, none) == (True, [])
    bad = {"sensitivity": 0.948, "subtype_sensitivity": {**none["subtype_sensitivity"], "epidural": 0.91}}
    good, reasons = fa.eligible(bad, none)
    assert not good and len(reasons) == 2


def test_paired_gain_is_paired_by_patient():
    labels = np.array([0, 0, 0, 0, 1, 1])
    patients = np.array(["a", "a", "b", "c", "c", "d"])
    a = np.array([0, 0, 0, 1, 1, 1])  # 3 of 4 negatives kept
    b = np.array([1, 0, 1, 1, 1, 1])  # 1 of 4
    g = fa.paired_specificity_gain(a, b, labels, patients, n=200, seed=1)
    assert g["gain"] == pytest.approx(0.5) and g["ci95"][0] <= 0.5 <= g["ci95"][1]


def test_heights_and_reference_sample():
    from strokecopilot.vision.haemorrhage.aggregate import Calibrator

    slices = fake_slices(n_patients=40, seed=2)
    slices["height"] = fa.heights(slices)
    one = slices[slices["series_id"] == "s0_0"].sort_values("z_mm")
    assert one["height"].iloc[0] == 0 and one["height"].iloc[-1] == 1 and one["height"].is_monotonic_increasing
    rows = fa.reference_slices(slices, Calibrator(method="topk", k=2), threshold=0.3, n=5, seed=3)
    assert set(rows["group"]) == {"tp", "fp"} and len(rows) == 10
    assert all(rows.loc[i, "prob"] == slices[slices["series_id"] == rows.loc[i, "series_id"]]["prob"].max() for i in rows.index)
    explained = [{"prob": 0.5, "brain": 0.2, "bone": 0.5, "other": 0.2, "outside": 0.1}] * len(rows)
    summary = fa.reference_summary(rows, explained)
    assert set(summary) == {"tp", "fp"} and summary["fp"]["slices"] == 5 and summary["tp"]["bone"] == 0.5


def test_measure_reads_slices_through_the_shard_readers():
    from strokecopilot.data.prep_rsna import window_image

    yy, xx = np.mgrid[:96, :96]
    r = np.hypot(yy - 48, xx - 48)
    hu = np.full((96, 96), -1000.0)
    hu[r < 40] = 30
    hu[(r >= 32) & (r < 36)] = 1000
    hu[r < 30] = 35
    images = {"a": window_image(hu, 1, 0, size=96), "b": window_image(np.full((96, 96), -1000.0), 1, 0, size=96)}

    class Reader:
        def read(self, image_id):
            return images[image_id]

    items = pd.DataFrame({"shard": [0, 0], "image_id": ["a", "b"], "any": [0, 0]}, index=[7, 9])
    out = fa.measure(items, {0: Reader()}, workers=0, batch_size=2)
    assert list(out.index) == [7, 9] and out.loc[9, "head_px"] == 0 and out.loc[9, "brain_fraction"] == 0
    assert 0.3 < out.loc[7, "brain_fraction"] < 0.8
