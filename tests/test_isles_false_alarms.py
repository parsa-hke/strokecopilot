"""EXP-50 (isles_false_alarms.py) on synthetic NIfTI heads through EXP-16's real slab preparation: no data."""

import json

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("scipy")
pytest.importorskip("cv2")

from strokecopilot.vision.haemorrhage import isles_false_alarms as ifa  # noqa: E402
from strokecopilot.vision.haemorrhage.aggregate import Calibrator  # noqa: E402


def head(n_z=20):
    """A head in HU, 64 x 64 x n_z at 3.5 x 3.5 x 2.5 mm: thick bone and little brain at the bottom (skull base),
    more brain higher up."""
    hu = np.full((64, 64, n_z), -1000.0)
    yy, xx = np.mgrid[:64, :64]
    r = np.hypot(yy - 32, xx - 32)
    for z in range(n_z):
        brain = 6 + 18 * min(1.0, z / (n_z * 0.5))
        hu[..., z][r < 30] = 30.0
        hu[..., z][(r >= brain) & (r < 28)] = 1000.0
        hu[..., z][r < brain] = 35.0
    return nib.Nifti1Image(hu, np.diag([3.5, 3.5, 2.5, 1.0]))


def fake_predict(hot):
    """Slab probabilities: 0.05 everywhere, `hot` gives {slab index: probability} (negative = from the top)."""
    def predict(images):
        p = np.full(len(images), 0.05)
        for i, v in hot.items():
            p[i] = v
        return p
    return predict


CONTRACT = Calibrator(method="topk", k=2, temperature=1.0, bias=0.0, threshold=0.4, kind="platt")


def test_raw_threshold_inverts_the_calibration():
    c = Calibrator(method="topk", k=2, temperature=0.99, bias=-2.07, threshold=0.175)
    assert float(c.apply(np.array([ifa.raw_threshold(c)]))[0]) == pytest.approx(0.175, abs=1e-9)
    ifa.check_aggregation(CONTRACT)
    with pytest.raises(ValueError):
        ifa.check_aggregation(Calibrator(method="max"))


def test_thresholds_take_none_from_dec28_and_the_rest_from_exp49():
    exp49 = {"selection": {"candidates": {
        "none": {"rule": "none", "ratio": None, "full_split": {"threshold_raw": 0.9}},
        "gate(0.40)": {"rule": "gate", "ratio": 0.4, "full_split": {"threshold_raw": 0.35}},
        "adjacent": {"rule": "adjacent", "ratio": None}}}}  # no export: left out
    rules = ifa.thresholds(CONTRACT, exp49)
    assert set(rules) == {"none", "gate(0.40)"} and rules["none"][2] == pytest.approx(0.4)
    assert set(ifa.thresholds(CONTRACT)) == {"none"}


def test_run_flags_the_skull_base_scan_under_none_but_not_under_the_gate(tmp_path):
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    torch.manual_seed(0)
    net = timm.create_model("resnet18", pretrained=False, num_classes=1)
    scans = {"case-base": fake_predict({0: 0.9, 1: 0.8}), "case-mid": fake_predict({5: 0.9, 6: 0.85}),
             "case-clear": fake_predict({})}
    rules = {"none": ("none", None, 0.4), "gate(0.40)": ("gate", 0.4, 0.4)}
    current = {}

    def load(case):
        current["case"] = case
        return head()

    table, cams, examples = ifa.run(list(scans), load, lambda imgs: scans[current["case"]](imgs), rules, explain_net=net)
    t = table.set_index("case")
    assert t.loc["case-base", "slabs"] == 10 and (t["error"] == "").all()
    assert t.loc["case-base", "none_called"] and not t.loc["case-base", "gate(0.40)_called"]  # skull-base slabs gated
    assert t.loc["case-mid", "none_called"] and t.loc["case-mid", "gate(0.40)_called"]
    assert not t.loc["case-clear", "none_called"]
    assert t.loc["case-base", "top_height"] == 0.0 and t.loc["case-base", "top_relative_bf"] < ifa.SKULL_BASE_RELATIVE_BF
    assert t.loc["case-mid", "top_adjacent"]
    assert len(cams) == 4 and len(examples) == 2  # two flagged scans x top 2 slabs; one example each
    summary = ifa.summarise(table, cams, rules, proposal="gate(0.40)")
    assert summary["candidates"]["none"]["flagged"] == 2 and summary["candidates"]["gate(0.40)"]["flagged"] == 1
    assert summary["candidates"]["gate(0.40)"]["vs_none"] == {"both": 1, "only_first": 1, "only_second": 0, "p_exact": 1.0}
    assert summary["top_slab"]["flagged"]["skull_base_share"] == 0.5 and summary["gradcam_flagged"]["slabs"] == 4
    assert "case-" not in json.dumps(summary)  # no case ID in what is saved
    assert ifa.gradcam_figure(examples, tmp_path / "g.png", n=2).exists()
    assert ifa.contract_check(table, expected=2)["reproduced"]


def test_a_failed_case_is_kept_with_its_error():
    def load(case):
        raise OSError("unreadable")
    table, cams, _ = ifa.run(["case-x"], load, None, {"none": ("none", None, 0.4)})
    assert table.loc[0, "error"].startswith("OSError") and cams == []
    summary = ifa.summarise(table, cams, {"none": ("none", None, 0.4)})
    assert summary["failed"] == 1 and summary["scored"] == 0 and summary["candidates"]["none"]["flagged"] == 0


def test_mcnemar_counts_flips():
    m = ifa.mcnemar([1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1])
    assert (m["both"], m["only_first"], m["only_second"]) == (2, 2, 1) and 0 < m["p_exact"] <= 1
