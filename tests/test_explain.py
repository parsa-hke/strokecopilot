"""Item 2: Grad-CAM, the density masks and the candidate scan-score rules, on CPU with a synthetic head phantom and a
small random ResNet. No data."""

import numpy as np
import pytest

pytest.importorskip("scipy")
cv2 = pytest.importorskip("cv2")

from strokecopilot.data.prep_rsna import window_image  # noqa: E402
from strokecopilot.vision.haemorrhage import explain as ex  # noqa: E402
from strokecopilot.vision.haemorrhage import slab_rules as sr  # noqa: E402

SIZE = 128


def phantom(brain_r=38, skull=(40, 46), head_r=50, size=SIZE):
    """HU slice: air, scalp (30 HU) to head_r, a skull ring (1000 HU), brain (35 HU) inside brain_r."""
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    hu = np.full((size, size), -1000.0)
    hu[r < head_r] = 30.0
    hu[(r >= skull[0]) & (r < skull[1])] = 1000.0
    hu[r < brain_r] = 35.0
    return hu, r


def windowed(hu):
    return window_image(hu, 1, 0, size=hu.shape[0])


def test_masks_find_brain_bone_and_outside_on_a_phantom():
    hu, r = phantom()
    m = ex.tissue_masks(windowed(hu))
    assert m["head"][r < 49].all() and not m["head"][r > 52].any()
    assert m["bone"][(r >= 41) & (r < 45)].all() and not m["bone"][r < 39].any()
    assert m["brain"][r < 30].all()  # well inside
    assert not m["brain"][r >= 39].any()  # never in or beyond the skull (the margin keeps it off the bone)
    assert not (m["brain"] & m["bone"]).any() and not (m["other"] & (m["brain"] | m["bone"])).any()


def test_heat_shares_sum_to_one_and_follow_the_map():
    hu, r = phantom()
    masks = ex.tissue_masks(windowed(hu))
    centre = (r < 20).astype(float)
    shares = ex.heat_shares(centre, masks)
    assert shares["brain"] == pytest.approx(1.0) and sum(shares.values()) == pytest.approx(1.0)
    ring = ((r >= 41) & (r < 45)).astype(float)
    assert ex.heat_shares(ring, masks)["bone"] == pytest.approx(1.0)
    corner = np.zeros_like(centre)
    corner[:5, :5] = 1
    assert ex.heat_shares(corner, masks)["outside"] == pytest.approx(1.0)
    assert ex.heat_shares(np.zeros_like(centre), masks) == dict.fromkeys(("brain", "bone", "other", "outside"))


def test_slice_measures_give_brain_fraction_and_height():
    empty = np.full((SIZE, SIZE), -1000.0)
    base, _ = phantom(brain_r=12, skull=(14, 30), head_r=34)  # skull base: thick bone, little brain
    mid, _ = phantom()
    rows = ex.slice_measures(np.stack([windowed(s) for s in (empty, base, mid, mid, base)]))
    assert rows[0]["head_px"] == 0 and rows[0]["brain_fraction"] == 0.0
    assert rows[1]["brain_fraction"] < 0.15 < 0.4 < rows[2]["brain_fraction"]
    assert [r["height"] for r in rows] == [0.0, 0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]


def test_grad_cam_maps_and_probabilities_match_the_model():
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    from strokecopilot.vision.haemorrhage import infer

    torch.manual_seed(0)
    net = timm.create_model("resnet18", pretrained=False, num_classes=1)
    hu, _ = phantom(size=64)
    images = np.stack([windowed(hu), windowed(hu[::-1].copy())])
    cams, probs = ex.grad_cam(net, images)
    assert cams.shape == (2, 64, 64) and cams.min() >= 0 and cams.max() <= 1 + 1e-6
    assert np.allclose(probs, infer.slice_probs(net, images, "cpu"), atol=1e-5)
    rows, maps = ex.explain_slices(net, images, batch_size=1)
    assert len(rows) == 2 and maps.shape == (2, 64, 64) and set(rows[0]) == {"prob", "brain", "bone", "other", "outside"}
    assert ex.target_layer(net) is net.layer4
    with pytest.raises(ValueError):
        ex.target_layer(torch.nn.Linear(2, 2))


def test_candidates_are_fixed_and_rules_score_as_specified():
    assert len(sr.candidates()) == 1 + 4 + 1 + 4 and sr.candidates()[0] == ("none", None)
    p = [0.1, 0.9, 0.2, 0.8, 0.1]
    bf = [0.05, 0.10, 0.60, 0.70, 0.65]  # the hot slice at index 1 is a low-brain (skull-base) slice
    assert sr.scan_score(p) == pytest.approx(0.85)  # EXP-6: top two, 0.9 and 0.8
    assert sr.scan_score(p, rule="adjacent") == pytest.approx(0.55)  # best neighbouring pair: (0.9, 0.2) or (0.2, 0.8)
    assert sr.scan_score(p, bf, "gate", 0.4) == pytest.approx(0.5)  # 0.9 left out (0.10 < 0.4 x 0.70): 0.8 and 0.2
    assert sr.scan_score(p, bf, "both", 0.4) == pytest.approx(0.5)  # runs that pass: (0.2, 0.8), (0.8, 0.1)
    assert sr.scan_score(p, bf, "gate", 0.99) == pytest.approx(0.85)  # under k slices pass: falls back to "none"
    assert sr.scan_score(p, bf, "both", 0.99) == pytest.approx(0.55)  # no passing run: falls back to "adjacent"
    assert sr.scan_score([0.3]) == pytest.approx(0.3) and sr.scan_score([0.3], rule="adjacent") == pytest.approx(0.3)
    with pytest.raises(ValueError):
        sr.scan_score(p, rule="gate")
    with pytest.raises(ValueError):
        sr.scan_score(p, bf[:3], "gate", 0.4)
    with pytest.raises(ValueError):
        sr.scan_score(p, rule="median")


def test_scan_table_orders_slices_along_the_body():
    pd = pytest.importorskip("pandas")
    slices = pd.DataFrame({"series_id": ["a"] * 4 + ["b"] * 2, "any": [0, 1, 0, 0, 0, 0],
                           "prob": [0.9, 0.1, 0.8, 0.1, 0.2, 0.3], "z_mm": [0, 10, 5, 15, 0, 5],
                           "brain_fraction": [0.5] * 6})
    t = sr.scan_table(slices, "adjacent").set_index("series_id")
    assert t.loc["a", "label"] == 1 and t.loc["a", "score"] == pytest.approx(0.85)  # 0.9 at z 0 and 0.8 at z 5 touch
    assert t.loc["b", "score"] == pytest.approx(0.25)
