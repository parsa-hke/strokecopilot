"""Vision B ISLES'24 domain-shift check on synthetic volumes and a fake model. No real data, no GPU."""

import json
import math

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
nib = pytest.importorskip("nibabel")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("pydicom")  # data.rsna imports prep_rsna, which needs it

from strokecopilot.data import isles24, splits  # noqa: E402
from strokecopilot.data.prep_rsna import IMAGE_SIZE, JPEG_QUALITY, window_image  # noqa: E402
from strokecopilot.vision.haemorrhage import isles_check as ic  # noqa: E402
from strokecopilot.vision.haemorrhage.aggregate import Calibrator  # noqa: E402

CALIBRATOR = Calibrator(method="topk", k=2, temperature=1.0, bias=0.0, threshold=0.278, kind="platt")


def _head(shape=(64, 64, 20), zooms=(0.5, 0.5, 2.0), blob=None):
    """RAS volume: air, an ellipse of brain at 35 HU, and an optional 70 HU blob (i, j, k slices)."""
    vol = np.full(shape, -1000.0, dtype=np.float32)
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    brain = ((i - shape[0] / 2) / (shape[0] * 0.4)) ** 2 + ((j - shape[1] / 2) / (shape[1] * 0.45)) ** 2 < 1
    vol[brain, 2:-2] = 35.0
    if blob is not None:
        vol[blob] = 70.0
    return nib.Nifti1Image(vol, np.diag([*zooms, 1.0]))


def _reoriented(img, flip_axes):
    """The same head stored with some axes flipped (e.g. LAS or LPS), as ISLES'24 sources are."""
    data = np.asarray(img.dataobj)
    affine = img.affine.copy()
    for axis in flip_axes:
        data = np.flip(data, axis)
        affine[:3, 3] += affine[:3, axis] * (data.shape[axis] - 1)
        affine[:3, axis] *= -1
    return nib.Nifti1Image(np.ascontiguousarray(data), affine)


def _fake_predict(images):
    """Probability rises with the share of very bright brain-window pixels (the 70 HU blob)."""
    bright = (images[..., 0] > 200).mean(axis=(1, 2))
    return np.clip(0.02 + 40 * bright, 0, 0.99)


# --- orientation and slab averaging ---------------------------------------------------------------


def test_axial_stack_puts_front_at_top_and_patient_right_on_the_left():
    shape = (40, 50, 6)
    vol = np.zeros(shape, dtype=np.float32)
    vol[-1, -1, 3] = 100.0  # patient's far right (high i), front (high j)
    stack, (slice_mm, row_mm, col_mm) = ic.axial_stack(nib.Nifti1Image(vol, np.diag([0.4, 0.5, 2.5, 1.0])))
    assert stack.shape == (6, 50, 40) and (slice_mm, row_mm, col_mm) == pytest.approx((2.5, 0.5, 0.4))
    assert stack[3, 0, 0] == 100.0  # top row (front), left column (patient's right)


@pytest.mark.parametrize("flips", [(0,), (0, 1), (1,), (2,)])
def test_axial_stack_is_the_same_whatever_the_stored_orientation(flips):
    ras = _head(blob=(slice(10, 14), slice(40, 44), slice(8, 10)))
    expected, zooms = ic.axial_stack(ras)
    got, got_zooms = ic.axial_stack(_reoriented(ras, flips))
    assert got_zooms == zooms and np.array_equal(got, expected)


def test_slab_weights_average_thin_slices_by_overlap():
    w = ic.slab_weights(10, 1.0, 5.0)
    assert w.shape == (2, 10) and np.allclose(w.sum(axis=1), 1)
    assert np.allclose(w[0, :5], 0.2) and np.allclose(w[0, 5:], 0)
    w2 = ic.slab_weights(10, 2.0, 5.0)  # 20 mm of coverage -> four 5 mm slabs
    assert w2.shape == (4, 10)
    assert np.allclose(w2[0, :3], [0.4, 0.4, 0.2]) and np.allclose(w2[1, 2:5], [0.2, 0.4, 0.4])


def test_slab_weights_leave_thick_or_unset_slices_alone():
    assert np.array_equal(ic.slab_weights(7, 5.0, 5.0), np.eye(7))
    assert np.array_equal(ic.slab_weights(7, 6.0, 5.0), np.eye(7))
    assert np.array_equal(ic.slab_weights(7, 1.0, None), np.eye(7))
    assert ic.slab_weights(3, 1.0, 5.0).shape == (1, 3)  # shorter than one slab: one slab of it all


def test_to_slabs_keeps_constant_volumes_and_averages_in_hu():
    stack = np.full((10, 4, 4), 35.0, dtype=np.float32)
    assert np.allclose(ic.to_slabs(stack, 1.0, 5.0), 35.0)
    ramp = np.arange(10, dtype=np.float32)[:, None, None] * np.ones((1, 2, 2), np.float32)
    assert np.allclose(ic.to_slabs(ramp, 1.0, 5.0)[:, 0, 0], [2.0, 7.0])
    assert ic.to_slabs(ramp, 1.0, None) is not None and ic.to_slabs(ramp, 1.0, None).shape == ramp.shape


def test_square_and_field_of_view_pad_with_air_about_the_centre():
    stack = np.zeros((1, 4, 6), dtype=np.float32)
    sq = ic.square(stack)
    assert sq.shape == (1, 6, 6) and (sq[0, 0] == ic.AIR_HU).all() and (sq[0, -1] == ic.AIR_HU).all() and (sq[0, 1:5] == 0).all()
    fov = ic.set_field_of_view(np.zeros((1, 10, 10), np.float32), 0.5, 0.5, 3.0)
    assert fov.shape == (1, 6, 6)


# --- windowing matches the training shards --------------------------------------------------------


def test_window_slices_match_the_shard_pipeline_exactly():
    rng = np.random.default_rng(0)
    hu = rng.normal(30, 200, size=(1, 512, 512)).astype(np.float32)
    ours = ic.window_slices(hu)
    image = window_image(hu[0], 1, 0, IMAGE_SIZE)  # what prep_rsna stored, before JPEG
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    shard = cv2.imdecode(np.frombuffer(jpeg.tobytes(), "uint8"), cv2.IMREAD_COLOR)
    assert ok and ours.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 3) and np.array_equal(ours[0], shard)


def test_window_channels_are_brain_subdural_bone():
    images = ic.window_slices(np.full((1, 64, 64), 40.0, np.float32), size=32, jpeg_quality=None)
    brain, subdural, bone = (int(images[0, 16, 16, c]) for c in range(3))
    assert brain == 128 and subdural == round((40 - (80 - 100)) / 200 * 255) and bone == round((40 - (600 - 1400)) / 2800 * 255)


def test_prepare_gives_training_sized_slabs_and_numbers_only():
    images, info = ic.prepare(_head(shape=(64, 80, 20), zooms=(0.5, 0.5, 2.0)))
    assert images.dtype == np.uint8 and images.shape == (8, IMAGE_SIZE, IMAGE_SIZE, 3)  # 40 mm -> eight 5 mm slabs
    assert info["native_slices"] == 20 and info["slabs"] == 8 and info["slab_mm"] == 5.0 and info["slice_mm"] == 2.0
    assert info["orientation"] == "RAS" and info["fov_mm"] == 40.0 and not info["thicker_than_target"]
    assert info["hu_p01"] == -1000.0
    native, native_info = ic.prepare(_head(), target_mm=None)
    assert native.shape[0] == 20 and native_info["slab_mm"] == 2.0


# --- scoring and summaries ------------------------------------------------------------------------


def test_score_uses_the_calibrators_aggregation_and_threshold():
    probs = np.array([0.1, 0.9, 0.3, 0.7])
    result = ic.score(np.zeros((4, 8, 8, 3), np.uint8), lambda images: probs, CALIBRATOR)
    assert result["raw_score"] == pytest.approx(0.8) and result["top_slabs"] == [1, 3]
    assert result["confidence"] == pytest.approx(float(CALIBRATOR.apply(np.array([0.8]))[0]))
    assert result["present"] is True and result["max_slab_prob"] == 0.9
    low = ic.score(np.zeros((2, 8, 8, 3), np.uint8), lambda images: np.array([0.05, 0.1]), CALIBRATOR)
    assert low["present"] is False
    with pytest.raises(ValueError):
        ic.score(np.zeros((3, 8, 8, 3), np.uint8), lambda images: np.array([0.1]), CALIBRATOR)


def test_wilson_interval_known_values():
    low, high = ic.wilson(0, 10)
    assert low == 0.0 and high == pytest.approx(ic.Z95 ** 2 / (10 + ic.Z95 ** 2))
    low, high = ic.wilson(5, 100)
    assert low == pytest.approx(0.0215, abs=1e-4) and high == pytest.approx(0.1118, abs=1e-4)
    assert ic.wilson(0, 0) == (None, None)


def test_summarise_counts_rates_and_histogram():
    conf = np.array([0.01, 0.05, 0.2, 0.3, 0.6, 0.95])
    summary = ic.summarise(conf, 0.278)
    at = summary["at_threshold"]
    assert at["called_positive"] == 3 and at["presumed_fp_rate"] == 0.5 and at["ci95"][0] < 0.5 < at["ci95"][1]
    assert 0.278 in [row["threshold"] for row in summary["by_threshold"]]
    assert sum(summary["histogram"]["counts"]) == 6 and summary["confidence"]["max"] == 0.95


# --- the whole check ------------------------------------------------------------------------------

CASES = ["sub-stroke9001", "sub-stroke9002", "sub-stroke9003", "sub-stroke9004"]  # synthetic: in no split


def _loader(missing=()):
    blob = (slice(20, 40), slice(20, 40), slice(8, 12))
    images = {CASES[0]: _head(), CASES[1]: _head(blob=blob), CASES[2]: _reoriented(_head(), (0, 1))}

    def load(case):
        if case in missing:
            raise FileNotFoundError(f"no ncct_native for {case}")
        return images[case]

    return load


def test_run_check_ranks_scans_counts_failures_and_keeps_ids_out_of_metrics():
    table, metrics = ic.run_check(CASES, _loader(missing={CASES[3]}), _fake_predict, CALIBRATOR)
    assert list(table["case"]) == [CASES[1], CASES[0], CASES[2], CASES[3]]  # the blob first; ties keep input order
    assert table.loc[0, "rank"] == 1 and bool(table.loc[0, "slab_5mm_present"]) and not table.loc[1:2, "slab_5mm_present"].any()
    assert table.loc[3, "case"] == CASES[3] and table.loc[3, "error"].startswith("FileNotFoundError") and math.isnan(table.loc[3, "rank"])
    assert metrics["cases"] == 4 and metrics["scored"] == 3 and metrics["failed"] == 1 and metrics["failure_types"] == {"FileNotFoundError": 1}
    at = metrics["variants"]["slab_5mm"]["at_threshold"]
    assert at["called_positive"] == 1 and at["scans"] == 3 and at["threshold"] == 0.278
    assert set(metrics["variants"]) == {"slab_5mm", "native"} and "native" in metrics["agreement"]
    assert metrics["native_ncct"]["orientation"] == {"RAS": 2, "LPS": 1}
    text = json.dumps(metrics)
    assert ic.ids_in(text, CASES) == 0 and ic.ids_in(table.to_csv(), CASES) == 4


def test_run_check_stops_on_a_heldout_case(tmp_path):
    folds = {"heldout": [CASES[3]], "folds": {"0": CASES[:3]}}
    load = lambda case: isles24.load_native_ncct(case, tmp_path, folds=folds)  # noqa: E731
    with pytest.raises(splits.HeldoutError):
        ic.run_check([CASES[3]], load, _fake_predict, CALIBRATOR)


def test_figures_are_written_without_case_ids(tmp_path):
    table, metrics = ic.run_check(CASES[:3], _loader(), _fake_predict, CALIBRATOR)
    hist = ic.histogram_figure(metrics, str(tmp_path / "hist.png"))
    top = ic.top_scans_figure(table, _loader(), n=2, out_png=str(tmp_path / "top.png"))
    assert (tmp_path / "hist.png").stat().st_size > 0 and (tmp_path / "top.png").stat().st_size > 0 and hist and top


def test_prepared_slabs_run_through_the_real_model_path():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from strokecopilot.vision.haemorrhage import infer
    from strokecopilot.vision.haemorrhage import model as hmodel

    torch.manual_seed(0)
    net = hmodel.build("resnet50", pretrained=False).eval()
    images, _ = ic.prepare(_head(shape=(48, 48, 10)))
    result = ic.score(images, lambda x: infer.slice_probs(net, x, torch.device("cpu"), batch_size=4), CALIBRATOR)
    assert 0 <= result["confidence"] <= 1 and len(result["top_slabs"]) == 2


# --- field of view (PR #3 review, item 3) -----------------------------------------------------------


def test_variant_spec_reads_numbers_none_and_dicts():
    assert ic.variant_spec(5.0) == (5.0, None) and ic.variant_spec(None) == (None, None)
    assert ic.variant_spec({"target_mm": 5.0, "fov_mm": 250.0}) == (5.0, 250.0)
    with pytest.raises(ValueError):
        ic.variant_spec({"target_mm": 5.0, "fov": 250.0})


def test_matching_the_field_of_view_shrinks_a_small_field_head_in_the_image():
    img = _head(shape=(64, 64, 20), zooms=(2.0, 2.0, 2.0))  # 128 mm field of view
    plain, _ = ic.prepare(img, 5.0, jpeg_quality=None)
    matched, info = ic.prepare(img, 5.0, jpeg_quality=None, fov_mm=256.0)  # padded with air to twice the field
    brain = lambda images: (images[..., 0] > 100).mean()  # noqa: E731  share of pixels in the brain window's bright range
    assert info["fov_mm"] == 128.0 and plain.shape == matched.shape
    assert brain(matched) == pytest.approx(brain(plain) / 4, rel=0.15)  # half the width and height


def test_run_check_with_fov_variants_pairs_the_two_readings():
    table, metrics = ic.run_check(CASES[:3], _loader(), _fake_predict, CALIBRATOR, variants=ic.FOV_VARIANTS)
    assert set(metrics["variants"]) == {"slab_5mm", "slab_5mm_fov250"} and "slab_5mm_fov250" in metrics["agreement"]
    assert metrics["variant_settings"]["slab_5mm_fov250"] == {"target_mm": 5.0, "fov_mm": ic.RSNA_FOV_MM}
    assert metrics["variant_settings"]["slab_5mm"] == {"target_mm": 5.0, "fov_mm": None} and metrics["target_slice_mm"] == 5.0
    assert {"slab_5mm_confidence", "slab_5mm_fov250_confidence"} <= set(table.columns)
    same, _ = ic.run_check(CASES[:3], _loader(), _fake_predict, CALIBRATOR)
    assert np.allclose(table.set_index("case")["slab_5mm_confidence"], same.set_index("case")["slab_5mm_confidence"])  # EXP-16 reading unchanged
    assert ic.ids_in(json.dumps(metrics), CASES) == 0


def test_top_scans_figure_uses_the_variants_it_is_given(tmp_path):
    table, _ = ic.run_check(CASES[:3], _loader(), _fake_predict, CALIBRATOR, variants=ic.FOV_VARIANTS, primary="slab_5mm_fov250")
    ic.top_scans_figure(table, _loader(), n=2, primary="slab_5mm_fov250", out_png=str(tmp_path / "top.png"), variants=ic.FOV_VARIANTS)
    assert (tmp_path / "top.png").stat().st_size > 0
