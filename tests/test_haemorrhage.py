"""Vision B: aggregation, calibration, inference and a tiny CPU training run on synthetic shards."""

import io
import json
import os
import tarfile

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
cv2 = pytest.importorskip("cv2")

from strokecopilot.contract.validate import validate_vision_output  # noqa: E402
from strokecopilot.vision.haemorrhage import aggregate as agg  # noqa: E402


def test_roc_auc_matches_known_values():
    assert agg.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert agg.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert agg.roc_auc([1, 1], [0.2, 0.3]) is None


def _slices(n_scans=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scans):
        bleed = rng.random() < 0.4
        for z in range(10):
            positive = bleed and z in (4, 5, 6)
            prob = float(np.clip(rng.normal(0.75 if positive else 0.3, 0.15), 0.01, 0.99))
            rows.append({"series_id": f"ID_s{s}", "any": int(positive), "prob": prob})
    return pd.DataFrame(rows)


def test_scan_scores_and_aggregation_choice():
    slices = pd.DataFrame({"series_id": ["a"] * 4 + ["b"] * 2, "any": [0, 1, 0, 0, 0, 0], "prob": [0.1, 0.9, 0.5, 0.2, 0.3, 0.4]})
    by_max = agg.scan_scores(slices).set_index("series_id")
    assert by_max.loc["a", "score"] == 0.9 and by_max.loc["a", "label"] == 1 and by_max.loc["b", "label"] == 0
    assert agg.scan_scores(slices, "topk", 2).set_index("series_id").loc["a", "score"] == pytest.approx(0.7)
    method, k, table = agg.choose_aggregation(_slices())
    assert len(table) == 1 + len(agg.TOP_K) and all(row["scan_auc"] > 0.9 for row in table)


def test_fit_logistic_recovers_temperature_and_bias():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 20000)
    labels = rng.random(len(z)) < 1 / (1 + np.exp(-(z / 2.0 - 1.0)))
    scores = 1 / (1 + np.exp(-z))
    temperature, bias = agg.fit_logistic(scores, labels, bias=True)
    assert temperature == pytest.approx(2.0, abs=0.15) and bias == pytest.approx(-1.0, abs=0.1)
    temperature_only, zero = agg.fit_logistic(scores, labels, bias=False)
    assert zero == 0.0 and temperature_only > 0


def test_calibrate_fixes_prior_shift_and_sets_threshold(tmp_path):
    val = _slices(800)
    method, k, _ = agg.choose_aggregation(val)
    calibrator, report = agg.calibrate(val, method, k)
    assert report["kinds"]["platt"]["ece"] < report["kinds"]["identity"]["ece"]
    assert report["operating_point"]["sensitivity"] >= agg.TARGET_SENSITIVITY
    metrics = agg.evaluate(_slices(400, seed=5), calibrator)
    assert metrics["scan_auc"] > 0.9 and metrics["slice_auc"] > 0.9 and 0 <= metrics["scan_ece"] < 0.1
    path = tmp_path / "calibration.json"
    calibrator.to_json(path)
    assert agg.Calibrator.from_json(path) == calibrator
    assert agg.reliability_plot(report["reliability"], str(tmp_path / "r.png")) and os.path.getsize(tmp_path / "r.png") > 0
    assert "ID_" not in json.dumps(report) + json.dumps(metrics)


def test_threshold_for_sensitivity():
    point = agg.threshold_for_sensitivity([0.9, 0.8, 0.7, 0.6, 0.2, 0.1], [1, 1, 1, 1, 0, 0], target=0.75)
    assert point["threshold"] == 0.7 and point["sensitivity"] == 0.75 and point["specificity"] == 1.0


def test_haemorrhage_output_fits_the_contract():
    from strokecopilot.vision.haemorrhage.infer import haemorrhage_output

    calibrator = agg.Calibrator(method="topk", k=2, temperature=1.0, bias=0.0, threshold=0.3, kind="platt")
    out = haemorrhage_output([0.1, 0.4, 0.2], calibrator)
    assert out == {"present": True, "confidence": 0.3}
    fixture = json.load(open(os.path.join(os.path.dirname(__file__), "..", "fixtures", "fx01_thrombolysis_clear.json")))
    fixture["haemorrhage"] = out
    validate_vision_output(fixture)


# Training on CPU with a randomly initialised model and tiny images. Needs torch and timm.

SIZE = 32


@pytest.fixture
def tiny_shards(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from strokecopilot.data import rsna

    rows = []
    tar_path = tmp_path / "s0.tar"
    with tarfile.open(tar_path, "w") as tar:
        for i in range(48):
            label = int(i % 4 == 0)
            image = np.full((SIZE, SIZE, 3), 200 if label else 40, dtype=np.uint8)
            data = cv2.imencode(".jpg", image)[1].tobytes()
            info = tarfile.TarInfo(f"ID_{i}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            rows.append({"image_id": f"ID_{i}", "shard": 0, "any": label, "series_id": f"ID_s{i // 6}"})
    return {0: rsna.ShardReader(str(tar_path))}, pd.DataFrame(rows)


def test_balanced_epoch_uses_every_negative_in_turn():
    from strokecopilot.vision.haemorrhage.train import balanced_epoch

    labels = np.array([1] * 5 + [0] * 20)
    epochs = [balanced_epoch(labels, e, seed=3) for e in range(4)]
    assert all(len(e) == 10 and labels[e].sum() == 5 for e in epochs)
    negatives = np.concatenate([e[labels[e] == 0] for e in epochs])
    assert sorted(negatives) == list(range(5, 25))
    assert list(balanced_epoch(labels, 1, seed=3)) == list(epochs[1])


def test_augment_keeps_shape_and_type():
    from strokecopilot.vision.haemorrhage.train import augment

    image = np.random.default_rng(0).integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    out = augment(image, np.random.default_rng(1))
    assert out.shape == image.shape and out.dtype == np.uint8


def test_fit_resumes_and_predicts(tiny_shards, tmp_path):
    import torch

    from strokecopilot.vision.haemorrhage import train
    from strokecopilot.vision.haemorrhage.infer import haemorrhage_output, slice_probs

    readers, items = tiny_shards
    config = train.TrainConfig(model="resnet50", pretrained=False, epochs=1, batch_size=4, warmup_steps=2, workers=0, log_every=0)
    device = torch.device("cpu")
    history = train.fit(config, items.iloc[:36], items.iloc[36:], readers, str(tmp_path / "run"), device, log=lambda _: None)
    assert len(history) == 1 and history[0]["train_slices"] == 16
    config.epochs = 2
    lines = []
    history = train.fit(config, items.iloc[:36], items.iloc[36:], readers, str(tmp_path / "run"), device, log=lines.append)
    assert [h["epoch"] for h in history] == [0, 1] and lines[0].startswith("Resuming after epoch 0")
    config.epochs = 4
    lines = []
    history = train.fit(config, items.iloc[:36], items.iloc[36:], readers, str(tmp_path / "run"), device, log=lines.append, time_budget_s=1e-9)
    assert len(history) == 3 and lines[-1].startswith("Stopping before epoch 3")
    net, state = train.load_best(str(tmp_path / "run" / "best.pt"), device)
    probs = train.predict(net, train.SliceDataset(items.iloc[36:], readers), device, batch_size=8, workers=0)
    assert probs.shape == (12,) and ((probs >= 0) & (probs <= 1)).all()
    images = np.stack([readers[0].read(f"ID_{i}") for i in range(6)])
    assert slice_probs(net, images, device).shape == (6,)
    assert set(haemorrhage_output(probs, agg.Calibrator())) == {"present", "confidence"}


def test_speed_probes_run_on_cpu(tiny_shards):
    import torch

    from strokecopilot.vision.haemorrhage import train

    readers, items = tiny_shards
    speed = train.loader_speed(train.SliceDataset(items, readers, train=True), batch_size=4, workers=0, batches=3)
    assert speed["slices"] == 12 and speed["slices_per_second"] > 0
    gpu = train.gpu_speed("resnet50", torch.device("cpu"), batch_size=2, size=SIZE, steps=2, warmup=1)
    assert gpu["params_m"] > 20 and gpu["peak_gib"] is None
