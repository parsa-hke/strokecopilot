"""nnU-Net dataset conversion, splits, trainer file and scoring on a small synthetic slim folder. No nnU-Net needed."""

import json
import os

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from strokecopilot.data import clinical_columns as cc  # noqa: E402
from strokecopilot.data import isles24_layout as layout  # noqa: E402
from strokecopilot.data import prep_isles24 as prep  # noqa: E402
from strokecopilot.data import splits  # noqa: E402
from strokecopilot.vision.perfusion import nnunet_wrapper as nw  # noqa: E402

SHAPE = (40, 44, 16)
AFFINE = np.diag([0.75, 0.75, 1.5, 1.0])
DEV = ["sub-stroke0001", "sub-stroke0002", "sub-stroke0004", "sub-stroke0005", "sub-stroke0006"]
HELD = ["sub-stroke0003"]
FOLDS = {"heldout": HELD, "folds": {str(k): [case] for k, case in enumerate(DEV)}}


def _save(root, case, name, array):
    path = layout.path_of(str(root), case, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(array, AFFINE), path)


@pytest.fixture(scope="module")
def slim(tmp_path_factory):
    base = tmp_path_factory.mktemp("nnunet")
    source = base / "train"
    for case in DEV + HELD:
        ncct = np.full(SHAPE, -1000, dtype=np.float32)
        ncct[4:36, 4:40, 2:14] = 35
        _save(source, case, "ncct", ncct)
        for name in ("cbf", "cbv", "mtt"):
            _save(source, case, name, np.where(ncct > 0, 20.0, 0.0).astype(np.float32))
        tmax = np.where(ncct > 0, 2.0, -30.0).astype(np.float32)
        tmax[20:30, 10:20, 4:8] = -8.0
        _save(source, case, "tmax", tmax)
        lesion = np.zeros(SHAPE)
        lesion[20:30, 10:30, 4:10] = 1
        _save(source, case, "lesion", lesion)
        _save(source, case, "lvo", np.zeros(SHAPE, dtype=np.uint8))
        _save(source, case, "cow", np.zeros(SHAPE, dtype=np.uint8))
    prep.build_parts(str(source), DEV + HELD, str(base / "parts"))
    row = {column: 1 for column in (*cc.BASELINE_COLUMNS, *cc.OUTCOME_COLUMNS)}
    prep.assemble(str(base / "parts"), DEV + HELD, pd.DataFrame([{"case": c, **row} for c in DEV + HELD]), str(base / "out"))
    return base / "out" / prep.SLIM_NAME


def test_channel_arrays_handle_negative_tmax():
    tmax = np.array([[[-30.0, -5.0, 0.0, 7.5]]], dtype=np.float32)
    mtt = np.array([[[np.nan, -2.0, 1.0, np.inf]]], dtype=np.float32)
    arrays = {"ncct": np.zeros_like(tmax), "cbf": np.ones_like(tmax), "cbv": np.ones_like(tmax), "mtt": mtt, "tmax": tmax}
    channels = nw.channel_arrays(arrays)
    assert len(channels) == len(nw.CHANNELS) == len(nw.CHANNEL_NAMES)
    assert channels[4].ravel().tolist() == [0.0, 0.0, 0.0, 7.5]
    assert channels[5].ravel().tolist() == [0.0, 0.0, 1.0, 1.0]
    assert channels[3].ravel().tolist() == [0.0, 0.0, 1.0, 0.0]


def test_convert_writes_development_cases_only(slim, tmp_path):
    dataset = nw.convert(str(slim), DEV, str(tmp_path / "raw"), folds=FOLDS, log_every=0)
    images = sorted(os.listdir(os.path.join(dataset, "imagesTr")))
    assert len(images) == len(DEV) * len(nw.CHANNELS)
    assert not any(h in "".join(images) for h in HELD)
    meta = json.load(open(os.path.join(dataset, "dataset.json")))
    assert meta["numTraining"] == len(DEV) and meta["channel_names"]["5"] == "noNorm"
    image = nib.load(os.path.join(dataset, "imagesTr", f"{DEV[0]}_0005.nii.gz"))
    label = nib.load(os.path.join(dataset, "labelsTr", f"{DEV[0]}.nii.gz"))
    assert image.shape == label.shape and np.allclose(image.affine, label.affine)
    assert set(np.unique(np.asanyarray(image.dataobj))) == {0.0, 1.0}
    assert nw.convert(str(slim), DEV, str(tmp_path / "raw"), folds=FOLDS, log_every=0) == dataset  # resumes


def test_convert_refuses_held_out(slim, tmp_path):
    with pytest.raises(splits.HeldoutError):
        nw.convert(str(slim), DEV + HELD, str(tmp_path / "raw"), folds=FOLDS, log_every=0)


def test_splits_final_matches_our_folds(tmp_path):
    out = nw.splits_final(FOLDS)
    assert len(out) == 5 and out[2]["val"] == [DEV[2]]
    assert sorted(out[2]["train"]) == sorted(set(DEV) - {DEV[2]})
    path = nw.write_splits(str(tmp_path), FOLDS)
    assert json.load(open(path)) == out and path.endswith(f"{nw.DATASET_NAME}/splits_final.json")


def test_trainer_file_and_environment(tmp_path):
    os.makedirs(tmp_path / "training" / "nnUNetTrainer" / "variants")
    path = nw.install_trainer(str(tmp_path))
    source = open(path).read()
    compile(source, path, "exec")
    assert "class nnUNetTrainer_StrokeCopilot(nnUNetTrainer)" in source
    assert "def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict" in source
    env = nw.environment("/r", "/p", "/x", epochs=50, iterations=250, save_every=5, n_proc_da=8)
    assert env["SC_NUM_EPOCHS"] == "50" and env["nnUNet_results"] == "/x" and env["nnUNet_n_proc_DA"] == "8"
    assert nw.train_command(0, resume=True)[-1] == "--c" and nw.train_command(0, resume=False)[-1] == nw.TRAINER


def test_epoch_times_and_checkpoints(tmp_path):
    (tmp_path / "training_log_2026_11_4_10_00_00.txt").write_text(
        "2026-11-04 10:01:00.1: Epoch 0\n2026-11-04 10:02:00.1: Epoch time: 61.25 s\n2026-11-04 10:03:00.1: Epoch time: 59.5 s\n")
    assert nw.epoch_times(str(tmp_path)) == [61.25, 59.5]
    (tmp_path / "checkpoint_latest.pth").write_text("x")
    assert nw.checkpoint_state(str(tmp_path)) == {"checkpoint_final.pth": False, "checkpoint_latest.pth": True, "checkpoint_best.pth": False}


def test_score_predictions(slim, tmp_path):
    dataset = nw.convert(str(slim), DEV, str(tmp_path / "raw"), folds=FOLDS, log_every=0)
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    for case in DEV[:2]:
        label = nib.load(os.path.join(dataset, "labelsTr", f"{case}.nii.gz"))
        nib.save(label, str(pred_dir / f"{case}.nii.gz"))  # a perfect prediction
    rows = nw.score_predictions(str(pred_dir), str(slim), DEV[:2], fold=0, folds=FOLDS)
    assert [r["dice"] for r in rows] == [1.0, 1.0]
    assert rows[0]["dice_reliable"] == 1.0 and 0 < rows[0]["infarct_reliable_share"] < 1
