"""RSNA preprocessing on synthetic DICOMs (plan §5.2, DEC-17). Needs the [rsna] extra."""

import json
import os
import tarfile

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
cv2 = pytest.importorskip("cv2")
pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid  # noqa: E402

from strokecopilot.data import prep_rsna  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:Invalid value for VR UI")

N_SLICES = 24
SIZE = 64


def _write_dicom(path, i):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    pydicom.config.settings.writing_validation_mode = pydicom.config.IGNORE
    ds.PatientID = f"ID_patient{i % 3}"
    ds.StudyInstanceUID = f"ID_study{i % 3}"  # RSNA-style, not a valid UID
    ds.SeriesInstanceUID = f"ID_series{i % 3}"
    ds.ImagePositionPatient = [0, 0, float(i) * 5]
    ds.PixelSpacing = [0.5, 0.5]
    ds.Rows = ds.Columns = 128
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    ds.PixelData = np.random.default_rng(i).integers(0, 2000, (128, 128), dtype=np.int16).tobytes()
    ds.save_as(str(path))


@pytest.fixture
def dataset(tmp_path):
    image_dir = tmp_path / "stage_2_train"
    image_dir.mkdir()
    rows = []
    for i in range(N_SLICES):
        image_id = f"ID_{i:05d}"
        if i == 3:
            (image_dir / f"{image_id}.dcm").write_bytes(b"corrupt")
        elif i != 5:  # ID_00005 stays missing
            _write_dicom(image_dir / f"{image_id}.dcm", i)
        for name in prep_rsna.LABEL_NAMES:
            rows.append((f"{image_id}_{name}", int(name == "any" and i % 4 == 0)))
    rows.append(("ID_00000_any", 1))  # the real CSV has duplicates
    csv_path = tmp_path / "stage_2_train.csv"
    pd.DataFrame(rows, columns=["ID", "Label"]).to_csv(csv_path, index=False)
    return prep_rsna.load_labels(csv_path), str(image_dir), tmp_path / "out"


def test_find_data_handles_nested_mounts(dataset, tmp_path):
    csv_path, image_dir = prep_rsna.find_data(tmp_path)
    assert csv_path.endswith("stage_2_train.csv") and image_dir.endswith("stage_2_train")
    with pytest.raises(FileNotFoundError, match="Add Input"):
        prep_rsna.find_data(tmp_path / "out_absent")


def test_load_labels_gives_one_row_per_slice(dataset):
    labels, _, _ = dataset
    assert len(labels) == N_SLICES
    assert list(labels.columns) == ["image_id", *prep_rsna.LABEL_NAMES]
    assert labels["any"].sum() == 6


def test_load_labels_rejects_missing_label_column(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("ID,Label\nID_1_any,1\n")
    with pytest.raises(ValueError, match="missing"):
        prep_rsna.load_labels(path)


def test_shards_are_stable_and_cover_everything(dataset):
    labels, _, _ = dataset
    assignment = [prep_rsna.shard_of(i) for i in labels["image_id"]]
    assert assignment == [prep_rsna.shard_of(i) for i in labels["image_id"]]
    assert set(assignment) <= set(range(prep_rsna.N_SHARDS))
    assert prep_rsna.shard_of("ID_00000") == 2  # pinned: changing the hash would reshuffle a built dataset


def test_window_image_maps_hounsfield_units_to_each_channel():
    # pixel 1064 with intercept -1024 is 40 HU, the centre of the brain window
    image = prep_rsna.window_image(np.full((128, 128), 1064, dtype=np.int16), 1, -1024, size=SIZE)
    assert image.shape == (SIZE, SIZE, 3) and image.dtype == np.uint8
    assert image[0, 0, 0] == 128  # brain (40/80): the middle
    assert image[0, 0, 1] in (76, 77)  # subdural (80/200): (40 + 20) / 200 = 0.3
    assert image[0, 0, 2] in (76, 77)  # bone (600/2800): (40 + 800) / 2800 = 0.3


@pytest.mark.parametrize("workers", [0, 2])
def test_all_shards_cover_all_slices_and_record_failures(dataset, workers):
    labels, image_dir, out = dataset
    ok_total, failed = 0, []
    for shard in range(prep_rsna.N_SHARDS):
        if not any(prep_rsna.shard_of(i) == shard for i in labels["image_id"]):
            continue
        report = prep_rsna.process_shard(labels, image_dir, out, shard, size=SIZE, workers=workers)
        tar_path, index_path = prep_rsna.shard_paths(out, shard, SIZE)
        assert not os.path.exists(f"{tar_path}.part")
        assert prep_rsna.verify_shard(tar_path, index_path, size=SIZE)["images"] == report["slices_ok"]
        index = pd.read_csv(index_path)
        ok_total += report["slices_ok"]
        failed += list(index.loc[~index["ok"], "image_id"])
        json.loads(open(prep_rsna.save_report(report, out, shard)).read())
    assert ok_total == N_SLICES - 2
    assert sorted(failed) == ["ID_00003", "ID_00005"]


def test_index_holds_ids_positions_and_labels(dataset):
    labels, image_dir, out = dataset
    shard = prep_rsna.shard_of("ID_00000")
    prep_rsna.process_shard(labels, image_dir, out, shard, size=SIZE, workers=0)
    index = pd.read_csv(prep_rsna.shard_paths(out, shard, SIZE)[1]).set_index("image_id")
    row = index.loc["ID_00000"]
    assert (row["patient_id"], row["study_id"], row["series_id"]) == ("ID_patient0", "ID_study0", "ID_series0")
    assert row["z_mm"] == 0.0 and row["pixel_spacing_mm"] == 0.5
    assert row["any"] == 1 and row["subdural"] == 0


def test_reruns_give_identical_archives(dataset):
    labels, image_dir, out = dataset
    shard = prep_rsna.shard_of("ID_00000")
    prep_rsna.process_shard(labels, image_dir, out / "a", shard, size=SIZE, workers=0)
    prep_rsna.process_shard(labels, image_dir, out / "b", shard, size=SIZE, workers=0)
    a, b = (open(prep_rsna.shard_paths(out / d, shard, SIZE)[0], "rb").read() for d in "ab")
    assert a == b


def test_limit_projects_the_full_size(dataset):
    labels, image_dir, out = dataset
    shard = prep_rsna.shard_of("ID_00000")
    report = prep_rsna.process_shard(labels, image_dir, out, shard, size=SIZE, workers=0, limit=2)
    assert report["slices_processed"] == 2 and report["complete_shard"] is False
    assert report["projected_total_gb"] >= 0


def test_verify_rejects_tar_that_differs_from_index(dataset):
    labels, image_dir, out = dataset
    shard = prep_rsna.shard_of("ID_00000")
    prep_rsna.process_shard(labels, image_dir, out, shard, size=SIZE, workers=0)
    tar_path, index_path = prep_rsna.shard_paths(out, shard, SIZE)
    with tarfile.open(tar_path, "w"):
        pass  # now empty
    with pytest.raises(ValueError, match="differ"):
        prep_rsna.verify_shard(tar_path, index_path, size=SIZE)


def test_interrupted_run_leaves_no_finished_tar(dataset, monkeypatch):
    labels, image_dir, out = dataset
    shard = prep_rsna.shard_of("ID_00000")

    def boom(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(prep_rsna, "_write_jpeg", boom)
    with pytest.raises(KeyboardInterrupt):
        prep_rsna.process_shard(labels, image_dir, out, shard, size=SIZE, workers=0)
    assert not os.path.exists(prep_rsna.shard_paths(out, shard, SIZE)[0])
