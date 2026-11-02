"""RSNA shard reader on small synthetic shards. Needs the [rsna] extra."""

import io
import os
import tarfile

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("pydicom")

from strokecopilot.data import prep_rsna, rsna  # noqa: E402

SIZE = 32


def _write_shard(folder, shard, rows):
    os.makedirs(folder)
    tar_path, index_path = prep_rsna.shard_paths(str(folder), shard, SIZE)
    with tarfile.open(tar_path, "w") as tar:
        for row in rows:
            if row["ok"]:
                image = np.full((SIZE, SIZE, 3), row["value"], dtype=np.uint8)
                data = cv2.imencode(".jpg", image)[1].tobytes()
                info = tarfile.TarInfo(f"{row['image_id']}.jpg")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    frame = pd.DataFrame(rows).drop(columns="value").assign(shard=shard)
    frame.to_csv(index_path, index=False)


def _row(i, series, z, any_label=0, ok=True):
    labels = {name: 0 for name in prep_rsna.LABEL_NAMES}
    labels["any"] = any_label
    return {"image_id": f"ID_{i}", "ok": ok, "patient_id": f"P_{series}", "series_id": f"S_{series}", "z_mm": z, "value": 40 + i, **labels}


@pytest.fixture
def shards(tmp_path):
    _write_shard(tmp_path / "s0", 0, [_row(0, "a", 10.0), _row(1, "a", 5.0, any_label=1), _row(2, "b", 0.0, ok=False)])
    _write_shard(tmp_path / "s1", 1, [_row(3, "b", 5.0), _row(4, "b", 0.0)])
    return rsna.find_shards(str(tmp_path), n_shards=2)


def test_find_shards_pairs_tars_and_indexes(shards):
    assert sorted(shards) == [0, 1]
    assert shards[0][0].endswith(".tar") and shards[0][1].endswith("_index.csv.gz")


def test_find_shards_names_what_is_missing(tmp_path):
    _write_shard(tmp_path / "s0", 0, [_row(0, "a", 0.0)])
    with pytest.raises(FileNotFoundError, match=r"\[1\]"):
        rsna.find_shards(str(tmp_path), n_shards=2)


def test_load_index_drops_failed_slices(shards):
    index = rsna.load_index(shards)
    assert sorted(index["image_id"]) == ["ID_0", "ID_1", "ID_3", "ID_4"]


def test_scan_labels_are_positive_if_any_slice_is(shards):
    scans = rsna.scan_table(rsna.load_index(shards)).set_index("series_id")
    assert scans.loc["S_a", "any"] == 1 and scans.loc["S_b", "any"] == 0
    assert scans.loc["S_a", "slices"] == 2 and scans.loc["S_a", "patient_id"] == "P_a"


def test_slices_of_a_scan_are_in_body_order(shards):
    assert list(rsna.slices_of_scan(rsna.load_index(shards), "S_a")["image_id"]) == ["ID_1", "ID_0"]


def test_reader_reads_any_slice_in_any_order(shards):
    reader = rsna.ShardReader(shards[0][0])
    assert len(reader) == 2 and "ID_0" in reader and "ID_2" not in reader
    assert abs(int(reader.read("ID_1")[0, 0, 0]) - 41) <= 2
    assert reader.read("ID_0").shape == (SIZE, SIZE, 3)
    reader.close()


def test_decode_refuses_bad_bytes():
    with pytest.raises(ValueError, match="decode"):
        rsna.decode(b"not a jpeg")


def test_window_hu_matches_training_windows():
    hu = np.full((64, 64), 40.0, dtype=np.float32)
    image = rsna.window_hu(hu, size=32)
    assert image.shape == (32, 32, 3)
    assert image[0, 0, 0] == 128  # brain window: 40 HU is its centre


def test_measure_read_returns_numbers_only(shards):
    reader = rsna.ShardReader(shards[1][0])
    for threads in (1, 2):
        result = rsna.measure_read(reader, n=10, threads=threads)
        assert result["slices"] == 2 and result["threads"] == threads and result["slices_per_second"] > 0
    reader.close()


def test_copy_shard(shards, tmp_path):
    dest, seconds = rsna.copy_shard(shards[0][0], str(tmp_path / "local"))
    assert os.path.getsize(dest) == os.path.getsize(shards[0][0]) and seconds >= 0


@pytest.fixture
def unpacked(tmp_path):
    """Two shards as Kaggle leaves them: a folder of JPEGs and a plain CSV."""
    _write_shard(tmp_path / "a" / "s0", 0, [_row(0, "a", 10.0), _row(1, "a", 5.0), _row(2, "b", 0.0, ok=False)])
    _write_shard(tmp_path / "a" / "s1", 1, [_row(3, "b", 5.0)])
    for shard in (0, 1):
        folder = tmp_path / "a" / f"s{shard}"
        tar_path, index_path = prep_rsna.shard_paths(str(folder), shard, SIZE)
        with tarfile.open(tar_path) as tar:
            tar.extractall(tar_path[: -len(".tar")], filter="data")
        os.remove(tar_path)
        pd.read_csv(index_path).to_csv(index_path[: -len(".gz")], index=False)
        os.remove(index_path)
    return tmp_path


def test_unpacked_shards_are_found_and_read_like_tars(unpacked):
    shards = rsna.find_shards(str(unpacked), n_shards=2)
    assert os.path.isdir(shards[0][0]) and shards[0][1].endswith("_index.csv")
    assert sorted(rsna.load_index(shards)["image_id"]) == ["ID_0", "ID_1", "ID_3"]
    reader = rsna.open_shard(shards[0][0])
    assert isinstance(reader, rsna.FolderReader) and reader.ids() == {"ID_0", "ID_1"}
    assert abs(int(reader.read("ID_1")[0, 0, 0]) - 41) <= 2
    assert rsna.measure_read(reader, n=5)["slices"] == 2


def test_a_folder_is_packed_into_a_tar_that_reads_the_same(unpacked, tmp_path):
    shards = rsna.find_shards(str(unpacked), n_shards=2)
    dest, _ = rsna.copy_shard(shards[0][0], str(tmp_path / "local"))
    assert dest.endswith("_s00.tar")
    reader = rsna.open_shard(dest)
    assert isinstance(reader, rsna.ShardReader) and reader.ids() == {"ID_0", "ID_1"}
    assert abs(int(reader.read("ID_0")[0, 0, 0]) - 40) <= 2
    reader.close()


def test_a_folder_without_an_index_is_missing(unpacked):
    for name in os.listdir(unpacked / "a" / "s1"):
        if name.endswith(".csv"):
            os.remove(unpacked / "a" / "s1" / name)
    with pytest.raises(FileNotFoundError, match=r"\[1\]"):
        rsna.find_shards(str(unpacked), n_shards=2)
