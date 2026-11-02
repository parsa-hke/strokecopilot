"""RSNA ICH shard reader and CT windowing, for Kaggle notebooks (plan §5.2, DEC-17).

The index holds slice, patient and study IDs, so nothing here prints them, and none of this data may
leave Kaggle (docs/DATA_USE.md). Image channels are brain, subdural, bone, as `prep_rsna` stored them.

Kaggle unpacks archives when a dataset is made, so an attached shard can be a folder of JPEGs and a
plain CSV instead of a tar and a .csv.gz. Both layouts are read the same way.
"""

import os
import random
import re
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from strokecopilot.data.prep_rsna import IMAGE_SIZE, LABEL_NAMES, N_SHARDS, SLIM_VERSION, window_image

DATA_RE = re.compile(rf"^{SLIM_VERSION}_\d+px_s(\d+)(\.tar)?$")
INDEX_RE = re.compile(rf"^{SLIM_VERSION}_\d+px_s(\d+)_index\.csv(\.gz)?$")


def find_shards(root, n_shards=N_SHARDS):
    """{shard: (tar or folder path, index path)} under root, such as /kaggle/input. Raises if any is missing."""
    data, indexes = {}, {}
    for folder, subfolders, files in os.walk(root):
        for name in subfolders + files:
            path = os.path.join(folder, name)
            if (match := DATA_RE.match(name)) and (name in files) == name.endswith(".tar"):
                data.setdefault(int(match.group(1)), path)
            elif (match := INDEX_RE.match(name)) and name in files:
                indexes[int(match.group(1))] = path
        subfolders[:] = [d for d in subfolders if not DATA_RE.match(d)]  # 188,000 JPEGs need no listing here
    complete = {shard: (data[shard], indexes[shard]) for shard in data if shard in indexes}
    missing = sorted(set(range(n_shards)) - set(complete))
    if missing:
        raise FileNotFoundError(f"Shards {missing} lack images or an index under {root}. Attach every rsna-slim-v1 dataset with Add Input.")
    return dict(sorted(complete.items()))


def load_index(shards):
    """One row per readable slice across shards. Failed slices (`ok` false) are dropped."""
    frames = [pd.read_csv(index_path, low_memory=False) for _, index_path in shards.values()]
    index = pd.concat(frames, ignore_index=True)
    return index[index["ok"].astype(str) == "True"].reset_index(drop=True)


def scan_table(index):
    """One row per scan (DICOM series): patient, slice count and a label per type, positive if any slice is."""
    labels = index.groupby("series_id")[list(LABEL_NAMES)].max()
    info = index.groupby("series_id").agg(patient_id=("patient_id", "first"), slices=("image_id", "size"))
    return info.join(labels).reset_index()


def slices_of_scan(index, series_id):
    """A scan's rows, ordered by position along the body. Spacing comes from z_mm (plan §5.2)."""
    return index[index["series_id"] == series_id].sort_values("z_mm")


class ShardReader:
    """Random access to one uncompressed tar. Reads are thread-safe, because each states its offset."""

    def __init__(self, tar_path):
        self.path = tar_path
        with tarfile.open(tar_path) as tar:
            self.offsets = {
                os.path.basename(m.name)[: -len(".jpg")]: (m.offset_data, m.size)
                for m in tar
                if m.isfile() and m.name.endswith(".jpg") and not os.path.basename(m.name).startswith("._")  # macOS metadata
            }
        self._fd = os.open(tar_path, os.O_RDONLY)

    def __contains__(self, image_id):
        return image_id in self.offsets

    def __len__(self):
        return len(self.offsets)

    def ids(self):
        return set(self.offsets)

    def read_bytes(self, image_id):
        offset, size = self.offsets[image_id]
        return os.pread(self._fd, size, offset)

    def read(self, image_id):
        return decode(self.read_bytes(image_id))

    def close(self):
        os.close(self._fd)


class FolderReader:
    """The same reads from a shard Kaggle unpacked into a folder of JPEGs."""

    def __init__(self, folder):
        self.path = folder
        self.paths = {}
        for parent, _, files in os.walk(folder):
            for name in files:
                if name.endswith(".jpg"):
                    self.paths[name[: -len(".jpg")]] = os.path.join(parent, name)

    def __contains__(self, image_id):
        return image_id in self.paths

    def __len__(self):
        return len(self.paths)

    def ids(self):
        return set(self.paths)

    def read_bytes(self, image_id):
        with open(self.paths[image_id], "rb") as f:
            return f.read()

    def read(self, image_id):
        return decode(self.read_bytes(image_id))

    def close(self):
        pass


def open_shard(path):
    return FolderReader(path) if os.path.isdir(path) else ShardReader(path)


def decode(jpeg):
    """JPEG bytes -> height x width x 3 uint8 array: brain, subdural, bone."""
    image = cv2.imdecode(np.frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode a slice.")
    return image


def window_hu(hu, size=IMAGE_SIZE):
    """HU slice -> the same three windows as training, for running on ISLES'24 NCCT (plan §8.2)."""
    return window_image(hu, 1, 0, size)


def copy_shard(path, dest_dir):
    """Put a shard on local disk as one tar, and return the new path and seconds taken. A folder is packed."""
    os.makedirs(dest_dir, exist_ok=True)
    started = time.perf_counter()
    dest = os.path.join(dest_dir, f"{os.path.basename(path).removesuffix('.tar')}.tar")
    if os.path.isdir(path):
        subprocess.run(["tar", "-cf", dest, "-C", path, "."], check=True)
    else:
        shutil.copyfile(path, dest)
    return dest, time.perf_counter() - started


def measure_read(reader, n=2000, threads=1, seed=0):
    """Random reads and decodes per second. Returns numbers only."""
    ids = random.Random(seed).sample(sorted(reader.ids()), min(n, len(reader)))
    started = time.perf_counter()
    if threads == 1:
        for image_id in ids:
            reader.read(image_id)
    else:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(reader.read, ids))
    seconds = time.perf_counter() - started
    return {"slices": len(ids), "threads": threads, "slices_per_second": round(len(ids) / max(seconds, 1e-6), 1)}
