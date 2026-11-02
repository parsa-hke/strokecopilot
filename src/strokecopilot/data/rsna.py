"""RSNA ICH shard reader and CT windowing, for Kaggle notebooks (plan §5.2, DEC-17).

The index holds slice, patient and study IDs, so nothing here prints them, and none of this data may
leave Kaggle (docs/DATA_USE.md). Image channels are brain, subdural, bone, as `prep_rsna` stored them.
"""

import os
import random
import re
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from strokecopilot.data.prep_rsna import IMAGE_SIZE, LABEL_NAMES, N_SHARDS, SLIM_VERSION, window_image

SHARD_RE = re.compile(rf"^{SLIM_VERSION}_\d+px_s(\d+)(\.tar|_index\.csv\.gz)$")


def find_shards(root, n_shards=N_SHARDS):
    """{shard: (tar path, index path)} under root, such as /kaggle/input. Raises if any is missing."""
    found = {}
    for folder, _, files in os.walk(root):
        for name in files:
            match = SHARD_RE.match(name)
            if match:
                found.setdefault(int(match.group(1)), {})[match.group(2)] = os.path.join(folder, name)
    complete = {shard: (paths[".tar"], paths["_index.csv.gz"]) for shard, paths in found.items() if len(paths) == 2}
    missing = sorted(set(range(n_shards)) - set(complete))
    if missing:
        raise FileNotFoundError(f"Shards {missing} lack a tar or an index under {root}. Attach every rsna-slim-v1 dataset with Add Input.")
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
            self.offsets = {m.name[: -len(".jpg")]: (m.offset_data, m.size) for m in tar if m.isfile()}
        self._fd = os.open(tar_path, os.O_RDONLY)

    def __contains__(self, image_id):
        return image_id in self.offsets

    def __len__(self):
        return len(self.offsets)

    def read_bytes(self, image_id):
        offset, size = self.offsets[image_id]
        return os.pread(self._fd, size, offset)

    def read(self, image_id):
        return decode(self.read_bytes(image_id))

    def close(self):
        os.close(self._fd)


def decode(jpeg):
    """JPEG bytes -> height x width x 3 uint8 array: brain, subdural, bone."""
    image = cv2.imdecode(np.frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode a slice.")
    return image


def window_hu(hu, size=IMAGE_SIZE):
    """HU slice -> the same three windows as training, for running on ISLES'24 NCCT (plan §8.2)."""
    return window_image(hu, 1, 0, size)


def copy_shard(tar_path, dest_dir):
    """Copy a shard to local disk, and return the new path and seconds taken."""
    os.makedirs(dest_dir, exist_ok=True)
    started = time.perf_counter()
    dest = os.path.join(dest_dir, os.path.basename(tar_path))
    shutil.copyfile(tar_path, dest)
    return dest, time.perf_counter() - started


def measure_read(reader, n=2000, threads=1, seed=0):
    """Random reads and decodes per second. Returns numbers only."""
    ids = random.Random(seed).sample(sorted(reader.offsets), min(n, len(reader)))
    started = time.perf_counter()
    if threads == 1:
        for image_id in ids:
            reader.read(image_id)
    else:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(reader.read, ids))
    seconds = time.perf_counter() - started
    return {"slices": len(ids), "threads": threads, "slices_per_second": round(len(ids) / max(seconds, 1e-6), 1)}
