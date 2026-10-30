"""Raw RSNA DICOM -> windowed JPEG tar shards plus a scan-level index (plan §5.2, DEC-17).

Each slice is read once, so the index comes from the same pass as the images. Slices are assigned
to shards by a hash of their ID, so a shard's contents never depend on the order of the input.
Patient-level splits are made later, from the index. JPEG channels are brain, subdural, bone, in
the order cv2 stores them.
"""

import hashlib
import io
import json
import os
import tarfile
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import pydicom

SLIM_VERSION = "rsna_slim_v1"
WINDOWS = ((40, 80), (80, 200), (600, 2800))  # (centre, width): brain, subdural, bone
LABEL_NAMES = ("any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural")
IMAGE_SIZE = 320
JPEG_QUALITY = 95
N_SHARDS = 4
MAX_TOTAL_GB = 19.0  # all shards must fit on Kaggle's local disk during training

INDEX_COLUMNS = ["image_id", "shard", "ok", "error", "patient_id", "study_id", "series_id", "z_mm", "pixel_spacing_mm"]


def find_data(root):
    """Return (labels csv, image folder) under root, without listing the image folders."""
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = [d for d in subfolders if not d.startswith("stage_2_")]
        if "stage_2_train.csv" in files:
            return os.path.join(folder, "stage_2_train.csv"), os.path.join(folder, "stage_2_train")
    raise FileNotFoundError(f"stage_2_train.csv not found under {root}. Add the competition data with Add Input.")


def load_labels(csv_path):
    """Read stage_2_train.csv into one row per slice, with one column per label."""
    rows = pd.read_csv(csv_path).drop_duplicates("ID")
    parts = rows["ID"].str.rsplit("_", n=1, expand=True)
    rows = rows.assign(image_id=parts[0], label_name=parts[1])
    labels = rows.pivot(index="image_id", columns="label_name", values="Label")
    missing = set(LABEL_NAMES) - set(labels.columns)
    if missing:
        raise ValueError(f"labels file is missing {sorted(missing)}")
    if labels[list(LABEL_NAMES)].isna().any().any():
        raise ValueError("some slices do not have all six labels")
    return labels[list(LABEL_NAMES)].astype(int).sort_index().reset_index().rename_axis(columns=None)


def shard_of(image_id, n_shards=N_SHARDS):
    return int(hashlib.md5(image_id.encode()).hexdigest(), 16) % n_shards


def window_image(pixels, slope, intercept, size=IMAGE_SIZE):
    """Raw pixels -> size x size x 3 uint8 image, one CT window per channel."""
    hu = pixels.astype("float32") * float(slope) + float(intercept)
    channels = [np.clip((hu - (centre - width / 2)) / width, 0, 1) for centre, width in WINDOWS]
    image = (np.stack(channels, axis=-1) * 255).round().astype("uint8")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _read_slice(job):
    """Worker: read one DICOM. Returns (image_id, jpeg bytes or None, index fields)."""
    image_id, path, size, quality = job
    fields = {"ok": False, "error": ""}
    try:
        ds = pydicom.dcmread(path)
        image = window_image(ds.pixel_array, ds.get("RescaleSlope", 1), ds.get("RescaleIntercept", 0), size)
        encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not encoded:
            raise ValueError("JPEG encoding failed")
        position = ds.get("ImagePositionPatient")
        spacing = ds.get("PixelSpacing")
        fields.update(
            ok=True,
            patient_id=str(ds.get("PatientID", "")),
            study_id=str(ds.get("StudyInstanceUID", "")),
            series_id=str(ds.get("SeriesInstanceUID", "")),
            z_mm=float(position[2]) if position else None,
            pixel_spacing_mm=float(spacing[0]) if spacing else None,
        )
        return image_id, jpeg.tobytes(), fields
    except Exception as error:
        fields["error"] = f"{type(error).__name__}: {str(error)[:120]}"
        return image_id, None, fields


def _init_worker():
    # RSNA's de-identified IDs are not valid DICOM UIDs; validating each one floods the log.
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
    cv2.setNumThreads(1)


def _write_jpeg(tar, image_id, jpeg):
    info = tarfile.TarInfo(f"{image_id}.jpg")
    info.size = len(jpeg)
    tar.addfile(info, io.BytesIO(jpeg))  # mtime stays 0, so reruns give identical archives


def shard_paths(out_dir, shard, size=IMAGE_SIZE):
    stem = f"{SLIM_VERSION}_{size}px_s{shard:02d}"
    return os.path.join(out_dir, f"{stem}.tar"), os.path.join(out_dir, f"{stem}_index.csv.gz")


def process_shard(labels, image_dir, out_dir, shard, n_shards=N_SHARDS, size=IMAGE_SIZE, quality=JPEG_QUALITY, workers=None, limit=None, progress_every=20000):
    """Write one shard's tar and index, and return a report.

    limit processes only the first slices, for a quick size check. The tar keeps a .part name until
    it is complete, so an interrupted run never leaves a finished-looking file.
    """
    selected = labels[labels["image_id"].map(lambda i: shard_of(i, n_shards)) == shard]
    if selected.empty:
        raise ValueError(f"shard {shard} has no slices")
    complete = limit is None or limit >= len(selected)
    selected = selected.iloc[:limit] if limit else selected
    os.makedirs(out_dir, exist_ok=True)
    tar_path, index_path = shard_paths(out_dir, shard, size)
    jobs = [(i, os.path.join(image_dir, f"{i}.dcm"), size, quality) for i in selected["image_id"]]

    workers = workers if workers is not None else (os.cpu_count() or 2)
    pool = Pool(workers, initializer=_init_worker) if workers > 0 else None
    if pool is None:
        _init_worker()
    results = pool.imap(_read_slice, jobs, chunksize=16) if pool else map(_read_slice, jobs)

    start, jpeg_bytes, rows = time.perf_counter(), 0, []
    part_path = f"{tar_path}.part"
    try:
        with tarfile.open(part_path, "w") as tar:
            for done, (image_id, jpeg, fields) in enumerate(results, 1):
                rows.append({"image_id": image_id, "shard": shard, **fields})
                if jpeg is not None:
                    _write_jpeg(tar, image_id, jpeg)
                    jpeg_bytes += len(jpeg)
                if done % progress_every == 0:
                    seconds = time.perf_counter() - start
                    print(f"{done:,} / {len(jobs):,} slices, {done / seconds:.0f} per second, {jpeg_bytes / 1024 ** 3:.2f} GB")
    finally:
        if pool:
            pool.close()
            pool.join()
    os.replace(part_path, tar_path)

    index = pd.DataFrame(rows).reindex(columns=INDEX_COLUMNS).merge(selected, on="image_id", validate="one_to_one")
    index.to_csv(index_path, index=False)

    n_ok = int(index["ok"].sum())
    seconds = time.perf_counter() - start
    report = {
        "version": SLIM_VERSION, "shard": shard, "n_shards": n_shards, "image_size": size, "jpeg_quality": quality,
        "slices_in_labels": len(labels), "slices_processed": len(jobs), "slices_ok": n_ok, "slices_failed": len(jobs) - n_ok,
        "errors": index.loc[~index["ok"], "error"].str.split(":").str[0].value_counts().to_dict(),
        "complete_shard": complete,
        "seconds": round(seconds, 1), "slices_per_second": round(len(jobs) / seconds, 1),
        "tar_gb": round(os.path.getsize(tar_path) / 1024 ** 3, 3),
        "projected_total_gb": round(jpeg_bytes / max(n_ok, 1) * len(labels) / 1024 ** 3, 2),
        "index_file": os.path.basename(index_path),
    }
    return report


def verify_shard(tar_path, index_path, size=IMAGE_SIZE, sample=200):
    """Check a shard against its index, and time reading it straight through. Raises if they differ."""
    index = pd.read_csv(index_path, keep_default_na=False)
    expected = {f"{i}.jpg" for i in index.loc[index["ok"].astype(str) == "True", "image_id"]}
    start, found, decoded = time.perf_counter(), set(), 0
    with tarfile.open(tar_path) as tar:
        for member in tar:
            data = tar.extractfile(member).read()
            found.add(member.name)
            if decoded < sample:
                image = cv2.imdecode(np.frombuffer(data, "uint8"), cv2.IMREAD_COLOR)
                if image is None or image.shape != (size, size, 3):
                    raise ValueError(f"bad image in shard: {member.name}")
                decoded += 1
    if found != expected:
        raise ValueError(f"tar and index differ: {len(found - expected)} extra, {len(expected - found)} missing")
    seconds = time.perf_counter() - start
    return {"images": len(found), "decoded_sample": decoded, "read_images_per_second": round(len(found) / max(seconds, 1e-6), 1)}


def save_report(report, out_dir, shard):
    path = os.path.join(out_dir, f"{SLIM_VERSION}_s{shard:02d}_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
