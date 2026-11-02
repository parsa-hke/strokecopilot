"""ISLES'24 slim-format reader: list_cases(), load_case(), fetch_slim() (plan §5.4).

Held-out cases are refused before Phase 4, and list_cases() leaves them out. Arrays are RAS, on one
isotropic grid per manifest.json, except ncct_native. Tmax voxels at -30 have no perfusion data, and
map background comes from the manifest. Reference volumes are in `native_volumes`, measured before
resampling, because resampling moves Tmax > 6 s by up to about 10 mL (docs/DATA.md).
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

import nibabel as nib
import numpy as np
import pandas as pd

from strokecopilot.config import LOCAL_ROOT
from strokecopilot.data import clinical_columns, splits
from strokecopilot.data.archive_listing import copy_with_progress

SLIM_NAME = "isles24_slim_v1"
DEFAULT_ROOT = LOCAL_ROOT / SLIM_NAME
FILES = {
    "ncct": "ncct.nii.gz",
    "cbf": "cbf.nii.gz",
    "cbv": "cbv.nii.gz",
    "mtt": "mtt.nii.gz",
    "tmax": "tmax.nii.gz",
    "occlusion": "cta_occlusion.nii.gz",
    "cow": "cow.nii.gz",
    "infarct": "infarct_followup.nii.gz",
}
MASKS = ("occlusion", "infarct")


@dataclass
class Case:
    id: str
    arrays: dict
    spacing_mm: float
    affine: np.ndarray
    clinical: dict
    native_volumes: dict = field(default_factory=dict)

    @property
    def note_fields(self):
        """Clinical values allowed in a triage note (plan §8.3). `clinical` also holds forbidden ones."""
        return {column: value for column, value in self.clinical.items() if column != "case" and clinical_columns.tag_of(column) == "allowed"}


def fetch_slim(tar_path, parent=LOCAL_ROOT):
    """Copy the tarball to local disk and unpack it (DEC-3). Does nothing if already there."""
    root = os.path.join(parent, SLIM_NAME)
    if os.path.exists(os.path.join(root, "manifest.json")):
        return root
    os.makedirs(parent, exist_ok=True)
    local = os.path.join(parent, f"{SLIM_NAME}.tar")
    copy_with_progress(tar_path, local)
    subprocess.run(["tar", "-xf", local, "-C", parent], check=True)
    os.remove(local)
    return root


def load_manifest(root=DEFAULT_ROOT):
    with open(os.path.join(root, "manifest.json")) as f:
        return json.load(f)


def list_cases(root=DEFAULT_ROOT, folds=None):
    """Cases in the tarball, without held-out cases unless ALLOW_HELDOUT=1."""
    cases = load_manifest(root)["cases"]
    if splits.heldout_allowed():
        return cases
    heldout = set((folds or splits.load_folds())["heldout"])
    return [case for case in cases if case not in heldout]


@lru_cache(maxsize=4)
def _tables(root):
    clinical = pd.read_csv(os.path.join(root, "clinical.csv"), dtype=object).set_index("case")
    volumes = pd.read_csv(os.path.join(root, "native_volumes.csv")).set_index("case")
    return clinical, volumes


def _clean(row):
    return {column: (None if pd.isna(value) else value) for column, value in row.items()}


def load_case(case, root=DEFAULT_ROOT, names=tuple(FILES), folds=None):
    """Arrays, clinical row, spacing and native volumes for one case. Raises on a held-out case."""
    splits.check_ids([case], folds)
    root = str(root)
    if case not in load_manifest(root)["cases"]:
        raise KeyError(f"Case {case} is not in the tarball.")
    arrays, affine = {}, None
    for name in names:
        img = nib.load(os.path.join(root, case, FILES[name]))
        array = np.asanyarray(img.dataobj)
        arrays[name] = array > 0 if name in MASKS else array
        affine = img.affine
    clinical, volumes = _tables(root)
    return Case(
        id=case,
        arrays=arrays,
        spacing_mm=load_manifest(root)["spacing_mm"],
        affine=affine,
        clinical={"case": case, **_clean(clinical.loc[case])},
        native_volumes={k: float(v) for k, v in volumes.loc[case].items()},
    )


def load_native_ncct(case, root=DEFAULT_ROOT, folds=None):
    """The unchanged source NCCT, as a nibabel image, for Vision B (plan §8.2). Not RAS or resampled."""
    splits.check_ids([case], folds)
    return nib.load(os.path.join(str(root), case, "ncct_native.nii.gz"))
