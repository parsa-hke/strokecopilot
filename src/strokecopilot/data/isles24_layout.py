"""Where each ISLES'24 file lives in the archive, as found in Phase 1 (docs/DATA.md)."""

import os
import subprocess

FILES = {
    "ncct": "raw_data/{c}/ses-01/{c}_ses-01_ncct.nii.gz",
    "cbf": "derivatives/{c}/ses-01/perfusion-maps/{c}_ses-01_space-ncct_cbf.nii.gz",
    "cbv": "derivatives/{c}/ses-01/perfusion-maps/{c}_ses-01_space-ncct_cbv.nii.gz",
    "mtt": "derivatives/{c}/ses-01/perfusion-maps/{c}_ses-01_space-ncct_mtt.nii.gz",
    "tmax": "derivatives/{c}/ses-01/perfusion-maps/{c}_ses-01_space-ncct_tmax.nii.gz",
    "lvo": "derivatives/{c}/ses-01/{c}_ses-01_space-ncct_lvo-msk.nii.gz",
    "cow": "derivatives/{c}/ses-01/{c}_ses-01_space-ncct_cow-msk.nii.gz",
    "lesion": "derivatives/{c}/ses-02/{c}_ses-02_space-ncct_lesion-msk.nii.gz",
    "dwi": "derivatives/{c}/ses-02/{c}_ses-02_space-ncct_dwi.nii.gz",
    "baseline": "phenotype/{c}/ses-01/{c}_ses-01_demographic_baseline.xlsx",
    "outcome": "phenotype/{c}/ses-02/{c}_ses-02_outcome.xlsx",
}
STRAY_LESION = "derivatives/{c}/ses-02/{c}_ses-02_lesion-msk.nii"
DICTIONARY = "clinical_data-description.xlsx"

# What a working copy keeps. Everything else is decoded but not written, which saves about 85 GiB.
WORKING_MEMBERS = ("train/clinical_data-description.xlsx", "train/phenotype/*", "train/derivatives/*", "train/raw_data/*")
WORKING_EXCLUDE = (
    "*_ctp.nii.gz", "*_cta.nii.gz", "*_adc.nii.gz",
    "*_ses-01_cbf.nii.gz", "*_ses-01_cbv.nii.gz", "*_ses-01_mtt.nii.gz", "*_ses-01_tmax.nii.gz",
)  # fmt: skip


def path_of(root, case, name):
    return os.path.join(root, FILES[name].format(c=case))


def find_cases(root):
    phenotype = os.path.join(root, "phenotype")
    return sorted(d for d in os.listdir(phenotype) if os.path.isdir(os.path.join(phenotype, d)))


def missing_files(root, cases):
    """{name: number of cases without that file}, for names with at least one case missing."""
    counts = {name: sum(not os.path.exists(path_of(root, case, name)) for case in cases) for name in FILES}
    return {name: n for name, n in counts.items() if n}


def pack(parent, tar_path):
    """Tar `train/` under parent, uncompressed: the images are already gzipped."""
    subprocess.run(["tar", "-cf", tar_path, "-C", parent, "train"], check=True)


def unpack(tar_path, parent):
    subprocess.run(["tar", "-xf", tar_path, "-C", parent], check=True)
