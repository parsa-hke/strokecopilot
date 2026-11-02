# Data

What we actually hold, as found. Exact file layouts are added in Phase 1 (plan §7).

## ISLES'24

Stored at `StrokeCopilot/raw/isles24_train.7z` on the shared Drive. The md5 was computed on the copy stored on Drive, and it matches the value on Zenodo.

```json
{
  "record_id": 17652035,
  "doi": "10.5281/zenodo.17652035",
  "file": "train.7z",
  "size_bytes": 99022114670,
  "md5": "4959a5dd2438d53e3c86d6858484e781",
  "verified_utc": "2026-10-31T16:28:51.820341+00:00"
}
```

Licence: CC BY-NC-SA 4.0 (plan §5.1).

### Archive layout

From the archive's file list, read on 2 November 2026 with `01a_list_isles24.ipynb`. No file has been extracted yet, so image shapes, spacing and file contents are unconfirmed. The full listing and summary are in `StrokeCopilot/artifacts/isles24_listing/`.

- 2,982 files, 95.840 GiB unpacked. 92.222 GiB packed, because the images are already gzipped.
- 149 cases. IDs run from `sub-stroke0001` to `sub-stroke0189`, so they have gaps.
- All four perfusion maps (CBF, CBV, MTT, Tmax) exist for all 149 cases, in both `derivatives/` and `raw_data/`.
- Paths below start at `train/`. `{c}` is the case ID. GiB is the total for all 149 cases.

| Path | GiB | Slim file (§5.4) |
|---|---:|---|
| `raw_data/{c}/ses-01/{c}_ses-01_ncct.nii.gz` | 1.910 | `ncct_native`, and `ncct` after resampling |
| `raw_data/{c}/ses-01/{c}_ses-01_cta.nii.gz` | 5.921 | Not used |
| `raw_data/{c}/ses-01/{c}_ses-01_ctp.nii.gz` | 6.249 | Not used |
| `raw_data/{c}/ses-01/perfusion-maps/{c}_ses-01_{cbf,cbv,mtt,tmax}.nii.gz` | 0.398 | Not used; these are not in NCCT space |
| `derivatives/{c}/ses-01/perfusion-maps/{c}_ses-01_space-ncct_cbf.nii.gz` | 1.375 | `cbf` |
| `derivatives/{c}/ses-01/perfusion-maps/{c}_ses-01_space-ncct_cbv.nii.gz` | 1.288 | `cbv` |
| `derivatives/{c}/ses-01/perfusion-maps/{c}_ses-01_space-ncct_mtt.nii.gz` | 1.236 | `mtt` |
| `derivatives/{c}/ses-01/perfusion-maps/{c}_ses-01_space-ncct_tmax.nii.gz` | 1.096 | `tmax` |
| `derivatives/{c}/ses-01/{c}_ses-01_space-ncct_lvo-msk.nii.gz` | 0.012 | `cta_occlusion` |
| `derivatives/{c}/ses-01/{c}_ses-01_space-ncct_cow-msk.nii.gz` | 0.012 | `cow` |
| `derivatives/{c}/ses-01/{c}_ses-01_space-ncct_cta.nii.gz` | 2.160 | Not used |
| `derivatives/{c}/ses-01/{c}_ses-01_space-ncct_ctp.nii.gz` | 67.435 | Not used |
| `derivatives/{c}/ses-02/{c}_ses-02_space-ncct_lesion-msk.nii.gz` | 0.094 | `infarct_followup` |
| `derivatives/{c}/ses-02/{c}_ses-02_space-ncct_dwi.nii.gz` | 3.272 | Not used |
| `derivatives/{c}/ses-02/{c}_ses-02_space-ncct_adc.nii.gz` | 3.353 | Not used |
| `phenotype/{c}/ses-01/{c}_ses-01_demographic_baseline.xlsx` | < 0.001 | `clinical.csv` |
| `phenotype/{c}/ses-02/{c}_ses-02_outcome.xlsx` | < 0.001 | `clinical.csv`; see Task 2 in plan §7 |
| `clinical_data-description.xlsx` | < 0.001 | Data dictionary |

What this means:

- **No NCCT in `derivatives/`.** NCCT exists only in `raw_data/`, which fits NCCT being the reference space. Whether `space-ncct` files match its shape is still to be checked.
- **Clinical data is two Excel files per case,** not the CSVs the paper describes. Preparation has to merge 298 files.
- **Extracting a single modality saves almost nothing.** The archive is solid, with 26 blocks. Getting any `derivatives/` file type means decoding about 79 GiB, including the lesion masks. Clinical files need about 1.7 GiB and the data dictionary almost none. The `-x` option still saves disk, because skipped files are decoded but not written.
- **Disk:** a CPU runtime had 205 GiB free, and a full unpack needs 95.840 GiB. Other runtime types may differ.
- **Stray file:** one case also has `derivatives/{c}/ses-02/{c}_ses-02_lesion-msk.nii` (0.025 GiB, not gzipped, no `space-ncct`). All 149 cases have the normal mask. Which case it is and what it holds are unchecked. Preparation selects files by exact name, so it will not pick this one up.

## RSNA ICH

Terms limit where this data can go; see `DATA_USE.md`. Built on Kaggle from the competition's stage 2 training data into private datasets (DEC-17): `rsna_slim_v1`, 320 px, JPEG quality 95, 4 shards. 752,803 labelled slices in total.

| Shard | Dataset | Slices | Failed | Tar size (GiB) | Status |
|---|---|---:|---:|---:|---|
| 0 | `rsna-slim-v1-s0` | 188,251 | 0 | 4.371 | Dataset created |
| 1 | `rsna-slim-v1-s1` | 187,442 | 0 | 4.352 | Dataset created |
| 2 | `rsna-slim-v1-s2` | 188,381 | 1 | 4.373 | Built and verified; dataset not created yet |
| 3 | `rsna-slim-v1-s3` | 188,729 | 0 | 4.378 | Built and verified; dataset not created yet |
| Total | | 752,803 | 1 | 17.474 | |

Slices in the shards add up to the labelled total, so 752,802 images are usable. The failed slice raised a `ValueError` and is not in the tar. It stays in shard 2's index with `ok` false, so loaders must filter on `ok`. Its `z_mm` and `pixel_spacing_mm` are empty.

The reports' `projected_total_gb` (16.94) counts JPEG bytes only; the total above includes tar overhead. Reading speed from an attached dataset has not been measured. Excluded scans: none so far.
