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
- **Stray file:** `sub-stroke0142` also has `derivatives/sub-stroke0142/ses-02/sub-stroke0142_ses-02_lesion-msk.nii` (0.025 GiB, not gzipped, no `space-ncct`). All 149 cases have the normal mask. What this file holds is unchecked. Preparation selects files by exact name, so it will not pick this one up.

### Clinical data

Surveyed on 2 November 2026 with `01b_clinical_isles24.ipynb`, in aggregate only. The results are in `StrokeCopilot/artifacts/isles24_clinical/`. Extraction took 129 s on Colab. Note tags are enforced in `src/strokecopilot/data/clinical_columns.py`.

Each workbook has one sheet and the same columns. They hold no case ID, so the ID comes from the path. "Missing" counts rows out of 150, because of the extra row described below.

| Column | Source | Note tag | Missing | Remarks |
|---|---|---|---:|---|
| `Center` | baseline | Forbidden | 0 | 1 or 2. Not in the dictionary, which lists `Identifier` instead |
| `Sex` | baseline | Allowed | 1 | M 75, F 74 |
| `Age` | baseline | Allowed | 0 | Decimal, though the dictionary says integer |
| `Atrial fibrillation` | baseline | Allowed | 2 | |
| `Hypertension` | baseline | Allowed | 1 | |
| `Diabetes` | baseline | Allowed | 2 | The dictionary's description wrongly says hypertension |
| `Hyperlipidemia` | baseline | Allowed | 3 | Has one value of 99, read as unknown. Same dictionary error |
| `Anticoagulation` | baseline | Allowed | 3 | |
| `Lipid lowering drugs` | baseline | Allowed | 5 | The dictionary calls it `Statins` |
| `PAIs` | baseline | Allowed | 4 | Platelet aggregation inhibitors |
| `Glucose` | baseline | Allowed | 19 | mg/dl, altered ±5% |
| `Leucocytes` | baseline | Allowed | 60 | G/l, altered ±5% |
| `CRP` | baseline | Allowed | 60 | mg/dl, altered ±5% |
| `INR` | baseline | Allowed | 61 | Altered ±5% |
| `Wake-up` | baseline | Allowed | 2 | 42 are wake-up strokes |
| `In-House` | baseline | Allowed | 13 | 7 are in-house strokes |
| `Referral` | baseline | Allowed | 1 | 19 came from an external clinic |
| `Onset to door` | baseline | Allowed | 76 | Text, hrs:min per the dictionary |
| `Alert to door` | baseline | Allowed | 136 | Text. Wake-up strokes only |
| `NIHSS at admission` | baseline | Allowed | 4 | |
| `mRS at admission` | baseline | Forbidden | 51 | Not a §8.3 note field; not the same as premorbid mRS |
| `mRS premorbid` | baseline | Allowed | 60 | |
| `Door to imaging` | baseline | Allowed | 12 | Text |
| `Door to groin` | baseline | Forbidden | 12 | Reveals thrombectomy |
| `Door to first series` | baseline | Forbidden | 51 | Reveals thrombectomy |
| `Time of intervention` | baseline | Forbidden | 54 | Reveals thrombectomy |
| `Door to recanalization` | baseline | Forbidden | 55 | Reveals thrombectomy |
| `TICI postinterventional` | outcome | Forbidden | 51 | Text: 3 (75), 2c (22), 2b (1), 2a (1) |
| `NIHSS 24h` | outcome | Forbidden | 66 | After treatment |
| `mRS 24h` | outcome | Forbidden | 65 | After treatment |
| `NIHSS discharge` | outcome | Forbidden | 53 | After treatment |
| `mRS discharge` | outcome | Forbidden | 53 | After treatment |
| `mRS 3 months` | outcome | Forbidden | 20 | After treatment |

No ASPECTS column and no thrombolysis column exist.

What this means:

- **Extra rows:** one baseline and one outcome workbook have two rows. In baseline the extra row is nearly empty but has `Center` and `Age`. Which cases these are is unchecked. Preparation must handle them explicitly and report them.
- **Onset time is often unknown:** only 74 rows have `Onset to door` and 14 have `Alert to door`, so at least 61 of the 149 cases have neither. There is no last-known-well or sleep-midpoint column.
- **Rule inputs are often unknown:** `INR` is missing in 61 rows and `mRS premorbid` in 60. Notes must say these are not recorded, so the reasoner treats them as unknown.
- **Possible centre effect:** centre 2 has 50 rows, and several columns are missing in about 51. That suggests centre 2 does not record them, but it is not confirmed.
- **Reperfusion:** of the 99 rows with a TICI value, 97 are 2c or 3 and 2 are lower. The rest have none, so the data does not show that every patient reached 2c or 3.
- **±5% on times:** the dictionary states this for labs only, while the paper says it covers times too (plan §5.1). We treat times as altered.

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
