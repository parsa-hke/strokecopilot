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

From the archive's file list, read on 2 November 2026 with `01a_list_isles24.ipynb`. The full listing and summary are in `StrokeCopilot/artifacts/isles24_listing/`.

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

- **No NCCT in `derivatives/`.** NCCT exists only in `raw_data/`, which fits NCCT being the reference space. See the image survey below for how well `space-ncct` files match it.
- **Clinical data is two Excel files per case,** not the CSVs the paper describes. Preparation has to merge 298 files.
- **Extracting a single modality saves almost nothing.** The archive is solid, with 26 blocks. Getting any `derivatives/` file type means decoding about 79 GiB, including the lesion masks. Clinical files need about 1.7 GiB and the data dictionary almost none. The `-x` option still saves disk, because skipped files are decoded but not written.
- **Disk:** a CPU runtime had 205 GiB free, and a full unpack needs 95.840 GiB. Other runtime types may differ.
- **Stray file:** `sub-stroke0142` also has `derivatives/sub-stroke0142/ses-02/sub-stroke0142_ses-02_lesion-msk.nii` (0.025 GiB, not gzipped, no `space-ncct`). All 149 cases have the normal mask. The stray file is on the same grid and close to the normal mask (0.37 vs 0.34 mL, Dice 0.66). We ignore it, and preparation selects files by exact name, so it will not pick this one up.

### Image survey

From `01c_unpack_split_survey.ipynb`, run on 2 November 2026. Results are in `StrokeCopilot/artifacts/isles24_survey/`. The working copy of the files we use is `artifacts/isles24_work/isles24_work.tar` (10.32 GiB); making it took 9,845 s on a Colab CPU runtime, 6,089 s of that decoding. Alignment, labels, values and grids were measured on development cases only.

**Headers (all 149).** All images are 3D. Spacing is 0.305–0.562 mm in-plane (median 0.41) and 0.8–4.0 mm between slices (median 2.0). Shape is 512 × 512–686 × 34–177. Every file matches the NCCT grid except one Circle of Willis mask, which is in a held-out case. Preparation moves any off-grid file by world position and reports a count only.

**Alignment.** Median 99.6% of an infarct mask sits on soft-tissue NCCT voxels (p5 93.7%, lowest 81.9%, 127 cases). In the figure the outline matches the bright area on DWI in all six cases. No registration is needed. Two development infarct masks are empty, and two development occlusion masks are empty.

**Circle of Willis labels.** Labels 1–12, and 15 in 4 cases. By the TopCoW numbering (from memory, not confirmed for this dataset) there is one MCA label per side, 5 and 7, with a median of about 0.1 mL, and each is missing in 14 of 129 cases. **So M1 and M2 cannot be separated.**

Most occlusion masks do not overlap any label (median share outside 100%). The median distance to a label is 3.3 mm, p95 45 mm. Nearest labels over 127 masks: 7 (38), 5 (35), 1 (20), 3 (9), 6 (8), 4 (7), 12 (5), 2 (3), 11 (2). So about 57% are nearest an MCA label and become `unknown` (plan §8.1). The 20 nearest label 1, the basilar, look high and need the distance limit.

**Values (10 cases).**
- NCCT looks like HU: air at -1000, bone up to about 1400.
- Tmax background is -30. CBF, CBV and MTT background is 0.
- CBV and MTT have negative values (p0.1 -15 and -97). MTT has 8 non-finite voxels.
- Units are unconfirmed.
- Tmax has gaps inside the brain, in places a whole region (`sub-stroke0012`) or a whole slice (`sub-stroke0094`), and some infarct lies in them. It looks like limited CTP coverage. How much infarct this affects is measured in preparation.

**Grid.** Chosen: **1.5 mm isotropic, cropped to the brain.** Measured on 10 cases, so the p95 values are close to the worst case:

| Spacing | Lesion error p50 / p95 | Tmax > 6 s error p50 / p95 | Median crop | Projected slim |
|---|---:|---:|---:|---:|
| 1.0 mm | 0.8% / 5.3% | 3.9 / 6.7 mL | 145×178×140 | 5.36 GiB |
| 1.5 mm | 1.4% / 9.5% | 2.6 / 6.8 mL | 98×120×94 | 2.98 GiB |
| 2.0 mm | 1.1% / 13.3% | 1.8 / 6.0 mL | 73×90×70 | 2.37 GiB |

1.0 mm has about 3.3 times the voxels with no gain on Tmax, and 2.0 mm has the worst p95 lesion error. Tmax error getting worse at finer spacing is unexplained. Because a 2–7 mL resampling error matters near a volume threshold, `native_volumes.csv` keeps volumes measured before resampling.

**Split.** 20 held-out and folds of 26, 26, 26, 26 and 25, with centres balanced. Infarct p95 is 56 mL in held-out and 107–168 mL in the folds, so Phase 4 says little about very large infarcts, which may be chance with 20 cases. The split stays as it is.

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

- **Extra rows:** both workbooks of `sub-stroke0180` have two rows, with the same number of values in each (16 and 16 in baseline, 1 and 1 in outcome). Whether they agree is unchecked, because the survey's check compared a row with itself; that is fixed. Preparation merges them: gaps are filled, values that disagree become missing, and it reports what it found.
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
