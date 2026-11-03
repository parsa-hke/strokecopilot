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
- Tmax has gaps inside the brain, in places a whole region (`sub-stroke0012`) or a whole slice (`sub-stroke0094`), and some infarct lies in them. It looks like limited CTP coverage. See the slim section below for how much.

**Grid.** Chosen: **1.5 mm isotropic, cropped to the brain.** Measured on 10 cases, so the p95 values are close to the worst case:

| Spacing | Lesion error p50 / p95 | Tmax > 6 s error p50 / p95 | Median crop | Projected slim |
|---|---:|---:|---:|---:|
| 1.0 mm | 0.8% / 5.3% | 3.9 / 6.7 mL | 145×178×140 | 5.36 GiB |
| 1.5 mm | 1.4% / 9.5% | 2.6 / 6.8 mL | 98×120×94 | 2.98 GiB |
| 2.0 mm | 1.1% / 13.3% | 1.8 / 6.0 mL | 73×90×70 | 2.37 GiB |

1.0 mm has about 3.3 times the voxels with no gain on Tmax, and 2.0 mm has the worst p95 lesion error. Tmax error getting worse at finer spacing is unexplained.

**Split.** 20 held-out and folds of 26, 26, 26, 26 and 25, with centres balanced. Infarct p95 is 56 mL in held-out and 107–168 mL in the folds, so Phase 4 says little about very large infarcts, which may be chance with 20 cases. The split stays as it is.

### Slim tarball

`StrokeCopilot/slim/isles24_slim_v1.tar`, built on 2 November 2026 with `01_prep_isles24.ipynb` (layout in plan §5.4). Immutable: a change means `_v2`. Build report and figure are in `StrokeCopilot/artifacts/isles24_prep/`.

```json
{
  "cases": 149,
  "spacing_mm": 1.5,
  "size_bytes": 3066152960,
  "sha256": "b0afd16e4cbfb16a69fb5190083fc3869f5dcc6c92f86f10135b0de1f59f4728"
}
```

That is 2.86 GiB. It took 990 s to build (7 s per case) on a Colab CPU runtime, plus 209 s to copy the working copy from Drive. The final check found no problems in any file's grid, spacing, orientation or type. Median shape is 98×119×94, largest 112×135×110.

Measured over all 149 cases, as counts:

- **Orientation:** sources are 100 LAS and 49 LPS, so the slim files were turned to RAS. Those counts match centres 1 and 2, but that link is not confirmed.
- **Off-grid:** the one Circle of Willis mask was moved to the NCCT grid.
- **Crop:** it grew for 3 cases to hold a mask.
- **Rescued masks:** 4 occlusion masks and 1 infarct mask would have vanished, so each keeps its best voxel.
- **MTT:** 417,749 non-finite voxels became background, in 21 cases. Most are small (median 250 voxels), and the worst has 225,045, about 1% of a median volume. CBF, CBV and Tmax have none.
- **Label loss:** 5 cases lost a Circle of Willis label, for two different reasons, checked with `01d_check_slim.ipynb`. In 4 development cases one tiny label (9 or 10, about 0.005 mL, 13–15 voxels) vanished in resampling, which is expected. One held-out case lost 8 labels (1,051 voxels), all outside the crop, so its mask does not sit where the brain does. It is very likely the one off-grid mask, but that link is inferred. We do not look further at a held-out case. v1 stands, and occlusion site must return `unknown` when labels are absent or too far away.

Measured over 129 development cases:

| | p50 | p95 |
|---|---:|---:|
| Infarct volume error | 0.8% (0.13 mL) | 6.2% (0.45 mL) |
| Tmax > 6 s volume error | 2.8 mL | 9.6 mL |
| Tmax > 6 s error if maps had been averaged | 7.5 mL | 17.4 mL |

What this means:

- **Use `native_volumes.csv` for true volumes.** A Tmax error up to about 10 mL is too large near a volume threshold, so read reference `penumbra_ml` from that file, not from the resampled map. Averaging would have been clearly worse.
- **Some infarct has no usable perfusion input.** Counting only -30 as no data, the share of an infarct with Tmax has a median of 99.4% and p5 of 80.9%, and is under 90% in 15 of 127 cases. Counting only Tmax of 0 or more, it is a median of 95.3% and p5 of 65.1%. `sub-stroke0094` has none. Vision A cannot learn or be fairly scored on tissue it has no input for, so training and metrics need to take this into account in Phase 2.
- **Many Tmax values are negative, and they are not small.** Beyond -30, a median of 16% (p95 28%) of remaining voxels are below zero, typically from -28 to -0.7 with a median of -14 s, which has no physical meaning as a delay. 43% lie on soft tissue (20–60 HU) and 8% on CSF-like values, so about half lie on neither. A median of 4% (p95 16%) of an infarct lies on them. The cause is unknown; failed perfusion fits are a guess, and -30 may be the lower limit of the same values rather than a separate fill value. **Recommendation for Vision A:** treat all negative Tmax as no reliable data. Tmax > 6 s itself is unaffected, but tissue with a failed fit cannot count as penumbra, so `penumbra_ml` can undercount there.

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
| `Onset to door` | baseline | Allowed | 76 | Text, `hh:mm:ss`, some with microseconds. The dictionary says hrs:min |
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

- **Extra rows:** both workbooks of `sub-stroke0180` have two rows, with 16 values in each baseline row and 1 in each outcome row. Preparation found no gaps to fill and no disagreements, so they are duplicates, merged into one row.
- **Time text:** `Onset to door`, `Alert to door` and `Door to imaging` read as `hh:mm:ss`, about a quarter with microseconds as well (development cases). The microseconds fit the ±5% alteration. `clinical.csv` keeps the text, so note generation must parse it. `clinical.csv` also holds the forbidden columns, so note generation must filter with `tag_of`.
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
| 2 | `rsna-slim-v1-s2` | 188,381 | 1 | 4.373 | Dataset created |
| 3 | `rsna-slim-v1-s3` | 188,729 | 0 | 4.378 | Dataset created |
| Total | | 752,803 | 1 | 17.474 | |

Slices in the shards add up to the labelled total, so 752,802 images are usable. The failed slice raised a `ValueError` and is not in the tar. It stays in shard 2's index with `ok` false, so loaders must filter on `ok`. Its `z_mm` and `pixel_spacing_mm` are empty.

The reports' `projected_total_gb` (16.94) counts JPEG bytes only; the total above includes tar overhead. Excluded scans: none so far.

### What the datasets hold, and read speed

Measured on 3 November 2026 with `02b_rsna_read_speed.ipynb` on a 4-CPU Kaggle runtime. Kaggle unpacked each tar when the dataset was made, so each shard is a folder of JPEGs, and each index a plain CSV. Every shard's images match its index. There are 752,802 readable slices, 21,744 scans and 18,938 patients. 14.34% of slices and 40.85% of scans have a bleed. Scans have 28–44 slices (p5–p95), median 33.

| Reading from | Threads | Slices per second |
|---:|---:|---:|
| Attached folder, first read | 1 | 25 |
| Attached folder, same files again | 4 | 1,726 (cached, so not a real speed) |
| Local tar, just written | 1 | 847 |
| Local tar, same files again | 4 | 1,856 |

Packing one shard from its folder into a local tar took 7,579 s (2.1 hours), which is also 25 files per second. The first read from a folder is therefore too slow to train on: about 8.4 hours for one pass or to pack all four shards. A local tar is fast enough, at about 7 minutes a pass for reading and decoding alone. Local speeds are best-case, because the file had just been written and so was probably in memory. 

**Re-store test on shard 0, 3 November 2026.** Kaggle keeps a file named `.shard` whole, while a `.tar` next to it is unpacked, confirmed with a small test and then with the real 4.371 GiB file. Packing shard 0 with `02c_rsna_repack.ipynb` and 16 threads took 748 s, at 252 slices per second.

| Reading from | Threads | Slices per second |
|---:|---:|---:|
| Attached `.shard`, random reads | 1 | 16.5 |
| Attached `.shard`, random reads | 4 | 15.3 |
| Local copy, different slices each time | 1 | 877 |
| Local copy, different slices each time | 4 | 2,193 |

Copying the file to local disk took 103 s (about 43 MiB/s) and listing it 17.5 s. **Always copy first: never read directly from the attached file.** Reading and decoding alone takes about 5.7 minutes per pass, best-case because the copy is probably in memory. Augmentation and the GPU will add to this.

**All four as single files, 3 November 2026.** Datasets `rsna-slim-v1-file-s0` to `-s3` hold one `.shard` file and its index each, and all 752,802 readable slices match their indexes. Copying all four took 199 s (33–69 s each) and listing them 35 s, so **a training session sets up in about 4 minutes.** With all four copied, local random reads ran at 373 slices per second on 1 thread and 1,227 on 4, about 10 minutes per pass for reading and decoding alone. That is lower than the one-shard test, probably because 17.5 GiB no longer fits in memory; it is the more realistic figure. The old `rsna-slim-v1-s0` to `-s3` folder datasets are no longer needed.
