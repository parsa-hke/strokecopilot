# CHANGELOG

## 2026-11-04 — Claude Code session (Phase 2, task 3: nnU-Net set-up)
- **Code:** added `vision/perfusion/nnunet_wrapper.py`, pinned to nnU-Net v2 2.8.1 after reading its trainer and command-line code. It:
  - converts development cases only to six channels: NCCT, CBF, CBV, MTT, Tmax with negative values set to 0, and a Tmax-valid mask. The label is the whole final infarct.
  - writes our five folds as `splits_final.json`;
  - adds a trainer, `nnUNetTrainer_StrokeCopilot`, whose epochs, iterations and checkpoint interval come from the environment so a run can resume across Colab sessions;
  - parses epoch times from nnU-Net's logs, and scores the validation predictions with `evaluate.py`.
- **Notebook:** added `notebooks/10_visionA_nnunet.ipynb`.
  - The dummy mode (EXP-2) trains 2 epochs, resumes for a third, then predicts and scores fold 0, to measure epoch time and check that checkpoints load on Colab's PyTorch.
  - The train mode (EXP-3) needs its epoch count set from EXP-2. It checkpoints to Drive and refuses to resume if a session plans a different network.
- **Tested on synthetic data only**, with Colab, git, pip, nvidia-smi and nnU-Net's command line faked: both modes, a train session that stops after preprocessing and is rerun, a refused rerun of a finished run, and no token or held-out ID in any output.
- **Not tested at all:** real nnU-Net, real PyTorch and GPU behaviour, and real epoch time. nnU-Net 2.8.1 excludes PyTorch 2.9, so pip may replace Colab's PyTorch; the notebook prints the version before and after.
- Tests: `test_nnunet_wrapper.py`. pytest: 221 passed.
- **First EXP-2 attempt on an L4 failed:** the system killed planning with exit code -9 after 6 of 129 cases of fingerprint extraction, almost certainly from running out of memory. nnU-Net starts 8 fresh Python processes by default, and each imports PyTorch. Colab now runs Python 3.13.
  - Fix: fingerprinting and preprocessing use 2 workers (`N_PROC`), and training's data-augmentation workers are capped at 4 (`N_PROC_DA`, up to 12 by default). The notebook prints RAM, and a -9 exit now says it was a memory kill. Tested on synthetic data only.
  - pytest: 222 passed.
- **Notion sync:**
  - done: EXP-2 and EXP-3 rows (Planned).

## 2026-11-04 — Claude Code session (Phase 2: EXP-1 results)
- **EXP-1 ran on Colab** at `bf86e15` with all 129 development cases, at 0.21 s per case. Results are in `runs/EXP-1_thresholds_baseline/`.
- **Threshold core vs final infarct:**
  - Dice 0.111 ± 0.031 over folds; the pooled median is 0.027.
  - Absolute volume error 25.2 ± 5.0 mL. The median error is -4.2 mL, and the core underestimates large infarcts (p5 -106 mL).
  - The predicted core falls on the other side of a cut-off from the reference in 35, 29, 23 and 17 of 129 cases at 21, 31, 51 and 70 mL.
  - Dropping the Tmax > 6 s restriction changes little (Dice 0.117).
  - Penumbra matches native Tmax > 6 s to a median of 2.8 mL, as in Phase 1.
- **Not accepted yet:** a Dice this low needs checking for a bug before it stands. Added `thresholds.diagnose_case`, with a test, and `notebooks/11b_visionA_threshold_check.ipynb`. The notebook measures how much infarct lies inside Tmax > 6 s and on the side called affected, the rCBF inside the infarct, and Dice at rCBF 20–45%. Tested on synthetic data only.
- **Occlusion distance limit set to 5 mm** from the EXP-1 distances:
  - ICA-nearest occlusions sit at a median 0 mm (p95 6.0 mm);
  - basilar-nearest ones sit 17–51 mm away, except one under 3 mm;
  - 5 mm gives 13 ICA, 1 basilar and 115 `unknown` of 129, the `unknown` cases including 72 nearest an MCA label, 7 ambiguous and 2 empty.
- **Consequence:** about 89% of real cases will have `unknown` occlusion site, so most real-case thrombectomy decisions will be undetermined (§9).
- **11b ran at `869e47f` and EXP-1 is accepted.** The low Dice comes from the data, not from a bug that the checks can see:
  - the infarct is on the affected side in almost every case (median share 1.0);
  - only half of it lies inside Tmax > 6 s, and its median rCBF is 0.65;
  - small infarcts score near zero, while the largest third has a median Dice of 0.24.
  - The 30% cut-off is kept. Details are in `docs/DATA.md` under "Perfusion and the final infarct".
- **MCA-nearest occlusions:** 20 of 72 overlap the M1 label. They stay `unknown` under the approved rule.
- §7: the `thresholds.py` and `volumes.py` tasks are ticked.
- pytest: 214 passed.
- **Notion sync:**
  - done: EXP-1 row → Done, with headline numbers and the check result.
  - pending: the Decisions rows listed in the entry below.

## 2026-11-03 — Claude Code session (Phase 2 start: metric definition, threshold baseline code)
- **Phase 2 set-up, agreed with Parsa:**
  - **LLMs:** the frontier model is Claude, with about $100 of API credits. The open and medical models are to be chosen in Phase 3.
  - **Rule-claim check:** an LLM judge, not NLI.
  - **Retrieval embeddings:** `BAAI/bge-base-en-v1.5`.
  - **RSNA split:** a seeded hash of patient ID into 80/10/10 train/validation/test, checked to keep the natural bleed rate. No IDs leave Kaggle.
  - **Compute:** 180 Colab compute units and 30 Kaggle GPU hours this week.
  - **Notion Experiments** gained `Fold` and `Git commit` columns. Rows are created first, and their ID names the run folder.
- **Hallucination metric:** `docs/HALLUCINATION_METRIC.md` was written before any verifier code and approved with its seven open points, then mirrored to Notion. Additions to §8.4:
  - claims also come from blocking factors, fact-stating uncertainties and decision `rule_ids`;
  - a content hallucination rate (sub-types 1–5) is reported next to the full rate;
  - omission of uncertainty is a separate count;
  - the judge model is chosen by seeded-error recall between Haiku 5.5 and Opus 5.5, because of the budget.
  - §7 task ticked.
- **Temperature:** the current Claude API rejects `temperature`, so §8.3 and §8.4 now say "temperature 0 where the provider allows it", with a repeat sample to show run-to-run spread. The §10 cost risk was updated.
- **Vision A code:** added `vision/perfusion/volumes.py` (mask to mL, contract volumes, the occlusion-site rule), `thresholds.py` (rCBF < 30% core, Tmax > 6 s penumbra, Tmax < 0 is no data), and a new `evaluate.py`. `evaluate.py` holds Dice, Dice on reliable Tmax, volume error against `native_volumes.csv` and cut-off crossings, and is shared with nnU-Net. Tests: `test_volumes.py`, `test_thresholds.py`.
- **Notebook:** added `notebooks/11_visionA_thresholds.ipynb`, EXP-1, which also measures occlusion-to-label distances. It ran on synthetic data with Colab, git and pip stubbed: full run, refusal to replace a finished run, and no token or held-out ID in any output.
- **Not yet run on real data:** real Dice, volume errors, timing and the distance distribution are unknown. The occlusion distance limit stays provisional at 5 mm until the run, so the §7 `thresholds.py` and `volumes.py` tasks stay open.
- **Label numbering:** the TopBrain 2025 data page confirms that TopCoW labels 1–12 and 15 are kept, and that 5 and 7 are the M1 segments of the MCAs. MCA-nearest occlusions stay `unknown` under the approved rule.
- pytest: 213 passed.
- **Notion sync:**
  - done: Hallucination metric page; Experiments columns; EXP-1 row (Planned).
  - pending: Decisions rows for the LLM judge, embedding model, RSNA split and temperature wording; Tasks → hallucination metric done.

## 2026-11-03 — Claude Code session (Phase 1 closed)
- Shards 1–3 re-stored with `02c`. Final `02b` on the four `rsna-slim-v1-file-*` datasets: every shard is one file, 752,802 readable slices all match their indexes, copying all four takes 199 s and listing them 35 s. Local random reads: 373 slices per second on 1 thread, 1,227 on 4, about 10 minutes per pass reading only. Lower than the one-shard test, probably because 17.5 GiB no longer fits in memory.
- §7: loader task ticked. README badges ticked: rows for unwritten Phase 2+ notebooks no longer link to missing files, and `02_prep_rsna` is marked Kaggle-only.
- **Phase 1 acceptance (as changed under DEC-6) is met:** the ISLES'24 tarball and RSNA shards are ready, both loaders work on real data, and RSNA set-up and read speed fit training. Still deferred, not done: `00_template`, onboarding and Phase 2 assignment. Phase 0's open items are still Parsa's: professors' deliverables and deadline, the Kaggle no-copies question, the shared Drive folder and clinician review.
- Notion sync: pending. Tasks → `isles24.py` and `rsna.py` loaders done; RSNA preprocessing done, with datasets now `rsna-slim-v1-file-s0` to `-s3`. The old folder datasets and `dummy-unpack-test` can be deleted.

## 2026-11-03 — Claude Code session (Phase 1: re-store test passed on shard 0)
- Parsa suggested a quick test with dummy files first: Kaggle unpacked a `.tar` and kept a `.shard` whole, so the choice goes by file name. Shard 0 then packed in 748 s with 16 threads, and `rsna-slim-v1-file-s0` kept the 4.371 GiB file whole.
- `02b` on it: copy 103 s, list 17.5 s, local reads 877–2,193 slices per second without repeating slices, best-case. Random reads from the attached file are 15–17 per second, so training must copy first. All four shards should set up in about 8 minutes per session, projected from one shard.
- Next: `02c` for shards 1–3, then a final `02b` with all four files.
- Notion sync: pending until all four are done.

## 2026-11-03 — Claude Code session (Phase 1: re-storing RSNA shards)
- Parsa approved re-storing the shards. Added `pack_shard` to `rsna.py`, which packs a shard's folder into one tar named `.shard`, and `notebooks/02c_rsna_repack.ipynb`, run for shard 0 first as a test. A single file now wins over a folder if both are attached.
- Fixed `02b`: every measurement uses different slices. A single-file shard is copied locally and listed there, because listing it over Kaggle's storage may be very slow; it also reports copy time, the set-up cost of a session.
- Tested by running both notebooks' real cells on synthetic data, with Kaggle and pip stubbed. **Not yet run on Kaggle**: whether Kaggle keeps a `.shard` file whole is an assumption, and copy speed and multi-thread first-read speed are unknown.
- Notion sync: pending until the test shard has passed.

## 2026-11-03 — Claude Code session (Phase 1: RSNA read speed)
- Ran `02b_rsna_read_speed.ipynb` on Kaggle. All four datasets exist and match their indexes: 752,802 readable slices. All four are folders, because Kaggle unpacked the tars. Full table in `docs/DATA.md`.
- **First read from a folder is 25 slices per second**, the same as raw DICOM. One pass or packing all four shards would take about 8.4 hours. Packing one shard took 2.1 hours. A local tar reads at 847–1,856 per second, best-case.
- **My mistake:** `02b` measured 4 threads on the same files as 1 thread, so its 1,726 per second from a folder is cached and means nothing. To be fixed in the next notebook.
- Recommended: re-store the shards once in a form Kaggle does not unpack, starting with one shard as a test. Waiting for Parsa's go-ahead. The §7 loader task stays open.
- Notion sync: pending until the storage decision. RSNA preprocessing → all four datasets exist.

## 2026-11-02 — Claude Code session (DEC-6 changed: onboarding deferred)
- **Parsa changed DEC-6:** no fixed onboarding date. Parsa sends teammates instructions when assigning them work. Nobody has been onboarded yet; this is deferred, not done.
- **Phase 1 acceptance changed:** tarballs ready, loaders working on real data for both datasets, and RSNA read speed measured and fit for training. It was every teammate running `00_template`.
- `00_template`, onboarding and Phase 2 assignment moved to a deferred list in §7, unticked. `00_template` gets written when the first teammate is brought in. Week 1 in §7 now starts when the first Phase 2 tasks are assigned.
- Still needed before anyone joins: the shared Drive folder, and their acceptance of the RSNA competition rules.
- Notion sync: pending. Decisions → update DEC-6. Tasks → mark the onboarding task deferred.

## 2026-11-02 — Claude Code session (Phase 1: RSNA shards are unpacked on Kaggle)
- The first run of `02b_rsna_read_speed.ipynb` showed all four datasets attached, so `-s2` and `-s3` exist, but `-s0` holds a folder instead of a `.tar`: Kaggle unpacks archives when a dataset is made. The other three are assumed the same, because no shard was found.
- `rsna.py` now reads both layouts: `FolderReader`, `open_shard`, and `copy_shard` packs a folder into one local tar. `02b` measures speed from the attached folder and from a local tar, and the time to pack, which is the set-up cost of a session.
- **Not yet run on real data.** Whether the shards can stay as folders, or need re-storing in a form Kaggle does not unpack, depends on these numbers. The earlier `.tar` sizes in `docs/DATA.md` describe the files as built, not the datasets.
- Notion sync: none pending until the notebook has run.

## 2026-11-02 — Claude Code session (Phase 1: slim check results)
- Ran `01d_check_slim.ipynb` on Colab. All 129 development cases load through the loader at 0.13 s each with no problems, all 20 held-out cases are refused, and copy and unpack takes 54 s. The checksum on Drive matches the build.
- **Label loss explained:** in 4 development cases one tiny label vanished in resampling, as expected. One held-out case lost 8 labels, all outside the crop, very likely the one off-grid mask. Not looked at further. v1 stands, and occlusion site must return `unknown` when labels are absent or too far away.
- **MTT:** non-finite voxels are in 21 cases, median 250, worst about 1% of a volume.
- **Negative Tmax:** typically -28 to -0.7, median -14 s, and only about half lie on soft tissue or CSF-like values. Cause unknown. Infarct with Tmax of 0 or more is a median of 95.3%, p5 65.1%, lower than the 99.4% and 80.9% that counted only -30 as no data. Recommendation for Vision A: treat all negative Tmax as no reliable data.
- The §7 loader task stays open until `02b_rsna_read_speed.ipynb` has run.
- Notion sync: pending. Tell Vision A about negative Tmax when Phase 2 tasks are assigned.

## 2026-11-02 — Claude Code session (Phase 1, Task 7: loaders)
- Implemented `data/isles24.py` (`fetch_slim`, `list_cases`, `load_case`, `load_native_ncct`) and `data/rsna.py` (`find_shards`, `load_index`, `scan_table`, `slices_of_scan`, `ShardReader`, `decode`, `window_hu`, `copy_shard`, `measure_read`), with `tests/test_isles24_loader.py` and `tests/test_rsna_loader.py`. `load_case` and `load_native_ncct` refuse held-out cases, and `list_cases` leaves them out. `Case.note_fields` filters by note tag, because `clinical` also holds forbidden columns.
- Added `cow_label_check`, `mtt_non_finite_check` and `tmax_negative_check` to `data/isles24_survey.py`, and notebooks `01d_check_slim.ipynb` (Colab) and `02b_rsna_read_speed.ipynb` (Kaggle).
- Tested by running both notebooks' real cells on synthetic data with Colab, Kaggle and pip stubbed. **Not yet run on real data**: the real read speed, whether Kaggle kept each shard as a `.tar`, whether `-s2` and `-s3` exist, and the answers to the label, MTT and Tmax questions are unknown. The §7 loader task stays open until both notebooks have run.
- RSNA patient-level splits are not in `rsna.py`; they belong to Vision B in Phase 2.
- Notion sync: none pending until the notebooks have run.

## 2026-11-02 — Claude Code session (Phase 1: slim tarball built)
- Ran `01_prep_isles24.ipynb` on Colab: `slim/isles24_slim_v1.tar`, 149 cases, 2.86 GiB, sha256 `b0afd16e…4728`, read back from Drive and matching, no problems in the final check. 7 s per case, 990 s in all, well under my 30–90 minute estimate. §7 grid and prep tasks ticked; 1.5 mm is in `manifest.json`. Full results are in `docs/DATA.md`.
- **Results:** infarct volume error 0.8% median, 6.2% p95. Tmax > 6 s error 2.8 mL median, 9.6 mL p95, against 7.5 and 17.4 mL if averaged, so not averaging maps was right on real data. Reference volumes should come from `native_volumes.csv`.
- **New findings:** in 15 of 127 development cases over 10% of the infarct has no Tmax data, and one has none. A median of 16% of non-background Tmax voxels are negative, which is unexplained. MTT has 417,749 non-finite voxels, now background. Source orientation is 100 LAS and 49 LPS. Both `sub-stroke0180` rows are duplicates. Time text is `hh:mm:ss`, not hrs:min.
- **Corrected:** `01_prep`'s figure caption said black on Tmax is tissue without data; it can also be a negative value. Only -30 counts as no data in the coverage figures.
- **Open:** 5 cases lost a Circle of Willis label, including what looks like one case losing most of them, possibly outside the crop. Not diagnosed. v1 stands; occlusion site must return `unknown` when labels are absent. `01_prep` has not been rerun since the caption change, and a rerun refuses to replace the tarball.
- Notion sync: pending. Tasks → grid and ISLES'24 slim are done. Tell Vision A about Tmax coverage, negative Tmax and `native_volumes.csv`, and NLP A about time text format, when Phase 2 tasks are assigned.

## 2026-11-02 — Claude Code session (Phase 1: split in the repo)
- The `01c` section 9 `OK:` line confirms the results were read back from Drive, the archive is unchanged, and there are 129 development cases.
- Added `splits/folds.json`. Its sha256 matches the one the notebook recorded (`023813f2…d291`). §7 task ticked. The split was not regenerated.
- **Held-out exposure:** attaching `folds.json` to the chat loaded the whole file, held-out IDs included, into this Claude Code session's context and local transcript. That was the file's ID list only, with no images, volumes or clinical data for those cases. They were not repeated or used in any analysis or code, and DEC-5 is unchanged. From now on `folds.json` is checked by checksum and never opened or attached in a session.
- Notion sync: pending. Tasks → split generated and in the repo.

## 2026-11-02 — Claude Code session (Phase 1: survey results and slim preparation)
- Ran `01c_unpack_split_survey.ipynb` on Colab. Working copy on Drive (10.32 GiB) in 9,845 s, 6,089 s of that decoding. The split was generated and all 149 cases are complete. Full results are in `docs/DATA.md`. The `OK:` line from section 9 has not been seen yet, though the results are on Drive.
- **Alignment passes,** so no registration. **The Circle of Willis labels cannot separate M1 from M2**; §5.1 and §8.1 now use the nearest label with a distance limit, and MCA is `unknown`. That is about 57% of development masks. The schema is unchanged. Parsa approved both. §7 tasks ticked.
- **Grid: 1.5 mm isotropic, cropped to the brain.** Parsa approved. The §7 task stays open until `manifest.json` exists.
- **New findings:** Tmax has gaps inside the brain and some infarct lies in them, not yet quantified; CBV and MTT have negative values and MTT has non-finite voxels; one Circle of Willis mask, in a held-out case, is off the NCCT grid; 2 development infarct masks and 2 occlusion masks are empty; held-out has no very large infarcts.
- **Fixed a bug:** `extra_rows` compared a row with itself when two rows had equal counts, so the survey's "no conflicts" for `sub-stroke0180` means nothing. `docs/DATA.md` was also wrong to call the extra row nearly empty.
- Added `data/prep_isles24.py`, `tests/test_prep_isles24.py`, `notebooks/01_prep_isles24.ipynb` and `sha256_of` in `data/archive_listing.py`. Slim v1 gains `native_volumes.csv`, because resampling moves Tmax > 6 s volumes by 2–7 mL and native Tmax is not in the tarball.
- **Changed my own suggestion:** on irregular synthetic shapes, averaging before shrinking lost more mask volume than plain sampling (about -40% vs -20% at 0.03 mL, -8% vs -3% at 0.3 mL), so only NCCT is averaged. The report also measures Tmax error with averaging, for comparison.
- Tested by running the notebook's real cells on a synthetic 30-case working copy with Drive and pip stubbed: full run, refusal to replace a finished tarball, resuming from saved batches, and no held-out ID in the output or report. A full-size synthetic case took 1.6 s on a Mac. **Not yet run on Colab or real data**: real time per case, Tmax coverage, resampled volume errors, time text format and what the `sub-stroke0180` rows hold are all unknown. `splits/folds.json` is not in the repo yet, so that §7 task stays open.
- Notion sync: pending. Tasks → spot-check and M1/M2 check are done, and the grid is decided. Decisions → occlusion site rule (MCA is `unknown`), 1.5 mm grid, `native_volumes.csv` in slim v1. Tell Vision A about Tmax gaps and NLP A about `unknown` occlusion site when Phase 2 tasks are assigned.

## 2026-11-02 — Claude Code session (Phase 1: split, spot-check and grid notebook)
- Added `data/splits.py`, `tests/test_splits.py`, `data/isles24_layout.py`, `data/isles24_survey.py`, `tests/test_isles24_survey.py`, `splits/README.md` and `notebooks/01c_unpack_split_survey.ipynb`. The `[isles]` extra gains numpy, nibabel, scipy and matplotlib, and the wheel now ships `splits/`.
- The notebook makes a working copy of the files we use and saves it to Drive, checks the known anomalies before any split exists, generates the split once, checks headers for all cases, spot-checks alignment and labels on development cases only, and measures candidate grids.
- **Split:** §5.5 now stratifies by centre as well as infarct volume tertile, seed 20261102. DEC-5 is unchanged. Parsa approved this on 2 November 2026.
- Tested by running its real cells on a synthetic 30-case archive with Drive and pip stubbed, with 7-Zip 26.05 and p7zip 17.05: first run, rerun from the saved copy, a changed split being refused, and no held-out ID in the output or report. **Not yet run on Colab or real data**: real timing, disk, Drive upload of the tar, image contents and the survey's reading of real labels are untested. `folds.json` does not exist yet, and no §7 task is ticked.
- Notion sync: none pending until the notebook has run.

## 2026-11-02 — Claude Code session (Phase 1, Task 2 results)
- Ran `01b_clinical_isles24.ipynb` on Colab: extraction took 129 s, results saved to Drive and read back, archive unchanged. The printed PDF cut off long lines, so the full results were read from the saved JSON files on Drive through the Google Drive connector, read-only.
- Added `data/clinical_columns.py` and `tests/test_clinical_columns.py`: all 33 real columns are tagged allowed or forbidden for notes, and an untagged column raises an error. `mRS at admission` and `Center` are forbidden by our choice; the rest follow §8.3. Full table in `docs/DATA.md`. §7 task ticked.
- **Findings:** at least 61 of 149 cases have no onset or alert time, and there is no last-known-well column. `INR` is missing in 61 rows and `mRS premorbid` in 60. Of 99 rows with a TICI value, 2 are below 2c and the rest of the rows have none, so §5.1 and §8.1 now carry that caveat. One baseline and one outcome workbook have an extra row; which cases is unchecked. `Hyperlipidemia` has a 99 code. Actual column names differ from the dictionary, and `Center` is undocumented.
- The extra lesion mask belongs to `sub-stroke0142`. Its contents are unchecked.
- Fixed a small bug: `workbooks_holding_their_case_id` reached the JSON as a string. The value, 0, was right.
- Time columns are text and their format has not been parsed yet, so the `hrs:min` format is only what the dictionary states.
- Notion sync: done in this session for Tasks → column tagging. Still pending: tell NLP A about the missing-data findings when Phase 2 tasks are assigned.

## 2026-11-02 — Claude Code session (Phase 1, Task 2: clinical survey)
- Added `data/clinical_survey.py`, `tests/test_clinical_survey.py`, `notebooks/01b_clinical_isles24.ipynb`, `extract_members` in `data/archive_listing.py`, and an `[isles]` extra. The notebook extracts only the data dictionary and clinical workbooks, prints the dictionary, and surveys each column in aggregate: counts for columns with few distinct values, percentiles for numbers, and no case IDs or single-patient rows. It also names the case holding the extra lesion mask.
- Tested by running its real cells on a synthetic archive with Drive and pip stubbed, with 7-Zip 26.05 and p7zip 17.05, including a rerun and an archive with no clinical files. **Not yet run on Colab or the real files**: the real workbook structure, column names, extraction time and Excel value formats are untested. Columns are not tagged yet, so the §7 task stays open.
- Notion sync: none pending until the notebook has run.

## 2026-11-02 — Claude Code session (Phase 1, Task 1 results)
- Ran `01a_list_isles24.ipynb` on Colab: file list read in 3 s, saved to Drive and read back, archive unchanged. The Colab install and Drive read both worked. 2,982 files, 95.840 GiB unpacked, 149 cases with IDs from 0001 to 0189. All four perfusion maps exist for every case. Exact paths and the mapping to slim files are in `docs/DATA.md`. §7 task ticked.
- Corrected the plan: clinical data is two Excel files per case, not CSVs (§5.1).
- The archive is solid: any `derivatives/` file type costs about 79 GiB of decoding, including lesion masks, while clinical files cost about 1.7 GiB. So the slim build and the infarct volumes for the split both need what is effectively a full unpack.
- **Order change:** the split (Task 6) moves ahead of the spot-check (Task 3), so nobody looks at images of cases that later become held-out. The split, spot-check and grid survey share one unpack session. Task 2 stays next because it is cheap.
- Unexplained: one case has an extra `ses-02_lesion-msk.nii` without `space-ncct`. To be checked in Task 3. Nothing has been extracted, so shapes, spacing and contents are unconfirmed.
- Notion sync: pending. A search of Tasks found no row for the archive inspection, so no row was changed. Data guide → point to `docs/DATA.md` for the layout.

## 2026-11-02 — Claude Code session (Phase 1, Task 1: archive listing)
- Added `data/archive_listing.py`, `tests/test_archive_listing.py` and `notebooks/01a_list_isles24.ipynb`. The notebook reads the file list of `raw/isles24_train.7z` without extracting anything and reports layout, perfusion maps, unpacked size, whether the archive is solid, and how much must be decoded to extract each file type. It saves to `artifacts/isles24_listing/`, reads the result back from Drive, and checks the archive's size and modified time did not change.
- Tested by running its real cells on small synthetic archives with Drive and pip stubbed, with 7-Zip 26.05 and p7zip 17.05: solid, non-solid, rerun, wrong size, truncated archive and missing file. **Not yet run on Colab or the real archive**: the private install, the `apt-get` step (Colab's 7z version), reading the list through the Drive mount and the real layout are untested. The §7 task stays open.
- Added a local `.venv` (git-ignored). pytest: all pass.
- Notion sync: none pending until the notebook has run.

## 2026-11-02 — Claude Code session (Phase 0 wrap-up)
- Read the Kaggle competition rules for RSNA. They apply to our copy and are stricter than the plan said, which had relied on the looser challenge terms: academic use is allowed, but no redistribution and no access for anyone who has not accepted the rules. Corrected §5.2, and added `docs/DATA_USE.md` with what this means for teammates, Drive, the repo, the report and local copies.
- **Open question:** the rules' clause against making copies conflicts, taken literally, with storing private shards (DEC-17). DEC-17 is unchanged; this needs asking the professors. Guideline licence terms are not checked yet.
- Added `docs/CLINICIAN_REVIEW.md`: rules, the five cases and questions, for sending to a reviewer. Not sent. DAWN and DEFUSE-3 numbers are still unchecked against the trial papers.
- Replaced `<org>` in the README. Confirmed the repo is private. Ticked the terms and GitHub tasks in §7.
- Phase 0 acceptance criteria are met. Three tasks stay open and are Parsa's: record deliverables and deadline, finish and share the Drive folder, and get the clinician review. Datasets `rsna-slim-v1-s2` and `-s3` were still uploading and their contents are unchecked.
- Notion sync: done in this session (Data-use terms, Data guide, and the terms, ISLES'24, RSNA and clinician tasks). The calendar was not touched.

## 2026-11-02 — Claude Code session (RSNA shards complete)
- All four `rsna_slim_v1` shards built on Kaggle: 752,803 slices match the labels, 752,802 usable, and each tar matches its index. One slice in shard 2 failed with a `ValueError`; its ID is in that shard's index with `ok` false. Total size is 17.47 GiB including tar overhead, under the 19 GB limit.
- Datasets `rsna-slim-v1-s0` and `-s1` exist. `-s2` and `-s3` are not created yet: on those versions Kaggle's Output menu offers "New dataset version" on `rsna-slim-v1-s0`, which must not be used.
- Speed varied from 35 slices per second (shard 0) to 10 (shard 3), so a shard took 1.5–5.4 hours. The notebook's 2–3 hour estimate is not reliable. Reading speed from an attached dataset is still unmeasured.
- Shard 2's verify step printed a harmless pandas `DtypeWarning`, caused by the failed slice's empty fields.
- Notion sync: pending. Tasks → ISLES'24 download is done; RSNA preprocessing is built, with two datasets to create.

## 2026-10-31 — Claude Code session (data download records)
- ISLES'24 download finished on Colab: 99,022,114,670 bytes on Drive, md5 matches Zenodo. Resuming across sessions worked; one attempt stalled for about 15 minutes and the notebook's retry recovered. The archive has not been opened yet. Phase 0 task ticked.
- RSNA shard 0 built on Kaggle: 188,251 slices, 0 failed, 4.371 GiB, about 92 minutes, with the tar matching its index. Projected total is 16.94 GiB at 320 px, under the 19 GB limit. Shards 1–3 are not recorded yet. Reading speed from an attached dataset is still unmeasured; the 8,043 images per second in the report came from local disk and decoded only a sample.
- Added `docs/DATA.md` with both records.
- Notion sync: pending. Tasks → ISLES'24 download is done; RSNA preprocessing has shard 0 done.

## 2026-10-31 — Claude Code session (RSNA preprocessing)
- Implemented `data/prep_rsna.py`: one pass over the raw DICOM writes 320 px three-window JPEGs into hash-assigned tar shards, plus an index with patient, study and series IDs, slice position, spacing and all six labels. Added `tests/test_prep_rsna.py` and an `[rsna]` extra.
- Added `notebooks/02_prep_rsna.ipynb`, run once per shard with Save Version. It installs the package with a read-only GitHub token from Kaggle Secrets and stops early if the projected size would exceed 19 GB.
- Tested on synthetic DICOMs with the Kaggle secret client and pip stubbed. **Not yet run on Kaggle or real data**: the private `pip install`, real speed, real size, and tar reading from an attached dataset are untested.
- Notion sync: Tasks → RSNA preprocessing points to this notebook.

## 2026-10-31 — Claude Code session (RSNA timing results, DEC-17)
- First Kaggle timing run: 752,803 labelled slices, 14.3% with a bleed, no missing files or failed reads, patient/study/series IDs in all 2,000 sampled files, `SliceThickness` absent. Raw reading is about 20–28 files per second, or about 7.5 h per epoch.
- **Test flaw:** the 384 px loader speed (206 images per second) is inflated because it re-read files the earlier steps had cached. The report's 384 px epoch times should be ignored, and the notebook still has this ordering flaw; it is kept as the record of this run.
- DEC-17 decided: preprocess once to 320 px JPEG tar shards, then train on Kaggle; 256 px is the fallback. Plan §2, §4.2, §5.2 and §7 updated. Reading speed from tar shards is not measured yet.
- `02a_rsna_kaggle_timing.ipynb`: silence pydicom's warning about RSNA's non-standard IDs.
- Notion sync: done in this session (DEC-17, RSNA task, Data guide).

## 2026-10-31 — Claude Code session (Kaggle notebook fix)
- Fixed `02a_rsna_kaggle_timing.ipynb`: the first Kaggle run could not find `stage_2_train.csv`, because the search only went two folders deep. It now walks the input folder and skips image folders. Tested on five folder layouts; not yet rerun on Kaggle.
- Notion sync: none pending.

## 2026-10-31 — Claude Code session (run calendar)
- Created the Notion "Run Calendar" database under the hub, covering every planned run and its prerequisites from 31 Oct to 7 Nov 2026, split across Parsa and four teammates. Teammates are placeholder roles until they join the workspace. Estimates are unmeasured.
- This compresses the plan's 8-week phases into 8 days; §7 has not been changed. 7 Nov needs confirming as the real deadline before the plan is rewritten around it.
- Notion sync: done in this session.

## 2026-10-31 — Claude Code session (contract freeze)
- Parsa froze `schemas/vision_output.json` at v0.1 with no new fields (DEC-4). Mass effect, M2 dominance and extensive hypodensity stay out of the contract; the reasoner must name them as uncertainties. Later changes need a Decisions row and a version bump.
- Notion sync: done in this session (contract task marked Done).

## 2026-10-31 — Claude Code session (RSNA Kaggle timing notebook)
- Added `notebooks/02a_rsna_kaggle_timing.ipynb`. It runs on Kaggle, trains nothing, and saves only `rsna_timing_report.json`, with counts and timings and no IDs or images. It estimates epoch time per model and image size, preprocessing time and output size, to settle DEC-17. Self-contained, because the package cannot be installed there yet.
- Tested by running its real cells on synthetic DICOMs on a Mac CPU, including corrupt, missing and duplicate entries. Testing caught and fixed a JSON crash in the final cell. **Not yet run on Kaggle, on a GPU or on real RSNA files**, so the GPU training path, real file reading and Kaggle's limits are untested.
- Notion sync: Tasks → RSNA timing test now points to this notebook.

## 2026-10-30 — Claude Code session (RSNA full data)
- DEC-16: Parsa chose all labelled RSNA data, no subset. DEC-17 proposed: preprocess and train Vision B in Kaggle notebooks so no 180 GB transfer is needed, pending a timing test.
- Plan §2, §5.2, §7, §8.2 and §10 updated. Phase 0 acceptance now requires full RSNA attached and timed, not downloaded. The `rsna_slim` tarball is no longer assumed.
- Notion sync: done in this session (DEC-16, DEC-17, RSNA task, Data guide).

## 2026-10-30 — Claude Code session (guideline verification)
- Parsa confirmed DEC-14 and DEC-15, and supplied the 2026 guideline PDF and the RSNA challenge page. The PDF stays out of the repo.
- §8.3 rulebook rewritten from the primary text. **Corrections to the secondary summaries:** extended-window thrombolysis (§4.6.3 rec 2) applies only to patients not eligible for thrombectomy; M2 is not a single COR 2b, but COR 2a for dominant proximal M2 and COR 3 no benefit otherwise; ASPECTS ≥ 6 at 6–24 h is confirmed as COR 1.
- Fixture expectations revised: fx02 and fx05 thrombolysis return to no, reversing the borderline expectations added earlier today; fx01 thrombectomy depends on M2 dominance; fx04 stays a thrombectomy yes. Vision values and notes unchanged.
- DEC-7 supported: guideline Table 3 defines penumbra as Tmax > 6 s and uses penumbra − core as the mismatch volume.
- Salvageable-penumbra rule: the recommendations give no numbers, so we use the trial criteria in Table 3 (EXTEND for 4.5–9 h, TRACE-III for 4.5–24 h). Our choice; needs clinician review.
- RSNA terms: the challenge Terms of Use allow sharing with attribution, but the AWS listing forbids it. Plan §5.2 now prefers the challenge copy on Kaggle; the Kaggle rules are still unread. Earlier note that the licence bars sharing was incomplete. RSNA states 5 mm slices.
- No contract change. Mass effect, M2 dominance and extensive hypodensity stay out of the contract as named uncertainties, pending the freeze decision.
- Notion sync: done in this session (DEC-8, DEC-14, DEC-15, guideline task, RSNA tasks, Data guide).

## 2026-10-30 — Claude Code session (Notion sync and RSNA review)
- Notion synced by this session: DEC-1 Decided; DEC-7 to DEC-13 added as Decided; DEC-14 and DEC-15 added as Proposed. Nine tasks added, stale Phase 0 rows updated, Data guide and hub links updated. Plan §2 now matches.
- RSNA review: the AWS listing's licence bars sharing or redistributing the data, which affects third-party preprocessed copies and possibly the shared Drive. Source not chosen; plan §5.2 marks this and asks for whole-scan sampling.
- Notion sync: done in this session.

## 2026-10-30 — Claude Code session (git attribution)
- Parsa's rule: commits and PRs never credit Claude. Added to CLAUDE.md.
- Rewrote the three existing commits to remove the `Co-Authored-By` line and force-pushed `main` and `lead/phase0-scaffold`. File contents and dates are unchanged; commit hashes changed. Anyone with an earlier clone must re-clone or reset to `origin/main`.
- Notion sync: none pending.

## 2026-10-30 — Claude Code session (Phase 0 download notebook)
- Added `notebooks/00_download_isles24.ipynb`: downloads `train.7z` from Zenodo record 17652035 (99,022,114,670 bytes, md5 `4959a5dd2438d53e3c86d6858484e781`) to `raw/`. It resumes after interruptions, stops if the Drive folder is missing or too full, and keeps the file as `.part` until the md5 from Drive passes. It writes `raw/isles24_download.json` as the provenance record.
- The notebook is self-contained, an exception to "logic lives in src/": it must work before the repo is pushed, and it runs once. Plan §3 and the README table updated.
- Tested by running its real cells with a stubbed Drive against a local server: dropped connection and resume, rerun, recovery of a lost record, checksum mismatch, missing folder, changed Zenodo metadata. **Not yet run on Colab or against the full file**; only Zenodo metadata and a 1-byte range request were checked live.
- The download has not been started. The Phase 0 task stays open.
- Notion sync: Tasks → "Start ISLES'24 download" now points to this notebook. Once it has run, copy `isles24_download.json` into `docs/DATA.md`.

## 2026-10-30 — Claude Code session (Phase 0 architecture review)
- Plan v0.2. Professors approved the project (reported by Parsa), so DEC-1 is Decided. The Phase 0 confirmation task stays open until deliverables and the deadline are recorded.
- **Clinical rules:** the corpus baseline is now the 2026 AHA/ASA guideline, which replaced the 2018/2019 documents (DEC-8). §8.3 rulebook rewritten. The primary text returned 403, so rules come from secondary summaries and are marked for verification; single-source items and the numeric definition of salvageable penumbra are flagged as unconfirmed.
- **Contract (still draft v0.1):** added required, nullable `aspects` (DEC-9), in `schemas/vision_output.json`, the pydantic mirror and all fixtures. No ASPECTS model is planned.
- **`schemas/recommendation.json`:** a claim now has `sources` (one or more) and a required `derived` flag instead of a single `source` (DEC-10). §6.4 example updated.
- **Fixtures:** new expectations in §6.3 and `fixtures/README.md`. fx04 thrombectomy is now yes (large-core tier, ASPECTS 4), fx02 and fx05 thrombolysis are now borderline and must cite. Scores: fx01 9, fx02 8, fx03 null, fx04 4, fx05 null. All expectations still need primary-text and clinician review. Vision values and notes unchanged.
- **Data (§5):** ISLES'24 moves to Zenodo record 17652035, which includes the data dictionary (DEC-12). Recorded the licence, the mTICI 2c/3 cohort, tabular fields, ±5% noise on times and labs, and masks already in NCCT space. Slim format gains `ncct_native` and `cow`. The registration task became a spot-check; occlusion site now uses Circle of Willis labels.
- **Notes (§8.3):** treatment-revealing and post-treatment columns are forbidden, including door to groin and time of intervention.
- **Vision (§8):** the core that feeds the contract is chosen by comparison in Phase 3, because trial cut-offs were defined on perfusion core. Vision B runs on native-thickness NCCT.
- **Evaluation (§8.4, §9, DEC-11):** "exactly one source" replaced; added misapplied-rule and faulty-derivation sub-types; synthetic scenarios are the primary evaluation; added seeded errors, no-retrieval arm, ground-truth vs predicted vision arm, case-level CIs and recommendation correctness. New stubs: `nlp/verifier/scenarios.py`, `nlp/verifier/seeded_errors.py`.
- `tests/test_contract.py` updated and extended; all pass.
- Still open: full guideline text, clinician review, `mismatch_ratio` two decimals and `haemorrhage.confidence` = P(haemorrhage). The contract is **not** frozen.
- Notion sync:
  - Decisions → DEC-1 Decided. New rows: DEC-7 (`penumbra_ml`), DEC-8 (2026 guideline), DEC-9 (`aspects`), DEC-10 (multi-source claims, hallucination definition), DEC-11 (evaluation design), DEC-12 (Zenodo record). §5.4 slim format change needs a row too.
  - Tasks → new in Phase 0: verify rulebook against primary text; clinician review; confirm provisional contract choices and freeze. New in Phase 1: spot-check masks, tag clinical columns, check M1/M2 labels. New in Phase 2: scenarios, seeded errors.
  - Docs → update the ISLES'24 download source to record 17652035.

## 2026-10-29 — Claude Code session (Phase 0 scaffolding)
- Package skeleton per §3 under `src/strokecopilot/`: every module exists as a docstring stub naming its phase. `config.py` holds Drive paths and slim versions; `schemas/` and `fixtures/` resolve from the repo root (editable/source) or from inside the installed package (wheel).
- `pyproject.toml` (hatchling): deps `pydantic>=2.5`, `jsonschema>=4.18`; extra `dev` = pytest. The wheel ships `schemas/` and `fixtures/` so `pip install git+...` can validate on Colab without a checkout. Checked with a non-editable install into a fresh Python 3.11 venv, run from outside the repo.
- `schemas/vision_output.json` v0.1 (JSON Schema 2020-12) per §6.1, plus `contract/schema.py` (pydantic mirror) and `contract/validate.py` (JSON Schema first, then pydantic rules; raises `ContractError`). All fields required (null must be explicit); undeclared fields rejected; strict typing, so null is never coerced to 0.
- `contract.schema.derive_mismatch` is the single definition of the mismatch arithmetic; `vision/perfusion/volumes.py` should call it in Phase 2.
- Five fixtures + triage notes per §6.3, with `fixtures/README.md` holding the expected recommendations (kept out of the notes so they never reach the reasoner). fx01 has core 0.6 mL → `mismatch_ratio` null (null-handling case); fx03 has occlusion `unknown`; fx05 is a deliberate DEFUSE-3 near-miss (mismatch 14.5 mL) that passes DAWN.
- `schemas/recommendation.json` per §6.4. Added a `note` source type (`{"type": "note", "quote": ...}`), which §8.4(b) needs but §6.4's example does not show. Vision `field` is an enum of dotted vision_output paths.
- `tests/test_contract.py`: 59 tests, all passing — fixtures validate, derived fields, null handling, rejections, pydantic mirror in sync with the JSON Schema, recommendation schema.
- README.md with placeholder Open-in-Colab badges (`<org>` until the repo is pushed; notebooks do not exist yet, so the Phase 1 README task stays open). `data/README.md` added.
- `.gitignore` extended. The old `data/` pattern also matched `src/strokecopilot/data/`; patterns are now anchored to the repo root (`/data/*`, `!/data/README.md`, `/runs/`, `/artifacts/`).
- Decisions made here (contract-level, confirm before freezing v0.1):
  - `mismatch_ratio` rounded to **two** decimals and computed from the already-rounded volumes. §6.2 only fixes rounding for volumes; one decimal would push e.g. 1.76 over the DEFUSE-3 1.8 cut-off.
  - `haemorrhage.confidence` read as P(haemorrhage present), not confidence in the `present` call.
- **Decided by Parsa: `penumbra_ml` is the total Tmax > 6 s volume, core included.** §6.1 and §8.1 disagreed: §6.1's formulas (`mismatch_ratio = penumbra_ml / core_ml`, `mismatch_volume_ml = penumbra_ml - core_ml`) only match DEFUSE-3 under this reading, while §8.1 said "Tmax > 6 s minus core". §6.1 stands. Reworded §6.1 (meaning column), §8.1 (penumbra bullet), the §13 glossary, the `penumbra_ml` description in `schemas/vision_output.json`, and `fixtures/README.md`. No field, type or formula changed, and the fixtures are unchanged. fx05 therefore stays a DEFUSE-3 near-miss (mismatch 14.5 mL < 15 mL).
- Not done in this session: professor confirmation, Drive folder, downloads, licence notes, GitHub push.
- Notion sync:
  - Tasks → Done: "Scaffold repo", "Write vision_output schema + pydantic mirror", "Write five fixtures and notes", "tests/test_contract.py".
  - Decisions → new rows (Proposed): mismatch_ratio rounded to 2 dp; haemorrhage.confidence = P(haemorrhage); `note` source type added to recommendation.json.
  - Decisions → new row (Decided, Parsa): `penumbra_ml` = total Tmax > 6 s volume including core; salvageable tissue = `mismatch_volume_ml`. Plan §6.1/§8.1/§13 aligned.

## 2026-10-29 — Parsa
- Repo created. Added STROKE_COPILOT_PLAN.md v0.1 and CLAUDE.md.
- Notion hub created under Project HQ → Stroke Decision Copilot (tasks, decisions, experiments, docs).
- Notion sync: none pending.
