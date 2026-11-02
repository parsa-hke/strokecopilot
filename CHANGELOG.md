# CHANGELOG

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
