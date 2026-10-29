# STROKE_COPILOT_PLAN.md

**Stroke Decision Copilot — build plan and reference for every session**

MAAI 901 (Deep Learning for Visual Data) + MAAI 902 (NLP & LLMs) joint group project, MSc Applied AI, UOWD.
Team of five. Plan version 0.1, 29 October 2026.

This file is the single technical source of truth for the repo. The Notion hub (Project HQ → Stroke Decision Copilot) is the team coordination layer: tasks, decisions, experiments, meetings. When the two disagree, fix whichever one is wrong the same day and note it in CHANGELOG.md.

---

## 0. How to use this file

**If you are a Claude Code session:**

1. Read `CLAUDE.md`, then this file, then `CHANGELOG.md` — in that order, every session.
2. Find the current phase in §7. Work only on that phase's open tasks unless told otherwise.
3. Before writing code, re-read the component spec in §8 for the component you are touching and the contract in §6 if you touch anything that crosses the vision/NLP boundary.
4. At the end of the session: tick completed tasks here, append a dated entry to `CHANGELOG.md`, and list anything that should be mirrored to Notion (a new decision, a finished task, an experiment result) under a "Notion sync" heading in that entry.
5. Never silently change a fixed decision (§2). If one must change, stop, explain why, and wait for a human to record it in the Notions Decisions log.

**If you are a teammate:**

Read §1–§3 once, §5–§6 before touching data, §8 for your component, §11 for conventions. The rest is reference.

---

## 1. What we are building and why

### 1.1 The clinical problem

Acute ischaemic stroke: a clot blocks an artery in the brain and tissue starts dying within minutes. There are two main treatments.

- **Thrombolysis** (IV alteplase/tenecteplase): a clot-dissolving drug. Standard window is 4.5 hours from symptom onset. Absolutely contraindicated if there is an intracranial haemorrhage.
- **Thrombectomy**: a catheter physically removes the clot. Indicated for large-vessel occlusions; the window can extend to 24 hours *if imaging shows salvageable tissue* (DAWN and DEFUSE-3 trials).

Eligibility therefore depends on three things at once:

1. What the scan shows — how much tissue is already dead (**core**), how much is starving but salvageable (**penumbra**), whether there is a bleed, where the occlusion is.
2. Facts about the patient — time since onset (or last known well), age, NIHSS score, anticoagulant use, prior disability.
3. Published rules — the AHA/ASA guideline and the trial criteria.

Clinicians do this under time pressure. Our system does the imaging part with vision models, the reasoning part with a retrieval-grounded language model, and then audits the language model's output claim by claim.

### 1.2 The system in one diagram

```
          plain CT ──► [Vision B: haemorrhage?] ──┐
                                                  ├──► vision_output.json ──┐
   CT perfusion ──► [Vision A: core / penumbra] ──┘                         │
                                                                            ▼
   triage note (generated from clinical variables) ──────────────► [NLP A: reasoner] ──► recommendation + rationale
   guideline corpus (AHA/ASA, DAWN, DEFUSE-3) ──── retrieval ────┘                              │
                                                                                                ▼
                                                                  [NLP B: verifier] ──► every claim traced or flagged
```

### 1.3 The research claim

The verifier lets us measure, on a clinical decision task with known ground truth inputs, how often different language models assert things that are not in their inputs. The deliverable that matters for the report is a table: hallucination rate (and critical hallucination rate) per model, per clinical scenario, with confidence intervals.

Everything else — the segmentation Dice, the haemorrhage AUC, the demo — supports that table.

### 1.4 What "done" looks like

- A Colab notebook that takes an ISLES'24 case ID and a triage note and returns a recommendation where every claim is either linked to a vision field or a guideline passage, or flagged.
- Vision metrics reported per fold on ISLES'24 and on RSNA ICH.
- The hallucination comparison table across at least three language models.
- A report and presentation (format per professors — see §12).

---

## 2. Fixed decisions

These are recorded in the Notion Decisions log. Do not change them without a new Decisions row.

| # | Decision | Status |
|---|---|---|
| DEC-1 | Group project is the Stroke Decision Copilot, replacing Bias Lens | Proposed — pending written confirmation from both professors |
| DEC-2 | Datasets: ISLES'24 (Zenodo) for perfusion + infarct + clinical variables; RSNA Intracranial Haemorrhage (Kaggle) for bleed detection; AHA/ASA guideline + DAWN + DEFUSE-3 papers for the corpus | Decided |
| DEC-3 | Compute: Google Colab Pro. Storage: shared Google Drive folder `StrokeCopilot/`. Training data is copied to Colab local disk as one tarball per session — never read per-file through the Drive mount | Decided |
| DEC-4 | Contract-first: `schemas/vision_output.json` is frozen at the end of Phase 0. NLP builds against hand-written fixtures until Phase 3 | Decided |
| DEC-5 | Evaluation split is fixed on day one: 5 folds + ~20 held-out cases. Held-out cases are untouched until Phase 4 | Decided |
| DEC-6 | The lead builds Phase 0–1 alone, then onboards the team within 7–10 days of 29 Oct 2026 | Decided |

---

## 3. Repository layout

```
strokecopilot/
├── CLAUDE.md                      # session instructions for Claude Code (short; points here)
├── STROKE_COPILOT_PLAN.md         # this file
├── CHANGELOG.md                   # dated entries, newest first
├── README.md                      # for humans: what, why, how to run, Open-in-Colab badges
├── pyproject.toml                 # package metadata; pip install git+https://... works
├── .gitignore                     # data/, runs/, *.tar, secrets
│
├── src/strokecopilot/
│   ├── __init__.py
│   ├── config.py                  # paths, constants, Drive layout, slim version pins
│   ├── data/
│   │   ├── __init__.py
│   │   ├── isles24.py             # load_case(), list_cases(), slim format reader
│   │   ├── rsna.py                # RSNA slice/scan loaders, windowing
│   │   ├── prep_isles24.py        # raw archive → slim tarball (run once)
│   │   ├── prep_rsna.py           # raw/preprocessed → slim tarball (run once)
│   │   └── splits.py              # reads splits/folds.json; guards against held-out use
│   ├── vision/
│   │   ├── perfusion/             # Vision A
│   │   │   ├── thresholds.py      # rCBF<30% core, Tmax>6s penumbra baseline
│   │   │   ├── nnunet_wrapper.py  # dataset conversion, train, predict, resume
│   │   │   └── volumes.py         # mask → mL, mismatch ratio, occlusion site lookup
│   │   └── haemorrhage/           # Vision B
│   │       ├── model.py           # 2D classifier with pretrained backbone
│   │       ├── train.py
│   │       ├── aggregate.py       # slice → scan, calibration
│   │       └── infer.py
│   ├── contract/
│   │   ├── schema.py              # pydantic model of vision_output.json
│   │   ├── build.py               # assemble vision_output from A + B predictions
│   │   └── validate.py            # validate any JSON against schemas/vision_output.json
│   ├── nlp/
│   │   ├── corpus/                # NLP A
│   │   │   ├── ingest.py          # PDF → passages with stable IDs
│   │   │   ├── index.py           # embeddings + retrieval
│   │   │   └── passages.jsonl     # (generated) the indexed corpus
│   │   ├── notes/
│   │   │   └── generate.py        # clinical variables → triage note (templated)
│   │   ├── reasoner/
│   │   │   ├── prompt.py          # versioned prompts
│   │   │   ├── run.py             # inputs → structured recommendation
│   │   │   └── providers.py       # thin wrappers for each LLM API / local model
│   │   └── verifier/              # NLP B
│   │       ├── claims.py          # rationale → list of atomic claims
│   │       ├── trace.py           # claim → source (vision field / note / passage) or None
│   │       ├── metrics.py         # hallucination rate, critical rate, omission rate, CIs
│   │       └── adjudicate.py      # export samples for human labelling, agreement stats
│   ├── pipeline/
│   │   ├── end_to_end.py          # case_id + note → full output
│   │   └── report.py              # metrics.json writer, run folder conventions
│   └── demo/
│       └── app.py                 # Gradio app with clickable claims
│
├── schemas/
│   ├── vision_output.json         # THE contract (JSON Schema)
│   └── recommendation.json        # reasoner output schema
├── fixtures/
│   ├── fx01_thrombolysis_clear.json
│   ├── fx02_thrombectomy_large_mismatch.json
│   ├── fx03_haemorrhage_present.json
│   ├── fx04_large_core_no_mismatch.json
│   ├── fx05_borderline_onset.json
│   └── notes/                     # matching triage notes, one per fixture
├── splits/
│   └── folds.json                 # 5 folds + heldout list, generated once, committed
├── prompts/
│   ├── reasoner_v1.md
│   └── claim_extractor_v1.md
├── notebooks/
│   ├── 00_template.ipynb          # install, mount, copy tarball, load one case
│   ├── 01_prep_isles24.ipynb      # runs prep_isles24; produces slim tarball
│   ├── 02_prep_rsna.ipynb
│   ├── 10_visionA_nnunet.ipynb
│   ├── 11_visionA_thresholds.ipynb
│   ├── 20_visionB_train.ipynb
│   ├── 30_nlpA_corpus.ipynb
│   ├── 31_nlpA_reasoner.ipynb
│   ├── 40_nlpB_verifier.ipynb
│   ├── 41_nlpB_model_comparison.ipynb
│   └── 50_end_to_end.ipynb
├── tests/
│   ├── test_contract.py           # fixtures validate; volumes/ratios computed correctly
│   ├── test_splits.py             # no held-out leakage; folds disjoint
│   ├── test_claims.py             # claim extraction on known rationales
│   └── test_trace.py              # tracing on fixtures
├── data/
│   └── README.md                  # where data lives on Drive; nothing else committed here
└── docs/
    ├── DATA.md                    # exact file layout of ISLES'24 and RSNA as found (filled in Phase 1)
    ├── RULEBOOK.md                # human-readable summary of the clinical criteria with passage IDs
    └── REPORT_OUTLINE.md
```

Rules: logic lives in `src/`, notebooks are thin and call it. Nothing under `data/` or `runs/` is committed. Every notebook has an "Open in Colab" badge in the README.

---

## 4. Environment

### 4.1 Colab

- Runtime: GPU. A100 for training (Colab Pro), T4 is fine for `00_template` and NLP work.
- Sessions disconnect. Every training loop checkpoints to Drive every N steps and resumes from the latest checkpoint on restart. Treat every run as if it will be interrupted.
- Prefer models that finish a meaningful run inside one session: 2D/2.5D for haemorrhage, nnU-Net with a reduced epoch count for perfusion, resumed across sessions.

### 4.2 Drive layout

```
StrokeCopilot/
  raw/        ISLES'24 archive + md5, RSNA archive — untouched after download
  slim/       isles24_slim_v1.tar, rsna_slim_v1.tar — immutable; new version = new file
  runs/       EXP-<id>_<short-name>/  config.json, checkpoints/, metrics.json, logs/
  artifacts/  guidelines/ (PDFs), figures/ (exported for the report), corpus/ (passages.jsonl)
```

Teammates: add a shortcut to the shared folder in your own Drive so it mounts under `MyDrive/StrokeCopilot/`.

### 4.3 The one rule

Never train by reading individual files through the Drive mount. Every notebook starts with:

```python
from google.colab import drive
drive.mount('/content/drive')
SLIM = 'isles24_slim_v1.tar'
!cp /content/drive/MyDrive/StrokeCopilot/slim/{SLIM} /content/
!tar -xf /content/{SLIM} -C /content/
!pip install -q git+https://github.com/<org>/strokecopilot.git
```

Then everything reads from `/content/`.

### 4.4 Secrets

Kaggle token, Hugging Face token, LLM API keys: Colab Secrets panel only. The package reads them via `google.colab.userdata.get(...)` with a fallback to environment variables for local runs. Never in a notebook cell, never in the repo.

### 4.5 Local development (optional)

`pip install -e .` in a venv works for the NLP side and tests. Vision training is Colab-only.

---

## 5. Data

### 5.1 ISLES'24

- Source: Zenodo record 16748089, v2 (August 2025), open access. ~99 GB. 149 training cases from a multi-centre acute ischaemic stroke cohort. Cite the two preprints listed on the record (arXiv:2408.11142 and arXiv:2408.10966).
- Contents per case (to be confirmed against the archive in Phase 1 and recorded in `docs/DATA.md`): acute non-contrast CT, CT angiography with vessel-occlusion mask, CT perfusion (4D source and/or derived maps — CBF, CBV, MTT, Tmax), follow-up MRI with the delineated final infarct mask, and a row of clinical/demographic variables.
- What we use:
  - **Input to Vision A:** perfusion maps (and NCCT if useful).
  - **Target for Vision A:** final infarct mask on follow-up MRI, registered to CT space (check whether the archive already provides this registration; if not, it is a Phase 1 task).
  - **Occlusion site:** from the CTA mask, mapped to ICA / M1 / M2 / basilar / none / unknown.
  - **Clinical variables:** to generate triage notes (§8.3) and to stratify the split.
- Known issue: challenge test data is not public; our held-out set is carved from the 149 training cases.

### 5.2 RSNA Intracranial Haemorrhage

- Source: Kaggle competition dataset. Raw is ~180 GB of DICOM slices with per-slice labels (epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any).
- We do not download raw DICOM. Use a preprocessed PNG/NPZ version from Kaggle via the API, or subsample 10–20k slices with a balanced "any" label. Record exactly which in `docs/DATA.md`.
- Windowing: stack three CT windows as channels — brain (centre 40, width 80), subdural (80/200), bone (600/2800). This is the standard trick and matters for small bleeds.
- Scan-level labels: a scan is positive if any slice is positive. We need scan-level predictions for the contract.

### 5.3 Guideline corpus

- AHA/ASA acute ischaemic stroke guideline (latest edition at time of ingestion; record the year), DAWN trial paper, DEFUSE-3 trial paper. Add others only via a Decisions row.
- Chunked by criterion, not by page: each passage gets a stable ID like `AHA2019-3.5.2` or `DEFUSE3-inclusion-4`. The reasoner cites IDs; the verifier resolves them.
- Stored at `artifacts/corpus/passages.jsonl`; a human-readable summary with the same IDs lives in `docs/RULEBOOK.md`.

### 5.4 Slim format (v1)

One directory per case inside the tarball:

```
isles24_slim_v1/
  manifest.json              # version, case list, voxel spacing, which maps are present
  clinical.csv               # one row per case
  sub-strokeXXXX/
    ncct.nii.gz              # resampled to common grid
    cbf.nii.gz cbv.nii.gz mtt.nii.gz tmax.nii.gz
    cta_occlusion.nii.gz
    infarct_followup.nii.gz  # target, in CT space
```

Resample everything to one isotropic grid (start with 1.5 mm; decide in Phase 1 and record it in `manifest.json`). Crop to the brain bounding box to save space. Target size: 10–20 GB.

Slim tarballs are immutable. A preprocessing change produces `_v2`, never an overwrite. Every experiment records which version it used.

### 5.5 Splits

`splits/folds.json`:

```json
{
  "version": 1,
  "stratified_by": "infarct_volume_tertile",
  "heldout": ["sub-stroke0007", "..."],
  "folds": {"0": ["..."], "1": ["..."], "2": ["..."], "3": ["..."], "4": ["..."]}
}
```

- ~20 held-out cases, chosen once with a fixed seed, stratified by final infarct volume.
- 5 folds over the remaining ~129.
- `splits.py` raises if any training function is handed a held-out ID before `ALLOW_HELDOUT=1` is set, which only happens in Phase 4 notebooks.

---

## 6. The contract: `schemas/vision_output.json`

This is the only thing the vision side hands to the NLP side. Frozen at end of Phase 0. Changes after that need a Decisions row and a version bump.

### 6.1 Fields (v0.1 — draft until Phase 0 closes)

| Field | Type | Meaning | Producer |
|---|---|---|---|
| `schema_version` | string | `"0.1"` | pipeline |
| `case_id` | string | ISLES'24 subject ID | pipeline |
| `model_version` | string | git tag or commit of the vision models | pipeline |
| `haemorrhage.present` | bool | any intracranial haemorrhage on plain CT | Vision B |
| `haemorrhage.confidence` | float 0–1 | calibrated probability | Vision B |
| `core_ml` | float \| null | infarct core volume, mL, one decimal | Vision A |
| `penumbra_ml` | float \| null | total hypoperfused volume (Tmax > 6 s), mL, one decimal — **includes the core**; salvageable tissue is `mismatch_volume_ml` | Vision A |
| `mismatch_ratio` | float \| null | `penumbra_ml / core_ml`; null if core < 1 mL | Vision A |
| `mismatch_volume_ml` | float \| null | `penumbra_ml - core_ml` | Vision A |
| `occlusion_site` | enum | `ICA`, `M1`, `M2`, `basilar`, `none`, `unknown` | Vision A |
| `confidence.core` | float 0–1 | model confidence | Vision A |
| `confidence.penumbra` | float 0–1 | model confidence | Vision A |
| `warnings` | string[] | e.g. `"motion artefact"`, `"incomplete perfusion coverage"` | pipeline |

### 6.2 Rules

- Volumes in millilitres, rounded to one decimal. The reasoner must quote them exactly; the verifier compares after rounding.
- `null` means not computed. Zero is `0.0`. The reasoner must treat `null` as "unknown", never as zero.
- Fields may be added in a minor version. Existing fields are never renamed or removed before the end of the project.
- `contract/validate.py` validates every JSON against the schema; the pipeline refuses to pass an invalid object to the reasoner.

### 6.3 Fixtures

Five hand-written outputs in `fixtures/`, each with a matching triage note in `fixtures/notes/`. They are the NLP side's test bed until Phase 3 and remain the regression set afterwards.

| Fixture | Scenario | Expected recommendation (to be confirmed against the corpus) |
|---|---|---|
| fx01 | 2.5 h from onset, NIHSS 8, no bleed, small core, M2 occlusion | thrombolysis: yes; thrombectomy: consider |
| fx02 | 9 h from LKW, NIHSS 16, no bleed, core 18 mL, penumbra 95 mL, M1 | thrombolysis: no (window); thrombectomy: yes (DEFUSE-3 mismatch) |
| fx03 | 2 h, NIHSS 12, haemorrhage present | thrombolysis: no (absolute contraindication); thrombectomy: no |
| fx04 | 6 h, NIHSS 18, core 110 mL, penumbra 120 mL, M1 | thrombolysis: no; thrombectomy: no (core too large) |
| fx05 | wake-up stroke, LKW 10 h ago, NIHSS 11, age 82, core 15 mL, M1 | thrombolysis: no; thrombectomy: DAWN criteria — borderline, must cite |

### 6.4 Reasoner output: `schemas/recommendation.json`

```json
{
  "case_id": "...",
  "model": "gpt-4o-2024-08-06",
  "prompt_version": "reasoner_v1",
  "thrombolysis": {"recommend": true, "rule_ids": ["AHA2019-3.5.1"], "blocking_factors": []},
  "thrombectomy": {"recommend": false, "rule_ids": ["DEFUSE3-inclusion-2"], "blocking_factors": ["core_ml above 70"]},
  "rationale": [
    {"claim": "Infarct core volume is 18.0 mL.", "source": {"type": "vision", "field": "core_ml"}},
    {"claim": "DEFUSE-3 requires core volume under 70 mL.", "source": {"type": "passage", "id": "DEFUSE3-inclusion-2"}}
  ],
  "uncertainties": ["onset time is last-known-well, not witnessed"]
}
```

The reasoner is asked to produce this structure directly. The verifier checks the `source` the model claimed *and* independently re-traces each claim; disagreement between the two is itself a recorded error type.

---

## 7. Phases

Weeks are relative to the kickoff (week 1 starts when the team is onboarded). Deadline: _fill in once professors confirm_. If the deadline is shorter than 8 weeks, cut Phase 3 scope first (fewer LLMs, no occlusion-site model), never Phase 4.

### Phase 0 — Setup (this week; lead alone)

Goal: decisions recorded, contract frozen, data downloading, repo scaffolded.

- [ ] Get written confirmation from both professors for the switch (DEC-1 → Decided); agree deliverables (§12)
- [ ] Create shared Drive folder with the layout in §4.2; share with team
- [ ] Start ISLES'24 download to `raw/` from a Colab session; verify md5; record sizes
- [ ] Start RSNA preprocessed download to `raw/`
- [ ] Read Zenodo licence and RSNA terms; fill `docs/` note and the Notion "Data-use terms" page
- [x] Scaffold repo per §3: package skeleton, `pyproject.toml`, `CLAUDE.md`, `CHANGELOG.md`, `README.md`, `.gitignore`
- [x] Write `schemas/vision_output.json` (§6.1) and `contract/schema.py` (pydantic mirror)
- [x] Write the five fixtures and their notes (§6.3)
- [x] `tests/test_contract.py`: fixtures validate; mismatch ratio and volume computed correctly; null handling
- [ ] Push to GitHub (private), add repo URL to Notion hub Links

Acceptance: `pytest tests/test_contract.py` passes; the five fixtures validate; both downloads are complete or in progress with md5 checks queued.

### Phase 1 — Data (week 1; lead alone, then onboard)

Goal: slim tarballs on Drive; Colab template runs end to end on one case; team onboarded.

- [ ] Inspect ISLES'24 archive layout; write `docs/DATA.md` with exact paths and which perfusion maps exist
- [ ] Decide the common grid (spacing, crop) → record in `manifest.json` and CHANGELOG
- [ ] Check whether follow-up infarct masks are already in CT space; if not, implement registration (ANTs/SimpleITK) in `prep_isles24.py` and note its cost
- [ ] `prep_isles24.py` + `01_prep_isles24.ipynb` → `slim/isles24_slim_v1.tar`
- [ ] `prep_rsna.py` + `02_prep_rsna.ipynb` → `slim/rsna_slim_v1.tar` (windowed 3-channel slices, scan-level index)
- [ ] `splits.py` + `splits/folds.json` generated with fixed seed; `tests/test_splits.py`
- [ ] `isles24.py`: `load_case(id)` returns arrays + clinical row + spacing; `rsna.py` equivalents
- [ ] `00_template.ipynb`: install, mount, copy tarball, load one case, show maps + mask + clinical row. Must run top to bottom in five minutes on a T4
- [ ] README with Open-in-Colab badges
- [ ] Onboard the team: share repo + Notion; fill the Person column; everyone runs `00_template`
- [ ] Assign Phase 2 tasks per role

Acceptance: every teammate has run `00_template` successfully and screenshotted the output into the Notion onboarding task.

### Phase 2 — Baselines (weeks 1–2; whole team)

Goal: every box in the diagram does *something*, measured, even if badly.

**Vision A (perfusion)**
- [ ] `thresholds.py`: core = rCBF < 30% of contralateral, penumbra = Tmax > 6 s; compute volumes; evaluate Dice of thresholded core vs final infarct per fold — this is the clinical-standard baseline
- [ ] nnU-Net v2 dataset conversion (perfusion maps as channels, final infarct as label); train fold 0 with reduced epochs; checkpoint/resume to Drive; report Dice
- [ ] `volumes.py`: mask → mL using spacing; mismatch ratio/volume; occlusion-site lookup from CTA mask (rule-based for now)

**Vision B (haemorrhage)**
- [ ] 2D classifier (pretrained ResNet-50 or EfficientNet-B3) on windowed slices, "any" label; slice AUC
- [ ] `aggregate.py`: slice → scan (max and top-k mean); scan AUC; temperature scaling for calibration; reliability diagram
- [ ] `infer.py`: scan → `{present, confidence}`

**NLP A (reasoner)**
- [ ] `ingest.py`: guideline PDF + two trial papers → passages with stable IDs; `docs/RULEBOOK.md` generated from the same source
- [ ] `index.py`: embed passages; retrieve top-k by query; sanity-check that each fixture's expected rule IDs are retrievable
- [ ] `notes/generate.py`: clinical row → triage note (templated with controlled variation); produce notes for all 149 cases
- [ ] `reasoner/run.py` v0 with `prompts/reasoner_v1.md`: fixture JSON + note + retrieved passages → `recommendation.json`; run on all five fixtures with one model

**NLP B (verifier & eval)**
- [ ] Write the hallucination metric definition (§8.4) in `docs/` and Notion — before code
- [ ] `claims.py`: rationale → atomic claims (LLM-assisted with `prompts/claim_extractor_v1.md`, plus a rule-based numeric extractor)
- [ ] `trace.py`: numeric claims vs vision JSON (after rounding); patient facts vs note; rule claims vs passage text (embedding similarity + NLI or LLM judge — pick one, record it)
- [ ] `metrics.py`: hallucination rate, critical hallucination rate, omission rate; bootstrap CIs
- [ ] Run verifier on the five fixture outputs; `tests/test_claims.py`, `tests/test_trace.py`

**Everyone**
- [ ] Every run above has an Experiments row and a `runs/EXP-*` folder with `metrics.json`

Acceptance: a single notebook (`50_end_to_end.ipynb`, fixture mode) runs fixture → reasoner → verifier and prints a hallucination rate. Vision baselines have fold-0 numbers in Experiments.

### Phase 3 — Build (weeks 3–5)

Goal: each component made good; real vision outputs replace fixtures; the multi-LLM table exists.

**Vision A**
- [ ] All 5 folds for nnU-Net; mean ± std Dice; compare against threshold baseline
- [ ] Core/penumbra heads or a two-stage approach (predict final infarct = core proxy; penumbra from Tmax) — decide and record
- [ ] Confidence estimates (`confidence.core/penumbra`) from fold ensemble disagreement or MC dropout
- [ ] Error analysis: where does the model fail (small lesions, posterior circulation, artefacts)?
- [ ] `contract/build.py` produces valid `vision_output.json` for every non-held-out case; validate all

**Vision B**
- [ ] Tune aggregation and threshold for high sensitivity at acceptable specificity (false "no bleed" is the worst failure in the system — state the operating point and why)
- [ ] Domain shift check: run on ISLES'24 NCCT (which should mostly be negative); report the false-positive rate
- [ ] Optional: per-subtype outputs if time allows

**NLP A**
- [ ] Switch reasoner inputs from fixtures to real `vision_output.json` + generated notes
- [ ] Prompt v2 informed by Phase 2 verifier findings (what did the model get wrong?)
- [ ] `providers.py`: at least three models — one frontier API model, one open mid-size model (e.g. Llama-class via HF or API), one small medical model. Exact list → Decisions row
- [ ] Run all three on all fixtures and on a sample of real cases; outputs to `runs/`

**NLP B**
- [ ] `41_nlpB_model_comparison.ipynb`: the hallucination table — per model, per scenario, with CIs
- [ ] Human adjudication: two team members label ~100 sampled claims; report inter-annotator agreement and verifier-vs-human agreement
- [ ] Error taxonomy with examples: wrong number, invented finding, wrong rule, uncited rule, mis-attributed source, omission

Acceptance: the hallucination table exists with ≥3 models and CIs; vision outputs validate for all training cases; adjudication agreement reported.

### Phase 4 — Integration and evaluation (weeks 6–7)

Goal: end-to-end on held-out cases; error propagation analysis; demo.

- [ ] Unlock held-out cases (`ALLOW_HELDOUT=1` in Phase 4 notebooks only); never retrain after this
- [ ] `pipeline/end_to_end.py` on all ~20 held-out cases × 3 models
- [ ] Error propagation: for each wrong recommendation, was the cause vision error, retrieval error, reasoning error, or verifier miss? Table + examples
- [ ] Sensitivity analysis: perturb `core_ml` by ±20% and see which recommendations flip — shows how much vision accuracy matters clinically
- [ ] `demo/app.py`: pick a case → show maps + masks → show note → show recommendation with each claim clickable, linking to the vision field or the passage; flagged claims in red; "research prototype, not for clinical use" banner on every screen
- [ ] Export figures to `artifacts/figures/`

Acceptance: held-out results in Experiments; demo runs from a fresh Colab session in under 10 minutes.

### Phase 5 — Report and presentation (week 8)

- [ ] `docs/REPORT_OUTLINE.md` agreed by week 6; one owner per section
- [ ] Report written against the outline (§12)
- [ ] Presentation deck; rehearsed once with the demo
- [ ] Repo cleaned: README accurate, all notebooks run, CHANGELOG complete
- [ ] Final Notion sync: all tasks closed, all decisions final, experiments complete

---

## 8. Component specifications

### 8.1 Vision A — perfusion → core / penumbra / final infarct

**Inputs.** CT perfusion maps (CBF, CBV, MTT, Tmax) resampled to the common grid; optionally NCCT as an extra channel.

**Targets.** Final infarct mask from follow-up MRI (ISLES'24 ground truth). Note the conceptual gap: final infarct ≠ core at imaging time — it includes penumbra that was not saved. Report this limitation and handle it as follows:
- *Core proxy:* model trained to predict final infarct, evaluated as the "core" estimate. Also report the clinical threshold baseline (rCBF < 30%).
- *Penumbra:* `penumbra_ml` is the total Tmax > 6 s volume, core included — the clinical convention (DEFUSE-3 used RAPID with these thresholds), so `mismatch_ratio` and `mismatch_volume_ml` (§6.1) match the DEFUSE-3 definitions directly. Salvageable tissue is `mismatch_volume_ml`. Document this as a rule, not a learned output, unless a learned version clearly beats it.

**Model.** nnU-Net v2, 3D full-resolution configuration, reduced epochs to fit Colab sessions, resume from Drive. Do not build a custom U-Net in Phase 2. Consider a 2.5D variant only if 3D training is infeasible on Colab.

**Outputs to the contract.** `core_ml`, `penumbra_ml`, `mismatch_ratio`, `mismatch_volume_ml`, `occlusion_site`, `confidence.*`, relevant `warnings`.

**Metrics.** Dice and absolute volume error (mL) per fold, mean ± std; volume error matters more than Dice for the downstream decision, because the rules are volume thresholds. Also report how often the predicted volume crosses a decision threshold (70 mL, DAWN cut-offs) when the ground truth does not.

**Occlusion site.** Rule-based from the CTA occlusion mask location relative to a vascular atlas or simple anatomical heuristics; `unknown` if the mask is empty or ambiguous. A learned classifier is out of scope unless time allows.

### 8.2 Vision B — haemorrhage detection

**Inputs.** 2D CT slices, three windows stacked as channels (§5.2). Train on RSNA ICH. The ISLES'24 NCCT is the deployment domain and is mostly negative — use it for a false-positive check only.

**Model.** Pretrained ResNet-50 or EfficientNet-B3, binary "any haemorrhage". Slice-level training with standard augmentation (flips, small rotations, brightness). Class balance via sampling.

**Aggregation.** Slice probabilities → scan score: max, and mean of top-k; pick by validation. Temperature scaling for calibration; report a reliability diagram.

**Operating point.** Choose for high sensitivity. State the chosen threshold, the sensitivity/specificity at that point, and the reasoning in the report. The contract carries `confidence` so the reasoner can express uncertainty rather than a hard yes/no.

**Metrics.** Slice AUC, scan AUC, sensitivity at the chosen operating point, expected calibration error, false-positive rate on ISLES'24 NCCT.

### 8.3 NLP A — corpus, notes, reasoner

**Corpus ingestion.** PDFs → text → passages split at criterion boundaries (not fixed token windows). Each passage: `{id, source, section, text, page}`. IDs are stable strings that survive re-ingestion. `docs/RULEBOOK.md` is generated from the same records so humans and the model read the same thing.

**Retrieval.** Embedding index (a sentence-embedding model from HF; record which). Query = structured summary of the vision JSON + note. Return top-k passages with IDs. Also always include a small fixed set of "must-have" passages (thrombolysis window, haemorrhage contraindication, DEFUSE-3 and DAWN inclusion criteria) so the reasoner can never claim it was not shown the key rule.

**Triage note generation.** `notes/generate.py` turns the ISLES'24 clinical row into a short ED-style note: age, sex, last-known-well or witnessed onset time, NIHSS, anticoagulation if present, pre-stroke mRS if present, relevant comorbidities. Templated with a few phrasings per field so the reasoner does not overfit to one format. The note is an *input*; we are not learning from it. Every fact in the note is traceable back to a clinical column, which is what lets the verifier check patient-fact claims.

**Reasoner.** Prompt (`prompts/reasoner_vN.md`) instructs the model to:
1. Read the vision JSON, the note, and the retrieved passages.
2. Decide thrombolysis and thrombectomy separately.
3. Return `recommendation.json` (§6.4) with every rationale claim carrying a `source`.
4. Quote numbers exactly as given; treat `null` as unknown; list uncertainties.

Providers: thin wrappers with identical signatures for each model; temperature 0; log the full prompt and response to the run folder.

**Clinical rulebook summary (for orientation only — the corpus is the source of truth; verify every item against the primary documents during ingestion):**
- Thrombolysis: within 4.5 h of onset/LKW; absolute contraindication if intracranial haemorrhage; other contraindications include recent anticoagulant use with elevated INR/DOAC timing, recent major surgery, very large established infarct.
- Thrombectomy, 0–6 h: large-vessel occlusion (ICA/M1 ± M2), NIHSS ≥ 6, small core (ASPECTS ≥ 6 on NCCT).
- Thrombectomy, 6–16 h (DEFUSE-3): core < 70 mL, mismatch ratio ≥ 1.8, mismatch volume ≥ 15 mL.
- Thrombectomy, 6–24 h (DAWN): clinical–core mismatch by age/NIHSS/core volume (e.g. age ≥ 80: NIHSS ≥ 10 and core < 21 mL; age < 80: NIHSS ≥ 10 and core < 31 mL, or NIHSS ≥ 20 and core 31–51 mL).

### 8.4 NLP B — verifier and evaluation

**Claim.** An atomic assertion in the rationale about a number, a clinical finding, a patient fact, or a rule. Extracted by an LLM with `prompts/claim_extractor_v1.md`, cross-checked by a regex extractor for numbers and units. Hedges, restatements and formatting are not claims.

**Traceable.** A claim is traceable if it is supported by exactly one of:
- (a) a field in `vision_output.json`, matching after rounding to one decimal;
- (b) a statement in the triage note (string/number match against the generating clinical row);
- (c) a passage in the corpus — checked by embedding similarity above a threshold *and* an entailment judgement (NLI model or LLM judge; pick one in Phase 2 and record it).

**Hallucination.** A claim that is not traceable. Sub-types: wrong number, invented finding, wrong rule (cited passage does not say that), uncited rule (true rule, no source), mis-attributed source (claim is true but the model's declared source is wrong).

**Omission.** An input fact that should have changed the recommendation and was ignored (haemorrhage present but thrombolysis recommended; core > 70 mL but DEFUSE-3 thrombectomy recommended). Detected by rule checks on the recommendation, independent of the rationale.

**Critical hallucination.** A hallucination that, if believed, would change a treatment decision. Determined by re-running the rule checks with the hallucinated value substituted.

**Metrics.** Per model, per scenario: hallucination rate = hallucinated claims / total claims; critical hallucination rate; omission rate; source-attribution accuracy. Bootstrap 95% CIs over claims and over cases. Human adjudication on a sample: inter-annotator agreement (Cohen's κ) and verifier-vs-human agreement.

**Models compared.** Minimum three: one frontier API model, one open mid-size model, one small medical model. Same prompt version, same retrieval, same inputs, temperature 0. Exact list → Decisions row.

### 8.5 Demo

Gradio app in `demo/app.py`, runnable from Colab. Case picker → maps and masks → generated note → recommendation → rationale list with each claim as a button that opens its source (vision field highlighted in the JSON, passage shown with ID) or shows the flag reason. Persistent banner: *Research prototype — not for clinical use.* No patient-identifying content exists in the datasets, but the banner is non-negotiable.

---

## 9. Evaluation plan (summary)

| What | Data | Metric | Reported as |
|---|---|---|---|
| Core/final infarct segmentation | ISLES'24, 5 folds | Dice, volume error (mL), threshold-crossing rate | mean ± std over folds |
| Threshold baseline | same | same | same; compared to learned model |
| Haemorrhage detection | RSNA ICH val; ISLES'24 NCCT | slice/scan AUC, sensitivity at operating point, ECE, FPR on ISLES | single table |
| Retrieval | fixtures + sampled cases | recall of expected rule IDs @k | single number per k |
| Reasoner + verifier | fixtures; sampled train cases; held-out cases | hallucination rate, critical rate, omission rate, attribution accuracy | per model × scenario, with CIs |
| Verifier validity | ~100 adjudicated claims | κ (human–human), agreement (verifier–human) | two numbers |
| End-to-end | ~20 held-out cases × 3 models | recommendation correctness vs rule-derived ground truth; error-propagation table | table + examples |
| Sensitivity | held-out | % recommendations flipped under ±20% core perturbation | single figure |

Ground truth for "correct recommendation" is derived mechanically from the ground-truth masks + clinical row + the rulebook, so it is reproducible and does not depend on anyone's clinical opinion. Say so in the report.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Professors do not approve the switch | low–medium | Ask this week with the charter page; keep Bias Lens materials until confirmed |
| ISLES'24 follow-up masks not in CT space; registration is hard | medium | Check in Phase 1 day 1; use ANTs rigid+affine; if it fails, train on perfusion-space labels provided by the challenge if available |
| nnU-Net too slow on Colab | medium | Reduced epochs, resume across sessions, 2.5D fallback; the threshold baseline is always available |
| 149 cases is small; noisy Dice | high | Report CIs; lean on volume error; cross-validation; never claim SOTA |
| Guideline PDFs hard to parse | medium | Manual cleanup of the ~50 key passages is acceptable; record it |
| Verifier itself hallucinates (LLM judge) | medium | Human adjudication sample; rule-based numeric checks that do not use an LLM; report verifier agreement |
| LLM API cost | low | Temperature 0, cached prompts, small sample of real cases; one API model only |
| Team member drops or is unavailable | medium | Every component has a baseline another person can finish; contract isolates failures |
| Scope creep (occlusion classifier, per-subtype bleeds, patient explainer) | high | Anything not in §7 needs a Decisions row; cut Phase 3 extras before touching Phase 4 |

---

## 11. Conventions

**Git.** `main` is always runnable. Branch per task: `<role>/<short-name>` (e.g. `visionA/nnunet-fold0`). PR to `main`; one teammate reviews. Squash merge. Commit messages: imperative, one line, optional body.

**Notebooks.** Thin. Import from the package, call functions, display results. If a cell is more than ~20 lines of logic, it belongs in `src/`. Clear outputs before committing, except for `00_template` which keeps its outputs as documentation.

**Experiments.** Before a run: Experiments row (Planned) + `runs/EXP-<id>_<name>/config.json`. After: `metrics.json` (every metric, fold, data version, git commit) + row updated. A number that is not in Experiments does not exist for the report.

**Prompts.** Versioned files in `prompts/`. Never edit a version in place; create `_v2`. Every run records the version.

**CHANGELOG.md.** Newest first. Each entry: date, who/which session, what changed, what was decided, "Notion sync" list.

**Decisions.** Anything that changes §2, §5.4, §6, the phase scope, or the model list → Notion Decisions row first, then this file, then code.

**Tests.** `pytest` runs in under a minute without data: contract, splits, claims, trace on fixtures. Data-dependent checks live in notebooks.

**Language.** British spelling in docs and the report (haemorrhage, ischaemic), to match the clinical literature we cite. Code identifiers: whatever is shortest and consistent (`haemorrhage` throughout).

---

## 12. Report outline (draft — finalise by week 6)

Format, length and dates: _fill in once professors confirm._

1. Introduction — the clinical problem, why vision + NLP, the research claim
2. Related work — perfusion-based infarct prediction, haemorrhage detection, clinical LLMs and hallucination, grounded generation / fact-checking
3. Data — ISLES'24, RSNA ICH, corpus; preprocessing; splits; data-use terms
4. System — architecture, the contract, each component (§8)
5. Experiments — vision results, retrieval, hallucination table, adjudication, end-to-end, sensitivity
6. Discussion — error propagation, what the hallucination table says, limitations (final infarct ≠ core, small N, synthetic notes, verifier validity)
7. Ethics and safety — prototype only; de-identified data; failure modes that would matter clinically
8. Conclusion and future work
9. Contributions — who did what (per role)
Appendices: contract schema, prompts, rulebook with passage IDs, fixture definitions

---

## 13. Glossary

- **Ischaemic stroke** — artery blocked by a clot; tissue starves. **Haemorrhagic stroke** — bleeding into the brain. The first question on any stroke CT is which one, because the treatments are opposite.
- **NCCT** — non-contrast CT, the fast first scan. **CTA** — CT angiography, shows the vessels and the occlusion. **CTP** — CT perfusion, dye tracked over time; yields CBF (blood flow), CBV (blood volume), MTT (mean transit time), Tmax (time to peak of the residue function — delay).
- **Core** — tissue already infarcted; conventionally rCBF < 30% of the healthy side. **Penumbra** — hypoperfused but salvageable; conventionally Tmax > 6 s. In the contract, `penumbra_ml` is the full Tmax > 6 s volume including the core (§6.1). **Mismatch** — penumbra relative to core; large mismatch means surgery can still help.
- **LVO** — large-vessel occlusion (ICA, M1, proximal M2, basilar). Thrombectomy target.
- **NIHSS** — 0–42 stroke severity score. **mRS** — 0–6 disability scale; pre-stroke mRS matters for eligibility.
- **LKW** — last known well; the clock starts here when onset is unwitnessed (wake-up strokes).
- **ASPECTS** — 0–10 score of early ischaemic change on NCCT; ≥ 6 usually required for early thrombectomy.
- **DWI/FLAIR** — MRI sequences; DWI shows acute infarct within minutes, FLAIR within hours. ISLES'24 follow-up masks are drawn on MRI.
- **Dice** — overlap between predicted and true masks, 0–1.
- **Hallucination (ours)** — a rationale claim not traceable to the vision JSON, the note, or a corpus passage. See §8.4 for the precise definition.

---

## 14. Session checklist (copy into CLAUDE.md)

Start of session:
- [ ] Read `CLAUDE.md`, `STROKE_COPILOT_PLAN.md` §0, §2, current phase in §7, relevant §8 spec, and the latest `CHANGELOG.md` entry
- [ ] State which phase and which tasks this session will work on

During:
- [ ] Logic in `src/`, notebooks thin
- [ ] No held-out IDs before Phase 4
- [ ] No contract changes without stopping to ask
- [ ] Every run → `runs/EXP-*` folder with `config.json` and `metrics.json`

End of session:
- [ ] Tests pass (`pytest`)
- [ ] Tasks ticked in §7
- [ ] `CHANGELOG.md` entry with a "Notion sync" list
- [ ] Nothing under `data/` or `runs/` staged for commit; no secrets anywhere
