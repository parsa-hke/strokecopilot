# Stroke Decision Copilot

MAAI 901 (Deep Learning for Visual Data) + MAAI 902 (NLP & LLMs) group project, MSc Applied AI, UOWD.

Vision models read the stroke CT (haemorrhage on plain CT; core and penumbra on CT perfusion), a language model reasons over those findings, a triage note and the AHA/ASA guideline plus the DAWN and DEFUSE-3 trials, and a verifier checks every claim in its rationale against its sources. The research output is a hallucination-rate comparison across language models on a clinical decision task with known inputs.

**Research prototype — not for clinical use.**

The full plan is in [STROKE_COPILOT_PLAN.md](STROKE_COPILOT_PLAN.md); progress is in [CHANGELOG.md](CHANGELOG.md).

## How it fits together

```
plain CT ──► Vision B (haemorrhage) ──┐
                                      ├──► vision_output.json ──┐
CT perfusion ──► Vision A (core/penumbra) ┘                      ▼
triage note ─────────────────────────────────────────► NLP A: reasoner ──► recommendation.json
guideline rulebook (whole) ───────────────────────────┘                          │
                                                                                 ▼
                                                           NLP B: verifier ──► every claim traced or flagged
```

The vision and NLP sides meet only at [`schemas/vision_output.json`](schemas/vision_output.json). Until real vision outputs exist, the NLP side builds against the five hand-written cases in [`fixtures/`](fixtures/).

## Install

On Colab (or anywhere):

```bash
pip install git+https://github.com/parsa-hke/strokecopilot.git
```

For local development (NLP side and tests; vision training is Colab-only):

```bash
pip install -e ".[dev]"
pytest
```

Validate a vision output:

```python
from strokecopilot.contract import load_vision_output
out = load_vision_output("fixtures/fx02_thrombectomy_large_mismatch.json")
```

## Data

Nothing under `data/` or `runs/` is committed. Datasets live in the shared Google Drive folder `StrokeCopilot/`; see [data/README.md](data/README.md).

## Notebooks

Badges point at the notebooks once they exist.

| Notebook | Purpose | |
|---|---|---|
| `00_download_isles24` | Phase 0: download ISLES'24 to Drive and verify md5 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/00_download_isles24.ipynb) |
| `00_template` | Install, mount Drive, copy tarball, load one case | Not written yet; comes when the first teammate is brought in |
| `01a_list_isles24` | Phase 1: list the ISLES'24 archive without extracting it | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/01a_list_isles24.ipynb) |
| `01b_clinical_isles24` | Phase 1: survey the ISLES'24 clinical columns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/01b_clinical_isles24.ipynb) |
| `01c_unpack_split_survey` | Phase 1: working copy, fixed split, spot-check and grid evidence | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/01c_unpack_split_survey.ipynb) |
| `01_prep_isles24` | ISLES'24 raw → slim tarball | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/01_prep_isles24.ipynb) |
| `01d_check_slim` | Phase 1: read the tarball through the loader; label, MTT and Tmax questions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/01d_check_slim.ipynb) |
| `02_prep_rsna` | RSNA ICH → 320 px JPEG tar shards (runs on Kaggle) | Upload to Kaggle; this only runs there |
| `02a_rsna_kaggle_timing` | Phase 0: time full RSNA on Kaggle | Upload to Kaggle; this one does not run on Colab |
| `02b_rsna_read_speed` | Phase 1: check the RSNA shards and measure read speed on Kaggle | Upload to Kaggle; this only runs there |
| `02c_rsna_repack` | Phase 1: store an RSNA shard as one file on Kaggle | Upload to Kaggle; this only runs there |
| `10_visionA_nnunet` | Phase 2: nnU-Net v2 fold 0; dummy mode (EXP-2) checks timing and resume first, train mode is EXP-3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/10_visionA_nnunet.ipynb) |
| `11_visionA_thresholds` | Phase 2: threshold baseline (EXP-1) and occlusion-to-label distances | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/11_visionA_thresholds.ipynb) |
| `11b_visionA_threshold_check` | Phase 2: checks the low EXP-1 Dice (infarct vs Tmax > 6 s, rCBF, MCA overlap) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/11b_visionA_threshold_check.ipynb) |
| `20_visionB_train` | Phase 2: RSNA patient split and Vision B slice classifier with scan calibration; dummy mode (EXP-5) measures speed first | Upload to Kaggle; this only runs there |
| `30_nlpA_corpus` | NLP A: corpus ingestion and retrieval | Not needed: the corpus is built in the repo (`python -m strokecopilot.nlp.corpus.ingest build`, DEC-18) and sent whole (DEC-19) |
| `31_nlpA_reasoner` | Phase 2: reasoner on the five fixtures; dummy mode (EXP-8) checks schema, caching and cost first, run mode is EXP-9 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/31_nlpA_reasoner.ipynb) |
| `40_nlpB_verifier` | Phase 2: verifier on the seeded errors; judge bake-off (Haiku vs Opus) and τ; dummy mode (EXP-10) first, run mode is EXP-11 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/40_nlpB_verifier.ipynb) |
| `41_nlpB_model_comparison` | NLP B: hallucination table | Not written yet (Phase 2 or later) |
| `50_end_to_end` | Phase 2 acceptance: fixture → reasoner → verifier → hallucination rate (EXP-12); real cases in Phase 4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/50_end_to_end.ipynb) |

## Repository layout

`src/strokecopilot/` holds all logic (`data/`, `vision/`, `contract/`, `nlp/`, `pipeline/`, `demo/`); notebooks stay thin and call it. `schemas/` holds the contracts, `fixtures/` the hand-written test cases, `tests/` runs in under a minute without data. Full layout: plan §3.
