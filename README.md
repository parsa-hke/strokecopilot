# Stroke Decision Copilot

MAAI 901 (Deep Learning for Visual Data) + MAAI 902 (NLP & LLMs) group project, MSc Applied AI, UOWD.

Vision models read the stroke CT (haemorrhage on plain CT; core and penumbra on CT perfusion), a retrieval-grounded language model reasons over those findings, a triage note and the AHA/ASA guideline plus the DAWN and DEFUSE-3 trials, and a verifier checks every claim in its rationale against its sources. The research output is a hallucination-rate comparison across language models on a clinical decision task with known inputs.

**Research prototype — not for clinical use.**

The full plan is in [STROKE_COPILOT_PLAN.md](STROKE_COPILOT_PLAN.md); progress is in [CHANGELOG.md](CHANGELOG.md).

## How it fits together

```
plain CT ──► Vision B (haemorrhage) ──┐
                                      ├──► vision_output.json ──┐
CT perfusion ──► Vision A (core/penumbra) ┘                      ▼
triage note ─────────────────────────────────────────► NLP A: reasoner ──► recommendation.json
guideline corpus ──── retrieval ──────────────────────┘                          │
                                                                                 ▼
                                                           NLP B: verifier ──► every claim traced or flagged
```

The vision and NLP sides meet only at [`schemas/vision_output.json`](schemas/vision_output.json). Until real vision outputs exist, the NLP side builds against the five hand-written cases in [`fixtures/`](fixtures/).

## Install

On Colab (or anywhere):

```bash
pip install git+https://github.com/<org>/strokecopilot.git
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

Badges point at the notebooks once they exist; replace `<org>` with the GitHub owner after the repo is pushed.

| Notebook | Purpose | |
|---|---|---|
| `00_download_isles24` | Phase 0: download ISLES'24 to Drive and verify md5 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/00_download_isles24.ipynb) |
| `00_template` | Install, mount Drive, copy tarball, load one case | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/00_template.ipynb) |
| `01_prep_isles24` | ISLES'24 raw → slim tarball | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/01_prep_isles24.ipynb) |
| `02_prep_rsna` | RSNA ICH → 320 px JPEG tar shards (runs on Kaggle) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/02_prep_rsna.ipynb) |
| `02a_rsna_kaggle_timing` | Phase 0: time full RSNA on Kaggle | Upload to Kaggle; this one does not run on Colab |
| `10_visionA_nnunet` | Vision A: nnU-Net training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/10_visionA_nnunet.ipynb) |
| `11_visionA_thresholds` | Vision A: threshold baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/11_visionA_thresholds.ipynb) |
| `20_visionB_train` | Vision B: haemorrhage classifier | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/20_visionB_train.ipynb) |
| `30_nlpA_corpus` | NLP A: corpus ingestion and retrieval | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/30_nlpA_corpus.ipynb) |
| `31_nlpA_reasoner` | NLP A: reasoner | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/31_nlpA_reasoner.ipynb) |
| `40_nlpB_verifier` | NLP B: verifier | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/40_nlpB_verifier.ipynb) |
| `41_nlpB_model_comparison` | NLP B: hallucination table | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/41_nlpB_model_comparison.ipynb) |
| `50_end_to_end` | Full pipeline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<org>/strokecopilot/blob/main/notebooks/50_end_to_end.ipynb) |

## Repository layout

`src/strokecopilot/` holds all logic (`data/`, `vision/`, `contract/`, `nlp/`, `pipeline/`, `demo/`); notebooks stay thin and call it. `schemas/` holds the contracts, `fixtures/` the hand-written test cases, `tests/` runs in under a minute without data. Full layout: plan §3.
