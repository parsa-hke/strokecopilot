# Stroke Decision Copilot

MAAI 901 (Deep Learning for Visual Data) and MAAI 902 (NLP & LLMs) group project, MSc Applied AI, University of Wollongong in Dubai.

Vision models read the stroke CT (haemorrhage on plain CT; core and penumbra on CT perfusion). A language model reasons over those findings, a triage note and a rulebook written in our own words from the 2026 AHA/ASA guideline and the DAWN and DEFUSE-3 trials. A verifier then checks every claim in its rationale against its sources. The research output is a hallucination-rate comparison across six language models on a clinical decision task with known inputs, and how imaging errors carry through to treatment decisions.

**Research prototype — not for clinical use.**

- **Live demo:** [huggingface.co/spaces/parsa-hke/stroke-copilot](https://huggingface.co/spaces/parsa-hke/stroke-copilot): the pipeline story, a case explorer where every claim links to its source, the results as charts, and a scenario builder that runs the reasoner and the verifier live.
- **Plan and decisions:** [STROKE_COPILOT_PLAN.md](STROKE_COPILOT_PLAN.md) (§2 lists every fixed decision). **Progress:** [CHANGELOG.md](CHANGELOG.md).

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

The vision and NLP sides meet only at the contract, [`schemas/vision_output.json`](schemas/vision_output.json).

## Results at a glance

Development results are on 129 ISLES'24 cases (five folds, each case scored by the model that did not train on it) and on RSNA validation. Held-out results are on 20 ISLES'24 cases opened once, after the analyses were fixed in advance ([`docs/PHASE4_PREREGISTRATION.md`](docs/PHASE4_PREREGISTRATION.md)).

| Component | Result |
|---|---|
| **Vision A** (nnU-Net 3d_fullres) | Dice 0.215 ± 0.046 against the rCBF < 30% threshold's 0.111 ± 0.031 (paired +0.104 [+0.078, +0.133]); absolute volume error 23.5 against 25.2 mL, not a clear difference. Held-out: Dice 0.18 (five-fold ensemble). |
| **Vision B** (ResNet-50, RSNA) | Test split, opened once: scan AUC 0.9805 [0.974, 0.986], sensitivity 0.962, specificity 0.885 at the chosen threshold, ECE 0.015. On ISLES'24 it flags 68 of 129 presumed bleed-free scans, a domain shift traced to brain tissue, not the skull base (Grad-CAM and SHAP). |
| **Hallucination rate** (with the rulebook; 5 fixtures, 150 scenarios, 20 real cases per model; verifier rates, upper bounds) | Sonnet 5.5 4.0%, Qwen3.5-27B 5.0%, Gemma 3 27B 17.1%, MedGemma 27B 17.7%, Gemma 3 4B 40.6%, Gemma 3 4B LoRA-tuned 13.3%. |
| **Rulebook effect** | Giving the whole rulebook cuts false-content claims by 8 to 15 percentage points for every model. |
| **Verifier validity** | Seeded errors: recall ≥ 0.80 on every sub-type, 1% false flags. 100 human-labelled claims: verifier–human agreement 0.61; humans confirmed 18 of its 54 flags; human–human κ 0.27. |
| **End to end, held-out** (predicted vision) | Decisions right: Sonnet 22/23, Gemma 3 27B 22/23, MedGemma 21/21, Qwen 13/13 (9 format failures), tuned 4B 21/23, base 4B 3/23. Sonnet's one wrong decision came from a false bleed flag the verifier could not see. |

**Phase 5 additions** (report only, after the freeze; all run by 20 Nov 2026 except the clinicians' answers): exact Shapley values of the rule engine's inputs (the occlusion site and the clock decide most treatments; the core volume rarely does), of Vision A's input channels, Vision B's CT windows and the small reasoners' inputs (the base Gemma 3 4B recommends thrombolysis whatever it is told; the tuned one follows the clock as the rules do), pixel attributions beside Grad-CAM, a surrogate of the verifier's flags, token attributions of the retriever, 35 Student's t-tests beside the bootstrap results, a subgroup analysis, a retrieval re-test, an ONNX export, a re-judging of the 100 human-labelled claims by a non-Claude judge (GPT-6 Luna: 3 of 100 verdicts changed, agreement with the humans 0.62 against Haiku's 0.61), a BiomedCLIP comparison (frozen BiomedCLIP: scan AUC 0.90 as a linear probe and 0.75 zero-shot, against the ResNet-50's 0.98), and a clinician review pack. Summaries for the reports: [`docs/PHASE5_HANDOVER.md`](docs/PHASE5_HANDOVER.md).

## Quick start

```bash
pip install git+https://github.com/parsa-hke/strokecopilot.git
```

For local development and the tests (no data needed, under a minute):

```bash
pip install -e ".[dev]"
pytest
```

The extras group what each part needs: `isles` and `rsna` (data and vision), `retrieval`, `llm` (Anthropic and OpenAI-compatible clients), `space` (the demo server), `finetune`, and `analysis` (SHAP, surrogates and t-tests).

**Reproduce the hallucination table without an API key.** The replay bundle holds the saved answers and verifier caches for the fixtures and the 150 synthetic scenarios (no patient data). Download `replay_bundle.tar.gz` from the [release](https://github.com/parsa-hke/strokecopilot/releases) of the submission tag, then:

```bash
python -m strokecopilot.pipeline.replay run --bundle replay_bundle.tar.gz
```

It checks the bundle's sha256, rescores every model and arm offline from the cached verdicts, checks each rate against the reported one, and prints the table.

**Check a vision output against the contract:**

```python
from strokecopilot.contract import load_vision_output
out = load_vision_output("fixtures/fx02_thrombectomy_large_mismatch.json")
```

## Data

Nothing under `data/` or `runs/` is committed, and no data or model weights are distributed. ISLES'24 (Zenodo record 17652035, CC BY-NC-SA 4.0) and the RSNA Intracranial Haemorrhage data (Kaggle, competition rules) must be obtained from their sources; the team kept them on a shared Google Drive and, for RSNA, on Kaggle only. The guideline and trial texts stay with their publishers: the reasoner sees only our own-words rulebook ([`corpus/RULEBOOK.md`](corpus/RULEBOOK.md)). Terms: [`docs/DATA_USE.md`](docs/DATA_USE.md).

## Notebooks

Thin: each calls the package. Colab notebooks open with the badge; Kaggle notebooks are uploaded to Kaggle with the RSNA datasets attached.

| Notebook | What it does | Where |
|---|---|---|
| `00_download_isles24` | Download ISLES'24 to Drive and verify it | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/00_download_isles24.ipynb) |
| `00_template` | The team template: install, mount Drive, load one case | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsa-hke/strokecopilot/blob/main/notebooks/00_template.ipynb) |
| `01a`–`01d`, `01_prep_isles24` | ISLES'24 listing, clinical survey, split, slim tarball and its check | Colab |
| `02_prep_rsna`, `02a`–`02c` | RSNA preprocessing into shards, timing and read-speed checks | Kaggle |
| `10_visionA_nnunet` | nnU-Net v2 training per fold | Colab (rented GPU for the long runs, DEC-21) |
| `11_visionA_thresholds`, `11b` | The rCBF threshold baseline and its check | Colab |
| `12`, `13_visionA_error_analysis*` | Vision A error analysis, fold 0 and all five folds | Colab |
| `14_visionA_confidence` | `confidence.core` and `confidence.penumbra` | Colab |
| `15_visionA_channel_shap` | Phase 5: Shapley values of Vision A's input channels | Colab A100 |
| `20_visionB_train` | RSNA patient split and the ResNet-50 slice classifier | Kaggle |
| `21_visionB_isles_check` | Vision B on ISLES'24: the domain-shift check | Colab |
| `22_visionB_operating_point` | Cross-fitted operating point on RSNA validation | Kaggle |
| `23_visionB_rsna_geometry` | RSNA slice spacing and field of view | Kaggle |
| `24_visionB_false_alarms` | The pre-registered false-alarm rules on RSNA validation, Grad-CAM | Kaggle |
| `25_visionB_rsna_test` | The RSNA test split, opened once | Kaggle |
| `26_visionB_window_shap` | Phase 5: Shapley values of the three CT windows | Kaggle |
| `27_visionB_biomedclip` | Phase 5: BiomedCLIP (linear probe and zero-shot) against the ResNet-50; per-subtype probes | Kaggle |
| `31_nlpA_reasoner` | The reasoner on the five fixtures | Colab |
| `40_nlpB_verifier`, `51_verifier_recheck`, `52_haiku_extractor_check` | Verifier: seeded errors, judge choice, re-checks | Colab |
| `41_nlpB_model_comparison` | The hallucination table (C9) | Colab |
| `42_nlpB_small_reasoner`, `54_finetune_colab`, `55_small_reasoner_colab` | The LoRA-tuned Gemma 3 4B: training, answers, report; Phase 5 reasoner SHAP | Colab A100 |
| `50_end_to_end` | Fixture to reasoner to verifier, end to end | Colab |
| `53_paid_runs_colab` | The Gemma 27B pair served with vLLM | Colab A100 |
| `60_heldout_contracts` | The held-out contract path, checked on development cases first | Colab A100 |

## Repository layout

`src/strokecopilot/` holds all logic: `data/`, `vision/`, `contract/`, `nlp/` (corpus, notes, reasoner, verifier), `pipeline/` (runs, comparisons, held-out analysis, replay), `demo/` (the Space) and `analysis/` (Phase 5). `schemas/` holds the contracts, `fixtures/` the hand-written test cases, `corpus/` the rulebook, `prompts/` every prompt version, `docs/` the decisions' working papers, and `tests/` runs without data. Full layout: plan §3.

## Licence and citation

Code: [MIT](LICENSE). Data and model weights are not covered and not distributed (see above). Please cite with [CITATION.cff](CITATION.cff), and cite ISLES'24 (arXiv:2408.11142, arXiv:2408.10966) and RSNA ICH (Flanders et al., *Radiol Artif Intell* 2020) when you use their data.
