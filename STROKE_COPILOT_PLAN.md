# STROKE_COPILOT_PLAN.md

**Stroke Decision Copilot — build plan and reference for every session**

MAAI 901 (Deep Learning for Visual Data) + MAAI 902 (NLP & LLMs) joint group project, MSc Applied AI, UOWD.
Team of five. Plan version 0.2, 30 October 2026.

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

- **Thrombolysis** (IV alteplase/tenecteplase): a clot-dissolving drug. Standard window is 4.5 hours from symptom onset, extendable to 9 hours in selected patients who cannot have thrombectomy, if imaging shows salvageable tissue. Absolutely contraindicated if there is an intracranial haemorrhage.
- **Thrombectomy**: a catheter physically removes the clot. Indicated for large-vessel occlusions; the window can extend to 24 hours in selected patients (DAWN and DEFUSE-3 trials, 2026 AHA/ASA guideline), and eligibility now reaches large-core strokes.

Eligibility therefore depends on three things at once:

1. What the scan shows — how much tissue is already dead (**core**), how much is starving but salvageable (**penumbra**), whether there is a bleed, where the occlusion is.
2. Facts about the patient — time since onset (or last known well), age, NIHSS score, anticoagulant use, prior disability.
3. Published rules — the 2026 AHA/ASA guideline and the trial criteria.

Clinicians do this under time pressure. Our system does the imaging part with vision models, the reasoning part with a language model grounded in a guideline rulebook, and then audits the language model's output claim by claim.

### 1.2 The system in one diagram

```
          plain CT ──► [Vision B: haemorrhage?] ──┐
                                                  ├──► vision_output.json ──┐
   CT perfusion ──► [Vision A: core / penumbra] ──┘                         │
                                                                            ▼
   triage note (generated from clinical variables) ──────────────► [NLP A: reasoner] ──► recommendation + rationale
   guideline rulebook (AHA/ASA, DAWN, DEFUSE-3) ── whole ────────┘                              │
                                                                                                ▼
                                                                  [NLP B: verifier] ──► every claim traced or flagged
```

### 1.3 The research claim

The verifier lets us measure, on a clinical decision task with known ground truth inputs, how often different language models assert things that are not in their inputs. The deliverable that matters for the report is a table: hallucination rate (and critical hallucination rate) per model, per clinical scenario, with confidence intervals.

Guideline-grounded claim checking already exists in the literature, so the distinctive contribution is the vision link: how errors in imaging-derived values carry through to treatment decisions, measured by comparing ground-truth and predicted vision inputs, and whether retrieval reduces hallucination at all, measured against a no-retrieval arm.

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
| DEC-1 | Group project is the Stroke Decision Copilot, replacing Bias Lens | Decided — professors approved (reported by Parsa, 30 Oct 2026) |
| DEC-2 | Datasets: ISLES'24 (Zenodo) for perfusion + infarct + clinical variables; RSNA Intracranial Haemorrhage (Kaggle) for bleed detection; AHA/ASA guideline + DAWN + DEFUSE-3 papers for the corpus | Decided |
| DEC-3 | Compute: Google Colab Pro. Storage: shared Google Drive folder `StrokeCopilot/`. Training data is copied to Colab local disk as one tarball per session — never read per-file through the Drive mount | Decided |
| DEC-4 | Contract-first: `schemas/vision_output.json` is frozen at the end of Phase 0. NLP builds against hand-written fixtures until Phase 3 | Decided |
| DEC-5 | Evaluation split is fixed on day one: 5 folds + ~20 held-out cases. Held-out cases are untouched until Phase 4 | Decided |
| DEC-6 | The lead builds Phase 0–1 alone, then sends teammates instructions when assigning them work, with no fixed onboarding date. Changed by Parsa on 2 Nov 2026; it was within 7–10 days of 29 Oct 2026 | Decided |
| DEC-7 | `penumbra_ml` is the total Tmax > 6 s volume, core included; salvageable tissue is `mismatch_volume_ml` | Decided |
| DEC-8 | The rule baseline is the 2026 AHA/ASA guideline plus DAWN and DEFUSE-3, not the 2018/2019 guideline | Decided |
| DEC-9 | The contract carries a nullable `aspects` field. No ASPECTS model is built: it is null from the real pipeline and set only in fixtures and synthetic scenarios | Decided |
| DEC-10 | A rationale claim can cite several sources and be marked `derived`. The hallucination definition changes to match (§6.4, §8.4) | Decided |
| DEC-11 | The primary hallucination evaluation uses 100–200 synthetic scenarios; held-out cases are the realism check. Added: seeded errors, no-retrieval arm, ground-truth vs predicted vision arm (§9) | Decided |
| DEC-12 | ISLES'24 uses the latest Zenodo record (17652035), not 16748089, because the data dictionary was only added after that version | Decided |
| DEC-13 | Slim format v1 gains `ncct_native` and `cow` volumes (§5.4) | Decided |
| DEC-14 | `mismatch_ratio` is rounded to two decimals, from the rounded volumes | Decided |
| DEC-15 | `haemorrhage.confidence` means the calibrated probability that a haemorrhage is present | Decided |
| DEC-16 | Vision B uses all labelled RSNA data, with no subset | Decided |
| DEC-17 | Vision B preprocesses once to 320 px JPEG tar shards on Kaggle, then trains there, as an exception to DEC-3; only checkpoints and metrics move to Drive | Decided |
| DEC-21 | Vision A training (nnU-Net on ISLES'24) and, later, Phase 4 open-model inference if an API cannot serve the model, run on a GPU pod rented per job, as an amendment to DEC-3. Provider: Vast.ai, on-demand, one RTX 4090, from verified hosts with reliability ≥ 0.98, ≥ 16 CPU cores, ≥ 32 GB RAM, ≥ 100 GB disk, ≥ 200 Mbps download and a driver for CUDA 12.8, preferring hosts in the US or EU. Amended 5 Nov 2026: the first draft also required datacenter hosts, but no datacenter 4090 was offered under 0.50 USD/h (cheapest 0.72 USD/h), while verified non-datacenter hosts started near 0.40 USD/h. Fallback: RunPod Community Cloud RTX 4090, if no Vast offer passes at under 0.50 USD/h. Budget: 8–15 USD of GPU rental for the whole project, with no persistent volume; every run ends by destroying its pod, and a watchdog on the pod enforces an hour limit per run. No A100: EXP-4 ran this plan (3d_fullres, patch 96 × 128 × 112, batch 4) on a 24 GB L4. Vision A trains 300 epochs per fold, all five folds, fold 0 first. Kaggle keeps RSNA (DEC-17); Drive stays the store; Colab stays for the demo and the thin notebooks (§4.6). Amended 16 Nov 2026 (Hugging Face token): for the gated Gemma and MedGemma weights (DEC-24) a Hugging Face token may go to the serve pod, which the rule above forbade. It must be fine-grained and read-only for those two repositories; it travels on SSH's stdin into a file only root can read, is never logged or put on a command line, is removed from the pod as soon as the last model's weights are downloaded, and is revoked by Parsa after the run; the pod is destroyed at the end in every case | Decided — Parsa, 5 Nov 2026; host filter relaxed by Parsa, 5 Nov 2026; Hugging Face token on the serve pod allowed by Parsa, 16 Nov 2026 |
| DEC-20 | The verifier's judge is `claude-haiku-5-5` (effort medium, `judge_v2`; prompt `judge_v3` since DEC-23) with τ = 0.6; the claim extractor is `claude-sonnet-5-5` (effort medium). Chosen by the EXP-11 bake-off: both judges reach the seeded-error bar with the same recall, Haiku with fewer false flags (0.010 vs 0.019) at about a twentieth of the cost per call | Decided — Parsa, 4 Nov 2026 |
| DEC-19 | The reasoner gets the whole rulebook in its prompt, not retrieved passages. Retrieval stays only for the verifier's candidate passages (`BAAI/bge-base-en-v1.5`). Experiment arm 1 becomes with vs without the rulebook. Reason: EXP-7 found top-k retrieval missed 6 of 15 key fixture rules, because embeddings cannot tell apart rule rows that differ only in numbers. The rulebook is about 4,600 tokens, so sending it whole is cheap | Decided — Parsa, 4 Nov 2026 |
| DEC-18 | The corpus is a rulebook in our own words, one passage per criterion, each pointing to its source, section, recommendation and page. It keeps the numeric criteria as facts and copies no wording. The reasoner and judge see only our text; the PDFs stay on Drive for checking. Reason: AHA allows no reproduction without permission, and NEJM reserves text and data mining and AI use (`docs/DATA_USE.md`). The sources in DEC-2 are unchanged | Decided — Parsa, 4 Nov 2026 |
| DEC-22 | The RSNA split is by patient, 80/10/10 train/validation/test, ranked by a SHA-256 hash of seed 20261103 and patient ID and cut separately for patients with and without a bleed, so validation and test keep the natural bleed rate (EXP-6: 0.4058 and 0.4119 against 0.4085). It is rebuilt in each Kaggle session and never saved. The test split stays closed (`ALLOW_RSNA_TEST`) until the final Vision B model is chosen | Decided — Parsa, 12 Nov 2026 |
| DEC-23 | The judge prompt is `judge_v3`, amending DEC-20 (model, effort and τ unchanged). It adds how to judge a requirement that is not needed, criteria that are not met, conditional or counterfactual claims, and a reason for a decision. EXP-23: the EXP-11 seeded-error bar is unchanged (min recall 0.80, false flags 0.010), and EXP-12 drops from 23 to 19 flags of 350 against `judge_v2`, with every reasoner error of the earlier reading still flagged. Its examples come from EXP-12, so that gain is in-sample | Decided — Parsa, 12 Nov 2026 |
| DEC-24 | The reasoner models compared (§8.4) are Claude Sonnet 5.5 (frontier API), Qwen3.5-27B (open mid-size), MedGemma 27B text (small medical) and Gemma 3 27B, MedGemma's base, so medical training is compared in a pair. Open models run on a rented RTX 4090 at 4-bit with vLLM and constrained JSON (DEC-21). MedGemma needs Parsa's reading of the HAI-DEF terms §3.2 first; if they rule it out, Baichuan-M2-32B and its base Qwen2.5-32B replace the pair. No non-Claude frontier model: three of the four are non-Claude, judge bias is reported as verifier-vs-human agreement per model, and GPT-6.1 Sol fits the API budget only if the contingency goes unused. The claim extractor becomes Claude Haiku 5.5 if EXP-24 shows it keeps the seeded-error bar; otherwise Sonnet stays and the matrix shrinks, because four models with the Sonnet extractor cost about 150 USD against about 96 USD left. Shortlist by Hana (Notion, 9 Nov 2026) Amended 12 Nov 2026 after EXP-24 (Haiku failed the uncited-rule bar, recall 0.30, and missed 3 of the 9 EXP-12 reasoner errors): the extractor stays Claude Sonnet 5.5, and the extractor and the Sonnet reasoner run through the Message Batches API (half price, same model). The matrix shrinks: per model, the 5 fixtures with and without the rulebook, 150 scenarios with the rulebook and 75 without, 20 real development cases with predicted vision, ground-truth vision and no rulebook; the 20-case triple repeat for Sonnet only. Estimated about $65 for the main runs and about $85 for the project's API with 20% contingency Amended 15 Nov 2026 (hosting): the RTX 4090 cannot hold three of the four at 4-bit in a form vLLM loads (the official Qwen3.5-27B GPTQ-Int4 is about 30 GB; Gemma 3 and MedGemma have no official vLLM 4-bit checkpoint). So Qwen3.5-27B runs through the OpenRouter API at full precision (DEC-21 prefers an API when one serves the model; about $0.20/$1.56 per million tokens in/out), and Gemma 3 27B and MedGemma 27B text run together on one rented 48 GB GPU (L40S or RTX A6000, about $0.40-0.60/h) at the same 8-bit setting, so the medical-training pair is compared like for like. Estimated open-model cost about $5-7 (not measured); the Sonnet estimate is unchanged. Needed before the runs: Parsa's OpenRouter key in his shell environment (OPENROUTER_API_KEY); a Hugging Face token on the pod for the gated Gemma models, which needs a DEC-21 change (pending); Parsa's reading of the HAI-DEF terms §3.2 (pending). If either pending item is no, the fallback pair Baichuan-M2-32B and Qwen2.5-32B-Instruct (ungated, official 4-bit) replaces the Gemma pair on the same 48 GB GPU Amended 16 Nov 2026 (judge): the Haiku judge also runs through the Message Batches API, like the reasoner and extractor: the same model, prompt, settings and requests (a test checks the batched requests are byte for byte the live ones, and that every trace matches), at half price, and with no call that needs the laptop to stay on. Batches are submitted and fetched in separate sessions (`pipeline/batched.py`, `pipeline/paid_runs.py`); the open models are still answered live, through OpenRouter and the SSH tunnel, with each answer saved on the Mac as it arrives. 16 Nov 2026: Parsa read the HAI-DEF terms §3.2 and cleared MedGemma, so the Gemma pair stays and the fallback pair is not needed. 16 Nov 2026 (Gemma pair on Colab): the Gemma pair runs on a Colab A100 instead of a rented 48 GB GPU, to keep the Vast credit for Phase 4 (Parsa's choice): vLLM serves each model at the same 8-bit setting on the notebook's own machine (notebook 53), which does only the reasoner pass and saves each answer to Drive; the Mac imports the answers and runs the extraction and judge batches. The Hugging Face token comes from Colab Secrets into the server's environment only. The Vast serve-pod code (EXP-36/37 in compute/jobs.py) stays for Phase 4 | Decided — Parsa, 12 Nov 2026; hosting amended 15 Nov 2026; judge batching amended 16 Nov 2026; MedGemma cleared 16 Nov 2026; Gemma pair moved to Colab 16 Nov 2026 |
| DEC-25 | The core that feeds the contract (`core_ml`) is the nnU-Net prediction, not the rCBF < 30% threshold. For each development case it is the validation prediction of the fold model that did not train on it, so no case is scored by a model that saw it; how the five fold models are combined for held-out cases is left to Phase 4. EXP-25 (all 5 folds, 129 development cases, paired): Dice 0.215 ± 0.046 against the within-Tmax > 6 s threshold's 0.111 ± 0.031 (paired +0.104, 95% CI +0.078 to +0.133); abs volume error 23.5 against 25.2 mL (paired −1.6 mL, CI −4.5 to +1.3, p 0.35), and wrong side of 21/31/51/70 mL in 30/28/20/17 against 35/29/23/17 cases, none significant. So nnU-Net is no worse on any contract measure and clearly better at locating the infarct, but its volume is not measurably better: the report says so, and keeps the threshold as the comparison. Both under-call lesions of 21 mL or more by a wide margin (abs error about 53 mL); that is the open weakness for B4 and calibration. The penumbra stays Tmax > 6 s (§8.1) | Decided — Parsa, 13 Nov 2026 |
| DEC-26 | Vision A is two-stage: `core_ml` is the nnU-Net final-infarct prediction used as the core proxy (DEC-25), and `penumbra_ml` is the Tmax > 6 s rule (DEC-7). No core or penumbra heads, because ISLES'24 labels only the final infarct, and no volume calibration. EXP-26 (129 development cases, cross-fitted by fold): the nnU-Net core never exceeds the Tmax > 6 s volume, so no clipping is needed. Large infarcts are under-called (≥ 70 mL: median reference 118 mL, nnU-Net 35 mL, threshold 25 mL, Tmax > 6 s 188 mL; in 7 of 22 the final infarct exceeds the whole Tmax > 6 s region), and not from missing Tmax (median 92% reliable). Of 12 cross-fitted corrections, linear ones made things worse (abs error +2.4 to +2.6 mL; 54–61 cases on the wrong side of 21 mL against 30) and the best robust one gained 0.6 mL with a CI crossing zero; averaging nnU-Net and threshold volumes did best on 31 mL but was chosen after the fact from 16 variants and is not adopted. Scaling the core toward final infarct would also make `core_ml` a predicted final infarct, against trial cut-offs defined on acute core. The gap is reported as a limit (final infarct ≠ acute core, reperfused cohort) and is input to `confidence.core`. A cross-fitted lower probability threshold stays an optional later test | Decided — Parsa, 13 Nov 2026 |
| DEC-27 | `confidence.core` is the cut-off distance alone: the distance from nnU-Net's core volume to the nearest decision cut-off (21, 31, 51, 70 mL), on a log scale, mapped to a confidence by a small logistic regression cross-fitted by fold. It replaces Lorna's v2 (four signals), which it matches on EXP-32 (129 development cases, cross-fitted): AUC for the wrong side of a cut-off 0.736 vs 0.722 (v2 minus distance −0.013, 95% interval −0.048 to +0.019), Brier 0.188 vs 0.193 (no signal 0.222); v2 already weighted the distance most. Chosen for simplicity after seeing the results, as v2 was. The reasoner reads it as the chance that the core volume is on the right side of every cut-off: about 0.9 at 0 mL, about 0.41 from 21 to 70 mL, higher for very large cores; it does not measure outline quality. Development cases use EXP-32's cross-fitted values; held-out cases (Phase 4) its mapping fitted on all 129. Near-empty predictions (core under 1 mL, 30 cases) get median confidence 0.90 but are right only 25 of 30 times, and no tested signal finds the wrong ones, so the contract carries the warning "Little or no core predicted: a missed infarct cannot be ruled out" for them. `confidence.penumbra` stays Lorna's data-quality score (1 minus the share of perfusion-covered tissue with no reliable Tmax). The schema is unchanged | Decided — Parsa, 15 Nov 2026 |
| DEC-28 | The Vision B threshold for the contract is 0.175 on the calibrated scale (`haemorrhage.present` = calibrated confidence ≥ 0.175; `haemorrhage.confidence` carries the probability), replacing EXP-6's provisional 0.278. EXP-17 chose it on RSNA validation by cross-fitting on patient halves (96% target, 20 seeds): sensitivity 0.961 (95% CI 0.946–0.972), specificity 0.878 (0.859–0.895), 3.9 missed bleeds per 100 bleeds and 12.2 false alarms per 100 non-bleeds, against about 0.945, 5.5 and 8.6 at 0.278 cross-fitted (0.278 was chosen and measured on the same split). This follows §8.2: a false "no bleed" is the worst failure, and the threshold is set where bleed labels exist (RSNA), not by counting flags on ISLES'24's presumed bleed-free development cases. On ISLES'24 it flags 68 of 129 scans at 5 mm (52.7%; 64 with the field of view matched) against 45 at 0.278: domain shift, mostly mimics in the top-8 review (calcification, hyperdense vessel, artefact). Since `haemorrhage.present` is an absolute stop in the rules, the predicted-vision arm will refuse treatment in about half the real cases; the ground-truth vs predicted arms (DEC-11) report that cost. The primary reading stays 5 mm slabs without field-of-view matching. The RSNA test split stays closed until the final check (DEC-22). Mohamed's eye check of the scans between 0.175 and 0.278 describes the false alarms but cannot change the choice | Decided — Parsa, 15 Nov 2026 |
| DEC-29 | The without-rulebook arm (arm 1 of §8.4, DEC-11, DEC-19) gives the reasoner the same prompt with the rulebook removed (`prompts/reasoner_v2_norulebook.md`): rules come from its own knowledge of the current AHA/ASA guideline and the trials behind it, are named in the claim (guideline or trial, and the rule or threshold), and are cited with a fixed placeholder source `OWN-KNOWLEDGE` (also in `rule_ids`), so every output still fits the frozen recommendation schema. The verifier scores content only in this arm: the placeholder is dropped and each rule claim is checked against our corpus like an uncited rule (candidates the claim names, the retriever's top 5, the judge), so wrong numbers, invented findings, wrong and misapplied rules and faulty derivations count as in the rulebook arm; passage attribution (sub-types 6 and 7) is not charged, because citing a passage is impossible here, while imaging and note citations are still checked. In the rulebook arm the placeholder is a passage that does not exist (sub-type 3). Limits stated with the result: a true rule our 47-passage corpus does not hold counts as a wrong rule, and the arms' attribution numbers are not comparable. Matrix as in DEC-24: per model, the 5 fixtures, 75 of the 150 scenarios (every second one, ordered by decision path) and the 20 real cases with predicted vision | Decided — Parsa, 16 Nov 2026 |
| DEC-30 | A fine-tuned small reasoner extends DEC-24's model list with two rows: Gemma 3 4B (instruction-tuned) with prompt v3 and the rulebook, and the same model with a LoRA adapter trained on gold answers. The gold answers come from `nlp/reasoner/gold.py` (`rules.decide` -> the reply v3 asks for), pass `gold.check()` and, on a sample, the full verifier (EXP-40: 0 flags on 30). Training scenarios are synthetic, from a new seed and new value grids, with 4 of the 16 key routes chosen by seed and left out of training (amended by Parsa, 17 Nov 2026: drawn from the 14 routes other than "within 4.5 h with a disabling deficit" and "acute intracranial haemorrhage", which training always keeps; held out: beyond 4.5 h and eligible for thrombectomy, LVO 4.5-24 h without thrombectomy, INR above 1.7, no occlusion identified); no real case and no held-out case is used. Training and serving: bf16 LoRA on a Colab A100, loss on the answer only, a validation slice for early stopping, both models served through vLLM with identical settings. Evaluation: the rulebook arms only (5 fixtures, 150 scenarios, 20 real cases with predicted vision), seen and held-out routes reported apart, the held-out routes as the main result. Limits to report: the gold is templated, so the tuned model may write formulaic claims that are easier to verify; the verifier cost (about $12-15, estimate) comes from the API budget. | Decided — Parsa, 17 Nov 2026 |
| DEC-31 | Phase 4 pre-registration (`docs/PHASE4_PREREGISTRATION.md`), fixed before any held-out case is opened. **D1:** held-out cores come from the five fold models' nnU-Net ensemble (softmax averaged, `checkpoint_final.pth`), which settles DEC-25's open point; everything else in the contract is as for development cases; ground-truth contracts are built as in EXP-34; `plans.json` and `dataset.json` are rebuilt and matched to each fold's `debug.json` first. **D2:** all six reasoner models (DEC-24's four and DEC-30's pair), prompt v3 with the rulebook, predicted and ground-truth vision arms, 240 requests; this replaces §7's and §9's '× 3 models'. **D3:** correctness against `rules.decide` on the ground-truth contract, undetermined excluded; C9's metrics against EXP-37's development real cases; error propagation (a vision error if `rules.decide` on the predicted contract is already wrong, otherwise a reasoning error; verifier caught or missed); `core_ml` ×0.8 and ×1.2, at the rule level and for Sonnet; held-out Dice, volume error and cut-off crossings. Before unlocking: the EXP-41 consensus, the freeze list, Vision B's one RSNA test-split check (DEC-22), and the code tested on development cases. Estimate about $6 of API and 1–2 Colab A100 hours | Decided — Parsa, 19 Nov 2026 |
| DEC-32 | The demo is a public Hugging Face Space (`docs/DEC-32_PUBLIC_SPACE.md`), replacing §8.5's Gradio app run from Colab: a Docker Space on free CPU-basic hardware (Parsa's HF Pro account), a small FastAPI server serving a static page. It has a scrolling story of the pipeline, a case explorer where every claim is clickable and its source lights up, the results as interactive charts, and one live element: a scenario builder with structured inputs only (no free text). The project's note generator writes the note, and Sonnet 5.5 with prompt v3 and the rulebook, then the verifier (`judge_v3`, τ 0.6), run live with EXP-37's settings; any failure, slow call or spent budget switches to the nearest precomputed scenario (EXP-37's saved Sonnet answers), with a message. Spending is capped by a result cache, per-visitor and daily limits in the server, and a dedicated Anthropic key in its own console workspace with a small monthly limit Parsa sets (about $5 suggested; he expects fewer than 20 live runs). The key lives only in the Space's secrets, added to §4.4 for this one key. RSNA appears only as aggregate numbers; ISLES'24 images appear with attribution under CC BY-NC-SA 4.0 (checked on Zenodo, 19 Nov 2026), cases by alias, never by ID; guideline text only as our rulebook passages. Built in `demo/space/` on fixtures, scenarios and development results before unlocking, filled with held-out results after DEC-31's analyses; every push to the Space is a publish, only on Parsa's word. No live answer is counted in any result. Choices (§10): public; a private staging Space first; HF sign-in off at launch, behind a switch; continuous builder controls within the evaluated ranges; a model switch in the explorer; limits in memory with the workspace limit as the hard cap. Estimate 15–20 h of session work | Decided — Parsa, 19 Nov 2026 |
| DEC-34 | Phase 5 additions are report only; DEC-31's freeze holds (nothing retrained, nothing feeds the contract, the pipeline or the Space). The first is the clinician review (EXP-67, `docs/CLINICIAN_PACK.md`): two clinicians, two instruments answered independently (a rulebook and test-case review, and a written interview on nine cases chosen by a seeded rule over fixed strata), delivered as generated Google Forms. Their answers are expert opinion, never clinical validation. A rule error they find is counted in code (scenario and development decisions; held-out only with Parsa's OK) and reported as a limitation, not fixed. DEC-33 stays reserved for Item 1 (BiomedCLIP) | Decided — Parsa, 20 Nov 2026 |
| DEC-35 | The SHAP analyses (Phase 5 task B, `docs/PHASE5_SHAP_SURVEY.md`), report only under DEC-34. They run on rows 1, 3, 5, 6, 8, 9, 10, 11, 12 and 13 of the survey: Vision A input channels (five players: NCCT, CBF, CBV, MTT, Tmax with its mask) and `confidence.core` v2; Vision B windows and pixels; the rule engine on the 150 scenarios; the Gemma 3 4B pair on 24 scenarios; the retriever; three surrogates. Row 2 (Vision A spatial) runs if time allows; Sonnet only in the small form (about $9) and on a separate go-ahead. Exact Shapley values by enumeration where the players are few, the `shap` library elsewhere, named in each result. Baselines: Tmax absent as no reliable data (Tmax 0, mask 0), other perfusion maps mean-filled, NCCT at its normalised mean, a CT window at its RSNA training mean, a rule or reasoner input unknown. The values measure reliance on an input, not an ablation, because no model was trained without it; surrogates are labelled as such. No held-out case is used | Decided — Parsa, 20 Nov 2026 |
| DEC-36 | Student's t-tests (Phase 5 task C, `docs/PHASE5_TTESTS.md`): 35 tests beside the existing results, which stay as they are. The case is the unit; tests are paired where the same cases are on both sides and Welch otherwise. Each reports t, df, p, the mean difference with its interval, d_z or Hedges' g, skewness and Shapiro–Wilk, with a Wilcoxon or Mann–Whitney test beside every one. Holm's correction runs over the whole list as one family. The doc states that the data had already been seen; the list was fixed before any t statistic. T7 (development against held-out per-case rates, Welch) is a post hoc analysis of saved held-out results, by alias only, allowed by Parsa on 20 Nov 2026 | Decided — Parsa, 20 Nov 2026 |

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
│   │   ├── rsna_geometry.py       # RSNA slice spacing and field of view from the shard index
│   │   ├── prep_isles24.py        # raw archive → slim tarball (run once)
│   │   ├── prep_rsna.py           # raw DICOM → JPEG tar shards + scan index (run once, on Kaggle)
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
│   │       ├── infer.py
│   │       ├── isles_check.py     # Phase 3: false-positive check on ISLES'24 native NCCT
│   │       └── operating_point.py # Phase 3: cross-fitted threshold on validation patients
│   ├── contract/
│   │   ├── schema.py              # pydantic model of vision_output.json
│   │   ├── build.py               # assemble vision_output from A + B predictions
│   │   └── validate.py            # validate any JSON against schemas/vision_output.json
│   ├── nlp/
│   │   ├── corpus/                # NLP A
│   │   │   ├── ingest.py          # corpus/RULEBOOK.md → passages.jsonl, with checks (DEC-18)
│   │   │   └── index.py           # embeddings; candidate passages for the verifier (DEC-19)
│   │   ├── notes/
│   │   │   ├── facts.py           # fact sheets: what a note states, with verbatim spans
│   │   │   ├── render.py          # facts → note text with controlled variation (scenarios, generate.py)
│   │   │   └── generate.py        # clinical variables → triage note (templated)
│   │   ├── reasoner/
│   │   │   ├── prompt.py          # versioned prompts
│   │   │   ├── run.py             # inputs → structured recommendation
│   │   │   └── providers.py       # thin wrappers for each LLM API / local model
│   │   └── verifier/              # NLP B
│   │       ├── claims.py          # rationale → list of atomic claims
│   │       ├── trace.py           # claim → source (vision field / note / passage) or None
│   │       ├── metrics.py         # hallucination rate, critical rate, omission rate, CIs
│   │       ├── adjudicate.py      # export samples for human labelling, agreement stats
│   │       ├── rules.py           # the rulebook in code: decisions, omissions, criticality
│   │       ├── scenarios.py       # synthetic scenario matrix with rule-derived expected output
│   │       └── seeded_errors.py   # plant known fabrications to measure verifier recall
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
│   ├── claim_extractor_v1.md
│   ├── judge_v2.md                # LLM judge for rule claims (§8.4), DEC-20; v1 kept for the record
│   └── judge_v3.md                # the judge prompt since DEC-23 (12 Nov 2026)
├── notebooks/
│   ├── 00_download_isles24.ipynb  # Phase 0: resumable download to raw/ + md5 (self-contained)
│   ├── 00_template.ipynb          # install, mount, copy tarball, load one case
│   ├── 01a_list_isles24.ipynb     # Phase 1: lists the raw archive without extracting; layout, maps, decode cost
│   ├── 01b_clinical_isles24.ipynb # Phase 1: extracts the Excel files; data dictionary and aggregate column survey
│   ├── 01c_unpack_split_survey.ipynb # Phase 1: working copy, split, spot-check, label and grid evidence
│   ├── 01_prep_isles24.ipynb      # runs prep_isles24; produces slim tarball
│   ├── 01d_check_slim.ipynb       # Phase 1: reads the tarball through the loader; label, MTT and Tmax questions
│   ├── 02_prep_rsna.ipynb         # runs on Kaggle, once per shard
│   ├── 02b_rsna_read_speed.ipynb  # Phase 1: runs on Kaggle; checks the four shards and measures read speed
│   ├── 02c_rsna_repack.ipynb      # Phase 1: runs on Kaggle, once per shard; stores a shard as one file
│   ├── 02a_rsna_kaggle_timing.ipynb  # Phase 0: runs on Kaggle; times full RSNA for DEC-17 (self-contained)
│   ├── 10_visionA_nnunet.ipynb
│   ├── 11_visionA_thresholds.ipynb
│   ├── 20_visionB_train.ipynb     # runs on Kaggle: RSNA split, slice classifier, scan calibration
│   ├── 21_visionB_isles_check.ipynb # Colab: Vision B on ISLES'24 native NCCT (presumed no bleed)
│   ├── 22_visionB_operating_point.ipynb # Kaggle: cross-fitted operating point on the RSNA validation split
│   ├── 23_visionB_rsna_geometry.ipynb # Kaggle, CPU: RSNA training scans' slice spacing and field of view
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
    ├── DATA_USE.md                # what each data source's terms allow, and what that means for us
    ├── CLINICIAN_REVIEW.md        # rulebook and fixture expectations, written for a clinician reviewer
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
  slim/       isles24_slim_v1.tar — immutable; new version = new file (rsna_slim_v1 lives in private Kaggle datasets, DEC-17)
  runs/       EXP-<id>_<short-name>/  config.json, checkpoints/, metrics.json, logs/
  artifacts/  guidelines/ (PDFs, never in the repo), figures/ (exported for the report)
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

The Space's live Anthropic key (DEC-32) lives in the Space's secrets, set by Parsa; it is the only secret allowed there.

### 4.5 Local development (optional)

`pip install -e .` in a venv works for the NLP side and tests. Vision training runs on Colab or a rented pod (§4.6).

### 4.6 Rented GPUs (DEC-21)

Vision A training runs on a Vast.ai RTX 4090 pod rented for one job, driven from the Mac by `scripts/pod` (logic in `src/strokecopilot/compute/`):
- `scripts/pod up EXP-<id>` checks the repo for secrets, the Vast balance (at least 1.5 × the run's estimated cost) and the host's contract (it must run at least 1 h past the run's hour limit, because Vast stops the pod when the contract ends, as it did to EXP-14), creates the pod on an offer Parsa approved from `scripts/pod offers EXP-<id>` (`--offer <id> --go`), sends the tarball and an archive of the repo at the pinned commit, installs, and starts the job detached under `tmux` with its log on the pod. A watchdog on the pod destroys it after the run's hour limit.
- `scripts/pod collect EXP-<id>` copies `runs/EXP-<id>/` (config, metrics, best checkpoint, log) back to Drive, then runs `down`.
- `scripts/pod down EXP-<id>` destroys the pod and confirms by listing instances.

No Drive or GitHub token ever reaches a pod: data and code go up from the Mac over SSH, and results come back the same way. RSNA data never touches a pod (DEC-17, `docs/DATA_USE.md`). A pod is created only after Parsa approves the offer, price and hour limit for that run.

---

## 5. Data

### 5.1 ISLES'24

- Source: Zenodo, latest record 17652035 (20 November 2025, DOI 10.5281/zenodo.17652035), open access, ~99 GB, one `train.7z`. Record the record ID and md5 in `docs/DATA.md`. Earlier records lack `clinical_data-description.xlsx`, which first appears in 16813698. 149 training cases from two centres. Licence: CC BY-NC-SA 4.0, so non-commercial, and derivatives such as the slim tarball must stay under the same licence. Cite the dataset paper (arXiv:2408.11142) and the challenge paper (arXiv:2408.10966); the Zenodo text gives 2408.09259 for the dataset paper, but that ID resolves to an unrelated paper. Cite TopCoW too if the Circle of Willis masks are used.
- **Cohort.** The paper says every patient had interventional reperfusion therapy with near-complete recanalisation (mTICI 2c or 3), and follow-up MRI 2–9 days later. The data is close but not identical: of the 99 rows with a TICI value, 97 are 2c or 3 and 2 are lower, and the rest have no value (`docs/DATA.md`). Consequences: final infarct is a stronger core proxy than in an untreated cohort but will not generalise to patients who are not reperfused; real cases are not expected to include haemorrhage and offer little variety in recommendations, which is why §9 relies on synthetic scenarios.
- Contents per case (file names confirmed against the archive's file list; exact paths in `docs/DATA.md`): acute NCCT, CTA, 4D CTP, perfusion maps (CBF, CBV, MTT, Tmax, from icobrain cva), follow-up DWI/ADC with the final infarct mask, a vessel-occlusion mask, an automatically generated Circle of Willis mask, and baseline and outcome Excel files, one of each per case. Preprocessed images and masks are already in NCCT space.
- Tabular fields (paper): age, sex, atrial fibrillation, hypertension, diabetes, hyperlipidaemia, anticoagulation, statins, antiplatelets, glucose, leucocytes, CRP, INR, referral type (wake-up, in-house, external), times (onset to door, alert to door, door to imaging, door to groin, door to first series, time of intervention, door to recanalisation), NIHSS, mRS (including premorbid), mTICI. No ASPECTS field exists, confirmed from the data dictionary and workbooks, and neither does a thrombolysis field. Column names and missing counts are in `docs/DATA.md`; many cases lack onset time, INR or premorbid mRS. Lab values and times were altered by ±5% for anonymisation according to the paper (the dictionary states it for labs only), which matters near hard cut-offs such as 4.5 h.
- What we use:
  - **Input to Vision A:** perfusion maps (and NCCT if useful).
  - **Target for Vision A:** final infarct mask on follow-up MRI, in NCCT space (linear registration by the data providers; spot-check quality in Phase 1).
  - **Occlusion site:** nearest Circle of Willis label to the occlusion mask, within a distance limit, mapped to ICA / basilar / none / unknown. The labels cannot separate M1 from M2 (Phase 1, `docs/DATA.md`), so occlusions nearest an MCA label are `unknown`.
  - **Clinical variables:** to generate triage notes (§8.3) and to stratify the split.
- Known issue: challenge test data is not public; our held-out set is carved from the 149 training cases.

### 5.2 RSNA Intracranial Haemorrhage

- Source: Kaggle competition dataset. Raw is ~180 GB of DICOM slices with per-slice labels (epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any).
- We use all labelled data (DEC-16) but do not move the raw DICOM to Colab or Drive. The competition data is attached to Kaggle notebooks, where preprocessing and training run (DEC-17). Timing test, 31 Oct 2026: 752,803 labelled slices, 14.3% with a bleed, about 20–28 raw files per second, no failed reads. Raw reading is too slow to train on directly, so preprocessing runs once to 320 px three-window JPEGs in tar shards (`rsna_slim_v1`, estimated 17 GB; 256 px is the fallback). Kaggle unpacks `.tar` files when a dataset is made, and reads from such a folder are too slow, so datasets hold each shard as one `.shard` file (Phase 1, `docs/DATA.md`). Copy shards to Kaggle's local disk each session, about 4 minutes for all four; random reads from the attached file run at only about 16 slices per second. Record the data version, chosen image size and any excluded scans in `docs/DATA.md`.
- Terms: we use the Kaggle copy, so the Kaggle competition rules apply (read 2 November 2026; see `docs/DATA_USE.md`). They allow non-commercial use including academic research and education, and forbid redistribution, access by anyone who has not accepted the rules, and re-identification. Teammates accept the rules before getting access. RSNA data, including shards and indexes, stays in private Kaggle datasets and never goes on Drive, in the repo, or in the report or demo as images. Cite Flanders et al., *Radiol Artif Intell* 2020;2:3. The rules' clause against making copies is an open question for the professors. The RSNA challenge Terms of Use are looser, but they are not the terms for this copy.
- Split by patient, and keep validation and test at the natural bleed rate; only training is balanced. Slices group into scans through DICOM metadata. RSNA's dataset description gives axial 5 mm slices, 512 × 512, with one site resampled to 5 mm; a later preprint reports spacing of about 2.5–5 mm. `SliceThickness` was absent in all 2,000 sampled files, so derive spacing from `ImagePositionPatient`.
- Windowing: stack three CT windows as channels — brain (centre 40, width 80), subdural (80/200), bone (600/2800). This is the standard trick and matters for small bleeds.
- Scan-level labels: a scan is positive if any slice is positive. We need scan-level predictions for the contract.

### 5.3 Guideline corpus

- 2026 AHA/ASA guideline for the early management of acute ischaemic stroke (*Stroke*, doi:10.1161/STR.0000000000000513, published 26 January 2026; it replaces the 2018 guideline and 2019 update), DAWN trial paper, DEFUSE-3 trial paper. Add others only via a Decisions row.
- **Own words (DEC-18).** The corpus is a rulebook we write: one passage per criterion, with numeric criteria kept as facts and no copied wording. Each passage points to its source document, section, recommendation (with COR and LOE) and page, so a reader can check it against the PDF. The PDFs stay on Drive and are never sent to an LLM. DEFUSE-3 passages may be drafted from the PMC author manuscript (PMC6590673), which allows text mining. DAWN passages are drafted from the guideline's description of the trial and checked by a person reading the NEJM paper.
- Each passage gets a stable ID: the source, the section and a short slug, like `AHA2026-4.6.1-within-4.5h`, `AHA2026-T8-ich` (Table 8) or `DEFUSE3-inclusion-core`. The reasoner cites IDs; the verifier resolves them. An ID never changes once a passage is checked.
- The source is `corpus/RULEBOOK.md`, in the repo: humans read and review it, and `ingest.py` builds `corpus/passages.jsonl` from it, so people and the models read the same text. The corpus version is the sha256 of `passages.jsonl`.
- Each passage has a status: `draft`, `checked` (a team member compared it with the PDF) or `reviewed` (a clinician read it). Evaluation runs need every passage at least `checked`.
- The rulebook also holds guideline passages outside the two decisions (for example blood pressure and glucose management), so the reasoner has to pick the rules that apply (DEC-19) rather than being handed only those.

### 5.4 Slim format (v1)

One directory per case inside the tarball:

```
isles24_slim_v1/
  manifest.json              # version, case list, voxel spacing, which maps are present
  clinical.csv               # one row per case
  native_volumes.csv         # infarct and Tmax > 6 s mL, measured before resampling
  sub-strokeXXXX/
    ncct.nii.gz              # resampled to common grid
    ncct_native.nii.gz       # original spacing, HU preserved, for Vision B (§8.2)
    cbf.nii.gz cbv.nii.gz mtt.nii.gz tmax.nii.gz
    cta_occlusion.nii.gz
    cow.nii.gz               # Circle of Willis labels, for occlusion site
    infarct_followup.nii.gz  # target, in CT space
```

Everything is in RAS on one 1.5 mm isotropic grid, cropped to the brain bounding box, grown if needed to hold the masks; `ncct_native` stays the unchanged source file. Only NCCT is averaged before shrinking, because volumes are read from masks and maps by threshold. Map background stays exact: Tmax -30 means no perfusion data. The grid was chosen in Phase 1 on evidence in `docs/DATA.md` and is recorded in `manifest.json`. The tarball is 2.86 GiB, under the original 10–20 GB target.

Slim tarballs are immutable. A preprocessing change produces `_v2`, never an overwrite. Every experiment records which version it used.

### 5.5 Splits

`splits/folds.json`:

```json
{
  "version": 1,
  "stratified_by": "center+infarct_volume_tertile",
  "seed": 20261102,
  "heldout": ["sub-stroke0007", "..."],
  "folds": {"0": ["..."], "1": ["..."], "2": ["..."], "3": ["..."], "4": ["..."]}
}
```

- ~20 held-out cases, chosen once with a fixed seed, stratified by centre and final infarct volume tertile. Centre was added because the two centres differ in size and possibly in what they record.
- 5 folds over the remaining ~129.
- `splits.py` raises if any training function is handed a held-out ID before `ALLOW_HELDOUT=1` is set, which only happens in Phase 4 notebooks.

---

## 6. The contract: `schemas/vision_output.json`

This is the only thing the vision side hands to the NLP side. Frozen at v0.1 on 31 October 2026 (DEC-4). Changes need a Decisions row and a version bump.

### 6.1 Fields (v0.1 — frozen)

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
| `aspects` | int 0–10 \| null | ASPECTS on NCCT; null means unknown, never a low score | none: null from the real pipeline, set in fixtures and synthetic scenarios |
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

Expected recommendations were checked against the 2026 guideline's primary text on 30 October 2026 and still need clinician review. Details are in `fixtures/README.md`.

| Fixture | Scenario | Expected recommendation (to be confirmed against the corpus) |
|---|---|---|
| fx01 | 2.5 h from onset, NIHSS 8, no bleed, small core, M2 occlusion, ASPECTS 9 | thrombolysis: yes; thrombectomy: depends on M2 dominance, which is not reported: reasonable if dominant proximal, not recommended otherwise |
| fx02 | 9 h from LKW, NIHSS 16, no bleed, core 18 mL, penumbra 95 mL, M1, ASPECTS 8 | thrombolysis: no (eligible for thrombectomy, so the extended-window recommendation does not apply); thrombectomy: yes (6–24 h, ASPECTS ≥ 6; also meets DEFUSE-3 mismatch) |
| fx03 | 2 h, NIHSS 12, haemorrhage present, ASPECTS null | thrombolysis: no (absolute contraindication); thrombectomy: no |
| fx04 | 6 h, age 71, NIHSS 18, mRS 0, core 110 mL, penumbra 120 mL, M1, ASPECTS 4 | thrombolysis: no (outside 4.5 h, eligible for thrombectomy, little salvageable tissue); thrombectomy: yes (ASPECTS 3–5, must state mass effect is not reported). Fails DEFUSE-3, so it catches pre-2026 knowledge |
| fx05 | wake-up stroke, LKW 10 h ago, sleep midpoint about 5 h 45 min before imaging, NIHSS 11, age 82, core 15 mL, M1, ASPECTS null | thrombolysis: no while thrombectomy is possible; the sleep-midpoint rule applies only if it is not, must cite; thrombectomy: yes on DAWN criteria, borderline because ASPECTS is unknown and DEFUSE-3 is narrowly missed, must cite |

### 6.4 Reasoner output: `schemas/recommendation.json`

```json
{
  "case_id": "...",
  "model": "gpt-4o-2024-08-06",
  "prompt_version": "reasoner_v1",
  "thrombolysis": {"recommend": false, "rule_ids": ["AHA2026-4.6.3-rec2"], "blocking_factors": ["last known well 10 h ago", "eligible for thrombectomy"]},
  "thrombectomy": {"recommend": true, "rule_ids": ["DEFUSE3-inclusion-2"], "blocking_factors": []},
  "rationale": [
    {"claim": "Infarct core volume is 18.0 mL.", "sources": [{"type": "vision", "field": "core_ml"}], "derived": false},
    {"claim": "The patient was last known well 10 h before imaging.", "sources": [{"type": "note", "quote": "Time from last known well to imaging: 10 h 0 min."}], "derived": false},
    {"claim": "DEFUSE-3 requires core volume under 70 mL.", "sources": [{"type": "passage", "id": "DEFUSE3-inclusion-2"}], "derived": false},
    {"claim": "The core of 18.0 mL is within the DEFUSE-3 limit.", "sources": [{"type": "vision", "field": "core_ml"}, {"type": "passage", "id": "DEFUSE3-inclusion-2"}], "derived": true}
  ],
  "uncertainties": ["onset time is last-known-well, not witnessed"]
}
```

Passage IDs above are illustrative. A claim cites everything it rests on, and `derived: true` marks a calculation or comparison over its sources rather than something they state directly. The reasoner is asked to produce this structure directly. The verifier checks the `sources` the model claimed *and* independently re-traces each claim; disagreement between the two is itself a recorded error type.

---

## 7. Phases

Weeks are relative to the kickoff (week 1 starts when the lead assigns the first Phase 2 tasks). Deadline: _fill in once professors confirm_. If the deadline is shorter than 8 weeks, cut Phase 3 scope first (fewer LLMs, no occlusion-site model), never Phase 4.

### Phase 0 — Setup (this week; lead alone)

Goal: decisions recorded, contract frozen, data downloading, repo scaffolded.

- [ ] Get written confirmation from both professors for the switch (DEC-1 → Decided); agree deliverables (§12)
- [ ] Create shared Drive folder with the layout in §4.2; share with team
- [x] Start ISLES'24 download (record 17652035) to `raw/` with `00_download_isles24.ipynb`; verify md5; record sizes
- [x] Attach full RSNA to a Kaggle notebook; run `02a_rsna_kaggle_timing.ipynb`; settle DEC-17
- [x] Read Zenodo licence and RSNA terms; fill `docs/` note and the Notion "Data-use terms" page
- [x] Obtain the 2026 AHA/ASA guideline full text; verify every rule in §8.3 and the fixture expectations against it
- [ ] Ask a clinician to review the rulebook and fixture expectations once (via the professors if the team has no contact)
- [x] Freeze contract v0.1 (DEC-14 and DEC-15 are confirmed)
- [x] Scaffold repo per §3: package skeleton, `pyproject.toml`, `CLAUDE.md`, `CHANGELOG.md`, `README.md`, `.gitignore`
- [x] Write `schemas/vision_output.json` (§6.1) and `contract/schema.py` (pydantic mirror)
- [x] Write the five fixtures and their notes (§6.3)
- [x] `tests/test_contract.py`: fixtures validate; mismatch ratio and volume computed correctly; null handling
- [x] Push to GitHub (private), add repo URL to Notion hub Links

Acceptance: `pytest tests/test_contract.py` passes; the five fixtures validate; the ISLES'24 download is complete or in progress with its md5 check queued; full RSNA is attached on Kaggle and timed.

### Phase 1 — Data (week 1; lead alone)

Goal: slim tarballs ready, and loaders working on real data for both datasets. Teammates are brought in later (DEC-6).

- [x] Inspect ISLES'24 archive layout; write `docs/DATA.md` with exact paths and which perfusion maps exist
- [x] Decide the common grid (spacing, crop) → record in `manifest.json` and CHANGELOG
- [x] Spot-check that the provided infarct masks align with NCCT (the paper says they are already in NCCT space); implement registration only if this fails
- [x] Read `clinical_data-description.xlsx`: confirm the fields in §5.1, and tag each column as allowed or forbidden for note generation (§8.3)
- [x] Check whether the Circle of Willis labels separate M1 from M2
- [x] `prep_isles24.py` + `01_prep_isles24.ipynb` → `slim/isles24_slim_v1.tar`
- [x] `prep_rsna.py` + `02_prep_rsna.ipynb` on Kaggle: 320 px windowed JPEG tar shards plus a scan-level index with patient IDs and slice positions; patient-level splits are made from that index
- [x] `splits.py` + `splits/folds.json` generated with fixed seed; `tests/test_splits.py`
- [x] `isles24.py`: `load_case(id)` returns arrays + clinical row + spacing; `rsna.py` equivalents
- [x] README with Open-in-Colab badges

Acceptance: `slim/isles24_slim_v1.tar` and the RSNA shards are ready, `isles24.py` and `rsna.py` work on the real data, and RSNA read speed is measured and fit for training. Changed by Parsa on 2 November 2026 (DEC-6); it was every teammate running `00_template`.

**Deferred, not done (DEC-6).** Do these when the lead first brings someone in. Nobody has been onboarded yet:

- [ ] `00_template.ipynb`: install, mount, copy tarball, load one case, show maps + mask + clinical row, and warn against opening or attaching `splits/folds.json`. Must run top to bottom in five minutes on a T4 — written 6 Nov 2026 (`data/template.py`, tested; the figure checked locally on one development case); awaits its first T4 run
- [ ] Onboard each teammate as they are assigned work: share repo + Notion + the Drive folder; fill the Person column; they accept the RSNA competition rules before getting dataset access; they run `00_template` — started 6 Nov 2026: Lorna, Mohamed, Hana and Benny invited to Notion; the access steps are in `docs/team/README.md`
- [x] Assign Phase 2 tasks per role, with the notes in `docs/DATA.md`: Tmax and `native_volumes.csv` for Vision A; missing clinical data, time text and `unknown` occlusion site for NLP A — superseded: Phase 2 was done without the team. Phase 3 roles agreed by Parsa on 6 Nov 2026, with one brief each in `docs/team/`

### Phase 2 — Baselines (weeks 1–2; whole team)

Goal: every box in the diagram does *something*, measured, even if badly.

**Vision A (perfusion)**
- [x] `thresholds.py`: core = rCBF < 30% of contralateral, penumbra = Tmax > 6 s; compute volumes; evaluate Dice of thresholded core vs final infarct per fold — this is the clinical-standard baseline (EXP-1: Dice 0.111 ± 0.031, abs volume error 25.2 mL; `docs/DATA.md`)
- [x] nnU-Net v2 dataset conversion (perfusion maps as channels, final infarct as label); train fold 0 with reduced epochs; checkpoint/resume to Drive; report Dice — done 5 Nov 2026 as EXP-14, at the full 300 epochs on a rented RTX 4090 (DEC-21) instead of reduced epochs: fold 0 Dice 0.218 ± 0.249 (median 0.111), abs volume error 19.6 mL, against the threshold baseline's 0.123 and 20.5 mL on the same 26 cases. Checkpoint/resume proven live: Vast stopped the pod at epoch 270 when the host's contract ended, and training resumed from the epoch-269 checkpoint. Checkpoints on Drive `runs/EXP-14_nnunet_fold0/`. Cost 2.14 USD
- [x] `volumes.py`: mask → mL using spacing; mismatch ratio/volume via `contract.schema.derive_mismatch`; occlusion-site lookup from the occlusion mask and Circle of Willis labels (distance limit 5 mm, from EXP-1)

**Vision B (haemorrhage)**
- [x] 2D classifier (pretrained ResNet-50 or EfficientNet-B3) on windowed slices, "any" label; slice AUC — EXP-6 (ResNet-50, 10 epochs, Kaggle T4): validation slice AUC 0.975
- [x] `aggregate.py`: slice → scan (max and top-k mean); scan AUC; temperature scaling for calibration; reliability diagram — EXP-6: top-2 mean chosen (scan AUC 0.9772 vs max 0.9771), Platt calibration chosen over temperature (ECE 0.014 vs 0.131 out of half; the bias undoes the balanced-epoch prevalence shift), validation scan AUC 0.977, ECE 0.010, sensitivity 0.951 at specificity 0.917 (2,166 scans; threshold 0.278 set on the same split for 95% sensitivity, so specificity is optimistic)
- [x] `infer.py`: scan → `{present, confidence}` — code and tests done 4 Nov 2026; the EXP-6 calibrator plugs in through `calibration.json`

**NLP A (reasoner)**
- [x] `ingest.py`: guideline PDF + two trial papers → passages with stable IDs; `docs/RULEBOOK.md` generated from the same source — changed by DEC-18: `corpus/RULEBOOK.md` (own words) → `corpus/passages.jsonl`, with checks. Code, tests and 47 passages done 4 Nov 2026: all 43 AHA passages checked against the guideline, off-topic passages added, copy check clean. Done 5 Nov 2026: Parsa checked the 2 DAWN and 2 DEFUSE-3 passages against the trial protocols (supplementary appendices), with pages; 15 of 18 facts matched, 1 found on a second look (DEFUSE-3 pre-stroke mRS 0–2), and 2 corrected: DEFUSE-3 ratio above 1.8 and mismatch above 15 mL (strict), in the passage and in `rules.py` (no scenario decision changed). All 47 passages checked
- [x] `index.py`: embed passages; retrieve top-k by query; sanity-check that each fixture's expected rule IDs are retrievable — done 4 Nov 2026. The check (EXP-7, bge-base) found 9 of 15 fixture rules, so the reasoner gets the whole rulebook (DEC-19) and `index.py` serves only the verifier's candidates
- [x] `notes/generate.py`: clinical row → triage note (templated with controlled variation), using allowed columns only; test that nothing post-treatment leaks; produce notes for all 149 cases — done 5 Nov 2026 for the 129 development cases (the 20 held-out cases follow in Phase 4, under the held-out rule). Durations only, no invented clock times; missing rule inputs written as not recorded. Run on Drive `artifacts/isles24_notes_v1/`: onset time known in 45, wake-up 36 (last known well never recorded), no onset time in 44, onset to door without door to imaging in 4; not recorded: pre-stroke mRS 52, INR 53, NIHSS 4, anticoagulation 3
- [x] `reasoner/run.py` v0 with `prompts/reasoner_v1.md`: fixture JSON + note + the whole rulebook (DEC-19) → `recommendation.json`; run on all five fixtures with one model — code, prompt, tests and `31_nlpA_reasoner.ipynb` done 4 Nov 2026 with a fake model; EXP-8 (dummy) and EXP-9 (five fixtures × 2, `claude-sonnet-5-5`): 0 format failures, 18/18 judged decisions right, $0.37

**NLP B (verifier & eval)**
- [x] Write the hallucination metric definition (§8.4) in `docs/` and Notion — before code (`docs/HALLUCINATION_METRIC.md`, approved 3 Nov 2026)
- [x] `claims.py`: rationale → atomic claims (LLM-assisted with `prompts/claim_extractor_v1.md`, plus a rule-based numeric extractor) — code and tests done 4 Nov 2026; real extractor (`claude-sonnet-5-5`) run in EXP-10 and EXP-11 on 75 outputs with no failures
- [x] `trace.py`: numeric claims vs vision JSON (after rounding); patient facts vs note; rule claims vs passage text (embedding similarity + NLI or LLM judge — done 4 Nov 2026: LLM judge (`prompts/judge_v2.md`), `claude-haiku-5-5` with τ = 0.6 (DEC-20, from the EXP-11 bake-off)
- [x] `metrics.py`: hallucination rate, critical hallucination rate, omission rate; bootstrap CIs (criticality and omissions are filled in once `scenarios.py` exists)
- [x] `scenarios.py`: 100–200 synthetic scenarios with expected recommendations derived from the rulebook in code; review the rule code against the corpus — done 5 Nov 2026: `rules.py` (rulebook in code, omissions, criticality), 150 scenarios (seed 20261105) covering 16 decision routes at least 3 times each; a test checks every rule threshold against its passage. The rule conventions await clinician review
- [x] `seeded_errors.py`: plant known fabrications of each sub-type and measure verifier recall and false-flag rate — done 4 Nov 2026 (EXP-11): 5 gold outputs and 70 seeds; Haiku judge, τ ≤ 0.7: strict recall 1.0 on five sub-types, 0.9 wrong rule, 0.8 uncited rule, false flags 0.010; bar met
- [x] Run verifier on the five fixture outputs; `tests/test_claims.py`, `tests/test_trace.py` — EXP-12 (`50_end_to_end.ipynb`, 4 Nov 2026): 371 claims, verifier rate 0.235. Not yet adjudicated, and most flags look like verifier issues on real output (see CHANGELOG)

**Everyone**
- [x] Every run above has an Experiments row and a `runs/EXP-*` folder with `metrics.json` — checked 5 Nov 2026 for EXP-1 to EXP-14 (EXP-11 splits its metrics across files; EXP-3 and EXP-5 got their `metrics.json` in this check)

Acceptance: a single notebook (`50_end_to_end.ipynb`, fixture mode) runs fixture → reasoner → verifier and prints a hallucination rate. Vision baselines have fold-0 numbers in Experiments.

### Phase 3 — Build (weeks 3–5)

Goal: each component made good; real vision outputs replace fixtures; the multi-LLM table exists.

Owners (agreed by Parsa, 6 Nov 2026; briefs in `docs/team/`): Parsa keeps the long GPU runs, the paid model runs, `contract/build.py`, the verifier fixes and the decisions. Lorna: Vision A error analysis and confidence. Mohamed: Vision B operating point and ISLES'24 check. Hana: model shortlist, prompt v2, providers, real inputs. Benny: error taxonomy, comparison notebook, labelling kit. Lorna and Mohamed are the blind labellers.

**Vision A**
- [x] All 5 folds for nnU-Net; mean ± std Dice; compare against threshold baseline — done 13 Nov 2026 as EXP-14, EXP-18..21 and the summary EXP-25: Dice 0.215 ± 0.046 over 5 folds against the threshold's 0.111 ± 0.031 (paired +0.104, 95% CI +0.078 to +0.133); abs volume error 23.5 ± 3.9 mL against 25.2 ± 5.0 mL (paired −1.6 mL, CI −4.5 to +1.3, not a clear difference); wrong side of 21/31/51/70 mL in 30/28/20/17 of 129 against 35/29/23/17. Fold costs measured from the Vast credit, about $8.8 for folds 1–4 with failed attempts
- [x] Core/penumbra heads or a two-stage approach (predict final infarct = core proxy; penumbra from Tmax) — decide and record — DEC-26, 13 Nov 2026: two-stage (nnU-Net core proxy, Tmax > 6 s penumbra), no heads and no volume calibration; EXP-26
- [x] Decide which core feeds the contract: rCBF < 30% threshold or learned model, compared on volume error and decision-threshold crossings (§8.1) — DEC-25, 13 Nov 2026: nnU-Net (each development case from the fold model that did not train on it). On EXP-25 it is no worse on volume error or crossings and clearly better on Dice; its volume is not measurably better, which the report states
- [x] Confidence estimates (`confidence.core/penumbra`). Done by Lorna (EXP-30, EXP-31; `docs/CONFIDENCE_VISION_A.md`).
  - `confidence.core` comes from inference-time signals: missing Tmax near the at-risk area, predicted core size, disagreement with the threshold core, and distance to the nearest cut-off. That is version v2, chosen from three. A logistic map, cross-fitted by fold, turns them into a score.
  - `confidence.penumbra` is a data-quality score: 1 minus the share of the perfusion-covered tissue with no reliable Tmax.
  - Fold-ensemble disagreement and MC dropout were not used, because each development case was trained on by 4 of the 5 fold models.
  - DEC-27 (15 Nov 2026): the contract uses the cut-off distance alone, not v2 (equivalent cross-fitted, one input; EXP-32), plus a warning when the core is under 1 mL.
- [x] Error analysis: where does the model fail (small lesions, posterior circulation, artefacts)? Done by Lorna: EXP-15 (fold 0) and EXP-27 (all five folds); write-up in `docs/ERROR_ANALYSIS_VISION_A.md`.
- [x] `contract/build.py` produces valid `vision_output.json` for every non-held-out case; validate all — done 15 Nov 2026 as EXP-33 at `5acea14`: all 129 development cases built and validated, on Drive `artifacts/vision_outputs_v1/`. Core from the nnU-Net fold runs (DEC-25), penumbra and occlusion evidence from EXP-1 (DEC-7, DEC-26), haemorrhage from EXP-16 at 0.175 (DEC-28), confidence from EXP-32 (DEC-27), ASPECTS null (DEC-9). Counts: bleed flagged 68; core under 1 mL 30 (near-empty warning, mismatch ratio null); occlusion site ICA 13, basilar 1, unknown 115 (the §8.1 rule: MCA-nearest is unknown); no negative mismatch volume

**Vision B**
- [x] Tune aggregation and threshold for high sensitivity at acceptable specificity (false "no bleed" is the worst failure in the system — state the operating point and why) — EXP-17 (Mohamed, Kaggle, `ff522bd`): EXP-6's top-2 aggregation kept; threshold cross-fitted on validation patient halves (20 seeds). Proposed 0.175 (96% target): sensitivity 0.961 (95% CI 0.946–0.972), specificity 0.878 (0.859–0.895); on ISLES'24 it would flag 68 of 129 presumed bleed-free scans (52.7%). **DEC-28 (15 Nov 2026): 0.175 adopted** (`infer.CONTRACT_THRESHOLD`, applied over EXP-6's calibration by `contract_calibrator`)
- [x] Domain shift check: run on ISLES'24 NCCT (which should mostly be negative); report the false-positive rate — EXP-16 (Mohamed, 8 Nov 2026, `23e2f92`): at the provisional 0.278, 45 of 129 development scans called positive (34.9%, 95% CI 27.2–43.4%) on 5 mm slabs, 61 of 129 on native thin slices; review of the top scans and a check of RSNA's slice thickness on Kaggle still open; EXP-28 (14 Nov 2026, notebook 23): RSNA training scans' slice spacing median 5.037 mm (66.0% within 0.25 mm of 5 mm), field of view median 250.0 mm (p5–p95 216.0–253.3); EXP-29 (14 Nov 2026, `fea2c15`): with the field of view matched to 250 mm, 42 of 129 (32.6%) against 45 of 129 reproduced as in EXP-16; paired flips 11 and 8 (p = 0.65), so the field of view does not explain the rate. At EXP-17's proposed 0.175: 68 of 129 (52.7%, 95% CI 44.1–61.1%) unmatched against 64 of 129 (49.6%, 41.1–58.1%) matched; 54 under both, flips 14 and 10 (p ≈ 0.54). ISLES'24's field of view is median 229.0 mm (p25 215.6, p75 250.2), and 33 of 129 scans are over 250 mm, so matching crops about a quarter rather than padding them; whether any head was cut is unverified. Top-scan review, partial (read from Mohamed's Drive sheet at review, not in `per_case.csv`): of the 8 highest-scoring scans at 5 mm (confidence 0.66–0.99, above both thresholds), calcification 3, hyperdense vessel 2, artefact 2, nothing obvious 1, none read as a bleed. Open, needs the images: the scans between 0.175 and 0.278 by cause, and a check of the 33 cropped scans. **Done 16 Nov 2026 as EXP-35** (Parsa's session, Mohamed's two open checks): centre-cropping to 250 mm cut no head or brain on any of the 33 large-field scans (heads span about 18 x 15 cm); the 23 scans flagged at 0.175 but not 0.278, read from their top two slabs by Claude (not a radiologist): skull-base slab 12, normal calcification 4, dense falx 3, dense vessels 1, nothing obvious 3, none read as a bleed
- [x] Item 2 (minimal): why Vision B flags presumed bleed-free ISLES'24 scans, and whether a slab rule fixes it. Done 19 Nov 2026, pre-registered in `docs/ITEM2_FALSE_ALARMS.md` (approved by Parsa before any score). EXP-49 and EXP-51 (RSNA validation): of 10 candidate scan-score rules, gate(0.55) met the adoption rule, with specificity +0.47 pts [+0.15, +0.86] at 96% sensitivity and subtypes unchanged. EXP-50 (ISLES'24): it clears 4 of the 68 flags (p = 0.125). The flagged scans' top slabs sit in brain tissue (Grad-CAM median 76% of heat in brain), not at the skull base, so this is a domain shift no slab rule fixes. **DEC-28 stands (Parsa, 19 Nov 2026)**; Item 2 is reported as an explanation, with that negative result
- [ ] Optional: per-subtype outputs if time allows — not part of Phase 3's acceptance. To be done report-only, as linear probes on EXP-6's frozen features with Item 1, in the Phase 3 session after the close (Parsa, 19 Nov 2026). EXP-6, the contract and DEC-28 stay unchanged (DEC-31's freeze)
- [ ] Item 1 (minimal, report only): BiomedCLIP zero-shot and as a linear probe against EXP-6's ResNet-50 on RSNA validation, with ISLES'24 development flag counts. It needs a Decisions row; it never touches held-out cases or the RSNA test split, and does not feed the pipeline (DEC-31). Phase 3 session, after the close

**NLP A**
- [x] Switch reasoner inputs from fixtures to real `vision_output.json` + generated notes — done 16 Nov 2026 (Hana's task, done by Parsa's session): `end_to_end.load_cases("real", ...)` with rule-derived expectations, `sample_real_ids` (20 cases, 4 per fold, seeded); ground-truth contracts for the DEC-11 arm built as EXP-34 (`artifacts/vision_outputs_gt_v1/`: reference final infarct as core, no bleed, confidence 1.0). All 129 notes load and pass their fact check
- [x] Prompt v2 informed by Phase 2 verifier findings (what did the model get wrong?) — `prompts/reasoner_v2.md` written 9 Nov 2026 (Hana, from the EXP-12 labels) — v1 vs v2 checked 16 Nov 2026 as EXP-38 (5 fixtures, Sonnet 5.5, verified live): both 10/10 decisions right, no format failures or omissions; v2 writes 154 claims against v1's 339, with no wrong numbers, invented findings, faulty derivations or citation errors (v1: 15), at a similar flag rate (8.4% vs 8.0%). v2 chosen. Its 9 critical flags were all verifier mis-scores (checked 16 Nov 2026): 3 from thrombectomy passages listed under thrombolysis's `rule_ids`, 6 from the judge on "eligible for thrombectomy, so extended-window thrombolysis does not apply". **v3** (`prompts/reasoner_v3.md`: v2 plus one sentence, rule_ids name only passages about their own treatment; without-rulebook prompt unchanged) checked as EXP-39: 10/10 decisions, 142 claims, 5 flagged, 4 critical, no `rule_ids` flags, no wrong numbers, invented findings or faulty derivations. **v3 is the paid runs' prompt** (`paid_runs.PROMPT`)
- [x] `providers.py`: at least three models — one frontier API model, one open mid-size model (e.g. Llama-class via HF or API), one small medical model. Exact list → Decisions row — DEC-24; `VLLMJSON` added 14 Nov 2026 (Hana) for Qwen3.5-27B, MedGemma 27B text and Gemma 3 27B on vLLM, tested with fakes — adapter in PR #8 — done 16 Nov 2026: four models answered in EXP-37 (Claude Sonnet 5.5 through the Batch API, Qwen3.5-27B through OpenRouter pinned to Alibaba with the system prompt merged into the user message, Gemma 3 27B and MedGemma 27B text on vLLM on a Colab A100, DEC-24 amended)
- [x] Run all three on all fixtures, the synthetic scenarios and a sample of real cases, with and without the rulebook; outputs to `runs/` — done 16 Nov 2026 as EXP-37 (`runs/EXP-37_paid_runs/`, on Drive): 4 models × 7 arms (DEC-29) plus two Sonnet repeats, all verified; Anthropic about $26.90 (estimate), OpenRouter about $2.95, Colab about 2 A100 hours. Qwen's first pass (system prompt dropped by the provider) set aside and redone

- [x] Fine-tuned small reasoner (DEC-30, Item 4) — done 19 Nov 2026. Gemma 3 4B, LoRA-tuned on 1,080 synthetic gold answers (EXP-43, EXP-45), and the same model prompted, both on vLLM (EXP-46, EXP-47), reported as EXP-48. Hallucination rate tuned against prompted, paired: 14.3% against 41.6% over all 175 cases; 17.4% against 42.0% on the 51 held-out-route scenarios (−24.6 pts [−28.6, −20.7]). Lower than Gemma 3 27B and MedGemma 27B over all 175 cases, level with them on held-out routes, above Sonnet and Qwen. **Reported as a finding (Parsa, 19 Nov 2026), not tuned away:** the tuned model under-recommends treatment (thrombectomy yes caught 10/35, 3/18 on held-out routes; thrombolysis 24/42), and its critical rate is higher (+1.9 pts [+0.4, +3.4]). The likely cause is the training set's base rate (20% thrombectomy yes), which is unverified. No rebalanced variant is run. Verifier rates, not yet checked against the human labels

**NLP B**
- [x] `41_nlpB_model_comparison.ipynb`: the hallucination table — per model, per scenario, with CIs — done 17 Nov 2026 as EXP-42 (`pipeline/compare.c9_report`, no API calls): every model × arm of EXP-37 with 95% case-bootstrap intervals, the pooled headline per model, and paired comparisons on the same cases (rulebook effect on the content rate, DEC-29; predicted vs ground-truth vision, DEC-11; each model vs Sonnet). With the rulebook, rescored 19 Nov 2026 under metric points 19 and 20 (from EXP-41; no API calls; `runs/EXP-42_c9_table/rescore_points_19_20/`): Sonnet 4.0% [3.3, 4.8], Qwen 5.0% [4.2, 5.8], Gemma 17.1% [15.7, 18.6], MedGemma 17.7% [15.9, 19.5]. Qwen minus Sonnet is +1.0 pts [−0.1, +2.1], so not clearly different. Content and critical rates are unchanged. First scoring: 4.4%, 6.2%, 18.6% and 18.3%. Verifier rates; the human check (EXP-41) is added when the labels are in
- [x] Human adjudication: two team members label ~100 claims, sampled across models and blind to the model; report inter-annotator agreement and verifier-vs-human agreement — done 19 Nov 2026 as EXP-41. 100 claims from EXP-37's rulebook arms, four labellers in two mixed pairs (Lorna + Benny, Mohamed + Hana), 50 each, every claim labelled twice. Human–human κ 0.27 (pairs 0.05 and 0.33); most disagreements were true-but-uncited claims. Parsa settled the 28 disagreements as third rater; S035 was changed to no error after he checked the passage. Verifier–human agreement on all 100: **0.61** with the verifier rescored under metric points 19 and 20, which came from these labels (0.48 at the first scoring); weighted by the sampling strata, 0.87. Verifier-clean claims agree 43/46. The verifier caught 18 of the 21 human-judged errors, and humans confirmed 18 of its 54 flags (content 14/33, citation 4/21), so its rates are upper bounds. Population human error rates per model are not estimated: about 8 clean claims per model give intervals too wide to use
- [x] Error taxonomy with examples: wrong number, invented finding, wrong rule, misapplied rule, faulty derivation, uncited rule, mis-attributed source, omission — done: `docs/ERROR_TAXONOMY.md` (Benny, PR #5, merged 15 Nov 2026)

Acceptance: the hallucination table exists with ≥3 models and CIs; vision outputs validate for all training cases; adjudication agreement reported.

**Phase 3 closed 19 Nov 2026 (Parsa).** All three acceptance criteria are met: C9 (EXP-42, rescored), all 129 development contracts validated (EXP-33), and adjudication agreement reported (EXP-41). Item 4 (DEC-30) and Item 2 are done; DEC-28 stands. Phase 4 runs in its own session from DEC-31 (`docs/PHASE4_PREREGISTRATION.md`). Item 1 and the per-subtype probes continue in this session, report only

### Phase 4 — Integration and evaluation (weeks 6–7)

Goal: end-to-end on held-out cases; error propagation analysis; demo.

- [x] Vision B's RSNA test split, opened once (DEC-22, DEC-31 §1 item 3) — done 19 Nov 2026 as EXP-60 (Kaggle T4, `63b3390`, 8.8 min, free; a dry run on validation first reproduced EXP-6: scan AUC 0.9772). EXP-6 with DEC-28's 0.175, 2,180 test scans of 1,893 patients (bleed rate 0.4119): scan AUC **0.9805** [0.9743, 0.9859], sensitivity **0.962** [0.948, 0.975], specificity **0.885** [0.867, 0.904] (patient bootstrap), 3.8 missed bleeds per 100 bleeds and 11.5 false alarms per 100 non-bleeds, ECE **0.015**. By subtype: epidural 27/28, intraparenchymal 521/534, intraventricular 368/370, subarachnoid 341/353, subdural 400/409. Agrees with EXP-17's cross-fitted validation estimate (0.961, 0.878). The test split is not to be opened again
- [x] Held-out contract path (DEC-31 D1), built and tested on development cases — done 19 Nov 2026 as EXP-61 (Colab A100, `5b5594b`, `contract/heldout.py`, notebook 60 `dev_check`), 7/7 checks. The five checkpoints' plans agree with each other, their `debug.json` and their recorded plans, and DEC-31's rebuild (`nnUNetv2_plan_and_preprocess --no_pp`) matches them. Re-predicted by their validating fold, 10 development cases reproduce the recorded masks (8 identical; at most 3 voxels differ; Dice ≥ 0.9998; volume within 0.01 mL). The ensemble runs at 3.0 s per case. The held-out code rebuilds all 129 EXP-33 contracts with no field different (all but `confidence.core`, in-sample on development cases), and Vision B reproduces EXP-16's 68 flags (largest confidence difference 0.0037). The run folders are named exactly, after a first try matched EXP-3's older fold 0 folder
- [x] Error-propagation and sensitivity code (DEC-31 D3), tested on development cases — done 19 Nov 2026 as EXP-63 (`pipeline/heldout_analysis.py`); it reproduces EXP-25's and EXP-16's numbers. The supplementary scenario sensitivity was added before unlocking (DEC-31 §4)
- [x] Unlock held-out cases (`ALLOW_HELDOUT=1` in Phase 4 notebooks only); never retrain after this — done 19 Nov 2026 as EXP-62 (notebook 60 `heldout`, Colab A100, `e9bb941`, 4/4 checks). The 20 contracts (five-fold ensemble), ground-truth contracts and notes are on Drive (`artifacts/vision_outputs_heldout_v1`, `vision_outputs_gt_heldout_v1`, `isles24_notes_heldout_v1`). Bleed flagged 10/20; occlusion site unknown in 20/20; core median 1.85 mL predicted against 11.0 mL reference; 7 near-empty cores. DEC-31's freeze list holds from here. From 19 Nov 2026 (DEC-31 §4, Parsa), Phase 4 code paths on the Mac may also set the flag, per command
- [x] `pipeline/end_to_end.py` on all ~20 held-out cases × 3 models, with predicted vision values and with ground-truth (oracle) values — models, arms and analyses fixed by DEC-31 (`docs/PHASE4_PREREGISTRATION.md`, 19 Nov 2026): six models, 20 held-out cases — done 20 Nov 2026 as EXP-65 (`pipeline/phase4.py`; 14 cells: six models × two arms, plus Sonnet's core ×0.8 and ×1.2) and EXP-66 (the D3 report). Correctness rests on 23 scored decisions (thrombectomy "no" in all 20 truths; thrombolysis undetermined in 17): predicted vision Sonnet 22/23, Qwen 13/13 (9 format failures), Gemma 3 27B 22/23, MedGemma 21/21, Gemma 3 4B 3/23, tuned 4B 21/23. Hallucination rate held-out vs development real cases (predicted vision): Sonnet 3.2% vs 3.3%, Qwen 6.3% vs 3.3%, Gemma 11.6% vs 10.9%, MedGemma 10.0% vs 7.9%, 4B 40.4% vs 35.1%, tuned 4B 21.5% vs 12.3%. Spend: Anthropic $4.73, OpenRouter $0.28 (estimates)
- [x] Error propagation: for each wrong recommendation, was the cause vision error, reasoning error, or verifier miss? Table + examples — done 20 Nov 2026 in EXP-66 (`runs/EXP-66_heldout_analysis/report.md`, aliases only): Sonnet 1 wrong (vision error, a false bleed flag, missed by the verifier); Gemma 3 27B 1 (reasoning, caught); tuned 4B 2 (one vision error from a bleed flag, caught; one reasoning, caught); base 4B 20 (all reasoning, all caught); Qwen and MedGemma 0
- [x] Sensitivity analysis: perturb `core_ml` by ±20% and see which recommendations flip — shows how much vision accuracy matters clinically — done 20 Nov 2026 in EXP-66: 0 of 40 rule decisions and 0 of 40 of Sonnet's own answers change on the held-out contracts (×0.8 and ×1.2 each). The DEC-31 §4 supplement on the 150 scenarios: 9/300 and 10/300 rule decisions change (EXP-63)
- [x] The public HF Space (DEC-32, `docs/DEC-32_PUBLIC_SPACE.md`): story, case explorer with clickable claims and flags in red, results charts, live scenario builder with precomputed fallback; banner on every screen. Built on development results before unlocking, filled with held-out results after the analyses, load-tested, and published on Parsa's word. Replaces the Gradio `demo/app.py` (changed 19 Nov 2026) — done 20 Nov 2026: public at https://huggingface.co/spaces/parsa-hke/stroke-copilot (Parsa's account; private staging first, then public on Parsa's word). 195 cases (20 held-out by alias), EXP-66 results cards; tree scan clean for every case ID; load test on the Space 50/50, 0 errors (live off, $0); one live run 51.6 s, $0.108; daily cap $5 (Space variable).
- [x] Export figures to `artifacts/figures/` — done 20 Nov 2026 (`pipeline/figures.py`, Drive `artifacts/figures/`, PNG and SVG with a README): seven development figures and the held-out realism check (`fig_heldout_realism_check`)

Acceptance: held-out results in Experiments. The public Space is live: its live element is load-tested and falls back to precomputed results, the not-for-clinical-use banner is on every screen, and its data follows the licence check (DEC-32 §6). (Changed by DEC-32, 19 Nov 2026; it was "demo runs from a fresh Colab session in under 10 minutes".) **Met 20 Nov 2026** (EXP-65 and EXP-66 in Experiments; the Space public, load-tested, falling back, banner on every screen).

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

**Targets.** Final infarct mask from follow-up MRI (ISLES'24 ground truth). Note the conceptual gap: final infarct ≠ core at imaging time — it includes tissue lost before recanalisation, and the mask is drawn 2–9 days later. Because the cohort is reported to have reached mTICI 2c/3 (§5.1 has the caveat), the gap is smaller than in an untreated cohort, but the model is only valid for reperfused patients. Report this limitation and handle it as follows:
- *Core proxy:* model trained to predict final infarct, evaluated as the "core" estimate. Also report the clinical threshold baseline (rCBF < 30%). Trial cut-offs such as 70 mL were defined on perfusion-derived core, not final infarct, so the core that feeds the contract is chosen by comparison in Phase 3, not assumed to be the learned one.
- *Penumbra:* `penumbra_ml` is the total Tmax > 6 s volume, core included — the clinical convention (DEFUSE-3 used RAPID with these thresholds), so `mismatch_ratio` and `mismatch_volume_ml` (§6.1) match the DEFUSE-3 definitions directly. Salvageable tissue is `mismatch_volume_ml`. Document this as a rule, not a learned output, unless a learned version clearly beats it.

**Model.** nnU-Net v2, 3D full-resolution configuration, reduced epochs to fit Colab sessions, resume from Drive. Do not build a custom U-Net in Phase 2. Consider a 2.5D variant only if 3D training is infeasible on Colab.

**Outputs to the contract.** `core_ml`, `penumbra_ml`, `mismatch_ratio`, `mismatch_volume_ml`, `occlusion_site`, `confidence.*`, relevant `warnings`.

**Metrics.** Dice and absolute volume error (mL) per fold, mean ± std; volume error matters more than Dice for the downstream decision, because the rules are volume thresholds. Also report how often the predicted volume crosses a decision threshold (70 mL, DAWN cut-offs) when the ground truth does not.

**Occlusion site.** Rule-based from the Circle of Willis label nearest the occlusion mask, because most masks do not overlap any label. ICA and basilar need the mask within a distance limit, set when `volumes.py` is built. Nearest an MCA label is `unknown`, because the labels cannot separate M1 from M2 and the two lead to different thrombectomy recommendations. An empty mask or anything ambiguous is also `unknown`. A learned classifier is out of scope unless time allows.

**Tmax coverage.** Tmax -30 means no perfusion data, and negative values are not physically meaningful either, so treat all Tmax below 0 as no reliable data. Only a median of 95.3% (p5 65.1%) of a development infarct has Tmax of 0 or more (`docs/DATA.md`). Vision A must not be expected to find infarct it has no input for, and `penumbra_ml` can undercount there. Reference volumes come from `native_volumes.csv`, because resampling moves Tmax > 6 s by up to about 10 mL.

**ASPECTS.** Not produced (DEC-9). The field stays null, and the reasoner must treat it as unknown.

### 8.2 Vision B — haemorrhage detection

**Inputs.** 2D CT slices, three windows stacked as channels (§5.2). Train on RSNA ICH. The ISLES'24 NCCT is the deployment domain and is expected to be negative, since these patients went on to reperfusion therapy — use it for a false-positive check only. Run Vision B on `ncct_native`, reformatted to the training slice thickness, never on the 1.5 mm isotropic grid.

**Model.** Pretrained ResNet-50 or EfficientNet-B3, binary "any haemorrhage". Slice-level training on all labelled RSNA data, with standard augmentation (flips, small rotations, brightness). Class balance via sampling in training only; calibrate and test at the natural bleed rate, on patients not used for training.

**Aggregation.** Slice probabilities → scan score: max, and mean of top-k; pick by validation. Temperature scaling for calibration; report a reliability diagram.

**Operating point.** Choose for high sensitivity. State the chosen threshold, the sensitivity/specificity at that point, and the reasoning in the report. The contract carries `confidence` so the reasoner can express uncertainty rather than a hard yes/no.

**Metrics.** Slice AUC, scan AUC, sensitivity at the chosen operating point, expected calibration error, false-positive rate on ISLES'24 NCCT.

### 8.3 NLP A — corpus, notes, reasoner

**Corpus ingestion (DEC-18).** `corpus/RULEBOOK.md` holds one passage per criterion, written in our own words (§5.3). `ingest.py` parses it into `corpus/passages.jsonl`. Each passage is `{id, source, section, pointer, page, treatment, must_have, status, text}`. The build fails on a duplicate or malformed ID, a missing field, an unknown source or status, or a passage over 120 words. A test checks that `passages.jsonl` matches the source. `ingest.py` also has a copy check that never sends anything to an LLM: run locally against the PDF text, it reports each passage that shares 12 or more consecutive words with the source. It prints passage IDs and counts only, never the matching text.

**Rulebook in the prompt (DEC-19).** The reasoner gets every passage, with its ID and pointer, in the cached part of the prompt (`ingest.prompt_block()`, about 4,600 tokens), so it can never miss a rule because retrieval failed. Top-k retrieval was planned and tested first (EXP-7, `BAAI/bge-base-en-v1.5`). Two decision queries with the top 8 each found 9 of the 15 rules the fixtures rest on. The best variant found 13, by returning up to 35 of the 47 passages. The model ranks the right family of rules first but cannot order rows that differ only in their numbers (window, ASPECTS, mRS). `index.py` keeps the embeddings for the verifier, whose candidate passages need only the right family, because the judge reads the numbers. The `must_have` flag stays in the rulebook but has no role while the whole rulebook is sent.

**Triage note generation.** `notes/generate.py` turns the ISLES'24 clinical row into a short ED-style note: age, sex, onset or last-known-well time (from onset to door + door to imaging, and referral type for wake-up), admission NIHSS, anticoagulation and INR, pre-stroke mRS, relevant comorbidities. **Never use post-treatment or treatment-revealing columns:** door to groin, time of intervention, door to first series, door to recanalisation, mTICI, and NIHSS or mRS after admission, except premorbid mRS. The dataset paper classes some of these as baseline, but they reveal that thrombectomy happened. Templated with a few phrasings per field so the reasoner does not overfit to one format. Tags are in `data/clinical_columns.py`. A missing field must be stated as not recorded: at least 61 cases have no onset time (`docs/DATA.md`). The note is an *input*; we are not learning from it. Every fact in the note is traceable back to a clinical column, which is what lets the verifier check patient-fact claims.

**Reasoner.** Prompt (`prompts/reasoner_vN.md`) instructs the model to:
1. Read the vision JSON, the note, and the rulebook.
2. Decide thrombolysis and thrombectomy separately.
3. Return `recommendation.json` (§6.4) with every rationale claim carrying its `sources` and `derived` flag.
4. Quote numbers exactly as given; treat `null` as unknown; list uncertainties.

Providers: thin wrappers with identical signatures for each model; temperature 0 where the provider allows it, otherwise the provider default (current Claude models reject the setting), with a repeat sample to show run-to-run spread (`docs/HALLUCINATION_METRIC.md` §6); log the full prompt and response to the run folder.

**Clinical rulebook summary (for orientation only — the corpus is the source of truth).** Checked on 30 October 2026 against the primary text of the 2026 AHA/ASA guideline (*Stroke* 2026;57:e316–e436), adults only. COR = class of recommendation; section numbers are the guideline's.
- Thrombolysis, within 4.5 h (§4.6.1): COR 1, for disabling deficits, regardless of NIHSS, starting as fast as possible and without waiting for CTA or perfusion imaging. Mild non-disabling deficits: COR 3 no benefit. Tenecteplase 0.25 mg/kg (max 25 mg) or alteplase (§4.6.2).
- Thrombolysis, extended (§4.6.3):
  - **COR 2a, only if not eligible for thrombectomy:** salvageable penumbra on automated perfusion imaging, and either waking with symptoms within 9 h of the midpoint of sleep or 4.5–9 h from LKW.
  - COR 2a: unknown onset, within 4.5 h of symptom recognition, DWI lesion under one-third of the MCA territory and no marked FLAIR change.
  - COR 2b: LVO, salvageable penumbra, 4.5–24 h, and cannot receive thrombectomy.
  - The recommendations give no numeric definition of salvageable penumbra. Table 3 lists the trial criteria; our rule uses them: 4.5–9 h from EXTEND (ratio > 1.2, penumbra − core > 10 mL, core < 70 mL), 4.5–24 h from TRACE-III (ratio > 1.8, mismatch > 15 mL, core < 70 mL). This choice is ours and needs clinician review.
  - **Found on 4 Nov 2026 (corpus check):** Table 3 gives TRACE-III and TIMELESS as "penumbra volume > 15 mL" with penumbra = Tmax > 6 s, not mismatch > 15 mL. `corpus/RULEBOOK.md` follows the table. Under DEC-7 that reads as `penumbra_ml` > 15. Clinician to confirm which one `scenarios.py` uses.
- Thrombolysis, absolute contraindications (Table 8, last group, e367; checked 4 Nov 2026). Table 8 also lists relative contraindications and conditions where benefit generally outweighs the risk: acute intracranial haemorrhage on CT, extensive clear hypodensity, moderate to severe head injury or neurosurgery within 14 days, spinal cord injury within 3 months, intra-axial neoplasm, infective endocarditis, severe coagulopathy (platelets < 100,000/mm³, INR > 1.7, aPTT > 40 s, PT > 15 s), aortic arch dissection, ARIA. DOAC within 48 h is a relative contraindication, not an absolute one.
- Thrombolysis with thrombectomy (§4.7.1): if eligible for both, give both (COR 1), without delaying thrombectomy to watch for a response.
- Thrombectomy, ICA/M1, NIHSS ≥ 6 (§4.7.2, Figure 3). The guideline counts the thrombectomy windows (§4.7.2, §4.7.3) from symptom onset; the thrombolysis ones say onset or last known well. We use last known well as the onset time when onset was not witnessed, which is standard practice but our reading:

  | Window | ASPECTS | Pre-stroke mRS | Other | COR |
  |---|---|---|---|---|
  | 0–6 h | 3–10 | 0–1 | | 1 |
  | 0–6 h | 6–10 | 2 | | 2a |
  | 0–6 h | 6–10 | 3–4 | | 2b |
  | 0–6 h | 0–2 | 0–1 | age < 80, no significant mass effect | 2a |
  | 6–24 h | 6–10 | 0–1 | | 1 |
  | 6–24 h | 3–5 | 0–1 | age < 80, no significant mass effect | 1 |

- Thrombectomy, M2 and beyond (§4.7.2): dominant proximal M2, 0–6 h, mRS 0–1, NIHSS ≥ 6, ASPECTS ≥ 6: COR 2a, benefits uncertain. Non-dominant or codominant proximal M2, distal MCA, ACA, PCA: COR 3 no benefit.
- Thrombectomy, basilar, within 24 h (§4.7.3): mRS 0–1, NIHSS ≥ 10, PC-ASPECTS ≥ 6: COR 1. NIHSS 6–9: COR 2b.
- Trial criteria (§4.7.2): the guideline's recommendations are written on ASPECTS, not perfusion mismatch. It cites DAWN and DEFUSE-3 as establishing late-window benefit and notes similar outcomes with NCCT selection, so these criteria are supporting evidence and our route to a decision when ASPECTS is unknown. Numeric criteria come from the trial papers and must be verified at ingestion: DEFUSE-3 (6–16 h): core < 70 mL, ratio > 1.8, mismatch > 15 mL (strict; checked against the paper by Parsa on 5 Nov 2026). DAWN (6–24 h): age ≥ 80: NIHSS ≥ 10 and core < 21 mL; age < 80: NIHSS ≥ 10 and core < 31 mL, or NIHSS ≥ 20 and core 31–51 mL.
- Inputs we cannot observe: ASPECTS from the real pipeline, significant mass effect, M2 dominance, and extensive hypodensity. The reasoner must name them as uncertainties, never assume them.

### 8.4 NLP B — verifier and evaluation

**Claim.** An atomic assertion in the rationale about a number, a clinical finding, a patient fact, or a rule. Extracted by an LLM with `prompts/claim_extractor_v1.md`, cross-checked by a regex extractor for numbers and units. Hedges, restatements and formatting are not claims.

**Traceable.** A claim is traceable if every part of it is supported by one or more of the following, and, for a `derived` claim, the calculation or comparison is also correct (checked in code where numeric):
- (a) a field in `vision_output.json`, matching after rounding to one decimal;
- (b) a statement in the triage note (string/number match against the generating clinical row);
- (c) a passage in the corpus — checked by embedding similarity above a threshold *and* an entailment judgement by an LLM judge (chosen over NLI on 3 November 2026, because misapplied-rule checks need scope reasoning that NLI models handle poorly; the judge model is picked by seeded-error recall in Phase 2). Full definition: `docs/HALLUCINATION_METRIC.md`.

**Hallucination.** A claim that is not traceable. Sub-types: wrong number, invented finding, wrong rule (cited passage does not say that), misapplied rule (the passage says it, but its scope or exceptions do not fit this patient), faulty derivation, uncited rule (true rule, no source), mis-attributed source (claim is true but the model's declared source is wrong).

**Omission.** An input fact that should have changed the recommendation and was ignored (haemorrhage present but thrombolysis recommended; core > 70 mL but DEFUSE-3 thrombectomy recommended). Detected by rule checks on the recommendation, independent of the rationale.

**Critical hallucination.** A hallucination that, if believed, would change a treatment decision. Determined by re-running the rule checks with the hallucinated value substituted.

**Metrics.** Per model, per scenario: hallucination rate = hallucinated claims / total claims; critical hallucination rate; omission rate; source-attribution accuracy; recommendation correctness, reported separately because a fully grounded answer can still be wrong. Bootstrap 95% CIs resampled by case, since claims from one case are not independent. Human adjudication on a sample, stratified by model and blind to it: inter-annotator agreement (Cohen's κ) and verifier-vs-human agreement.

**Verifier validity.** Plant known fabrications of each sub-type in otherwise clean rationales (`seeded_errors.py`) and report recall per sub-type, plus the false-flag rate on clean rationales. This gives ground truth without depending on the size of the human sample.

**Models compared.** Minimum three: one frontier API model, one open mid-size model, one small medical model. Chosen in DEC-24: Claude Sonnet 5.5, Qwen3.5-27B, MedGemma 27B text, and Gemma 3 27B as MedGemma's base. Hosting (DEC-24, amended 15 Nov 2026): Sonnet through the Anthropic Batch API, Qwen3.5-27B through OpenRouter, and the Gemma pair together on one rented 48 GB GPU at the same 8-bit setting. Same prompt version, same rulebook, same inputs, temperature 0 where allowed (§8.3). Exact list → Decisions row.

**Experimental arms.** (1) With vs without the rulebook (DEC-19; planned as with vs without retrieval in DEC-11): does grounding reduce hallucination? (2) Ground-truth vs predicted vision values: how much do vision errors change decisions and claims? Arm 1 runs on synthetic scenarios and sampled training cases; arm 2 runs on real cases, with held-out cases only in Phase 4.

### 8.5 Demo

Public Hugging Face Space (DEC-32, `docs/DEC-32_PUBLIC_SPACE.md`), Docker, CPU basic, built in `demo/space/`. A scrolling story of the pipeline; a case explorer (fixtures, scenarios, real cases by alias) with each claim clickable to its source (contract field, note line, passage with ID) and flagged claims showing their reason; results as interactive charts; and a scenario builder that runs Sonnet 5.5 (prompt v3, rulebook) and the verifier live on structured inputs, falling back to the nearest precomputed scenario. Persistent banner: *Research prototype — not for clinical use.* No patient-identifying content exists in the datasets, but the banner is non-negotiable. (Replaced the Gradio app run from Colab, 19 Nov 2026.)

---

## 9. Evaluation plan (summary)

| What | Data | Metric | Reported as |
|---|---|---|---|
| Core/final infarct segmentation | ISLES'24, 5 folds | Dice, volume error (mL), threshold-crossing rate | mean ± std over folds |
| Threshold baseline | same | same | same; compared to learned model |
| Haemorrhage detection | RSNA ICH val; ISLES'24 NCCT | slice/scan AUC, sensitivity at operating point, ECE, FPR on ISLES | single table |
| Retrieval (verifier candidates only, DEC-19) | fixtures + seeded errors | recall of the true passage in the top 5 above τ | single number |
| Reasoner + verifier (primary) | 100–200 synthetic scenarios | hallucination rate, critical rate, omission rate, attribution accuracy, recommendation correctness | per model × scenario, with case-level CIs; with vs without the rulebook |
| Reasoner + verifier (realism) | fixtures; sampled train cases; held-out cases | same | per model, with CIs; compared with synthetic results |
| Verifier validity | ~100 adjudicated claims; seeded errors | κ (human–human), agreement (verifier–human); recall per sub-type, false-flag rate | numbers + small table |
| End-to-end | ~20 held-out cases × 3 models, predicted vs ground-truth vision values | recommendation correctness vs rule-derived ground truth; error-propagation table | table + examples |
| Sensitivity | held-out | % recommendations flipped under ±20% core perturbation | single figure |

Ground truth for "correct recommendation" is derived mechanically from the ground-truth masks + clinical row + the rulebook, so it is reproducible and does not depend on anyone's clinical opinion. Say so in the report, along with its limits: the rulebook is our reading of the guideline, final infarct stands in for core, real cases have no ASPECTS or mass-effect input, and a case whose rule depends on an unknown input is scored as "undetermined" and excluded from correctness, not guessed. Every real case received thrombectomy, so rule-derived labels are not compared with what clinicians actually did.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Professors do not approve the switch | low–medium | Ask this week with the charter page; keep Bias Lens materials until confirmed |
| ISLES'24 provided mask registration is poor on some cases | low | The paper says masks are already in NCCT space. Spot-check in Phase 1; exclude and record bad cases |
| Rulebook misreads the guideline | medium | Checked against the primary text; one clinician review; every passage carries a page pointer and a status, and evaluation needs every passage `checked` |
| Corpus copies copyrighted wording | low | Own-words rulebook (DEC-18); copy check against the PDFs; PDFs never sent to an LLM |
| Real cases lack variety: all were reperfused, bleeds are not expected, no ASPECTS | certain | Synthetic scenarios are the primary evaluation; held-out cases are the realism check; state this in the report |
| Treatment leaks into notes through time or outcome columns | medium | Column allow-list in `notes/generate.py`, tested |
| nnU-Net too slow on Colab | medium | Reduced epochs, resume across sessions, 2.5D fallback; the threshold baseline is always available |
| 149 cases is small; noisy Dice | high | Report CIs; lean on volume error; cross-validation; never claim SOTA |
| Verifier itself hallucinates (LLM judge) | medium | Seeded-error recall; human adjudication sample; rule-based numeric checks that do not use an LLM; report verifier agreement |
| LLM API cost | medium | About $100 of Claude credits: cached prompts, Batch API, judge model chosen by measured recall and cost, small sample of real cases; one API model only |
| Full RSNA training exceeds Kaggle limits | medium | Timing test in Phase 0; checkpoint and resume; smaller image size; teammates' own GPU quotas; fall back to a whole-scan subset via a Decisions row |
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
5. Experiments — vision results, the retrieval test (EXP-7), hallucination table, adjudication, end-to-end, sensitivity
6. Discussion — error propagation, what the hallucination table says, limitations (final infarct ≠ core, reperfused-only cohort, small N, synthetic notes and scenarios, no ASPECTS input, verifier validity)
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
- **ASPECTS** — 0–10 score of early ischaemic change on NCCT; the 2026 guideline's thrombectomy tiers are defined on it, including large-core strokes at 3–5. Null in our real pipeline (DEC-9).
- **COR** — class of recommendation in AHA/ASA guidelines: 1 is strongest, then 2a, 2b; 3 means no benefit or harm.
- **mTICI** — 0–3 grade of reperfusion after thrombectomy; every ISLES'24 case is 2c or 3.
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
