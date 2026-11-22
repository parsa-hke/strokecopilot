# Phase 5 additions: hand-over notes for the report, paper and presentation sessions

Owner: Parsa (all tasks). Each finished task has a result paragraph, **one sentence** a report can use, its figure in
Drive `artifacts/figures/` (PNG and SVG; captions in that folder's `README.md`) and its limits. The detail is in the
named doc and run folder. Everything here is report only: nothing was retrained and nothing feeds the contract, the
pipeline or the Space (DEC-31, DEC-34). No held-out case was used except T7, labelled post hoc.

Status on 21 Nov 2026: A to I done (A's pack is ready; no answers yet); J is this file. The submission is tagged
`v1.0.0` (commit `a0cb697`) with the replay bundle as a release asset.

## Where each result goes (task J)

The section names are the ones the two subject outlines expect: 901's report, 902's report and the one research paper
for the whole project (which stands in for both courses' paper assessments). Every entry is report only and carries its
limits wherever it is used.

| Task | Run | 901 report | 902 report | Paper |
|---|---|---|---|---|
| A. Clinician review | EXP-67 | Challenges & Mitigation | Evaluation & Results; Ethical Issues | Discussion |
| B. Rule engine SHAP | EXP-68 | | Solution Architecture | Results |
| B. Vision A channels (row 1) | EXP-75 | Results & Analysis | | Ablation Studies |
| B. `confidence.core` v2 | EXP-70 | Training Strategy & Evaluation | | |
| B. Vision B windows (row 5) | EXP-76, EXP-77 | Results & Analysis; Challenges & Mitigation | | Ablation Studies |
| B. Vision B pixels (row 6) | EXP-72 | Results & Analysis (explainability) | | Results |
| B. Reasoner inputs (row 9) | EXP-74 | | Evaluation & Results | Ablation Studies |
| B. Retriever tokens (row 10) | EXP-71 | | Solution Architecture | |
| B. Verifier surrogate (row 11) | EXP-73 | | Evaluation & Results | |
| B. Vision error surrogates (rows 12, 13) | EXP-80 | Results & Analysis | | |
| C. t-tests | EXP-69 | Training Strategy & Evaluation | Evaluation & Results | Results |
| D. BiomedCLIP and subtypes | EXP-82, EXP-83 | Model Architecture & Implementation; Results & Analysis | | Ablation Studies |
| E. Subgroups | EXP-80 | Results & Analysis | Ethical Issues | Discussion |
| F. Non-Claude judge | EXP-81 | | Evaluation & Results | Results |
| G. Retrieval re-test | EXP-79 | | Solution Architecture | Ablation Studies |
| H. ONNX export | EXP-78 | Model Architecture & Implementation | | |
| I. Reproducibility | replay bundle | Links | Links (code quality & reproducibility) | |

This is a suggestion for the report sessions, not a fixed outline (`docs/REPORT_OUTLINE.md` decides).

## A. Clinician pack (EXP-67, DEC-34)

Fixed and approved before sending (`docs/CLINICIAN_PACK.md`): a 76-item rulebook and test-case review (69 scored items),
delivered as a generated Google Form. Agreement between the
two clinicians will be reported as percent agreement, Cohen's κ and PABAK. **Not yet sent; no answers.**
- **Sentence (once answered):** "Two clinicians reviewed the rulebook, five test cases and twelve conventions
  independently."
- **Limit, always:** expert opinion from two clinicians, never clinical validation.

## B. SHAP (DEC-35; `docs/PHASE5_SHAP_SURVEY.md` §5)

**Rule engine, EXP-68** (figure `fig_rule_engine_shap`). Exact Shapley values over 13 inputs on the 150 scenarios:
thrombectomy turns on the occlusion site (mean |value| 0.57), thrombolysis on time (0.41); the core volume carries little
(0.06 and 0.007). Where the rules say yes, vision inputs carry 28% (thrombolysis) and 50% (thrombectomy).
- **Sentence:** "In our rulebook, the occlusion site and the clock decide most treatment decisions; the core volume, the
  quantity our segmentation model estimates, rarely does."
- **Limits:** synthetic scenarios with even route coverage; "unknown" is not neutral for these rules.

**Vision A's inputs, EXP-75** (`fig_visionA_channel_shap`). Exact Shapley values of nnU-Net's five inputs over all 32
combinations on the 129 development cases; the full combination reproduces every recorded volume within 0.02 mL. Tmax
carries the most (Dice 38%, volume 35%), then plain CT (23%, 24%), CBF (16%, 19%), MTT (13%, 12%) and CBV (10%, 11%).
Every input raises Dice on average; Tmax and plain CT matter more the larger the infarct. With every input 33% of cases
are on the wrong side of a cut-off, against 35% with none.
- **Sentence (901, results and explainability):** "Exact Shapley values over nnU-Net's five inputs show it relies most
  on Tmax (38% of the attribution for Dice) and then on plain CT (23%), more than on CBF (16%), the map the rCBF < 30%
  baseline uses."
- **Limits:** reliance, not ablation (no model was trained without an input); the perfusion maps share information,
  which Shapley values split between them; development cases; means hide case-to-case variation.

**`confidence.core` v2, EXP-70** (`fig_confidence_v2_shap`). The distance to the nearest cut-off carries 73% of v2's
attribution, which supports DEC-27's choice of that signal alone. In-sample.

**Vision B pixels, EXP-72** (`fig_visionB_pixel_shap`). Expected gradients on the 68 flagged ISLES'24 scans' top slabs:
median 50% of the attribution in brain tissue (Grad-CAM 76%), brain over bone in 127 of 136 slabs.
- **Sentence (901, explainability):** "A second attribution method (SHAP, expected gradients) agrees with Grad-CAM that
  Vision B's false alarms on ISLES'24 come from brain tissue, not from the skull base."
- **Limit:** a reference distribution of unflagged ISLES'24 slabs; SHAP weights bone edges more (26% vs 9%).

**Vision B's three CT windows, EXP-76 (RSNA) and EXP-77 (ISLES'24)** (`fig_visionB_window_shap`). Exact Shapley values
over the 8 window combinations; all three reproduce EXP-6 and EXP-50's 68 flags. On RSNA validation the brain window
carries 67% of the scan AUC above chance, subdural 23%, bone 10%. At DEC-28's threshold the brain window alone flags
almost every scan on both datasets (RSNA specificity 0.017; ISLES'24 129 of 129), so the context windows' job is to pull
clean scans back down. On clean RSNA scans they do (subdural −0.05, bone −0.10); on RSNA's own 152 false alarms they
do not (+0.06, −0.03), and ISLES'24's 68 flags show the same pattern (+0.02, −0.02).
- **Sentence (901, results and challenges):** "Shapley values over the three CT windows show that the haemorrhage model's
  false alarms on ISLES'24 follow the same pattern as its false alarms on its own RSNA data, where the subdural and bone
  windows fail to pull a clean scan's probability down; the domain shift changes how often this happens, not how."
- **Limits:** reliance, not ablation (an absent window is set to its RSNA training mean); group means.

**Retriever, EXP-71** (`fig_retrieval_token_shap`). Number tokens carry at least their share of bge's similarity in 10
of the 12 key passages.
- **Sentence (902, architecture):** "Token-level SHAP shows the embedding model does weigh the numbers; it fails to rank
  sibling rules because they share the same numbers in different combinations, which is why the reasoner gets the
  whole rulebook (DEC-19)."

**Verifier flags, EXP-73, a surrogate** (`fig_surrogate_flags_shap`). A model of the verifier's flags from claim features
known before verification (AUC 0.92): rule claims (44% flagged) and non-numeric comparisons (26%) push towards a flag,
restated imaging values (2%) away. It explains the surrogate, not the verifier.

**Rows 12 and 13, surrogates of vision errors, EXP-80.** Vision A's wrong-side errors are predictable from what is known
at presentation (AUC 0.73 [0.63, 0.82]): mostly the predicted core volume (48%) and the Tmax > 6 s volume (20%), then
age (14%); centre 7%, sex 4%. Vision B's false alarms are not (AUC 0.34 [0.24, 0.43], below chance, the usual sign of no
signal under cross-validation), so no SHAP values are reported for them.

**Small reasoners' inputs, EXP-74** (`fig_reasoner_shap`). Exact Shapley values of six input groups on 24 synthetic
scenarios (64 versions each, the bleed fixed) for the rule engine and Gemma 3 4B, prompted and LoRA-tuned; the full-input
versions reproduce EXP-47's answers (41 of 42 and 36 of 36 decisions). The base model recommends thrombolysis in every
version, so no input moves its answer. The tuned model responds to its inputs: for thrombolysis it ranks time first, as
the rules do (26% of its attribution against the rules' 43%); for thrombectomy it leans on the occlusion site (28%) and
as much on thrombolysis contraindications (28%), which the rules never use for thrombectomy (0%).
- **Sentence (902, fine-tuning and evaluation):** "Shapley values over the reasoner's inputs show that the base Gemma 3
  4B recommends thrombolysis whatever it is told, while the LoRA-tuned model does respond to its inputs and ranks the
  clock first for thrombolysis as the rules do, but also lets thrombolysis contraindications sway its thrombectomy
  decision, which the rules never do."
- **Limits:** 24 synthetic scenarios (21 and 18 complete; a format failure drops a scenario's game); the value shows what
  moves the answer, not whether it is right; a hidden group is rendered as "not recorded".

## C. Student's t-tests (EXP-69, DEC-36; `docs/PHASE5_TTESTS.md` §7)

All 35 tests, case-level (T9 per patient), paired or Welch, Holm over the family, a rank test beside each; every
reported bootstrap was recomputed and matched first (figure `fig_ttests_forest`).
12 hold after Holm, T9 among them (BiomedCLIP's probe against EXP-6: per-patient Brier 0.116 against 0.044, +0.073
[+0.064, +0.081], 1,894 patients); adding T9 changed no other adjusted p-value.
- **Sentence:** "Student's t-tests on per-case values, Holm-corrected over 35 tests, confirm the rulebook effect,
  nnU-Net's Dice gain and the model ranking; they do not support a volume gain, an effect of vision errors on
  hallucination, or any subgroup difference, and they weaken the tuned model's higher critical rate."
- **Must be said:** the data had been seen before the tests were fixed; T7 is post hoc on saved held-out results.

## D. BiomedCLIP and per-subtype probes (EXP-82, EXP-83, DEC-33)

Method fixed before any score (`src/strokecopilot/analysis/biomedclip.py`, notebook 27): BiomedCLIP's image embeddings
of a seeded 60,000-slice training sample, as a linear probe and zero-shot with two fixed prompts, against EXP-6's
ResNet-50, every scan scored by EXP-6's top-2 aggregation and Platt-calibrated on validation (figure `fig_biomedclip`).

| On RSNA validation (2,166 scans; 879 bleeds) | EXP-6 ResNet-50 | BiomedCLIP probe | BiomedCLIP zero-shot |
|---|---|---|---|
| Scan AUC | **0.977** | 0.899 | 0.753 |
| Specificity at 96% sensitivity | **0.882** | 0.416 | 0.162 |
| Per-patient Brier score (T9, 1,894 patients) | **0.044** | 0.116 | — |
| ISLES'24 development flags (129, presumed bleed-free; EXP-83) | **68** | 117 | 94 |

- **T9:** the probe's Brier score is higher by +0.073 [+0.064, +0.081] (paired t, d_z 0.38; holds after Holm over 35).
- **ISLES'24 (McNemar against EXP-6):** the probe flags 54 scans EXP-6 does not and misses 5 it flags (p < 0.0001);
  zero-shot 40 and 14 (p 0.0005). At its operating point BiomedCLIP also flags most clean RSNA scans (58% probe, 84%
  zero-shot), so its ISLES'24 flags mostly reflect that lower specificity.
- **Per subtype:** a linear probe on EXP-6's frozen features detects each subtype at scan AUC 0.91 (epidural, 31
  scans) to 0.98 (intraventricular, 389); EXP-6's own "any" score, used as a detector of one subtype, gives 0.77 to
  0.93. So EXP-6's features carry the subtype, and a per-subtype output would cost one linear layer.
- **Sentence (901, model choice):** "A frozen biomedical foundation model did worse than our ResNet-50 trained on RSNA
  (scan AUC 0.90 as a linear probe and 0.75 zero-shot, against 0.98), was worse calibrated, and flagged more of
  ISLES'24's presumed bleed-free scans (117 and 94 of 129, against 68): task-specific training wins here, and switching to
  a general model does not remove the domain shift."
- **Limits, always:** BiomedCLIP is not fine-tuned (a probe on frozen embeddings, 224 px), so this compares
  representations, not best achievable models; the operating point is in-sample on validation for every model alike;
  the AUCs have no intervals (T9 is the paired test); the "any" score was not trained to separate subtypes, so the
  subtype comparison shows what the features hold, not a better model; the RSNA test split was not read.

## E. Subgroups (EXP-80, DEC-37)

From saved results only. **Vision, 129 development cases** (figure `fig_subgroups_vision`): Vision A's Dice is lower at
80 and over (0.16 [0.10, 0.22]) than under 80 (0.25 [0.19, 0.30]), and at centre 1 (0.18) than centre 2 (0.29); its
absolute volume error is larger in women (31 mL [21, 42]) than in men (16 mL [11, 23]). The intervals overlap or nearly
do, the differences are unadjusted (infarct size differs between groups) and none survives Holm in the t-tests (T8).
Vision B flags about half of every group (47% to 61%), with no group standing out. **Reasoners, 150 scenarios:** no
model's hallucination rate or correctness differs by age band, sex or onset pattern (all intervals overlap). **Real
cases (20):** every group is too small to conclude anything.
- **Sentence (902, ethics):** "Across age band, sex, centre and onset pattern we found no reliable difference in the
  language models' error rates; Vision A's segmentation was nominally worse in patients 80 and over and its volumes in
  women, which our small sample cannot confirm or rule out."
- **Limits:** 129 development cases from two centres; one patient's sex not recorded; synthetic scenarios do not carry
  centre; no adjustment for infarct size.

## F. A non-Claude judge (EXP-81, DEC-38)

EXP-41's 100 human-labelled claims re-judged by GPT-6 Luna (`openai/gpt-6-luna` through OpenRouter; no family shared
with any of the six reasoners), with the same `judge_v3` prompt, τ 0.6 and reasoning effort, from an empty judge cache
(469 judge calls, $0.29, no provider errors; figure `fig_judge_agreement`). The two judges flag the same 97 of 100 claims
(κ 0.94; 95 of 100 with the same verdict type). Agreement with the human consensus is the same for both: Haiku 0.61
(weighted 0.87), Luna 0.62 (0.87); per reasoner model the two match except Sonnet's claims (Haiku 0.63, Luna 0.67). All
three disagreements are on Sonnet's claims, where Haiku flagged and Luna did not, and the humans sided with Luna on two.
- **Sentence (902, evaluation):** "Re-judging the 100 human-labelled claims with a judge from a different model family
  (GPT-6 Luna) changed 3 of 100 verdicts (κ 0.94) and left agreement with the humans unchanged (0.62 against 0.61), so the
  verifier's limited agreement with people comes from what it checks, not from which model judges; and the Claude judge
  was, if anything, stricter on the Claude reasoner."
- **Must be said with it:** metric points 19 and 20 were adjusted on these same labels (EXP-41); 100 claims, about 25
  per model, so per-model differences are small-sample; the code checks are shared by both judges, so this tests the
  judge, not the whole verifier.

## G. Retrieval, tested properly (EXP-79, DEC-37)

EXP-7's check of the 15 key fixture rules, repeated with four methods fixed before any score (top 8 per decision
query; a rule found if either query of its fixture returns it). EXP-7's own queries were not saved; the rebuilt ones
do not reproduce its 9 of 15 with bge (they find 5), so the counts compare methods on the same queries, not with EXP-7.

| Method | Key rules in top 8 (of 15) | Top 5 | Top 12 | Median best rank |
|---|---|---|---|---|
| bge-base (dense) | 5 | 2 | 9 | 12 |
| BM25 | 6 | 4 | 8 | 12 |
| BM25 + bge, reciprocal rank fusion | **8** | 3 | **10** | **8** |
| fusion top 20, reranked by bge-reranker-base | 3 | 3 | 6 | 14 |

- **Sentence (902, architecture):** "Re-testing retrieval with lexical (BM25), hybrid and cross-encoder methods, the best
  still found only 8 of the 15 rules our test cases rest on, and the reranker did worst, so sending the whole
  4,600-token rulebook (DEC-19) remains the safer design."
- **Limits:** 15 rules from five fixtures; rebuilt queries; no method or setting tuned (by design).

## H. ONNX export (EXP-78, DEC-37)

EXP-6's ResNet-50 exported to ONNX (opset 17, dynamic batch; 94.0 MB against 94.3 MB for the PyTorch checkpoint). On
every 5 mm slab of 20 ISLES'24 development scans (601 slabs) ONNX Runtime matches PyTorch to 8.8 × 10⁻⁶ in probability,
and all 20 scans get the same DEC-28 call (10 flagged by each). On the Mac's CPU, one slab takes about 25 ms in either
runtime; a batch of 32 takes 2.1 s with ONNX Runtime on all 10 threads against 6.7 s with PyTorch (about 3.2 times
faster).
- **Sentence (901, deployment):** "Exported to ONNX, the haemorrhage model gives the same probabilities as PyTorch (largest
  difference 9 × 10⁻⁶, identical calls on 20 scans) and runs a 32-slab batch about three times faster on a laptop CPU."
- **Limits:** one machine (Apple silicon CPU), median of 30 runs; per-slab cost was higher inside a batch of 32 than
  alone, so latency depends on memory as much as on compute here. The weights stay private to the team (DATA_USE); the
  `.onnx` file is on Drive only.

## I. Reproducibility (DEC-37; done 21 Nov 2026)

- **Replay bundle:** `python -m strokecopilot.pipeline.replay run --bundle replay_bundle.tar.gz` rebuilds the
  hallucination table for the fixtures and the 150 scenarios, all six models, with and without the rulebook, from saved
  answers and verifier caches, with no API key: all 20 cells match their reported rates exactly (about 3 s). 3.2 MB,
  reproducible (sha256 recorded in the code), no patient data. Attached to the release of the submission tag.
- **README** rewritten (results, quick start, every notebook, the Space); **MIT licence** and **CITATION.cff** added.
- **Sentence (902, code and reproducibility):** "Anyone can rebuild our hallucination table for the synthetic cases
  without an API key: a 3 MB bundle of saved answers and verifier caches is rescored offline and checked against every
  reported rate."
- **Submission tag and release (21 Nov 2026):** annotated tag `v1.0.0` on `a0cb697` (after the Space redesign) and
  the GitHub release "v1.0.0: submission" with `replay_bundle.tar.gz` attached. Checked at the tag in a clean
  worktree: 910 tests passed, 1 skipped; the bundle matches its sha256 (`44c8d3ab…b799`) and every replay cell its
  reported rate; the asset downloaded back from the release has the same sha256.
- **Still Parsa's:** markers' access. The repository is private, so the release and the bundle are visible only to
  collaborators until markers are added or the repository is made public.
