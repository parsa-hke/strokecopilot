# Phase 5 additions: hand-over notes for the report, paper and presentation sessions

Owner: Parsa (all tasks). Each finished task has a result paragraph, **one sentence** a report can use, its figure in
Drive `artifacts/figures/` (PNG and SVG; captions in that folder's `README.md`) and its limits. The detail is in the
named doc and run folder. Everything here is report only: nothing was retrained and nothing feeds the contract, the
pipeline or the Space (DEC-31, DEC-34). No held-out case was used except T7, labelled post hoc.

Status on 20 Nov 2026: A, C (34 of 35 tests), E, G and H done; B in progress (rows 1, 5 and 9 running on Colab
and Kaggle); I in progress; D and F to come.

## A. Clinician pack (EXP-67, DEC-34)

Fixed and approved before sending (`docs/CLINICIAN_PACK.md`): a 76-item rulebook and test-case review (69 scored items)
and a nine-case written interview whose cases were drawn by a seeded rule over fixed strata (right, flagged,
wrong-and-caught, wrong-and-missed), delivered as generated Google Forms with a Word case document. Agreement between the
two clinicians will be reported as percent agreement, Cohen's κ and PABAK. **Not yet sent; no answers.**
- **Sentence (once answered):** "Two clinicians reviewed the rulebook, five test cases and twelve conventions
  independently, and gave written opinions on nine system answers chosen by a fixed rule."
- **Limit, always:** expert opinion from two clinicians, never clinical validation.

## B. SHAP (DEC-35; `docs/PHASE5_SHAP_SURVEY.md` §5)

**Rule engine, EXP-68** (figure `fig_rule_engine_shap`). Exact Shapley values over 13 inputs on the 150 scenarios:
thrombectomy turns on the occlusion site (mean |value| 0.57), thrombolysis on time (0.41); the core volume carries little
(0.06 and 0.007). Where the rules say yes, vision inputs carry 28% (thrombolysis) and 50% (thrombectomy).
- **Sentence:** "In our rulebook, the occlusion site and the clock decide most treatment decisions; the core volume, the
  quantity our segmentation model estimates, rarely does."
- **Limits:** synthetic scenarios with even route coverage; "unknown" is not neutral for these rules.

**`confidence.core` v2, EXP-70** (`fig_confidence_v2_shap`). The distance to the nearest cut-off carries 73% of v2's
attribution, which supports DEC-27's choice of that signal alone. In-sample.

**Vision B pixels, EXP-72** (`fig_visionB_pixel_shap`). Expected gradients on the 68 flagged ISLES'24 scans' top slabs:
median 50% of the attribution in brain tissue (Grad-CAM 76%), brain over bone in 127 of 136 slabs.
- **Sentence (901, explainability):** "A second attribution method (SHAP, expected gradients) agrees with Grad-CAM that
  Vision B's false alarms on ISLES'24 come from brain tissue, not from the skull base."
- **Limit:** a reference distribution of unflagged ISLES'24 slabs; SHAP weights bone edges more (26% vs 9%).

**Vision B windows on ISLES'24, EXP-77.** The brain window alone flags all 129 scans; with all three windows, 68. On the
flagged scans the subdural and bone windows fail to pull the probability down, as they do on unflagged scans (−0.07,
−0.10). The RSNA half (EXP-76) and its figure follow the full Kaggle run.

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

## C. Student's t-tests (EXP-69, DEC-36; `docs/PHASE5_TTESTS.md` §7)

34 of 35 tests (T9 waits for task D), case-level, paired or Welch, Holm over the family, a rank test beside each; every
reported bootstrap was recomputed and matched first (figure `fig_ttests_forest`).
- **Sentence:** "Student's t-tests on per-case values, Holm-corrected over 34 tests, confirm the rulebook effect,
  nnU-Net's Dice gain and the model ranking; they do not support a volume gain, an effect of vision errors on
  hallucination, or any subgroup difference, and they weaken the tuned model's higher critical rate."
- **Must be said:** the data had been seen before the tests were fixed; T7 is post hoc on saved held-out results.

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

## I. Reproducibility (DEC-37; in progress)

- **Replay bundle:** `python -m strokecopilot.pipeline.replay run --bundle replay_bundle.tar.gz` rebuilds the
  hallucination table for the fixtures and the 150 scenarios, all six models, with and without the rulebook, from saved
  answers and verifier caches, with no API key: all 20 cells match their reported rates exactly (about 3 s). 3.2 MB,
  reproducible (sha256 recorded in the code), no patient data. To be attached to the submission tag's release.
- **README** rewritten (results, quick start, every notebook, the Space); **MIT licence** and **CITATION.cff** added.
- **Sentence (902, code and reproducibility):** "Anyone can rebuild our hallucination table for the synthetic cases
  without an API key: a 3 MB bundle of saved answers and verifier caches is rescored offline and checked against every
  reported rate."
- **Still Parsa's:** the submission tag and its release, and markers' access to the repository.

