# SHAP survey (Phase 5, task B)

Written 20 Nov 2026, before any SHAP value is computed. **Status: approved by Parsa, 20 Nov 2026, with the proposed defaults (§3 items 1–4): rows 1, 3, 5, 6, 8, 9 (the 4B pair), 10, 11, 12 and 13 run, row 2 if time allows, row 9b only in its small form on a separate go-ahead; Vision A with five players and the §2 baselines; Vision B's absent window at its RSNA training mean.** DEC-35 (Parsa, 20 Nov 2026). Owner: Parsa, for every row (20 Nov 2026); he runs everything, on his accounts where a run needs Colab, Kaggle or an API. Every row was checked
against the code and the saved outputs named in it. Nothing here is retrained, and nothing feeds the contract, the
pipeline or the Space (DEC-31, DEC-34). Held-out cases are not used anywhere in this survey.

**Two kinds of method, named in every result:**
- **Exact Shapley values by enumeration** where the players are few (up to about 13): every coalition is evaluated, so
  there is no sampling error. Written in our own code (about 20 lines), checked against `shap` on a small case in a test.
- **The `shap` library** (not installed yet; it would go in a new optional `analysis` extra in `pyproject.toml`) where an
  explainer fits: `LinearExplainer`, `TreeExplainer`, `GradientExplainer` (expected gradients) and the text masker.

**One limit applies to every "remove an input" analysis below.** No model was trained with an input missing, so a
coalition with an input replaced by a baseline is off the training distribution. These values measure how much the
trained model **relies** on an input, not how much the input would be worth to a model trained without it. That is not
an ablation study, and the reports must not call it one.

## 1. The table

Cost is time and money as estimated, not measured. A dummy run measures it before any long run (DEC-21 rule).

| # | Where | Method | Question it answers | Data (saved, checked) | Cost | Feasible? | Recommend |
|---|---|---|---|---|---|---|---|
| 1 | **Vision A, input channels** | Exact Shapley, 5 players → 32 coalitions | How much does each input drive the predicted core volume, Dice and cut-off crossings? | 129 development cases, slim tarball; each scored by the fold model that did not train on it (EXP-14, EXP-18–21 `checkpoint_final.pth`) | about 4,000 fold predictions; **1–2 Colab A100 hours** | Yes | **Yes** |
| 2 | Vision A, spatial | `GradientExplainer` (expected gradients) on one 3D patch | Where in the scan does the network look for the core? | 3 development cases by alias, one patch each | about 0.5 A100 hour | Yes, at patch scale | Optional |
| 3 | Vision A, `confidence.core` v2 | `LinearExplainer` (exact for a logistic model) | Which of the four signals moves v2's confidence for each case? | EXP-31 `per_case.csv` and `mapping_v2.json` (Drive) | seconds, Mac | Yes | Yes, small |
| 4 | Vision A, `confidence.core` under DEC-27 | — | — | EXP-32 | — | Trivial | **No**: one input, so it gets all the attribution. The mapping curve (0 mL → 0.90, 21–70 mL → about 0.41) already says everything |
| 5 | **Vision B, CT windows** | Exact Shapley, 3 players → 8 coalitions | Which window (brain, subdural, bone) drives the scan score, on true bleeds and on ISLES'24's false alarms? | RSNA validation (Kaggle, numbers only); ISLES'24's 129 development scans, 68 flagged (EXP-50 path, Mac) | Kaggle T4 about 30 min, free; Mac about 10 min | Yes | **Yes** |
| 6 | **Vision B, pixels** | `GradientExplainer` on ResNet-50, beside Grad-CAM (EXP-50) | Do pixel attributions agree with Grad-CAM that the false alarms sit in brain tissue? | the top 2 slabs of the 68 flagged ISLES'24 scans; background from unflagged scans | Mac CPU 1–2 h, or minutes on Colab | Yes | **Yes** |
| 7 | Vision B, slabs | — | Which slabs drive the scan score? | — | — | Trivial | **No**: the score is the mean of the top 2 slab probabilities, so the attribution falls on those two slabs by construction. One sentence in the report, no figure |
| 8 | **Rule engine** `rules.decide` | Exact Shapley, 13 players (8,192 coalitions) | How much does each input, and vision as a whole, decide the treatment? (§1.3's claim) | the 150 scenarios (`scenarios.generate`), optionally the 129 development contracts; no API | under 10 min, Mac | Yes | **Yes, first** |
| 9 | **Reasoners**, grouped inputs | Exact Shapley, 7 groups (128 coalitions), next to row 8 on the same cases | Do the language models rely on the same inputs as the rules? | 24 scenarios (seeded, one or two per route), Gemma 3 4B and tuned 4B on vLLM | about 6,100 requests; **2–4 Colab A100 hours**, $0 | Yes | **Yes, the 4B pair** |
| 9b | Reasoners, API models | as 9 | as 9, for Sonnet (or Qwen) | as 9, smaller | Sonnet by batch, reasoner only, about $0.02 per request (estimate): 24 cases × 128 = about **$65**; 12 cases × 5 groups (32) = about **$9** | Yes | Only the small version, if wanted |
| 10 | **Retriever**, token level | `shap` text masker on the bge cosine similarity | Do the numbers in a passage carry weight in retrieval? (explains EXP-7) | the 15 key fixture rules and their nearest wrong rows (the ICA/M1 and M2 families) | minutes, Mac CPU | Yes | **Yes** |
| 11 | Surrogate: verifier flags | Gradient boosting + `TreeExplainer`, **labelled a surrogate** | Which claim features go with being flagged? | EXP-37's rulebook cells and EXP-47, about 30,000 claims in 22 cells (rescored traces, local); grouped cross-validation by case | minutes, Mac | Yes | Yes, if its cross-validated AUC is clearly above 0.5 |
| 12 | Surrogate: Vision A errors | as 11 | Which case features go with a core on the wrong side of a cut-off? Feeds task E | EXP-27 `per_case.csv` plus age, sex, centre, onset pattern; 129 cases | minutes | Yes, but small | Yes, with E |
| 13 | Surrogate: Vision B false alarms | as 11 | Which case features go with a false bleed flag on ISLES'24? Feeds task E | EXP-16/EXP-50 per-case flags plus case features; 129 cases | minutes | Yes, but small | Yes, with E |
| 14 | Surrogate: human-judged errors | — | — | EXP-41, 100 claims, 21 human-judged errors | — | Too small | **No**: 21 positives cannot support a surrogate |
| 15 | `confidence.penumbra` | — | — | — | — | Not applicable | **No**: a formula (1 minus the missing Tmax share), nothing to explain |
| 16 | Vision B calibration, threshold | — | — | — | — | Not applicable | **No**: Platt scaling and a threshold are monotonic in one score |
| 17 | Linear probes (task D) | — | — | — | — | Not meaningful | **No**: SHAP over 2,048 or 512 embedding dimensions says nothing a reader can use |
| 18 | The LLM judge | `shap` text masker over the judge's inputs | Which words of a claim or passage decide the verdict? | EXP-41's 100 claims | many API calls per claim | Possible | **No**: costly, and the judge's validity is measured directly by task F and EXP-41 |
| 19 | rCBF < 30% threshold baseline | — | — | — | — | Not applicable | **No**: a rule on one map |

## 2. Notes per recommended row

**Row 1, Vision A channels: a correction to the starting list.** nnU-Net reads **six** channels, not four
(`nnunet_wrapper.channel_arrays`): NCCT, CBF, CBV, MTT, Tmax (negative values set to 0) and a Tmax-valid mask. The
proposal has five players: **NCCT, CBF, CBV, MTT, and Tmax with its mask** (one player, because the mask only describes
Tmax). 32 coalitions, all evaluated, so the values are exact. The full coalition reproduces the recorded validation
masks (EXP-61 showed this path reproduces them), which the dummy run checks first.
- **Outputs:** predicted core volume (mL), Dice against the reference, and "wrong side of 21/31/51/70 mL" (0/1), per
  case; summarised as the mean |Shapley value| per player, and per size band.
- **Baseline for an absent input (to decide).** Proposed:
  - **Tmax absent:** Tmax 0 and its mask 0 everywhere. That is exactly how the network was trained to see missing Tmax
    (no reliable data), so this coalition stays closest to what it knows;
  - **CBF, CBV, MTT absent:** the map set to a constant. nnU-Net's z-score scheme normalises each image by its own mean and standard deviation, so any
    constant becomes an all-zero channel: the case's own mean everywhere, with no spatial information;
  - **NCCT absent:** the constant that nnU-Net's CT normalisation maps to 0 (its dataset mean).

  Not proposed: raw zeros for CBF or CBV, which mean "no blood flow" and would push the network towards infarct, so the
  value would measure the baseline, not the map. nnU-Net is not installed on the Mac, so the per-image normalisation is
  taken from its documented behaviour and checked in the dummy run before anything else.

**Row 2, spatial (optional).** nnU-Net predicts by sliding window, so a whole-volume attribution is expensive. One patch
of the network's own size around the reference infarct, for three development cases chosen by rule (the largest
correctly found infarct, a miss, an over-call; by alias), with the target the summed infarct probability in the patch.
Shown as overlays on ISLES'24 slices with attribution.

**Row 5, Vision B windows.** The three channels are three windows of the same HU values (`prep_rsna.WINDOWS`), so a
window carries no information the others lack in principle; the question is which one the network uses. Baseline for an
absent window: that channel set to its RSNA training mean (a constant). On RSNA validation the scan score is recomputed
for each coalition (top 2 slices may change), and results are given apart for true bleeds, true negatives and false
positives, and per subtype. On ISLES'24: the 68 flagged scans and the 61 unflagged ones.

**Row 6, Vision B pixels.** Expected gradients on the logit, background 50 slabs from unflagged ISLES'24 scans, read
out with the same tissue masks as Item 2 (`explain.heat_shares`) so the two methods are compared on the same terms (share
of attribution in brain, bone, other, outside). ISLES'24 images with attribution; no RSNA image leaves Kaggle.

**Row 8, rule engine: the cheapest row and the one closest to the research claim.**
- **Players (13):** bleed; core; Tmax > 6 s volume; occlusion site; ASPECTS; time (onset or last known well, with its
  basis, wake-up and sleep midpoint); age; NIHSS; pre-stroke mRS; INR; anticoagulant; head trauma or surgery;
  previous haemorrhage or recent bleeding. The mismatch ratio and volume are derived from core and Tmax > 6 s, so they
  are recomputed when both are present and set unknown otherwise, as the contract would be.
- **Baseline:** the input unknown (`None`), which is what "not recorded" or a null contract field means to the rules.
- **Value of a coalition:** 1 if the decision equals the decision on the full inputs, else 0, for each treatment. A
  player's Shapley value is then its share in reaching that decision. "Vision as a whole" is also reported as a
  two-player game (all vision fields against all note fields).
- **Limit:** the scenarios were generated to cover the decision routes evenly, so the attributions describe our
  scenario set, not the mix of patients in practice. On the 129 development contracts (optional) the unknown occlusion
  site will dominate, as EXP-63 found.

**Row 9, amended 20 Nov 2026 (Parsa), before any answer:** the bleed stays fixed at its real value, because the frozen
contract cannot say "unknown" for `haemorrhage.present`. Six groups instead of seven (64 versions per scenario; 1,280
distinct requests per model after identical versions merge); the rules are scored on the same six. Ready: notebook 55
`MODE = 'shap_dummy'` (one scenario) then `'shap'`, as EXP-74; checked end to end on the Mac with rule-derived gold
answers.

**Row 9, reasoners.** The same games as row 8, with the model's `recommend` booleans as the output and the rules'
values beside them on the very same coalitions. Seven groups, to keep 128 coalitions: bleed; core and Tmax > 6 s (with
`confidence.core`); occlusion site; ASPECTS; time; age, NIHSS and mRS; contraindication facts. A group is absent when its
contract fields are null and its note lines say "not recorded" (the note generator already writes that form). Prompt v3
and the rulebook, as in EXP-47. A format failure leaves that coalition's value missing; a case with any missing coalition
is reported apart and left out of the summary. 24 scenarios, chosen by seed, one or two per decision route. The Gemma 3
4B pair is the main analysis, because it is free on Colab and the pair asks whether tuning changed what the model reads.

**Row 10, retriever.** For each of the 15 key fixture rules: the query as EXP-7 built it, the right passage and the top
wrong passage from the same family; a text-masker SHAP of the cosine similarity over the passage's tokens. The question
is whether number tokens ("6", "24 h", "ASPECTS 3–5") carry weight compared with words. EXP-7's queries are not in `src/`
(only its metrics are on Drive), so they are rebuilt from its config and noted as rebuilt.

**Rows 11 to 13, surrogates.** A small gradient-boosted model of the target from case or claim features, fitted with
cross-validation grouped by case, its cross-validated AUC reported first. SHAP explains the **surrogate**, never the
verifier or the vision model, and every figure says "surrogate model". If the AUC's interval includes 0.5, the SHAP
result is not shown and the null result is stated instead.

## 3. Decisions for Parsa

1. **Which rows run.** Proposed: 1, 3, 5, 6, 8, 9 (the 4B pair), 10, 11, 12, 13; row 2 if time allows; row 9b only in
   its small form and only on a separate go-ahead.
2. **Vision A players:** five (with NCCT), as proposed, or the four perfusion maps only with NCCT and the mask always
   present.
3. **Vision A baseline:** as proposed in §2 (Tmax as missing, the other maps mean-filled), or another.
4. **Vision B baseline:** a window set to its RSNA training mean, as proposed, or another.
5. **Owners:** Parsa, for every row (decided 20 Nov 2026).
6. **Decisions row and EXP numbers:** a row for the SHAP analyses (DEC-35, say) and one EXP number per row that runs, from
   EXP-68 (EXP-67 is the clinician review).

## 4. What each result becomes

One sentence and one figure for a report, the detail in the paper and the repo:
- row 1: a bar of mean |Shapley| per input for core volume, with Dice beside it (901, model line and explainability);
- rows 5–6: windows bar plus a pixel-attribution panel next to Grad-CAM (901, explainability, "Grad-CAM or SHAP");
- rows 8–9: rule engine and reasoners side by side, per input group (902, architecture and evaluation; §1.3's claim);
- row 10: token attributions on two passages that differ only in numbers (902, why DEC-19 dropped retrieval);
- rows 11–13: one surrogate figure with its AUC, in task E's subgroup section (902, ethics).

## 5. Results

**Row 8, the rule engine: EXP-68 (done 20 Nov 2026; `runs/EXP-68_rule_shap/report.md`; figure
`fig_rule_engine_shap`).** On the 150 synthetic scenarios, thrombectomy turns mostly on the occlusion site (mean
|Shapley value| 0.57, against 0.14 each for time, NIHSS and pre-stroke mRS), and thrombolysis on time (0.41), then the
occlusion site (0.14) and NIHSS (0.13). The core and Tmax > 6 s volumes carry little (thrombolysis 0.06 and 0.05,
thrombectomy 0.007 and 0.003), which agrees with EXP-63's core ×0.8/×1.2 check (9 and 10 of 300 decisions change). Where
the rules say yes, vision inputs carry 28% of thrombolysis's attribution and 50% of thrombectomy's. Read with its limits:
the scenario set covers routes evenly by design, and "unknown" is not neutral for these rules (an unknown site already
means no thrombectomy, in 86 of 150 scenarios), so some single values are negative and the magnitudes are the summary.

**For the reports, one sentence:** "In our rulebook, the occlusion site and the clock decide most treatment decisions;
the core volume, the quantity our segmentation model estimates, rarely does (exact Shapley values over 150 synthetic
scenarios)."

**Row 3, `confidence.core` v2: EXP-70 (done 20 Nov 2026; figure `fig_confidence_v2_shap`).** Exact SHAP values of v2's
logistic mapping (closed form, matching `shap.LinearExplainer` exactly with all 129 cases as background) in log-odds of
a large error. The distance to the nearest cut-off carries **73%** of the mean |SHAP value|, predicted core size 14%,
missing Tmax near the at-risk area 12%, disagreement with the threshold core 1%. In-sample, since v2 was fitted on the
same cases. It backs DEC-27's choice: v2 was mostly the distance already.

**Row 10, the retriever: EXP-71 (done 20 Nov 2026; figure `fig_retrieval_token_shap`).** Partition SHAP over the
passage tokens of the bge similarity between each of the 12 key fixture passages and a gold claim citing it (EXP-7's
own queries were not saved; the gold claims stand in, so ranks differ from EXP-7's). Number tokens carry **at least
their share** of the similarity in 10 of the 12 passages (median 13% of the attribution on 10% of the tokens; ratio
1.25). So bge does not ignore numbers; EXP-7's misses (here, 4 of 12 passages not ranked first) come from sibling rows
that share the same numbers in other combinations (6 h, 24 h, ASPECTS ranges), which raise their similarity too. This
refines EXP-7's diagnosis and still supports DEC-19.

**Row 11, verifier flags: EXP-73 (done 20 Nov 2026; figure `fig_surrogate_flags_shap`). A surrogate model, not the
verifier.** A gradient-boosted classifier of "flagged" from claim features known before verification, 27,472 claims of
the six models' rulebook arms; out-of-fold AUC **0.92 [0.92, 0.93]** (grouped by case), so its SHAP values are reported.
Claim kind (32% of the attribution) and model (28%) dominate, then how many numbers a claim holds (10%). Rule claims (44%
flagged) and non-numeric comparisons or reasons (26%) push towards a flag; restated imaging values (2%), decision
citations (7%) and absence claims (9%) push away. Rule and non-numeric comparison claims are the ones the LLM judge
decides; imaging values and numbers are checked in code. The model ranking matches EXP-42's.

**Row 6, Vision B pixels: EXP-72 (done 20 Nov 2026; Mac MPS, 10.4 min; figure `fig_visionB_pixel_shap`).** Expected
gradients (`shap.GradientExplainer`, 100 samples) on the logit of the top 2 slabs of the 68 flagged ISLES'24 development
scans, background the top slab of 50 unflagged scans, read with Item 2's tissue masks. The run reproduces EXP-50 (68
flags; Grad-CAM median 76% of the heat in brain, 134 of 136 slabs brain over bone). SHAP agrees on the main point:
median **50%** of the |attribution| in brain tissue, and brain over bone in **127 of 136** slabs. It gives bone more
weight than Grad-CAM (26% against 9%), as pixel-level gradients pick out high-contrast edges, and the two maps agree
moderately (median Spearman 0.50, IQR 0.31–0.59, at 10 × 10). So a second, different method supports Item 2's reading:
the false alarms come from brain tissue, not from skull-base slabs.
