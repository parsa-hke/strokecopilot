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

**Row 1, Vision A's input channels: EXP-75 (done 20 Nov 2026; Colab A100, 4,128 predictions, median 0.3 s each;
figure `fig_visionA_channel_shap`).** The full combination reproduces every case's recorded volume (EXP-27) within
0.02 mL, on all 129 cases. Means over cases:

| Input | Predicted volume, mean \|value\| (share) | Signed mean | Dice, mean \|value\| (share) |
|---|---|---|---|
| Tmax (with its mask) | 12.2 mL (35%) | +11.6 mL | 0.096 (38%) |
| NCCT | 8.6 mL (24%) | +5.4 mL | 0.059 (23%) |
| CBF | 6.5 mL (19%) | +0.8 mL | 0.039 (16%) |
| MTT | 4.1 mL (12%) | −2.5 mL | 0.032 (13%) |
| CBV | 3.8 mL (11%) | −1.0 mL | 0.026 (10%) |

With every input the mean predicted volume is 17.5 mL and Dice 0.215 (the reported Dice); with none, 3.1 mL and 0.008.
Every input raises Dice on average. Tmax and NCCT carry more as the reference infarct grows (Dice attribution, 0–5 mL
against over 70 mL: Tmax 0.046 against 0.157, NCCT 0.032 against 0.116). **Wrong side of a cut-off** (21, 31, 51 or
70 mL): 43 of 129 cases (33%) with every input, 35% with none, because a near-zero volume is on the right side for most
small infarcts; Tmax pushes towards the wrong side on average (+0.05), the others slightly away. So nnU-Net relies most on
Tmax and then on plain CT, more than on CBF, the map the rCBF < 30% baseline uses.
- **Limits:** reliance, not ablation (no model was trained without an input, so a filled input is outside what the
  network saw); the four perfusion maps come from the same acquisition and share information, which Shapley values split
  between them, so Tmax's share may partly stand for the others; development cases, each scored by the fold model that
  did not train on it; means hide large case-to-case variation.

**Row 9, the small reasoners: EXP-74 (done 20 Nov 2026; Colab A100, 1,280 requests per model; figure
`fig_reasoner_shap`).** 24 synthetic scenarios, 64 versions each (every combination of six input groups, the bleed
fixed), answered by Gemma 3 4B prompted and LoRA-tuned. The full-input versions reproduce EXP-47's answers on the same
scenarios (41 of 42 decisions for the base model, 36 of 36 for the tuned one). Games with a format failure among their
64 versions are dropped: 21 of 24 complete for the base model, 18 for the tuned one. The value is agreement with the
model's own full-input decision, so it shows what moves an answer, not whether the answer is right.

| Share of mean \|value\| | Thrombolysis: rules | base 4B | tuned 4B | Thrombectomy: rules | base 4B | tuned 4B |
|---|---|---|---|---|---|---|
| Total (sum of mean \|value\|) | 1.06 | **0** | 0.45 | 0.99 | 0.17 | 0.21 |
| Time | 43% | — | 26% | 21% | 11% | 12% |
| Occlusion site | 16% | — | 21% | 48% | 22% | 28% |
| Patient (age, NIHSS, mRS) | 13% | — | 21% | 21% | 12% | 9% |
| Contraindications | 15% | — | 11% | **0%** | **41%** | **28%** |
| Perfusion | 11% | — | 16% | 5% | 8% | 17% |
| ASPECTS | 2% | — | 5% | 4% | 5% | 6% |

(Rules' shares on the tuned model's 18 complete scenarios; on the base model's 21 they are within 5 points.)
- **The base 4B ignores its inputs.** It recommends thrombolysis in every version of every scenario and thrombectomy
  in 94–100% of them, so no input moves its thrombolysis answer, and its thrombectomy answer rarely moves.
- **The tuned 4B responds to its inputs, at about half the rules' sensitivity for thrombolysis and a fifth for
  thrombectomy.** For thrombolysis it ranks time first, as the rules do, but spreads its reliance more evenly. For
  thrombectomy it leans on the occlusion site, as the rules do, **and as much on the thrombolysis contraindications
  (anticoagulants, INR, trauma, surgery, prior bleed), which the rules never use for thrombectomy**. The base model shows
  the same misplaced reliance in the little that moves its thrombectomy answer (41%).
- **Limits:** 24 synthetic scenarios, 18 and 21 complete; the game measures what moves the answer, not its
  correctness; the bleed fixed and the first phrasing throughout; reliance, not ablation (a hidden group is rendered as
  "not recorded", which the models may read differently from a value).

**Row 5 (the method):**
- **Row 5, EXP-76 and EXP-77:** `notebooks/26_visionB_window_shap.ipynb` on Kaggle (`MODE = 'dummy'`, 200 validation
  scans, then `'run'`): every validation slice under the 8 window combinations, the RSNA training means from a seeded
  sample of 3,000 training slices, aggregates only. Then the ISLES'24 half on the Mac with those means
  (`python -m strokecopilot.analysis.visionb_window_shap --means <EXP-76 metrics.json>`), EXP-77.

**Row 5, ISLES'24 half: EXP-77 (done 20 Nov 2026; Mac MPS, 5.7 min; means from EXP-76's sample).** All three windows
reproduce EXP-50's 68 flags. The brain window alone flags all 129 scans; subdural or bone alone flag 1 and 0; brain with
subdural 52, brain with bone 47, all three 68. On the 68 flagged scans the brain window carries the attribution (mean
Shapley value +0.39 on the calibrated probability) and subdural and bone net out near zero (+0.02, −0.02); on the 61
unflagged scans subdural and bone pull the probability down (−0.07, −0.10). So on ISLES'24's false alarms the two context
windows fail to do what they do on clean scans, and together they flag more scans than either pair. Reliance, not
ablation (DEC-35). *Read with EXP-76 below: the brain window alone flags nearly every clean RSNA scan too, so that
count reflects how an absent window is filled, not ISLES'24.*

**Row 5, RSNA half: EXP-76 (done 20 Nov 2026; Kaggle T4 x2, 26.3 min; figure `fig_visionB_window_shap`).** Every RSNA
validation scan (2,166; 879 with a bleed) under the 8 window combinations. All three windows reproduce EXP-6 (scan AUC
0.9772, sensitivity 0.959 and specificity 0.882 at DEC-28).
- **Discrimination:** of the scan AUC's 0.477 above chance, the brain window carries 0.320 (67%), subdural 0.108 (23%),
  bone 0.049 (10%) (exact Shapley values of the AUC over the 8 combinations). Alone: brain 0.858, subdural 0.620, bone
  0.563; brain with subdural 0.951, brain with bone 0.891, all three 0.977.
- **At DEC-28's threshold the brain window alone flags almost everything:** sensitivity 0.999, specificity 0.017. So
  each context window's job at the operating point is to pull clean scans back down, and EXP-77's "brain alone flags
  all 129" is the same effect, not an ISLES'24 one.
- **Per scan, mean Shapley value on the calibrated probability:** bleeds (879) brain +0.73, subdural +0.12, bone +0.03;
  no bleed (1,287) brain +0.23, subdural −0.05, bone −0.10. On subdural bleeds the subdural window carries the most of
  any subtype (+0.17, against +0.09 on intraventricular).
- **RSNA's own false alarms (152) look like ISLES'24's:** brain +0.44, subdural +0.06, bone −0.03; the two context
  windows do not pull the probability down as they do on clean scans. On ISLES'24's 68 flagged scans: +0.39, +0.02,
  −0.02. So ISLES'24's false alarms follow the same pattern as the model's ordinary false alarms in its own domain; what
  differs is how often it happens (68 of 129 against 152 of 1,287).
- **Limits:** reliance, not ablation (an absent window is set to its training mean, which the network never saw);
  group means; RSNA validation only (the test split was not read).

