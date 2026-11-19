# Student's t-tests (Phase 5, task C)

Written 20 Nov 2026, before any t statistic is computed. **Status: approved by Parsa, 20 Nov 2026, with the proposed defaults:** the list in §3, a rank test beside every
t-test, and Holm over the whole list as one family. Every test on it is reported, whatever it shows, and none is added
after a t result is seen. T7 (post hoc, held-out, aliases only) allowed by Parsa the same day. DEC-36. Owner: Parsa (20 Nov
2026), who runs everything.

## 1. What this is, and what it is not

- **Beside, never instead.** Every existing interval, exact test and number stays exactly as it is (the case bootstraps,
  McNemar, Wilson intervals, the paired bootstrap differences). The t-tests are added next to them, because the course
  asks for them. Where a t-test and the existing bootstrap disagree (one interval excludes 0 and the other does not),
  both are reported, and the disagreement is stated.
- **Said plainly: the data has already been seen.** Every comparison below already has a reported bootstrap result, so
  this list is not a pre-registration in the strict sense. What it fixes is the set of tests, their form and the
  correction, before any t statistic exists. No test is chosen, dropped or reshaped after seeing its result.
- **The unit is the case, never the claim.** Claims from one case are not independent (`docs/HALLUCINATION_METRIC.md`
  §6). The paired form is used whenever the same cases are on both sides.

## 2. What is reported for each test

- t, degrees of freedom, two-sided p, the mean difference with its 95% t interval, and an effect size: d_z (mean
  difference / SD of the differences) for paired tests, Hedges' g for unpaired ones;
- n, and the cases dropped and why (a format failure or a case with no claims on either side drops the pair);
- the Holm-adjusted p across the whole approved list (§4), next to the unadjusted p;
- the existing result it sits beside, quoted as reported;
- **assumption checks:** skewness and the Shapiro–Wilk p of the differences (paired) or of each group (unpaired), with a
  Q–Q plot in the appendix;
- **a rank test beside every t-test:** Wilcoxon signed-rank (paired) or Mann–Whitney U (unpaired). The task asks for it
  where the differences are skewed; reporting it for every test, with the skewness flagged, avoids a choice that depends
  on the data. Parsa may prefer the narrower rule.

**Which t-test.** Paired: Student's paired t-test. Unpaired: Welch's t-test, the form of Student's test that does not
assume equal variances, which the group sizes here make the safer default.

**Per-case rates are not the pooled rates.** The reported hallucination rates pool claims across cases, so a case with
more claims weighs more. A t-test compares the **mean of per-case rates**. Both are given, and the report says which is
which. `docs/HALLUCINATION_METRIC.md` §6 already lists the mean of per-case rates as a check.

## 3. The proposed list

All data are saved; no model is run and nothing is paid for. The rescored results (metric points 19 and 20) are used
wherever they exist, as in the reported tables.

| # | Comparison | Measure per case | Form | Cases | Sits beside |
|---|---|---|---|---|---|
| T1 | nnU-Net against the rCBF < 30% threshold | Dice | paired | 129 development (EXP-27 `per_case.csv`, each case scored by the fold model that did not train on it) | EXP-25: paired +0.104 [+0.078, +0.133] |
| T2 | nnU-Net against the threshold | absolute volume error, mL | paired | 129 | EXP-25: −1.6 mL [−4.5, +1.3], p 0.35 |
| T3a–d | With against without the rulebook, per model (Sonnet, Qwen, Gemma 3 27B, MedGemma) | content hallucination rate (sub-types 1–5, DEC-29) | paired | 100 per model: 5 fixtures, 75 scenarios, 20 real cases (EXP-37) | EXP-42 §3: −14.9, −12.9, −13.0, −8.0 pts |
| T4a–d | Predicted against ground-truth vision, per model (the same four) | hallucination rate | paired | 20 development real cases (EXP-37) | EXP-42 §3: −0.3, −1.3, −4.4, +0.5 pts |
| T5a–e | Each model against Sonnet (Qwen, Gemma 3 27B, MedGemma, Gemma 3 4B, tuned 4B) | hallucination rate, rulebook arm, predicted vision | paired | 175: 5 fixtures, 150 scenarios, 20 real cases (EXP-37, EXP-47) | EXP-42 §3 and EXP-48 §5 |
| T6a | Tuned against base Gemma 3 4B | hallucination rate | paired | 175 | EXP-48: −27.3 pts [−29.6, −25.1] |
| T6b | Tuned against base | hallucination rate | paired | 51 held-out-route scenarios (DEC-30's main result) | EXP-48: −23.5 pts [−27.4, −19.6] |
| T6c | Tuned against base | critical hallucination rate | paired | 175 | EXP-48: +1.9 pts [+0.4, +3.4] |
| T7a–f | Development against held-out real cases, per model (all six) | hallucination rate, predicted vision | **Welch** | 20 against 20 | EXP-66's realism check (unpaired, by eye) |
| T8a–b | Vision A by age band (under 80, 80 and over) | Dice; absolute volume error | Welch | 129 development | task E (counts and intervals) |
| T8c–d | Vision A by sex | Dice; absolute volume error | Welch | 129 development | task E |
| T8e–j | Scenario hallucination by age band (under 80, 80 and over), per model (all six) | hallucination rate | Welch | 150 scenarios | task E |
| T9 | BiomedCLIP linear probe against EXP-6's ResNet-50 | Brier score of the calibrated scan probability, averaged per patient | paired | RSNA validation patients (Kaggle, numbers only) | task D's AUCs |

**35 tests:** T1–T2 (2), T3 (4), T4 (4), T5 (5), T6 (3), T7 (6), T8 (10), T9 (1). T9 runs only if task D runs.

Notes:
- **T7 uses saved held-out results.** It is a new analysis of them, so it is **post hoc**, labelled so wherever it
  appears, done by alias with no IDs, and **needs Parsa's OK before it is computed** (the session's held-out rule). If
  he says no, T7 leaves the list and the family is 29 tests.
- **T8, the subgroups:** two groups only, so a t-test applies. Centre (more than two) and onset pattern (known, wake-up,
  none) are left to task E's counts and intervals; a one-way test there is not proposed, because the smallest groups
  will be too small to say anything. Scenario age bands use the scenario's own age, so T8e–j asks whether the models
  hallucinate more around the age-80 rules, in our synthetic set only.
- **T9:** a scan's score is a probability on different scales for the two models, so the per-scan Brier score (after
  each model's own calibration) is compared, averaged per patient, because some patients have several scans.

**Considered and not proposed:**
- **Binary outcomes** (recommendation correctness, a bleed flag, a human label, a seeded error caught): a t-test on 0/1
  per case adds nothing to the exact tests and Wilson or bootstrap intervals already reported.
- **The real-case correctness on held-out cases** (23 scored decisions): too few, and binary.
- **`confidence.core` v2 against distance alone** (per-case Brier, paired): possible, but DEC-27 is settled and its
  comparison is reported. It can be added if Parsa wants it.
- **The Sonnet repeat runs** (run-to-run spread): a repeatability question, not a difference to test.

## 4. Multiple comparisons

Holm's step-down correction over the whole approved list as one family (35 tests, or 29 without T7). Holm controls the
family-wise error rate like Bonferroni but is uniformly more powerful. Both adjusted and unadjusted p are shown. Holm is
computed in our own code (statsmodels is not installed), with a test against a worked example.

## 5. Code and output

- `src/strokecopilot/pipeline/ttests.py`: loads the per-case values from the saved runs, runs the tests with
  `scipy.stats` (`ttest_rel`, `ttest_ind(equal_var=False)`, `wilcoxon`, `mannwhitneyu`, `shapiro`, `skew`), applies Holm,
  and writes `runs/EXP-<n>_ttests/` (`config.json`, `metrics.json`, `report.md`, Q–Q plots). Tests with known answers
  run in `pytest` without data.
- Per-case values are checked first against the reported numbers: each pooled rate and each bootstrap difference is
  recomputed from the same files and must match before any t statistic is computed.
- One figure for the reports: a forest plot of the mean differences with their t intervals and the existing bootstrap
  intervals side by side.

## 6. Decisions for Parsa

1. Approve the list in §3, or strike or add tests now (after approval nothing is added).
2. T7: may saved held-out results be used for this post hoc test?
3. The rank test beside every t-test (proposed), or only where the differences are skewed.
4. Holm over the whole list as one family (proposed), or per group of tests.
5. A Decisions row and an EXP number for the t-tests (after EXP-67 and the SHAP rows).
6. Owner: Parsa (decided 20 Nov 2026).
