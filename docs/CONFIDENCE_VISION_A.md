# Vision A confidence (Task 2)

Lorna, Vision A. Task 2 of `docs/team/lorna_vision_a.md`. Design approved by Parsa, 15 Nov 2026; Parsa then left the final design to Lorna.

**Status: done.** A first run (EXP-30 at `aac4cb0`) showed two problems, so the method was changed: missing data is now counted only where the perfusion scan looks, and two new signals were added. The second run (EXP-31 at `217bc59`) fitted three versions side by side. All three pass the done check. **Lorna chose v2**; the reasons and the caveats are below.

## What this is for

The contract (`schemas/vision_output.json`) gives the reasoner two numbers from 0 to 1: `confidence.core` and `confidence.penumbra`. They should tell it when Vision A's numbers are less trustworthy, so it can say so instead of treating a doubtful volume as fact.

## Terms used

- **Core volume:** the AI's (nnU-Net's) estimate of the dead brain tissue, in millilitres (mL).
- **Threshold core:** the same estimate from the simple rule with no AI (blood flow under 30% of the healthy side, inside the late-blood area).
- **Tmax:** how late blood arrives at each spot on the first scan. Over 6 seconds marks tissue at risk.
- **No or unreliable data:** spots where the perfusion scan has no usable Tmax value.
- **Perfusion slab:** the band of slices the perfusion scan covers. The plain CT covers the whole head; the perfusion scan usually does not.
- **Cut-offs:** the treatment rules compare the core volume with 21, 31, 51 and 70 mL.
- **Under-call:** the AI's core volume is smaller than the true volume and lands on the wrong side of a cut-off.
- **Fold:** the 129 development cases are split into five groups. Each case's AI result comes from the fold model that did not train on it.

## How confidence.core is made

**What counts as a large error.** The main measure is a case whose AI core volume lands on the wrong side of at least one cut-off, because that is the error that can change a treatment decision. A volume error of 10 mL or more is reported too; 10 mL is the smallest gap between two cut-offs.

**Where missing data is counted.** Only in plain-CT tissue that the perfusion scan covers: the slices from the first to the last one with perfusion values, and in each slice the area the perfusion maps cover, with holes filled. Real gaps inside the brain still count. The scalp and the slices outside the slab do not. A slice inside the slab with no values at all takes the slab's outline, so a missing slice still counts as missing.

**Signals**, all available when a prediction is made, never the true infarct:
1. **Missing data near the at-risk area:** the share of the Tmax > 6 s area, plus a 10 mm margin, with no or unreliable Tmax. If a case has no Tmax > 6 s area, the whole measurement area is used.
2. **Predicted core size:** the AI's volume, as log(1 + mL), so a 0 mL prediction still works.
3. **Disagreement:** how far apart the AI's and the threshold's core volumes are, as the gap between their log(1 + mL) values.
4. **Distance to the nearest cut-off:** how far the AI's volume is from the nearest of 21, 31, 51 and 70 mL, on the same log scale. A volume close to a cut-off is where a small error flips a decision.
5. **At-risk volume:** the Tmax > 6 s volume, as log(1 + mL). It is meant to catch a tiny predicted core inside a large at-risk area, as in Miss 1 of the error analysis (the AI said 0 mL for a 101 mL infarct).

**Three versions**, all kept in the results:
- **v1:** signals 1 to 3 (the approved design).
- **v2:** signals 1 to 4. **The chosen version.**
- **v3:** signals 1 to 5.

**From signals to a score.** For each version, a small logistic regression learns how the signals relate to the chance of a large error. `confidence.core` is 1 minus that chance. Each signal is put on a common scale before fitting, so the weights can be compared. A positive weight means more risk, so lower confidence.

**No case marks its own homework.** Each version is fitted five times, each time on four folds, and scores only the fifth fold. So every case's confidence comes from a mapping that never saw that case. A final fit of each version on all 129 cases is saved for later use on new cases. Held-out cases are never used.

**What is not used.** Disagreement between the five fold models is not used, because each development case was in the training data of four of them, so they would agree too easily.

## How confidence.penumbra is made

`confidence.penumbra` is 1 minus the share of the measurement area with no or unreliable Tmax data.

**This is a data-quality score only.** ISLES'24 has no label for the penumbra (the tissue at risk that could still be saved), so there is nothing to check it against or fit it to. It says how complete the perfusion data is, not how right the penumbra volume is.

## First run, before the changes (EXP-30 at `aac4cb0`)

This run used v1 and counted missing data over all plain-CT tissue. It is kept here as a record.

- **The done check passed:** cases on the wrong side of a cut-off, low band against high band, 24 against 7; errors of 10 mL or more, 33 against 17. Spearman correlation between confidence and volume error: −0.345.
- **The middle band** had 12 cases on the wrong side of a cut-off and 15 errors of 10 mL or more. Its median volume error (4.7 mL) was lower than the high band's (5.7 mL), so the score mainly separated the low band from the rest.
- **Final weights** (on the chance of a large error; positive means lower confidence): missing data near the at-risk area 0.239, predicted core size 0.667, disagreement −0.212.

**Problem 1: two signals pointed the opposite way to the design.**
- A **larger** predicted core gave **lower** confidence. The design expected a small core to be the risky one. With "wrong side of a cut-off" as the measure, this makes sense: a small prediction for a small infarct cannot cross a cut-off even when it misses completely, and errors in mL grow with size.
- **More disagreement** gave **higher** confidence. This is probably because disagreement overlaps with core size (the threshold predicts little almost everywhere), so its own weight is small and noisy.
- **The risk:** a 0 mL prediction for a large infarct (like Miss 1) gets high confidence. Signals 4 and 5 were added to address this.

**Problem 2: the missing-data share was inflated for every case.** The share of plain-CT tissue with no Tmax had a median of 32%, ranging from 17% to 70%, and all 129 penumbra scores were under 0.9 (median 0.678). The plain CT covers the whole head, including the scalp, but the perfusion scan covers only a slab of brain slices (`docs/DATA.md` notes "limited CTP coverage"). So the score mostly measured scan coverage and head size, not data quality. The measurement area above fixes this.

## Results

**Run:** EXP-31, notebook `notebooks/14_visionA_confidence.ipynb`, commit `217bc59`. Inputs: EXP-27's `per_case.csv` and the slim tarball. Outputs on Drive in `runs/EXP-31_visionA_confidence/`: `config.json`, `per_case.csv`, `metrics.json` and `mapping_v1.json`, `mapping_v2.json`, `mapping_v3.json`. All numbers are from the 129 development cases; counts and summaries only.

### Where missing data is counted: before and after

| Share of tissue with no Tmax data | Median | Lowest | Highest |
|---|---|---|---|
| Before (EXP-30): all plain-CT tissue | 32.2% | 17.2% | 70.0% |
| After (EXP-31): the measurement area | 11.5% | 6.1% | 40.4% |

The fix cut the typical missing share from about a third to about a tenth. What is left is a mix of real gaps in the perfusion data and the limits of the measurement area (see Limits).

### Each signal on its own

Each signal fitted alone on all 129 cases. A positive weight means more risk, so lower confidence.

| Signal | Weight alone | Effect on confidence | Correlation with volume error |
|---|---|---|---|
| Distance to the nearest cut-off | −0.911 | Farther from a cut-off, higher confidence | −0.324 |
| Predicted core size | 0.703 | Bigger core, lower confidence | 0.353 |
| Disagreement | −0.437 | More disagreement, higher confidence | −0.08 |
| At-risk volume (Tmax > 6 s) | 0.17 | Bigger at-risk area, slightly lower confidence | 0.11 |
| Missing data near the at-risk area | 0.015 | Almost no effect | 0.074 |

- **Distance to a cut-off is the strongest signal.** A volume far from any cut-off is rarely on the wrong side of one.
- **Alone, core size still points the opposite way to the original design:** a bigger core means lower confidence. As explained under the first run, that fits the main measure. Inside v2 the sign flips (see v2's final weights).
- **Disagreement still raises confidence,** but its link to the volume error is close to zero (−0.08).
- **On its own, the missing-data signal adds almost nothing** once missing data is counted correctly. Inside v2 it does more (see below).

### v2's final weights

The final v2 mapping, fitted on all 129 cases (`mapping_v2.json` and `metrics.json`). Signals are on a common scale; a positive weight means more risk, so lower confidence.

| Signal | Weight in v2 | Effect on confidence, the other signals held fixed |
|---|---|---|
| Distance to the nearest cut-off | −1.188 | Farther from a cut-off, higher confidence |
| Missing data near the at-risk area | 0.248 | More missing data, lower confidence |
| Predicted core size | −0.249 | Bigger core, slightly higher confidence |
| Disagreement | −0.017 | Almost no effect |
| (intercept) | −0.838 | |

- **Together, the signals behave differently from alone.** Once distance to a cut-off is in the model, a bigger core gives slightly **higher** confidence, not lower: much of what core size said alone was really "near a cut-off or not".
- **Missing data carries a real weight in v2** (0.248, about the same size as core size), even though alone it does almost nothing.
- **Disagreement drops to almost nothing** in v2.

### Three versions side by side

All three are cross-fitted by fold and split into three equal bands (about 43 cases each).

| | v1 | v2 (chosen) | v3 |
|---|---|---|---|
| Signals | 3 | 4 | 5 |
| Wrong side of a cut-off, low band | 24 | **26** | 26 |
| Wrong side of a cut-off, high band | 7 | 7 | 7 |
| Errors of 10 mL or more, low band | 33 | 32 | 31 |
| Errors of 10 mL or more, high band | 17 | 16 | 17 |
| Under-calls in the high band (of 34) | 7 | 7 | 7 |
| Spearman, confidence against volume error | **−0.33** | −0.287 | −0.267 |
| Done check passes | Yes | Yes | Yes |

- **All three pass the done check:** each low band holds far more large errors than its high band.
- **v2 and v3 put the most cut-off errors in the low band** (26 against v1's 24). v3 adds nothing over v2.
- **v1 ranks the overall volume error a little better** (Spearman −0.33 against −0.287), but our main measure is the cut-off error, not the volume error.
- The full band tables for each version, including the middle band, are in `metrics.json` on Drive.

### The cut-off distance alone (checked at Parsa's review)

The review compared each version with the simplest score there is: the distance to the nearest cut-off itself, with no fitting (farther from a cut-off, more confident). The measure is the AUC for the wrong side of any cut-off: the chance that a random case on the wrong side gets lower confidence than a random case on the right side (0.5 is chance, 1 is perfect). Computed from EXP-31's `per_case.csv` with `baseline_check`, using the cross-fitted confidences; 43 of the 129 cases are on the wrong side. These numbers are not in EXP-31's `metrics.json`; a rerun of notebook 14 adds them.

| Score | AUC, wrong side of any cut-off |
|---|---|
| Cut-off distance alone (not fitted) | **0.755** |
| v2 (chosen) | 0.722 |
| v3 | 0.703 |
| v1 | 0.695 |
| confidence.penumbra | 0.538 |

- **v2 minus the distance alone: −0.033**, with a paired bootstrap 95% interval of −0.066 to −0.002 (5000 resamples, seed 0).
- **Said plainly:** v2 was chosen after seeing the results, and the simpler signal does at least as well. On these 129 cases it does a little better, and the interval sits below zero.
- **confidence.penumbra hardly ranks the cut-off errors** (0.538), as expected for a data-quality score.
- Which one the contract uses, the distance alone or v2, is Parsa's decision (pending a DEC). Until then the code's default stays v2.

### confidence.penumbra

After the fix: median 0.885, lowest 0.596, and 94 of 129 cases under 0.9. Before the fix it was median 0.678, with all 129 under 0.9. It now reflects missing perfusion data more than scan coverage, but it is still a data-quality score only.

### Near-empty predictions (complete misses, checked at Parsa's review)

From EXP-31's `per_case.csv` with `complete_miss_check`:
- **25 cases have an nnU-Net core under 0.5 mL.** Their median v2 confidence is 0.90, and 5 of them are on the wrong side of a cut-off. All 5 are under-calls, and all 5 got a confidence of 0.88 or more.
- **Can one signal flag them?** Tested cross-fitted by fold (fit on 4 folds, score the 5th): flag a case when nnU-Net predicts under 1 mL and its Tmax > 6 s volume reaches a cut-off learned on the other folds. 30 cases are under 1 mL, 5 of them on the wrong side. The flag marks 5 cases, and only 1 of them is on the wrong side. The learned cut-offs are 263 to 314 mL.
- **Why it fails:** under 1 mL, the wrong-side cases do not have bigger at-risk areas. Their Tmax > 6 s volumes run from 14 to 314 mL (median 93), against 25 to 382 mL (median 122) for the 25 on the right side. A large at-risk area with no core is common in cases that turn out fine.
- **Other settings do no better:** under 0.5 mL, 9 flagged and 1 on the wrong side; under 2 mL, 10 flagged and 3 on the wrong side (of 7). Using missing data near the at-risk area instead: 10 flagged, 1 on the wrong side.
- **Options for Parsa's decision:** leave confidence.core as it is and say in the contract notes that a near-empty prediction gets high confidence; cap confidence.core for near-empty predictions at a fixed value, accepting that 20 of the 25 are right; or wait for nnU-Net's per-voxel probabilities (Future work). None of these is applied: `confidence_core` and the default method are unchanged.

### Version chosen: v2

**Lorna's choice**, with Parsa's go-ahead to decide the design. `confidence.py` uses v2 by default, and `mapping_v2.json` from EXP-31 is the mapping for scoring new cases.

**Why v2:**
- Our main measure is a case landing on the wrong side of a cut-off, and v2 puts the most of those in the low band.
- Distance to a cut-off is the clearest clinical reason for doubt: near a cut-off, a small error changes the decision.
- v3 adds nothing over v2.

**Said honestly:**
- The choice was made after seeing all three results.
- v1 and v2 differ by only 2 cases in the low band.
- v1 is slightly better on the overall volume-error ranking.
- The cut-off distance alone, with no fitting, does at least as well (AUC 0.755 against 0.722; see above).

So v2's advantage is small, and Phase 4's held-out cases are its real test.

### Which values build.py uses

- **Development cases (the 129):** the cross-fitted `confidence_core_v2` and `confidence_penumbra` in EXP-31's `per_case.csv`, read with `development_confidence`. Each comes from a mapping that never saw the case.
- **Held-out cases (Phase 4 only):** `mapping_v2.json`, loaded with `load_default_mapping` and applied with `case_confidence`.
- **Why not `mapping_v2.json` for everything:** it was fitted on all 129 development cases, so scoring one of them with it is marking its own homework. It looks better than it is (AUC 0.757 against 0.722 cross-fitted).

## What confidence.core means

For the reasoner: read `confidence.core` as roughly the chance that the core volume is on the right side of every cut-off (21, 31, 51 and 70 mL), that is, that the volume does not put the case in the wrong treatment bracket. It is not a measure of how many mL the volume is off by.

- **It is roughly calibrated on the 129** (cross-fitted): mean confidence 0.665, against 0.667 of cases on the right side. By band: high 0.87 against 0.84 on the right side, middle 0.68 against 0.77, low 0.45 against 0.40.
- **It does not rise or fall steadily with volume.** From `mapping_v2.json`, with average missing data near the at-risk area (11.9%) and nnU-Net and the threshold agreeing: 0 mL gives 0.91, 5 mL 0.66, 10 mL 0.52, 21 mL 0.36, 31 mL 0.38, 51 mL 0.40, 70 mL 0.41, 100 mL 0.53 and 150 mL 0.65. It dips near the cut-offs and rises away from them.
- **Missing data moves it a lot:** at 21 mL, 0.53 with no missing data and 0.17 with 30% missing.
- **So low confidence means "near a cut-off, or poor data", not "a big stroke".** And high confidence on a near-empty prediction does not rule out a complete miss (see above).

## Limits

- **Complete misses still get high confidence.** In every version, all 7 cases on the wrong side of a cut-off in the high band are under-calls: the AI predicted too little. A complete miss like Miss 1 (0 mL predicted for a 101 mL infarct) looks like a confident small prediction, far from any cut-off. The at-risk volume (v3) was meant to catch these, and it did not. Of 25 cases predicted under 0.5 mL, 5 are on the wrong side, with a median confidence of 0.90 for the 25; a cross-fitted Tmax > 6 s flag caught 1 of the 5 (see Near-empty predictions).
- **The missing-data signal is weak alone** (weight 0.015) once missing data is counted correctly, but it carries a weight of 0.248 inside v2, so its effect depends on the other signals.
- **confidence.penumbra is a data-quality score only.** It is not validated against any penumbra label, because none exists in ISLES'24. It says how complete the perfusion data is, not how right the penumbra volume is.
- **Only 129 cases,** about 43 per band, so differences between bands and between versions are rough.
- **The version was chosen on the same cases it was scored on.** Picking the best of three flatters the winner a little, v1 and v2 differ by 2 cases, and the cut-off distance alone does at least as well. The real test comes in Phase 4, on the held-out cases.
- **Simple signals.** They catch doubtful volumes, but not every reason the AI can be wrong, for example tissue that dies after the scan.
- **The measurement area is approximate.** The holes in the perfusion maps' area are filled slice by slice, so a gap that reaches the edge of the brain on a slice is not counted as missing.

## Future work

- **nnU-Net's per-voxel probabilities.** Before nnU-Net draws its outline, it gives every voxel a probability of being infarct. Those probabilities, before thresholding, could flag near-misses like Miss 1, where the AI may have been unsure but still marked nothing. Getting them needs GPU inference, so this is left for later.
