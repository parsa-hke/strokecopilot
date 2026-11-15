# Vision A confidence (Task 2)

Lorna, Vision A. Task 2 of `docs/team/lorna_vision_a.md`. Design approved by Parsa, 15 Nov 2026; Parsa then left the final design to Lorna.

**Status:** a first run (EXP-30 at `aac4cb0`) is recorded below. It showed two problems, so the method was changed: missing data is now counted only where the perfusion scan looks, and two new signals were added. Three versions are now fitted side by side. Their results are empty until the next Colab run (notebook 14). Lorna then picks a version and writes down why.

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
- **v2:** signals 1 to 4.
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

*To fill in after the next Colab run.*

**Run:** EXP-?, notebook `notebooks/14_visionA_confidence.ipynb`, commit `?`. Inputs: EXP-27's `per_case.csv` and the slim tarball. Outputs on Drive in `runs/EXP-?_visionA_confidence/`.

**Slab check** (share of tissue with no Tmax data):

| | Median | Lowest | Highest |
|---|---|---|---|
| Before: all plain-CT tissue | | | |
| After: measurement area | | | |

**Each signal on its own** (weight when fitted alone; positive means lower confidence):

| Signal | Weight alone | Raises confidence? | Correlation with volume error |
|---|---|---|---|
| Missing data near the at-risk area | | | |
| Predicted core size | | | |
| Disagreement | | | |
| Distance to the nearest cut-off | | | |
| At-risk volume | | | |

**Three versions side by side:**

| | v1 | v2 | v3 |
|---|---|---|---|
| Wrong side of a cut-off, low band | | | |
| Wrong side of a cut-off, high band | | | |
| Errors of 10 mL or more, low band | | | |
| Errors of 10 mL or more, high band | | | |
| Under-calls in the high band (of all under-calls) | | | |
| Spearman, confidence against volume error | | | |
| Done check passes | | | |

**Band tables per version:** *to copy from the notebook.*

**confidence.penumbra:** median *?*, lowest *?*, *?* of 129 cases under 0.9.

**Version chosen:** *Lorna's choice and why.*

## Limits

- **Only 129 cases,** about 43 per band, so differences between bands and between versions are rough.
- **Three versions compared on the same cases.** Choosing the best of three on these cases flatters the winner a little. The real test comes in Phase 4, on the held-out cases.
- **Simple signals.** They catch missing data and doubtful volumes, but not every reason the AI can be wrong, for example tissue that dies after the scan.
- **confidence.penumbra is not validated** against any penumbra label, because none exists in ISLES'24.
- **The measurement area is approximate.** The holes in the perfusion maps' area are filled slice by slice, so a gap that reaches the edge of the brain on a slice is not counted as missing.
