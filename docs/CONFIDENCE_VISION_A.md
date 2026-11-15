# Vision A confidence (Task 2)

Lorna, Vision A. Task 2 of `docs/team/lorna_vision_a.md`. Design approved by Parsa, 15 Nov 2026.

**Status:** method built and tested on fake data. The results below are empty until the Colab run (notebook 14, EXP-30).

## What this is for

The contract (`schemas/vision_output.json`) gives the reasoner two numbers from 0 to 1: `confidence.core` and `confidence.penumbra`. They should tell it when Vision A's numbers are less trustworthy, so it can say so instead of treating a doubtful volume as fact.

## Terms used

- **Core volume:** the AI's (nnU-Net's) estimate of the dead brain tissue, in millilitres (mL).
- **Threshold core:** the same estimate from the simple rule with no AI (blood flow under 30% of the healthy side, inside the late-blood area).
- **Tmax:** how late blood arrives at each spot on the first scan. Over 6 seconds marks tissue at risk.
- **No or unreliable data:** spots where the perfusion scan has no usable Tmax value.
- **Cut-offs:** the treatment rules compare the core volume with 21, 31, 51 and 70 mL.
- **Fold:** the 129 development cases are split into five groups. Each case's AI result comes from the fold model that did not train on it.

## How confidence.core is made

**What counts as a large error.** The main measure is a case whose AI core volume lands on the wrong side of at least one cut-off, because that is the error that can change a treatment decision. A volume error of 10 mL or more is reported too; 10 mL is the smallest gap between two cut-offs.

**Three warning signs**, all available when a prediction is made, never the true infarct:
1. **Missing data near the at-risk area:** the share of the Tmax > 6 s area, plus a 10 mm margin around it, with no or unreliable Tmax. If a case has no Tmax > 6 s area, the whole brain is used.
2. **A small predicted core:** the AI's volume, as log(1 + mL), so a 0 mL prediction still works and the difference between 1 and 5 mL counts more than between 101 and 105 mL.
3. **Disagreement:** how far apart the AI's and the threshold's core volumes are, as the gap between their log(1 + mL) values.

**From signs to a score.** A small logistic regression learns how the three signs relate to the chance of a large error. `confidence.core` is 1 minus that chance. It has only four numbers to learn, which suits about 100 training cases.

**No case marks its own homework.** The regression is fitted five times, each time on four folds, and scores only the fifth fold. So every case's confidence comes from a mapping that never saw that case. A final fit on all 129 cases is saved for later use on new cases. Held-out cases are never used.

**What is not used.** Disagreement between the five fold models is not used, because each development case was in the training data of four of them, so they would agree too easily.

## How confidence.penumbra is made

`confidence.penumbra` is 1 minus the share of the brain with no or unreliable Tmax data. The brain is the tissue range on the plain CT scan (−10 to 100 HU), as the threshold method uses.

**This is a data-quality score only.** ISLES'24 has no label for the penumbra (the tissue at risk that could still be saved), so there is nothing to check it against or fit it to. It says how complete the perfusion data is, not how right the penumbra volume is.

## Results

*To fill in after the Colab run.*

**Run:** EXP-30, notebook `notebooks/14_visionA_confidence.ipynb`, commit `?`. Inputs: EXP-27's `per_case.csv` and the slim tarball. Outputs on Drive in `runs/EXP-30_visionA_confidence/`.

**Confidence bands** (three equal groups of the 129 cases by `confidence.core`):

| Band | Cases | Confidence range | Mean volume error (mL) | Median volume error (mL) | Error of 10 mL or more | Wrong side of a cut-off |
|---|---|---|---|---|---|---|
| Low | | | | | | |
| Middle | | | | | | |
| High | | | | | | |
| All | 129 | | | | | |

**Wrong side of each cut-off, by band:**

| Band | 21 mL | 31 mL | 51 mL | 70 mL |
|---|---|---|---|---|
| Low | | | | |
| Middle | | | | |
| High | | | | |

**Done check:** passes when the low band has more cases on the wrong side of a cut-off, and more errors of 10 mL or more, than the high band. Result: *to fill in.*

**confidence.penumbra:** median *?*, lowest *?*, *?* of 129 cases under 0.9.

## Limits

- **Only 129 cases,** about 43 per band, so differences between bands are rough.
- **Three simple signs.** They catch missing data and doubtful volumes, but not every reason the AI can be wrong, for example tissue that dies after the scan.
- **confidence.penumbra is not validated** against any penumbra label, because none exists in ISLES'24.
- **The final mapping is fitted on all 129 development cases.** Its real test comes in Phase 4, on the held-out cases.
- **The brain mask is a tissue range on the plain CT,** so it can include some soft tissue outside the brain.
