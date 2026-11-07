# Vision A error analysis (EXP-15), draft

Lorna, Vision A. Task 1 of `docs/team/lorna_vision_a.md`. Draft: the visual readings section is still to be filled in.

## The question

Where and why does nnU-Net fail to find the final infarct? On fold 0, 9 of the 26 cases score Dice 0, including one 101 mL infarct that nnU-Net predicted as empty. Which cases are these, and is the problem the model or the input it was given?

## Data and method

- **Run:** EXP-15, notebook `notebooks/12_visionA_error_analysis.ipynb`, results at commit `c442a54`. Outputs on Drive in `runs/EXP-15_visionA_error_analysis/`: `config.json`, `per_case.csv`, `metrics.json` and `figures/`.
- **Cases:** the 26 development cases of fold 0, the nnU-Net validation fold. No held-out case was used.
- **Scores:**
  - nnU-Net 3d_fullres, 300 epochs (EXP-14);
  - the threshold baseline, rCBF < 30% within Tmax > 6 s (EXP-1, variant `within_tmax6`).

  Both are scored against the same reference volumes.
- **For each case, measured by code** (`strokecopilot.vision.perfusion.error_analysis`):
  - infarct size;
  - occlusion site and territory, from the Circle of Willis labels;
  - onset to imaging;
  - share of the infarct with no Tmax data;
  - **share of the infarct visible as Tmax > 6 s on the admission scan**, counting only voxels with reliable data.
- **Definitions:**
  - a *miss* is nnU-Net Dice 0;
  - "missed by both" means the threshold baseline also scored under 0.05.
- **Figures:** for each miss and for the 3 best nnU-Net cases, NCCT, CBF and Tmax at the slice with the most infarct, with the final infarct outlined in red. Figures carry anonymous labels only (Miss 1 is the largest miss).

All numbers below are counts from that run. No case IDs appear in this document.

## Results measured by code

| | Misses (Dice 0) | Rest |
|---|---|---|
| Cases | 9 | 17 |
| Median share of infarct visible as Tmax > 6 s | 30% | 64% |

- **No case was missed by nnU-Net alone.** On all 9 misses the threshold baseline also scored under 0.05, and exactly 0 on 6 of them. Both methods read the same perfusion maps, so this suggests the problem lies in the input more than in the network. It is weak evidence, though: the threshold baseline is itself poor (mean Dice 0.12), so failing where it fails proves little.
- **7 of the 9 misses are under 21 mL.**
- **4 of the 9 misses had under 25% of the infarct visible as Tmax > 6 s.**
- **The 101 mL miss (Miss 1)** had 22% of its infarct visible as Tmax > 6 s, while 88% of it had reliable data. So the data was there, but most of this infarct did not look delayed on the admission scan.
- **2 misses had over 75% visible.** The delay was there, and nnU-Net still found nothing.

## Volume error

The brief says volume error matters more than Dice, because the treatment rules are volume cut-offs (21, 31, 51 and 70 mL).

**Mean absolute volume error by size band** (mL, from `metrics.json` at `c442a54`):

| Size band | nnU-Net | Threshold |
|---|---|---|
| Under 5 mL | 7.6 | 6.8 |
| 5–21 mL | 8.7 | 5.7 |
| 21–70 mL | 28.2 | 19.8 |
| 70 mL and over | 50.7 | 70.2 |

- **The threshold has the lower volume error in every band except 70 mL and over**, even though nnU-Net has the higher Dice. A better overlap does not mean a better volume, and the treatment rules use the volume.
- The largest infarcts carry the largest errors in mL for both methods. Miss 1 alone is a 101 mL error for nnU-Net.

**Cases on the wrong side of each cut-off** (the prediction and the reference fall on different sides; of 26 cases):

| Cut-off | nnU-Net | Threshold | Both |
|---|---|---|---|
| 21 mL | 5 | 7 | 4 |
| 31 mL | 5 | 6 | 4 |
| 51 mL | 3 | 4 | 3 |
| 70 mL | 3 | 3 | 2 |
| Any | 7 | 8 | 6 |

- nnU-Net puts 7 of 26 cases on the wrong side of at least one cut-off, and the threshold 8. In 6 cases both do.
- nnU-Net's counts per cut-off (5, 5, 3, 3) match EXP-14's own crossing counts, which is an independent check of the join.
- At these sizes, nnU-Net is slightly less often on the wrong side than the threshold, but the two methods mostly fail on the same cases.

## Main failure groups

The groups overlap: a small infarct can also have little visible delay. Counts are of the 9 misses.

| Group | Misses | What it means |
|---|---|---|
| Small infarcts (under 21 mL) | 7 | Small targets are easy to miss outright. |
| Little visible delay (under 25% with Tmax > 6 s) | 4 | The admission scan barely shows the tissue that later died. Neither method had much to see. |
| Large infarct, little visible delay (Miss 1, 101 mL) | 1 | Most of the infarct had reliable data but no delay on admission. It is the most costly miss for the volume cut-offs. |
| Visible delay, still missed (over 75% visible) | 2 | Likely true model errors: the signal was in the input. |

**Context: why the final infarct and the delay can disagree.** Nearly every ISLES'24 patient was reperfused. Of the 99 cases with a TICI value, 97 were mTICI 2c or 3 and 2 were lower, and the rest have no value (plan §5.1). So the final infarct is, almost always, the tissue that died *despite* reperfusion. A large Tmax > 6 s area with a small final infarct is expected: most of the delayed tissue was saved. The reverse, an infarct with little delay on admission, may be tissue that died after the scan, or tissue whose delay the perfusion maps did not capture. This is an interpretation, not something the data here can prove.

## Example figures

The figures are on Drive in `runs/EXP-15_visionA_error_analysis/figures/`. They will be copied into this document once chosen for the report (ISLES'24 images may go in the report).

- **Miss 1** (`miss_1.png`), ≥70 mL: the 101 mL infarct nnU-Net predicted as empty, with 22% visible as Tmax > 6 s.
- **Best 1** (`best_1.png`): nnU-Net's highest-Dice case, for comparison.

## Visual readings

*To be filled in by Lorna from the figures.* These are one person's readings of one slice per case, kept apart from the measured numbers above. They are not a clinical opinion.

- Miss 1:
- Best 1:
- Other misses:

## What might fix each group

These are untested ideas.

- **Small infarcts.**
  - Check whether nnU-Net predicts anything at all on these cases once the masks are back.
  - Training that weights small lesions more.
  - A lower probability threshold.
  - Judge these cases by volume error rather than Dice: a 3 mL miss rarely changes a decision at the 21 mL cut-off.
- **Little visible delay.**
  - The perfusion input may not hold the answer.
  - Add NCCT as an extra channel, which plan §8.1 allows.
  - Use the visible-delay share as a warning: a low share should mean low `confidence.core` (Task 2).
- **Large infarct, little visible delay.**
  - Look at the prediction overlay when the masks arrive.
  - Check its onset-to-imaging time and occlusion site.
  - If the tissue died after imaging, no model of the admission scan can be expected to find it. Report it as a limit.
- **Visible delay, still missed.**
  - These are the cases a better model should catch. Check them first with overlays, then with folds 1–4 and an ensemble.

## Limits

- **One slice per figure.** The slice with the most infarct may not show what went wrong elsewhere.
- **No prediction overlays yet.** Fold 0's predicted masks were not kept and are being regenerated. Until then, we know *that* nnU-Net missed, not *where* it looked.
- **Only 26 cases**, one fold. Groups of 2–7 cases show patterns, not proven causes.
- **Visual readings are not a clinical opinion.**
- The territory and occlusion grouping use the TopCoW label numbering, which is not confirmed for this dataset (`docs/DATA.md`).
