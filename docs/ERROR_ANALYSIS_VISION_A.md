# Vision A error analysis (EXP-15), draft

Lorna, Vision A. Task 1 of `docs/team/lorna_vision_a.md`.

## Summary

- The AI model (nnU-Net) completely missed the dead brain tissue in 9 of 26 test cases. The simple rule-based method missed all 9 too.
- Most misses are small: 7 of the 9 are under 21 mL.
- In most misses, the first scan showed little sign of trouble where the tissue later died: a median of 30% of the dead area, against 64% in the other cases.
- One miss stands out as a likely AI error (Miss 4): the trouble sign was clear, and the AI still missed it.
- For the volume of dead tissue, which is what treatment decisions use, the simple method was closer than the AI in every size group except the largest.

## Terms used

- **Infarct:** brain tissue that died after the stroke. Here, it is the area seen on a follow-up scan taken 2 to 9 days later.
- **mL (millilitre):** the unit for the infarct's size. A teaspoon holds about 5 mL.
- **nnU-Net:** the AI model that tries to outline the infarct from the first scan.
- **Threshold baseline:** a simple method with no AI. It marks tissue with very low blood flow (under 30% of the healthy side) inside the area where blood arrives late. We compare the AI against it.
- **Dice:** a score for how well an outline overlaps the true infarct, from 0 (no overlap at all) to 1 (perfect match).
- **Tmax:** how late blood arrives at each spot in the brain on the first scan. Over 6 seconds (Tmax > 6 s) is a sign that the tissue is at risk.
- **Fold 0:** our cases are split into five groups (folds). The AI was trained on four of them and tested on fold 0, which it had not seen. Fold 0 has 26 cases.
- **Miss:** a case where the AI's Dice is 0, meaning its outline does not touch the infarct at all.

## The question

Where and why does the AI fail to find the infarct? On fold 0, 9 of the 26 cases score Dice 0. One of them is a 101 mL infarct that the AI marked as empty. Is the problem the AI, or what the first scan showed?

## How it was done

- **Run:** EXP-15, using notebook `notebooks/12_visionA_error_analysis.ipynb`, with results at commit `c442a54`. The outputs are on Drive in `runs/EXP-15_visionA_error_analysis/`: `config.json`, `per_case.csv`, `metrics.json` and `figures/`.
- **Cases:** the 26 fold-0 cases. No held-out case was used.
- **Two methods, scored against the same true volumes:**
  - the AI, nnU-Net 3d_fullres trained for 300 epochs (EXP-14);
  - the threshold baseline (EXP-1, variant `within_tmax6`).
- **Measured by code for each case** (`strokecopilot.vision.perfusion.error_analysis`):
  - infarct size;
  - where the blocked artery is, and whether it is in the front or back of the brain;
  - time from stroke onset to the first scan;
  - how much of the infarct had no Tmax data;
  - **how much of the infarct showed Tmax > 6 s on the first scan**, counting only spots with usable data. Below, this is called "visible".
- **"Missed by both"** means the threshold baseline also scored under 0.05.
- **Pictures:** for each miss and for the AI's 3 best cases, one slice of the scan (the one with the most infarct), with the infarct outlined in red. Pictures carry labels only. Miss 1 is the largest miss.

All numbers below come from that run. No case IDs appear here.

## Results measured by code

| | Misses (Dice 0) | Other cases |
|---|---|---|
| Cases | 9 | 17 |
| Median share of infarct visible (Tmax > 6 s) | 30% | 64% |

- **No case was missed by the AI alone.** On all 9 misses, the threshold baseline also scored under 0.05, and exactly 0 on 6 of them.
  - Both methods read the same scan, so this suggests the problem lies more in what the scan showed than in the AI.
  - This is weak evidence, though. The threshold baseline is poor itself (mean Dice 0.12), so failing where it fails proves little.
- **7 of the 9 misses are under 21 mL.**
- **4 of the 9 misses had under 25% visible.**
- **Miss 1 (101 mL)** had 22% visible, while 88% of it had usable data. So the data was there, but most of this infarct did not look at risk on the first scan.
- **2 misses had 75% or more visible:** Miss 4 (88%) and Miss 8 (75%). The sign was there, and the AI still missed it.

## Volume error

The treatment rules use the infarct's volume, with cut-offs at 21, 31, 51 and 70 mL. So getting the volume right matters more than the Dice score.

**Average volume error by size group** (mL; how far the predicted volume is from the true volume, ignoring direction):

| Size group | AI (nnU-Net) | Threshold baseline |
|---|---|---|
| Under 5 mL | 7.6 | 6.8 |
| 5–21 mL | 8.7 | 5.7 |
| 21–70 mL | 28.2 | 19.8 |
| 70 mL and over | 50.7 | 70.2 |

- **The threshold baseline has the smaller volume error in every group except 70 mL and over,** even though the AI has the higher Dice. A better overlap does not mean a better volume, and the rules use the volume.
- The largest infarcts have the largest errors in mL for both methods. Miss 1 alone is a 101 mL error for the AI.

**Cases on the wrong side of each cut-off** (the predicted and true volumes fall on different sides of it; out of 26 cases):

| Cut-off | AI | Threshold baseline | Both |
|---|---|---|---|
| 21 mL | 5 | 7 | 4 |
| 31 mL | 5 | 6 | 4 |
| 51 mL | 3 | 4 | 3 |
| 70 mL | 3 | 3 | 2 |
| Any | 7 | 8 | 6 |

- The AI puts 7 of 26 cases on the wrong side of at least one cut-off, and the threshold baseline puts 8. In 6 cases, both do.
- The AI's counts per cut-off (5, 5, 3, 3) match EXP-14's own counts. That is an independent check that the tables were joined correctly.
- The AI is wrong slightly less often, but the two methods mostly go wrong on the same cases.

## Main failure groups

The groups overlap: a small infarct can also have little of it visible. Counts are out of the 9 misses.

| Group | Misses | What it means |
|---|---|---|
| Small infarcts (under 21 mL) | 7 | Small targets are easy to miss completely. |
| Little of it visible (under 25%) | 4 | The first scan barely showed the tissue that later died. Neither method had much to see. |
| Large infarct, little of it visible (Miss 1, 101 mL) | 1 | The data was usable, but most of it did not look at risk. It is the most costly miss for the volume cut-offs. |
| Visible, still missed (75% or more) | 2 | Miss 4 (88%) is the clearest likely AI error. Miss 8 (75%) is borderline: a late-blood area marks tissue at risk, not tissue that will die. |

**Why the infarct and the late-blood area can disagree.**
- Nearly every patient in this dataset (ISLES'24) had their artery successfully reopened (reperfusion). Of the 99 cases with a TICI value (a 0 to 3 grade of how well blood flow was restored), 97 were 2c or 3 (near-complete or complete), 2 were lower, and the rest have no value (plan §5.1).
- So the infarct is, almost always, the tissue that died *even though* blood flow came back.
- A large late-blood area with a small infarct is therefore expected: most of the tissue at risk was saved.
- The reverse, an infarct that showed little late blood on the first scan, may be tissue that died after the scan, or tissue the scan did not capture well.
- This is an interpretation. The data here cannot prove it.

## Example figures

The pictures are on Drive in `runs/EXP-15_visionA_error_analysis/figures/`. They will be copied into this document once chosen for the report (ISLES'24 images may go in the report).

- **Miss 1** (`miss_1.png`, 70 mL and over): the 101 mL infarct the AI marked as empty, with 22% visible.
- **Best 1** (`best_1.png`): the AI's best case, for comparison.

## Visual readings

*Drafted with AI help, checked by Lorna; one slice per case, not a clinical opinion.*

How to read the pictures: the red outline is the brain tissue that died, seen on a scan days later. On the TMAX picture, bright means blood was arriving late, a sign of trouble at the time of the first scan. The % is how much of the dead area showed that sign, measured by the code across all slices of the dead area, not just this one slice.

Miss 1 (70+ mL, 22%): A large area died, but on the first scan most of it looked normal. Only a small corner showed trouble. The scan quality was fine, so the damage just wasn’t showing yet when the scan was taken.

Miss 2 (21–70 mL, 62%): The dead area is along the outer edge of the brain. On this slice, the blood-flow scans don’t seem to reach that edge, so much of the dead area appears to have no data here. Where data exists, across all slices, 62% showed trouble.

Miss 3 (5–21 mL, 65%): A thin, curvy strip near the front, plus two tiny spots. The trouble area on the scan is bigger than what died: it covers the strip and extends past it. So the sign was there, but it didn’t show exactly which part would die, and the thin shape may have made it even harder for the AI.

Miss 4 (5–21 mL, 88%): A small patch deep inside the brain. It only looks slightly bright on this slice, but overall it showed the clearest trouble sign of all the misses. This is the strongest case of the AI missing something it could have seen.

Miss 5 (5–21 mL, 3%): A few small pieces spread across both sides. Almost none of it showed trouble on the first scan, and the scan looks noisy in the middle.

Miss 6 (5–21 mL, 30%): Three small separate spots, one right at the edge of the brain. The scan looks noisy here, and the spots don’t stand out.

Miss 7 (under 5 mL, 9%): A few tiny spots. There’s a large trouble area nearby, but the tissue that died was mostly outside it.

Miss 8 (under 5 mL, 75%): One small spot inside a much larger trouble area. The trouble sign was there, but it covered far more brain than actually died, so it didn’t point to this spot. Most of that area likely recovered after treatment.

Miss 9 (under 5 mL, 13%): One tiny spot near the back. It doesn’t show trouble on the scan; the trouble areas are somewhere else.

Best 1 (70+ mL, 80%): A large dead area along one side that matches a large, obvious trouble area on the scan. Easy to spot.

Best 2 (70+ mL, 64%): A large dead area that shows clear trouble over most of it.

Best 3 (21–70 mL, 67%): A medium dead area, mostly covered by a clear trouble patch.

## What might fix each group

These are untested ideas.

- **Small infarcts.**
  - Check whether the AI predicts anything at all on these cases, once its outlines are back.
  - Train the AI to give small infarcts more weight.
  - Lower the AI's probability threshold, so it marks tissue it is less sure about.
  - Judge these cases by volume error rather than Dice: a 3 mL miss rarely changes a decision at the 21 mL cut-off.
- **Little of it visible.**
  - The first scan may simply not hold the answer.
  - Add the plain CT scan (NCCT) as an extra input, which plan §8.1 allows.
  - Use the visible share as a warning: a low share should mean low confidence in the AI's core estimate (`confidence.core`, Task 2).
- **Large infarct, little of it visible (Miss 1).**
  - Look at the AI's outline when it is back.
  - Check its onset-to-scan time and where the blocked artery is.
  - If the tissue died after the scan, no model of the first scan can be expected to find it. Report it as a limit.
- **Visible, still missed (Miss 4, and Miss 8 as borderline).**
  - These are the cases a better AI should catch.
  - Check them first with the AI's outlines, then with folds 1 to 4 and a combination of all five models.

## Limits

- **One slice per picture.** The slice with the most infarct may not show what went wrong elsewhere.
- **One slice can disagree with the number across all slices.** On Miss 2's slice, much of the dead area appears to have no data, but where data exists across all slices, 62% showed trouble.
- **No AI outlines in the pictures yet.** The AI's fold-0 outlines were not kept and are being regenerated. Until then, we know *that* the AI missed, not *where* it looked.
- **Only 26 cases,** from one fold. Groups of 2 to 7 cases show patterns, not proven causes.
- **Visual readings are not a clinical opinion.**
- The front/back grouping and the artery grouping use a label numbering (TopCoW) that is not confirmed for this dataset (`docs/DATA.md`).
