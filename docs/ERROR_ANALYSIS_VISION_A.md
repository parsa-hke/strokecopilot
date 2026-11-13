# Vision A error analysis (EXP-15), draft

Lorna, Vision A. Task 1 of `docs/team/lorna_vision_a.md`. Corrected at Parsa's review of PR #1 (13 Nov 2026).

This draft covers fold 0 only (26 cases). All five folds are now trained; EXP-25 has the 5-fold comparison with the threshold baseline.

## Summary

- The AI model (nnU-Net) scored Dice 0 on 9 of 26 test cases: its outline did not touch the dead brain tissue at all.
  - In 4 of the 9 it marked nothing, and in a fifth almost nothing (0.04 mL).
  - In the other 4 it marked tissue in the wrong place. One of these, Miss 8, is an over-call: 32.6 mL marked for a 1.1 mL infarct, enough to cross the 21 and 31 mL cut-offs.
- Small infarcts are missed more often: 7 of the 17 cases under 21 mL, against 2 of the 9 larger ones.
- Part of the infarct can have no perfusion data at all. Of the 4 cases with at least 5% of the infarct in no-data areas, 3 are misses. In Miss 2, half the infarct had no data.
- Misses tended to show less of the infarct as "at risk" on the first scan (a median of 30%, against 64% in the other cases), but the two groups overlap a lot, and this is not a clear separation. The share also goes with size.
- One miss stands out as a likely AI error (Miss 4): the trouble sign was clear, and the AI still missed it.
- Over all 26 cases, the AI's volume is slightly closer than the simple method's (19.6 against 20.5 mL). By size group, the simple method was closer in every group except the largest, partly because it predicts little everywhere.

## Terms used

- **Infarct:** brain tissue that died after the stroke. Here, it is the area seen on a follow-up scan taken 2 to 9 days later.
- **mL (millilitre):** the unit for the infarct's size. A teaspoon holds about 5 mL.
- **nnU-Net:** the AI model that tries to outline the infarct from the first scan.
- **Threshold baseline:** a simple method with no AI. It marks tissue with very low blood flow (under 30% of the healthy side) inside the area where blood arrives late. We compare the AI against it.
- **Dice:** a score for how well an outline overlaps the true infarct, from 0 (no overlap at all) to 1 (perfect match).
- **Tmax:** how late blood arrives at each spot in the brain on the first scan. Over 6 seconds (Tmax > 6 s) is a sign that the tissue is at risk.
- **No perfusion data:** spots where the perfusion scan has no value (Tmax −30 in the files). Spots with any Tmax below 0 are treated as unreliable (plan §8.1).
- **Fold 0:** our cases are split into five groups (folds). The AI was trained on four of them and tested on fold 0, which it had not seen. Fold 0 has 26 cases.
- **Miss:** a case where the AI's Dice is 0, meaning its outline does not touch the infarct at all.

## The question

Where and why does the AI fail to find the infarct? On fold 0, 9 of the 26 cases score Dice 0. One of them is a 101 mL infarct that the AI marked as empty. Is the problem the AI, or what the first scan showed?

## How it was done

- **Run:** EXP-15, using notebook `notebooks/12_visionA_error_analysis.ipynb`, with results at commit `c442a54`. The outputs are on Drive in `runs/EXP-15_visionA_error_analysis/`: `config.json`, `per_case.csv`, `metrics.json` and `figures/`. (`config.json` was written at an earlier commit and kept on reruns; `metrics.json` records both.)
- **Cases:** the 26 fold-0 cases. No held-out case was used.
- **Two methods, scored against the same true volumes:**
  - the AI, nnU-Net 3d_fullres trained for 300 epochs (EXP-14);
  - the threshold baseline (EXP-1, variant `within_tmax6`).
- **Measured by code for each case** (`strokecopilot.vision.perfusion.error_analysis`):
  - infarct size;
  - where the blocked artery is, and whether it is in the front or back of the brain;
  - time from stroke onset to the first scan;
  - how much of the infarct had no perfusion data, and how much had no reliable data;
  - **how much of the infarct showed Tmax > 6 s on the first scan**, counting only spots with reliable data. Below, this is called "visible".
- **"Missed by both"** means both methods scored under 0.05. That is 10 cases in `metrics.json`: the 9 misses, plus one case where the AI scored 0.03 rather than 0.
- **Pictures:** for each miss and for the AI's 3 best cases, one slice of the scan (the one with the most infarct), with the infarct outlined in red. Pictures carry labels only. Miss 1 is the largest miss.

All numbers below come from that run. No case IDs appear here.

## Results measured by code

| | Misses (Dice 0) | Other cases |
|---|---|---|
| Cases | 9 | 17 |
| Median share of infarct visible (Tmax > 6 s) | 30% | 64% |
| Under 25% visible | 4 | 5 |

- **The visible share differs, but not clearly.** The medians are far apart, but 4 of 9 misses and 5 of 17 other cases are under 25% visible. With groups this small, the difference could be chance (Mann-Whitney p ≈ 0.28). Smaller infarcts also tend to show less, so size and visibility are mixed up here.
- **Size has a base rate.** 17 of the 26 cases are under 21 mL, so most misses being small is partly what you would expect. The miss rate is what matters: 7 of 17 small cases, against 2 of 9 larger ones.
- **No case was missed by the AI alone.** On all 9 misses, the threshold baseline also scored under 0.05, and exactly 0 on 6 of them.
  - This does not show that the scan is the problem. The baseline only looks inside the Tmax > 6 s area, so when little of the infarct is in that area, it cannot find it by design.
  - The baseline is also poor itself (mean Dice 0.12), so failing where it fails proves little.
- **No perfusion data.** 4 of the 26 cases have at least 5% of the infarct in no-data areas, and 3 of them are misses (Miss 2, 3 and 7). Miss 2 is the extreme: 51% of its infarct had no data, and 66% no reliable data.
- **Miss 1 (101 mL)** had 22% visible, while 88% of it had reliable data. So the data was there, but most of this infarct did not look at risk on the first scan.
- **Miss 4 (88% visible)** showed the sign clearly, and the AI still missed it.

**What the AI marked on each miss** (true and predicted volume in mL):

| Miss | Size group | True | AI predicted | Visible | No perfusion data |
|---|---|---|---|---|---|
| 1 | 70 mL and over | 101.0 | 0 | 22% | 2% |
| 2 | 21–70 mL | 29.6 | 0 | 62% of the part with data | 51% |
| 3 | 5–21 mL | 11.7 | 0 | 65% | 5% |
| 4 | 5–21 mL | 11.0 | 0.8 | 88% | 0% |
| 5 | 5–21 mL | 8.6 | 0.04 | 3% | 0% |
| 6 | 5–21 mL | 5.7 | 0 | 30% | 3% |
| 7 | under 5 mL | 1.5 | 0.1 | 9% | 6% |
| 8 | under 5 mL | 1.1 | **32.6** | 75% | 0% |
| 9 | under 5 mL | 0.3 | 2.8 | 13% | 0% |

## Volume error

The treatment rules use the infarct's volume, with cut-offs at 21, 31, 51 and 70 mL. So getting the volume right matters more than the Dice score.

**Average volume error** (mL; how far the predicted volume is from the true volume, ignoring direction):

| Size group | Cases | AI (nnU-Net) | Threshold baseline |
|---|---|---|---|
| Under 5 mL | 5 | 7.6 | 6.8 |
| 5–21 mL | 12 | 8.7 | 5.7 |
| 21–70 mL | 4 | 28.2 | 19.8 |
| 70 mL and over | 5 | 50.7 | 70.2 |
| **All cases** | **26** | **19.6** | **20.5** |

- **Over all cases, the AI is slightly closer.** By size group, the threshold baseline has the smaller error in every group except 70 mL and over, even though the AI has the higher Dice. A better overlap does not mean a better volume, and the rules use the volume.
- **Part of the baseline's edge on small infarcts comes from predicting little.** In the 5–21 mL group, the baseline's median prediction is 8.2 mL against a true median of 8.5 mL, close by size, but its Dice is under 0.1 in 8 of the 12 cases (median 0.03), so mostly not in the right place.
- **The groups are small** (4 to 12 cases), so one case moves a group's average a lot. Miss 1 alone is a 101 mL error for the AI.

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
| Small infarcts (under 21 mL) | 7 | 7 of the 17 small cases are missed, against 2 of the 9 larger ones. Small targets are easier to miss completely. |
| No perfusion data for part of the infarct | 3 (Miss 2, 3, 7) | 3 of the 4 cases with at least 5% of the infarct in no-data areas are misses. In Miss 2, half the infarct had no data, so neither method had an input there. |
| Little of it visible (under 25%) | 4 | The first scan showed little of the tissue that later died. But 5 of the 17 other cases are also under 25%, so this alone does not explain a miss. |
| Large infarct, little of it visible (Miss 1, 101 mL) | 1 | The data was reliable, but most of it did not look at risk. It is the most costly miss for the volume cut-offs. |
| Visible, still missed (Miss 4, 88%) | 1 | The clearest likely AI error. |
| Over-call in the wrong place (Miss 8) | 1 | 32.6 mL marked for a 1.1 mL infarct, none of it on the dead spot. The infarct sits inside a large at-risk area that mostly survived. It puts the case on the wrong side of 21 and 31 mL. |

**Why the infarct and the late-blood area can disagree.**
- Nearly every patient in this dataset (ISLES'24) had their artery successfully reopened (reperfusion). Of the 99 cases with a TICI value (a 0 to 3 grade of how well blood flow was restored), 97 were 2c or 3 (near-complete or complete), 2 were lower, and the rest have no value (plan §5.1).
- So the infarct is, almost always, the tissue that died *even though* blood flow came back.
- A large late-blood area with a small infarct is therefore expected: most of the tissue at risk was saved. Miss 8 is an example.
- The reverse, an infarct that showed little late blood on the first scan, may be tissue that died after the scan, or tissue the scan did not capture well.
- This is an interpretation. The data here cannot prove it.

## Example figures

The pictures are on Drive in `runs/EXP-15_visionA_error_analysis/figures/`. They will be copied into this document once chosen for the report (ISLES'24 images may go in the report).

- **Miss 1** (`miss_1.png`, 70 mL and over): the 101 mL infarct the AI marked as empty, with 22% visible.
- **Best 1** (`best_1.png`): the AI's best case, for comparison.

## Visual readings

*Drafted with AI help, checked by Lorna; one slice per case, not a clinical opinion. "Noisy" is judged by eye, not measured.*

How to read the pictures: the red outline is the brain tissue that died, seen on a scan days later. On the TMAX picture, bright means blood was arriving late, a sign of trouble at the time of the first scan. The % is how much of the dead area showed that sign, measured by the code across all slices of the dead area, not just this one slice, and counting only spots with reliable data.

Miss 1 (70+ mL, 22%): A large area died, but on the first scan most of it looked normal. Only a small corner showed trouble. The scan quality was fine. The damage may not have been showing yet, or this tissue died after the scan; the data cannot tell which.

Miss 2 (21–70 mL, 62% of the part with data): The dead area is along the outer edge of the brain. On this slice, the blood-flow scans don't reach that edge, and across all slices half of the dead area has no perfusion data. Only about a third of the dead area has reliable data, and 62% of that third showed trouble: about a fifth of the whole dead area. Neither method had an input for the rest.

Miss 3 (5–21 mL, 65%): A thin, curvy strip near the front, plus two tiny spots. The trouble area on the scan is bigger than what died: it covers the strip and extends past it. So the sign was there, but it didn't show exactly which part would die, and the thin shape may have made it even harder for the AI.

Miss 4 (5–21 mL, 88%): A small patch deep inside the brain. It only looks slightly bright on this slice, but overall it showed the clearest trouble sign of all the misses. This is the strongest case of the AI missing something it could have seen.

Miss 5 (5–21 mL, 3%): A few small pieces spread across both sides. Almost none of it showed trouble on the first scan, and the scan looks noisy in the middle.

Miss 6 (5–21 mL, 30%): Three small separate spots, one right at the edge of the brain. The scan looks noisy here, and the spots don't stand out.

Miss 7 (under 5 mL, 9%): A few tiny spots. There's a large trouble area nearby, but the tissue that died was mostly outside it.

Miss 8 (under 5 mL, 75%): One small spot inside a much larger trouble area. The trouble sign was there, but it covered far more brain than actually died, so it didn't point to this spot. Most of that area likely recovered after treatment. The AI marked 32.6 mL, none of it on the dead spot, so this is an over-call rather than a case of seeing nothing. Where it marked it, the pictures do not show yet.

Miss 9 (under 5 mL, 13%): One tiny spot near the back. It doesn't show trouble on the scan; the trouble areas are somewhere else.

Best 1 (70+ mL, 80%): A large dead area along one side that matches a large, obvious trouble area on the scan. Easy to spot.

Best 2 (70+ mL, 64%): A large dead area that shows clear trouble over most of it.

Best 3 (21–70 mL, 67%): A medium dead area, mostly covered by a clear trouble patch.

## What might fix each group

These are untested ideas.

- **Small infarcts.**
  - In 5 of the 7 small misses (Miss 3 to 7), the AI marked at most 0.8 mL. In the other 2 it marked tissue in the wrong place (Miss 8 and 9).
  - Train the AI to give small infarcts more weight.
  - Lower the AI's probability threshold, so it marks tissue it is less sure about (also an option in DEC-26).
  - Judge these cases mostly by volume error rather than Dice: a few mL missed rarely changes a decision at the 21 mL cut-off. Over-calls can, though (Miss 8).
- **No perfusion data.**
  - No model can find infarct where the scan has no data. Report the no-data share with each case.
  - Use it as a warning and for confidence: a large no-data share inside the at-risk area should lower `confidence.core` (Task 2).
- **Little of it visible.**
  - The first scan may simply not hold the answer.
  - Add the plain CT scan (NCCT) as an extra input, which plan §8.1 allows.
  - Use the visible share as a warning: a low share should mean low confidence in the AI's core estimate (`confidence.core`, Task 2).
- **Large infarct, little of it visible (Miss 1).**
  - Look at the AI's outline (it marked nothing here).
  - Check its onset-to-scan time and where the blocked artery is.
  - If the tissue died after the scan, no model of the first scan can be expected to find it. Report it as a limit.
- **Visible, still missed (Miss 4).**
  - This is the case a better AI should catch.
  - Check it first with the AI's outline, then with folds 1 to 4.
- **Over-call (Miss 8).**
  - Look at where the AI's 32.6 mL sits relative to the at-risk area, with the EXP-22 outlines.
  - This is the costly kind of error for the cut-offs, so count over-calls across all five folds.

## Limits

- **One slice per picture.** The slice with the most infarct may not show what went wrong elsewhere.
- **The visible share counts only spots with reliable data.** Where much of the infarct has no data (Miss 2), a high share can still mean little of the infarct was seen.
- **No AI outlines in the pictures yet.** The AI's fold-0 outlines were regenerated as EXP-22 and are on Drive in `runs/EXP-22_nnunet_fold0_masks/`, but they are not drawn in these pictures yet. The predicted volumes above show *how much* the AI marked, not *where*.
- **Only 26 cases,** from one fold. Groups of 1 to 7 cases show patterns, not proven causes.
- **Visual readings are not a clinical opinion.** Noise and artefacts were judged by eye, not measured.
- **Some planned groupings are not reported yet:**
  - The occlusion site is "unknown" in 22 of 26 cases by the contract's own rule, so grouping by it says little. Grouping by the nearest artery label would need a distance limit: 12 occlusions sit more than 5 mm from their nearest label.
  - The front/back grouping uses the same nearest label with no distance limit, and a label numbering (TopCoW) that is not confirmed for this dataset (`docs/DATA.md`).
  - Onset to scan time is known for only 9 of 26 cases.
