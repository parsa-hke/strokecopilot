# Lorna: Vision A (perfusion), Phase 3 brief

Read [README.md](README.md) first: access, rules and how work is merged.

## Where Vision A stands

Vision A predicts the final infarct from CT perfusion. Two baselines exist on fold 0 (26 development cases):

| | Dice | Absolute volume error |
|---|---|---|
| Threshold baseline, rCBF < 30% (EXP-1) | 0.123 | 20.5 mL |
| nnU-Net, 300 epochs (EXP-14) | 0.218 ± 0.249 (median 0.111) | 19.6 mL |

Volume error matters more than Dice here, because the treatment rules are volume cut-offs (21, 31, 51 and 70 mL; plan §8.1). Parsa is training folds 1–4 on rented GPUs. You don't train models.

## Task 1: error analysis (start now)

**Question:** where and why does the model fail? On fold 0, 9 of the 26 cases score Dice 0, and one 101 mL infarct was predicted as empty. Lesions under 5 mL average 0.12, while three of the four above 95 mL score 0.60–0.70.

**Inputs (all on Drive):**
- `runs/EXP-14_nnunet_fold0/per_case.csv`: per case, Dice, predicted and reference volume, volume error, and threshold crossings.
- `runs/EXP-1_thresholds_baseline/per_case.csv`: the same for the threshold baseline, for comparison.
- `slim/isles24_slim_v1.tar`: images, masks (final infarct, occlusion, Circle of Willis labels) and clinical data. Load cases with `strokecopilot.data.isles24` as in `00_template`.

**Look at:**
- lesion size;
- territory (anterior or posterior circulation);
- occlusion site (`vision/perfusion/volumes.py` has the occlusion-site lookup);
- time from onset to imaging;
- perfusion data quality (Tmax voxels at −30 have no data);
- artefacts.

Look at the images of the failed cases. Also check where nnU-Net and the threshold disagree.

**Predicted masks:** fold 0's predicted masks weren't kept on the rented GPU. Folds 1–4 will bring theirs back, and fold 0's will be regenerated. Until then, start from the tables, the reference masks and the images. Add prediction overlays when the masks arrive.

**Deliverables:**
- `src/strokecopilot/vision/perfusion/error_analysis.py`, with tests on synthetic data;
- a thin notebook `notebooks/12_visionA_error_analysis.ipynb`;
- a short write-up, `docs/ERROR_ANALYSIS_VISION_A.md`, with figures (ISLES'24 images may go in the report);
- an Experiments row.

**Done when** the write-up names the main failure groups, with how many cases fall in each and example figures, and says what might fix each one.

## Task 2: confidence estimates (after folds 1–4)

The contract that Vision A hands to the reasoner (`schemas/vision_output.json`) has `confidence.core` and `confidence.penumbra`, each between 0 and 1. The plan suggests measuring how much the 5 fold models disagree.

**First design question, to agree with Parsa before coding:** every development case was in the training data of 4 of the 5 fold models. So ensemble disagreement on those cases will be too low. Options include:
- using only the out-of-fold model with test-time augmentation or MC dropout;
- or calibrating on development cases and checking on the held-out cases in Phase 4.

**Then:** write the code as functions with tests, and try it on a few cases on a Colab T4. Parsa runs inference over all cases on a rented GPU. **Done when** low confidence goes with large volume errors in a way you can show (for example, error by confidence band).

## Task 3: claim labelling (later, with Mohamed)

When the language-model runs are done, you and Mohamed each label about 100 claims as supported or not, without knowing which model wrote them. A short labelling guide will come with the task. About a day.

## Not yours

Training, the 5-fold results table, choosing which core feeds the contract, and `contract/build.py` are Parsa's.
