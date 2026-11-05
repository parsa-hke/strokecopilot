# Mohamed: Vision B (haemorrhage), Phase 3 brief

Read [README.md](README.md) first: access, rules and how work is merged. Before you touch RSNA, accept the competition rules on Kaggle and read `docs/DATA_USE.md`.

## Where Vision B stands

Vision B answers one question on plain CT: is there a bleed? It is a safety gate, because thrombolysis in a patient with a bleed can kill them. EXP-6 trained a ResNet-50 on RSNA ICH slices, with scans scored by the mean of the top 2 slices and Platt calibration:
- validation scan AUC 0.977, ECE 0.010;
- sensitivity 0.951 at specificity 0.917, at threshold 0.278.

The weights and calibration are on Drive in `runs/EXP-6_visionB_resnet50/` (`best.pt`, `calibration.json`). Code is in `src/strokecopilot/vision/haemorrhage/`, and training is in `notebooks/20_visionB_train.ipynb` (Kaggle).

## Task 1: the operating point (start now)

**Problem:** the 0.278 threshold was chosen on the validation split and measured on the same split, so the specificity is optimistic. The RSNA test split stays closed until the final Vision B model is chosen (`ALLOW_RSNA_TEST`). Don't open it.

**Suggested approach:**
1. Split the validation scans in two halves by patient, with a fixed seed.
2. Choose the threshold for at least 95% sensitivity on one half, and measure on the other. Then swap.
3. Report sensitivity and specificity with confidence intervals.

This needs the validation slice predictions, which aren't saved. Recompute them on Kaggle with `best.pt`, uploaded as a private Kaggle dataset (weights stay private to the team).

**Write the reasoning.** A missed bleed is the worst failure in the system. Say what specificity we give up for it, and how often that would wrongly hold back treatment.

**Deliverables:**
- code in `vision/haemorrhage/` with tests;
- a Kaggle notebook that prints counts and metrics only;
- an Experiments row;
- a proposed threshold, which Parsa records as a decision.

## Task 2: domain-shift check on ISLES'24 (start now, Colab)

The ISLES'24 patients are ischaemic strokes who went on to treatment, so their plain CT should mostly show no bleed. Running Vision B on them shows how a model trained on RSNA behaves on new data.

**Steps:**
1. Load the 129 development cases' original plain CT with `isles24.load_native_ncct` (never the resampled 1.5 mm grid; plan §8.2).
2. Reformat each to the training slice thickness and apply the same three windows as training (`window_hu` in `data/rsna.py`: brain, subdural, bone).
3. Run `best.pt` with `calibration.json`.
4. Report the false-positive rate at the chosen threshold and the distribution of scan scores, and look at the scans scored highest.

This uses ISLES'24 only, so it can run on Colab with Drive. No RSNA data is involved.

**Deliverables:**
- `vision/haemorrhage/isles_check.py` with tests;
- `notebooks/21_visionB_isles_check.ipynb`;
- an Experiments row.

## Task 3 (optional): per-subtype outputs

RSNA labels five haemorrhage subtypes. Only if Tasks 1–2 are done and time allows.

## Task 4: claim labelling (later, with Lorna)

When the language-model runs are done, you and Lorna each label about 100 claims without knowing which model wrote them. A guide will come with the task. About a day.

## Not yours

The final threshold decision (you propose it, Parsa decides) and `contract/build.py`.
