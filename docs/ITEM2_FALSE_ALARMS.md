# Item 2: Vision B's ISLES'24 false alarms — what is tested, fixed before any result

Written 19 Nov 2026, before any candidate below was scored on any data; **approved by Parsa the same day**. Code: `vision/haemorrhage/explain.py`
(Grad-CAM and the density measures) and `vision/haemorrhage/slab_rules.py` (the candidate rules). Changing anything
here after a candidate has been scored needs a dated note saying why.

## Why

At DEC-28's threshold (0.175 on the calibrated scale), Vision B flags 68 of 129 presumed bleed-free ISLES'24
development scans (EXP-16). EXP-35 read the 23 scans flagged at 0.175 but not at 0.278: 12 were driven by skull-base
slabs, 4 by normal calcification, 3 by a dense falx and 1 by dense vessels; none read as a bleed. These were read by
Claude, not by a radiologist. Because `haemorrhage.present` is an absolute stop in the rules, every false flag
withholds treatment in the predicted-vision arm.

## Minimal scope (agreed 17 Nov 2026)

No paid GPU, no API and no re-run of training. Two parts:

1. **Explain** (descriptive). Grad-CAM on ResNet-50's `layer4` for the single logit:
   - on the top 2 slabs of each of the 68 flagged ISLES'24 scans (on the Mac, from the 5 mm slabs as in EXP-16);
   - on a reference sample from RSNA validation: 20 true positives and 20 false positives at DEC-28's threshold,
     drawn by seed (on Kaggle; only numbers leave it).

   Read-outs per slab: the share of heat in brain-density tissue, in bone, in other head tissue and outside the head
   (`heat_shares`), and the slab's brain fraction and height (`slice_measures`). The figures use ISLES'24 slabs only.
   RSNA images and IDs never leave Kaggle (docs/DATA_USE.md).

2. **Test candidate fixes** (decides adoption). These are the four rules of `slab_rules.py`, and nothing else:
   - **none**: EXP-6's score, the mean of the top 2 slice probabilities, unchanged.
   - **gate(ratio)**: slices whose brain fraction is below `ratio` times the scan's own largest are left out of the
     top 2. If fewer than 2 slices pass, the score falls back to "none".
   - **adjacent**: the top 2 must be neighbouring slices along the body.
   - **both(ratio)**: neighbouring slices that both pass the gate. If no pair passes, the score falls back to
     "adjacent".
   - **Grid:** ratio ∈ {0.25, 0.40, 0.55, 0.70}. That makes 10 candidates in all.

## How a candidate is chosen (RSNA validation only)

The procedure is EXP-17's (`operating_point.py`):

1. Score every validation scan under each candidate from the same saved slice probabilities, which come from EXP-6's
   model. The model is not retrained.
2. Choose a threshold on the raw score for 96% sensitivity (DEC-28's target), cross-fitted on patient halves within
   bleed and no-bleed patients, 20 seeds. Platt calibration is monotonic, so the calls are the same as on the
   calibrated scale. The candidate's own calibration is refitted only if it is adopted, for the contract's
   `confidence`.
3. Read out the pooled held-out sensitivity and specificity, averaged over the 20 seeds, with patient-bootstrap
   intervals as in EXP-17. Also read out sensitivity per bleed subtype (epidural, intraparenchymal, intraventricular,
   subarachnoid, subdural), because a skull-base gate could miss low subarachnoid or posterior-fossa bleeds.
4. **A candidate is eligible** only if:
   - its cross-fitted sensitivity is no more than 1 point below "none"'s;
   - no subtype's sensitivity is more than 3 points below "none"'s.
5. **Among eligible candidates**, the one with the highest cross-fitted specificity is the proposal.
6. **Adoption needs**:
   - the proposal's specificity gain over "none" to have a paired patient-bootstrap 95% interval above 0;
   - Parsa's decision.

   Adopting amends DEC-28 (the scan score and threshold), rebuilds the 129 contracts as a new version, and reruns the
   predicted-vision arm on them before Phase 4 unlocks the held-out cases. If nothing is eligible or the gain's
   interval includes 0, DEC-28 stands and the false alarms are reported with part 1's explanation.

ISLES'24 has no bleed labels. Its flag count under the proposal (paired against "none", McNemar) is read out and
reported, but it never takes part in the choice (DEC-28).

## Runs

- **EXP-49 (Kaggle, CPU or the free T4):** RSNA validation, slice probabilities and slice measures, all 10
  candidates, the selection above, the 40-slice Grad-CAM reference. Numbers only leave Kaggle.
- **EXP-50 (Mac, CPU):** ISLES'24's 129 development scans:
  - slab probabilities and measures;
  - the 10 candidates' flag counts;
  - Grad-CAM on the 68 flagged scans' top 2 slabs;
  - figures.

  ISLES'24 case IDs stay on Drive.

Open before EXP-49:
- whether EXP-6's per-slice validation probabilities are saved on Kaggle (otherwise they are recomputed on the free
  T4);
- where EXP-6's checkpoint is for the Mac run.

## Results (19 Nov 2026)

**EXP-49 and EXP-51 (RSNA validation, Kaggle).** EXP-51 is EXP-49 rerun with the threshold export, and its selection
is identical. The scan AUC (0.9772) reproduces EXP-6, and all 10 candidates are eligible. The proposal is
**gate(0.55)**:
- specificity 0.8853 against "none" 0.8812, at sensitivity 0.9595 against 0.9597 (means over 20 seeds);
- paired gain on the main seed +0.0047 [+0.0015, +0.0086], which meets the adoption rule;
- subtype sensitivities unchanged within 0.001.

Requiring the top two slices to be neighbours lowers specificity. The Grad-CAM reference shows why a gate does little
on RSNA: its false positives put 73% of their heat in brain tissue at mid-head, as true positives do (85%).

**EXP-50 (ISLES'24, Mac, 1.4 min).** "none" reproduces the 68 flags of 129 (52.7%). Against that:
- gate(0.55) flags 64 (49.6%): it clears 4 and adds none (McNemar exact p = 0.125). The 4 it clears are
  low-brain slabs, three of them at the skull base (height 0.17–0.19);
- no candidate clears flags significantly. The adjacency rules swap about as many flags as they clear, and both(0.70)
  adds 9.

The flagged scans' top slab mostly sits in brain tissue: median relative brain fraction 0.87, and none below 0.40.
Its median height is 0.38. Grad-CAM puts a median 76% of the heat in brain tissue, and only 2 of 136 slabs have more
heat on bone than on brain. Only 16% of flagged scans have their top two slabs as neighbours, against 25% of
unflagged scans.

**Reading.** Skull-base slabs explain only a few of ISLES'24's false alarms: 4 of 68 under the gate. EXP-35's 12 of 23
came from the borderline band between 0.175 and 0.278 alone. Most flags come from single hot slabs inside the brain
tissue of acute-stroke patients. This is a domain shift in how the brain itself looks to the model, which no slab
rule can fix. These are density masks and Grad-CAM, read by Claude, not by a radiologist.

**Decision (Parsa, 19 Nov 2026): DEC-28 stands.** gate(0.55) met the RSNA rule, but on ISLES'24 it cleared 4 of 68
flags (p = 0.125). Adopting it would have meant amending DEC-28, rebuilding the 129 contracts and rerunning the
predicted-vision arms. Item 2 is reported as an explanation of the false alarms (Grad-CAM, tissue shares, the four
rules tested) with a negative result: no slab rule fixes them.
