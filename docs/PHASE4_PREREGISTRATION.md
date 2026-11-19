# Phase 4 pre-registration (DEC-31)

Written 19 Nov 2026, before any held-out case was opened; **approved as drafted by Parsa the same day**. The decisions below are fixed. A
change after the held-out cases are unlocked needs a dated note saying why. Plan §7 Phase 4 and §9 define the goals;
this document fixes the choices those sections leave open.

**Held-out set:** 20 ISLES'24 cases (`splits/folds.json → heldout`; checked by sha256 only, never opened). Development
folds: 26, 26, 26, 26 and 25. Unlocking means `ALLOW_HELDOUT=1` in Phase 4 notebooks only.

## 1. Close before unlocking

1. **EXP-41 consensus.** Parsa settles the 28 disagreements as third rater. Then human–human κ and
   verifier–human agreement are reported, the last Phase 3 acceptance criterion.
2. **Freeze list.** None of these changes after unlocking:
   - nnU-Net's five fold models (EXP-14, EXP-18 to EXP-21);
   - Vision B, EXP-6 with DEC-28's threshold (kept, Item 2);
   - prompt v3;
   - the verifier settings: `judge_v3`, τ 0.6, metric points 1–20;
   - the reasoner models of DEC-24 and DEC-30, the tuned 4B being EXP-45.

   Item 1 (BiomedCLIP) touches only RSNA and never feeds the pipeline, so it is not blocked, but if it is wanted it
   should be decided before Phase 4 starts.
3. **Vision B's RSNA test split (DEC-22).** It stays closed "until the final Vision B model is chosen". That is now
   EXP-6 with DEC-28. It is opened **once**, for the final figures: scan AUC, sensitivity and specificity at 0.175 with
   patient-bootstrap intervals, sensitivity per subtype, and ECE. This runs on Kaggle, free, and is independent of
   ISLES'24.
4. **Code, built and tested on development cases only:**
   - the held-out contract path (D1);
   - the error-propagation and sensitivity analyses (D3);
   - the demo (plan §8.5).

   Notes for held-out cases are generated after unlocking, with the same generator and seed scheme as
   `isles24_notes_v1`.

## 2. Decisions to approve

**D1. Vision for held-out cases.**
- **Core:** the five fold models' **nnU-Net ensemble**: their softmax averaged, nnU-Net's own `-f 0 1 2 3 4`, with
  each fold's `checkpoint_final.pth`. This is the open point DEC-25 left to Phase 4. Every fold model was trained on
  development cases only, so the held-out cases are unseen by all five.
- **Everything else as for development cases:**
  - penumbra Tmax > 6 s (DEC-26);
  - `confidence.core` by the cut-off distance (DEC-27);
  - haemorrhage from EXP-6 at 0.175 (DEC-28);
  - ASPECTS null (DEC-9);
  - occlusion evidence as EXP-1.
- **Outputs:**
  - the contracts as `vision_outputs_heldout_v1`;
  - the ground-truth contracts (DEC-11) as `vision_outputs_gt_heldout_v1`, built the EXP-34 way: the reference final
    infarct as core, no bleed, confidence 1.0.
- **Check first:** the trainer folders on Drive hold only `fold_N/`. `plans.json` and `dataset.json` are rebuilt by
  `nnUNetv2_plan_and_preprocess` from the development data, as each training run did, and must match each fold's
  `debug.json` before any held-out case is predicted.

**D2. Reasoner models and arms on held-out cases.**
- **Models:** DEC-24's four (Sonnet 5.5, Qwen3.5-27B, Gemma 3 27B, MedGemma 27B) and DEC-30's pair (Gemma 3 4B
  prompted and tuned).
- **Arms:** prompt v3 with the rulebook, in two vision arms, predicted and ground-truth (DEC-11). The without-rulebook
  arm is not repeated: DEC-29's comparison is on development cases.
- **Size:** 6 models × 2 arms × 20 cases = 240 requests. The same serving as EXP-37 and EXP-47: Sonnet by batch, Qwen
  through OpenRouter pinned to Alibaba with the system prompt merged, and the Gemma models on vLLM on Colab.

**D3. Analyses, defined now.** "Truth" is `rules.decide` on the ground-truth contract and the clinical row (§9).
Undetermined decisions are excluded and counted, never guessed.
- **Recommendation correctness:** per model and vision arm, as counts with case-bootstrap intervals. With 20 cases the
  intervals are wide, and the report says so.
- **Hallucination metrics:** as in C9 (points 1–20), per model, with held-out cases next to EXP-37's 20 development
  real cases. That comparison is §9's realism check.
- **Error propagation.** For each wrong decision in the predicted arm, with model answer A and `R(v)` meaning
  `rules.decide` on contract v:
  - **vision error** if `R(predicted)` ≠ truth: the rules applied to the predicted contract already give the wrong
    answer;
  - **reasoning error** if `R(predicted)` = truth but A ≠ truth;
  - separately, **verifier caught** if any claim of that decision is flagged, otherwise **verifier miss**.

  Reported as a table per model, with the predicted-contract fields that differ from ground truth (core across a
  cut-off, the bleed flag, the occlusion site) for each vision error.
- **Sensitivity:**
  - `core_ml` ×0.8 and ×1.2 on each predicted held-out contract; the share of rule-derived decisions that flip
    (`rules.decide` only, free);
  - the same for Sonnet's own answers: 40 extra requests.
- **Vision on held-out:**
  - core: ensemble Dice, absolute volume error, and how often the core falls on the wrong side of 21, 31, 51 and
    70 mL, against the reference final infarct;
  - haemorrhage: flags on the 20 scans, presumed bleed-free as on development (EXP-16).

**D4. Order.**
1. Close §1 items 1 and 3.
2. Build and test the §1 item 4 code on development cases.
3. Parsa approves this document.
4. Unlock.
5. Contracts and notes.
6. Reasoner runs.
7. Verifier.
8. Analyses.
9. Demo on held-out cases.

## 3. Cost and time (estimates, unmeasured until run)

- **Anthropic:**
  - checking in EXP-37's 20-case real cells cost $0.16–0.64 per cell at batch prices, so the four DEC-24 models'
    eight cells come to about $3;
  - the DEC-30 pair about $1;
  - Sonnet's reasoner is included in its cells;
  - the ±20% Sonnet arm about $1.2;
  - **total about $5–6.**
- **OpenRouter:** Qwen's 40 requests, under $1.
- **Colab A100:** about 1–2 hours, for the nnU-Net ensemble on 20 cases plus the Gemma pair and the small pair.
- **Kaggle:** Vision B's RSNA test check, about 15 minutes, free.
