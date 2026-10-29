# Fixtures

Five hand-written `vision_output.json` objects (plan §6.3), each with a matching triage note in `notes/` (same filename stem, `.txt`). They are the NLP side's test bed until Phase 3 and the regression set afterwards.

- Case IDs are `fixture-fxNN`, never real ISLES'24 subject IDs.
- Notes contain clinical facts only (age, onset/LKW, NIHSS, mRS, medications, history). Imaging findings live only in the JSON, so every claim has exactly one place it can be traced to.
- **The expected recommendations below must never be passed to the reasoner.** They are for evaluation.
- `penumbra_ml` is the total hypoperfused (Tmax > 6 s) volume, core included (§6.1). Salvageable tissue is `mismatch_volume_ml = penumbra_ml - core_ml`.

| Fixture | Scenario | Key vision values | Expected recommendation (plan §6.3; to be confirmed against the corpus) | Also exercises |
|---|---|---|---|---|
| fx01 | Witnessed onset 2.5 h, NIHSS 8, M2 | core 0.6, penumbra 22.4, ratio **null**, mismatch 21.8 | thrombolysis: yes; thrombectomy: consider | `mismatch_ratio` null because core < 1 mL — must be read as unknown, not 0 |
| fx02 | LKW 9 h, NIHSS 16, M1 | core 18.0, penumbra 95.0, ratio 5.28, mismatch 77.0 | thrombolysis: no (window); thrombectomy: yes (DEFUSE-3 mismatch) | — |
| fx03 | Witnessed onset 2 h, NIHSS 12, haemorrhage present (p = 0.96) | core 5.2, penumbra 13.9, occlusion `unknown` | thrombolysis: no (absolute contraindication); thrombectomy: no | `unknown` occlusion site; warning "CTA not acquired" |
| fx04 | Witnessed onset 6 h, NIHSS 18, M1 | core 110.0, penumbra 120.0, ratio 1.09, mismatch 10.0 | thrombolysis: no; thrombectomy: no (core too large) | — |
| fx05 | Wake-up stroke, LKW 10 h, age 82, NIHSS 11, mRS 1, M1 | core 15.0, penumbra 29.5, ratio 1.97, mismatch 14.5 | thrombolysis: no; thrombectomy: DAWN criteria — borderline, must cite | DEFUSE-3 near-miss (mismatch volume 14.5 < 15 mL) vs DAWN pass (age ≥ 80, NIHSS ≥ 10, core < 21 mL); warning "motion artefact" |
