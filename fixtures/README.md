# Fixtures

Five hand-written `vision_output.json` objects (plan §6.3), each with a matching triage note in `notes/` (same filename stem, `.txt`). They are the NLP side's test bed until Phase 3 and the regression set afterwards.

- Case IDs are `fixture-fxNN`, never real ISLES'24 subject IDs.
- Notes contain clinical facts only (age, onset/LKW, NIHSS, mRS, medications, history). Imaging findings live only in the JSON, so every claim has one place it can be traced to.
- **The expected recommendations below must never be passed to the reasoner.** They are for evaluation.
- `penumbra_ml` is the total hypoperfused (Tmax > 6 s) volume, core included (§6.1). Salvageable tissue is `mismatch_volume_ml = penumbra_ml - core_ml`.
- `aspects` is null from the real pipeline, so fx03 and fx05 exercise the null case. Null means unknown, never a low score.

**Status of the expected recommendations:** drafted from secondary summaries of the 2026 AHA/ASA guideline plus the DAWN and DEFUSE-3 criteria in plan §8.3. The guideline's primary text has not been checked yet. Confirm them against the corpus during ingestion and have a clinician review them.

| Fixture | Scenario | Key vision values | Expected recommendation | Also exercises |
|---|---|---|---|---|
| fx01 | Witnessed onset 2.5 h, NIHSS 8, M2 | core 0.6, penumbra 22.4, ratio **null**, mismatch 21.8, ASPECTS 9 | thrombolysis: yes; thrombectomy: consider (M2 is a weaker recommendation, verify) | `mismatch_ratio` null because core < 1 mL: read as unknown, not 0 |
| fx02 | LKW 9 h, NIHSS 16, M1, mRS 0 | core 18.0, penumbra 95.0, ratio 5.28, mismatch 77.0, ASPECTS 8 | thrombectomy: yes (DEFUSE-3 mismatch); thrombolysis: borderline, at the very end of the 4.5–9 h window, must cite | Hard boundary at exactly 9 h, large salvageable volume |
| fx03 | Witnessed onset 2 h, NIHSS 12, haemorrhage present (p = 0.96) | core 5.2, penumbra 13.9, occlusion `unknown`, ASPECTS null | thrombolysis: no (absolute contraindication); thrombectomy: no | `unknown` occlusion site; warning "CTA not acquired" |
| fx04 | Witnessed onset 6 h, age 71, NIHSS 18, mRS 0, M1 | core 110.0, penumbra 120.0, ratio 1.09, mismatch 10.0, ASPECTS 4 | thrombectomy: yes, large-core tier (ASPECTS 3–5, age < 80), must state that mass effect is not reported; thrombolysis: no (little salvageable tissue) | **Outdated-knowledge trap:** fails DEFUSE-3 (core > 70 mL), so a model relying on pre-2026 rules wrongly refuses |
| fx05 | Wake-up stroke, LKW 10 h, sleep midpoint about 5 h 45 min before imaging, age 82, NIHSS 11, mRS 1, M1 | core 15.0, penumbra 29.5, ratio 1.97, mismatch 14.5, ASPECTS null | thrombectomy: DAWN criteria, borderline, must cite; thrombolysis: borderline, reasonable under the sleep-midpoint rule if tissue is salvageable, must cite | DEFUSE-3 near-miss (mismatch volume 14.5 < 15 mL) vs DAWN pass (age ≥ 80, NIHSS ≥ 10, core < 21 mL); null ASPECTS; warning "motion artefact" |
