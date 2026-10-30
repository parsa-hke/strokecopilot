# Fixtures

Five hand-written `vision_output.json` objects (plan §6.3), each with a matching triage note in `notes/` (same filename stem, `.txt`). They are the NLP side's test bed until Phase 3 and the regression set afterwards.

- Case IDs are `fixture-fxNN`, never real ISLES'24 subject IDs.
- Notes contain clinical facts only (age, onset/LKW, NIHSS, mRS, medications, history). Imaging findings live only in the JSON, so every claim has one place it can be traced to.
- **The expected recommendations below must never be passed to the reasoner.** They are for evaluation.
- `penumbra_ml` is the total hypoperfused (Tmax > 6 s) volume, core included (§6.1). Salvageable tissue is `mismatch_volume_ml = penumbra_ml - core_ml`.
- `aspects` is null from the real pipeline, so fx03 and fx05 exercise the null case. Null means unknown, never a low score.

**Status of the expected recommendations:** checked against the primary text of the 2026 AHA/ASA guideline on 30 October 2026, using the rules in plan §8.3. Numeric DAWN and DEFUSE-3 criteria still need checking against the trial papers, and a clinician should review the table.

| Fixture | Scenario | Key vision values | Expected recommendation | Also exercises |
|---|---|---|---|---|
| fx01 | Witnessed onset 2.5 h, NIHSS 8, M2 | core 0.6, penumbra 22.4, ratio **null**, mismatch 21.8, ASPECTS 9 | thrombolysis: yes (§4.6.1); thrombectomy: depends on M2 dominance, which is not reported: reasonable if dominant proximal (COR 2a), not recommended otherwise (COR 3) | `mismatch_ratio` null because core < 1 mL: read as unknown, not 0; unreported M2 dominance |
| fx02 | LKW 9 h, NIHSS 16, M1, mRS 0 | core 18.0, penumbra 95.0, ratio 5.28, mismatch 77.0, ASPECTS 8 | thrombectomy: yes (6–24 h, ASPECTS ≥ 6, COR 1; also meets DEFUSE-3); thrombolysis: no (eligible for thrombectomy, so §4.6.3 does not apply) | Tempting wrong answer: large mismatch at 9 h looks like extended-window thrombolysis |
| fx03 | Witnessed onset 2 h, NIHSS 12, haemorrhage present (p = 0.96) | core 5.2, penumbra 13.9, occlusion `unknown`, ASPECTS null | thrombolysis: no (absolute contraindication, Table 8); thrombectomy: no | `unknown` occlusion site; warning "CTA not acquired" |
| fx04 | Witnessed onset 6 h, age 71, NIHSS 18, mRS 0, M1 | core 110.0, penumbra 120.0, ratio 1.09, mismatch 10.0, ASPECTS 4 | thrombectomy: yes (ASPECTS 3–5, COR 1 in either window), must state that mass effect is not reported; thrombolysis: no (outside 4.5 h, eligible for thrombectomy, little salvageable tissue) | **Outdated-knowledge trap:** fails DEFUSE-3 (core > 70 mL), so a model relying on pre-2026 rules wrongly refuses |
| fx05 | Wake-up stroke, LKW 10 h, sleep midpoint about 5 h 45 min before imaging, age 82, NIHSS 11, mRS 1, M1 | core 15.0, penumbra 29.5, ratio 1.97, mismatch 14.5, ASPECTS null | thrombectomy: yes on DAWN criteria, borderline because ASPECTS is unknown and DEFUSE-3 is narrowly missed, must cite; thrombolysis: no while thrombectomy is possible; the sleep-midpoint rule applies only if it is not, must cite | DEFUSE-3 near-miss (mismatch 14.5 < 15 mL) vs DAWN pass (age ≥ 80, NIHSS ≥ 10, core < 21 mL); no age limit for ASPECTS ≥ 6; null ASPECTS; warning "motion artefact" |
