# Clinician review request

**Stroke Decision Copilot**: an MSc Applied AI group project at UOWD (MAAI 901/902). It is a research prototype and is not for clinical use.

The system reads CT perfusion and plain CT, combines the results with a triage note and published guidelines, and suggests whether thrombolysis and thrombectomy are indicated. Every claim in its explanation is then checked against its inputs. The rules and test cases below are what we score it against, so an error here would make our results wrong.

**What we need:** one read, about 30 minutes. Please mark anything wrong or misleading in parts 1 and 2, and answer the questions in part 3. Notes in any format are fine.

## 1. Rules

Our reading of the 2026 AHA/ASA guideline (*Stroke* 2026;57:e316–e436), adults only. COR = class of recommendation; section numbers are the guideline's.

**Thrombolysis**
- Within 4.5 h (§4.6.1): COR 1 for disabling deficits, regardless of NIHSS, without waiting for CTA or perfusion imaging. Mild non-disabling deficits: COR 3 no benefit.
- Extended window (§4.6.3):
  - COR 2a, only if not eligible for thrombectomy: salvageable penumbra on automated perfusion imaging, and either waking with symptoms within 9 h of the midpoint of sleep, or 4.5–9 h from last known well.
  - COR 2a: unknown onset, within 4.5 h of symptom recognition, DWI lesion under one-third of the MCA territory and no marked FLAIR change.
  - COR 2b: LVO, salvageable penumbra, 4.5–24 h, and cannot receive thrombectomy.
- Absolute contraindications (Table 8): acute intracranial haemorrhage on CT, extensive clear hypodensity, moderate to severe head injury or neurosurgery within 14 days, spinal cord injury within 3 months, intra-axial neoplasm, infective endocarditis, severe coagulopathy (platelets < 100,000/mm³, INR > 1.7, aPTT > 40 s, PT > 15 s), aortic arch dissection, ARIA. DOAC within 48 h is treated as relative.
- If eligible for both treatments, give both (§4.7.1, COR 1), without delaying thrombectomy.

**Thrombectomy, ICA/M1, NIHSS ≥ 6 (§4.7.2)**

| Window | ASPECTS | Pre-stroke mRS | Other | COR |
|---|---|---|---|---|
| 0–6 h | 3–10 | 0–1 | | 1 |
| 0–6 h | 6–10 | 2 | | 2a |
| 0–6 h | 6–10 | 3–4 | | 2b |
| 0–6 h | 0–2 | 0–1 | age < 80, no significant mass effect | 2a |
| 6–24 h | 6–10 | 0–1 | | 1 |
| 6–24 h | 3–5 | 0–1 | age < 80, no significant mass effect | 1 |

**Other thrombectomy rules**
- Dominant proximal M2, 0–6 h, mRS 0–1, NIHSS ≥ 6, ASPECTS ≥ 6: COR 2a. Non-dominant or codominant proximal M2, distal MCA, ACA, PCA: COR 3 no benefit.
- Basilar, within 24 h (§4.7.3): mRS 0–1, NIHSS ≥ 10, PC-ASPECTS ≥ 6: COR 1. NIHSS 6–9: COR 2b.

**Our own choices, which the guideline does not state directly**
- *Salvageable penumbra:* the recommendations give no number, so we use the trial criteria in Table 3. For 4.5–9 h, EXTEND: ratio > 1.2, penumbra − core > 10 mL, core < 70 mL. For 4.5–24 h, TRACE-III: ratio > 1.8, mismatch > 15 mL, core < 70 mL.
- *When ASPECTS is unknown:* our imaging pipeline does not produce ASPECTS, so we fall back on trial criteria. DEFUSE-3 (6–16 h): core < 70 mL, ratio ≥ 1.8, mismatch ≥ 15 mL. DAWN (6–24 h): age ≥ 80, NIHSS ≥ 10 and core < 21 mL; age < 80, NIHSS ≥ 10 and core < 31 mL, or NIHSS ≥ 20 and core 31–51 mL. We have not yet checked these numbers against the trial papers.
- *What the system cannot see:* significant mass effect, M2 dominance and extensive hypodensity. It must name these as uncertainties and never assume them.

## 2. Test cases and expected answers

Volumes are in mL. Penumbra is the total Tmax > 6 s volume, core included. Mismatch is penumbra minus core. All cases are invented.

| Case | Scenario | Imaging | Expected answer |
|---|---|---|---|
| 1 | Witnessed onset 2.5 h, NIHSS 8, M2 | core 0.6, penumbra 22.4, ratio not computed (core < 1 mL), mismatch 21.8, ASPECTS 9 | Thrombolysis: yes. Thrombectomy: depends on M2 dominance, which is not reported: reasonable if dominant proximal (COR 2a), otherwise not recommended (COR 3) |
| 2 | Last known well 9 h, NIHSS 16, M1, mRS 0 | core 18.0, penumbra 95.0, ratio 5.28, mismatch 77.0, ASPECTS 8 | Thrombectomy: yes (6–24 h, COR 1). Thrombolysis: no, because eligibility for thrombectomy rules out the extended window |
| 3 | Witnessed onset 2 h, NIHSS 12, haemorrhage on CT (p = 0.96), CTA not acquired | core 5.2, penumbra 13.9, occlusion unknown, ASPECTS unknown | Thrombolysis: no. Thrombectomy: no |
| 4 | Witnessed onset 6 h, age 71, NIHSS 18, mRS 0, M1 | core 110.0, penumbra 120.0, ratio 1.09, mismatch 10.0, ASPECTS 4 | Thrombectomy: yes (ASPECTS 3–5, COR 1), stating that mass effect is not reported. Thrombolysis: no |
| 5 | Wake-up stroke, last known well 10 h, sleep midpoint about 5 h 45 min before imaging, age 82, NIHSS 11, mRS 1, M1 | core 15.0, penumbra 29.5, ratio 1.97, mismatch 14.5, ASPECTS unknown, motion artefact | Thrombectomy: yes on DAWN, flagged as borderline because ASPECTS is unknown and DEFUSE-3 is narrowly missed. Thrombolysis: no while thrombectomy is possible |

## 3. Questions

1. Is anything in part 1 wrong, or misleading in practice?
2. Do you agree with each expected answer in part 2? Case 5 is the one we are least sure of.
3. Are EXTEND and TRACE-III a reasonable way to put numbers on "salvageable penumbra"?
   For TRACE-III, the guideline's Table 3 gives "penumbra volume > 15 mL", where penumbra is the total Tmax > 6 s volume. Should our rule read that as total hypoperfused volume > 15 mL, or as hypoperfused volume minus core > 15 mL?
4. When ASPECTS is unknown, is falling back on DAWN and DEFUSE-3 reasonable?
5. Does final infarct on follow-up MRI, in patients who all reperfused, seem an acceptable stand-in for core at presentation, given that we report this as a limitation?
6. Is there a common clinical trap that these five cases miss?
7. The thrombectomy recommendations count time from symptom onset. Is it right to use last known well as the onset time when onset was not witnessed?
8. Our rules in code score the synthetic test cases with these conventions. Please mark any you would change:
   - COR 1 and 2a count as "yes", COR 2b as "undetermined", COR 3 or no applicable recommendation as "no".
   - NIHSS 6 or more counts as a disabling deficit. Under 6 within 4.5 h is left undetermined.
   - Table 8 absolute contraindications give "no". Relative ones, including a DOAC within 48 h, give "undetermined".
   - Times are taken at imaging, since no treatment time is given. "Within 6 h" includes exactly 6 h.
   - With ASPECTS unknown at 6–24 h, a "yes" needs DAWN or DEFUSE-3 criteria. If neither is met we call it undetermined, not "no", because ASPECTS could still qualify.
   - Extended-window thrombolysis needs EXTEND criteria at 4.5–9 h (or within 9 h of the midpoint of sleep), and only when thrombectomy is not an option. TRACE-III criteria at 4.5–24 h give "undetermined" (COR 2b).
