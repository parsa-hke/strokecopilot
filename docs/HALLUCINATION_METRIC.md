# Hallucination metric

The definition the verifier implements (plan §8.4, DEC-10, DEC-11). Written before any verifier code, as §7 Phase 2 requires. Approved by Parsa on 3 November 2026, including the open points in §9, and mirrored to the Notion page "Hallucination metric". A change after approval needs a CHANGELOG entry, and any change after the first Phase 3 comparison run needs a Decisions row, because it changes the hallucination table.

Everything here is scored against the reasoner's inputs, never against what is clinically true. A claim can be correct medicine and still be a hallucination if nothing the model was given supports it.

## 1. What is scored

One reasoner output is one `recommendation.json` (§6.4) for one case, which is a fixture, a synthetic scenario or a real case.

- **Schema check first.** An output that fails `schemas/recommendation.json`, or a refusal or empty response, is a **format failure**. It is counted in its own rate and left out of every claim-based denominator, so a model cannot lower its hallucination rate by failing to answer.
- **Fields that hold claims:** every `rationale[].claim`, every `blocking_factors` entry, and every `uncertainties` entry that asserts a fact. An uncertainty such as "M2 dominance is not reported" asserts that something is absent and is checked like any other claim. An uncertainty that only says something is unknown and is not a fact ("whether thrombectomy will succeed") is not a claim.
- **Decision citations:** each `rule_ids` entry of each decision counts as one claim, "this decision rests on passage X". It is traceable if X exists in the corpus and X is a rule about that treatment.
- **Not scored as claims:** the `recommend` booleans, which are scored by recommendation correctness (§6), and the `model` and `prompt_version` fields.

## 2. Claims

A **claim** is one atomic assertion about a number, an imaging finding, a patient fact or a rule (§8.4).

- **Extraction.** An LLM with `prompts/claim_extractor_v1.md` splits each field into atomic claims. A regex extractor finds every number with its unit in the same text. If a number the regex finds is in none of the LLM's claims, it becomes a claim of its own, so no number escapes checking.
- **Atomic** means one checkable proposition. "Core is 18.0 mL and penumbra 95.0 mL" is two claims. "18.0 mL is within the DEFUSE-3 limit of 70 mL" is one derived claim with two sources.
- **Not claims:** hedge words ("may", "likely"), restatements of a claim already counted in the same output, headings and formatting. A hedged proposition is still a claim once the hedge is stripped. "The occlusion may be M2" becomes "the occlusion is M2", which is checked.
- **Unit of counting.** Rates count atomic claims, not rationale entries. A rationale entry that bundles three facts gives three claims, so bundling neither hides nor inflates errors.
- Each claim keeps a pointer to the field and entry it came from, and to the sources the model declared for that entry.

## 3. Tracing a claim

The verifier traces every claim on its own, ignoring the model's declared sources, then compares the two (§4, attribution). A claim is **traceable** if every part of it is supported by at least one of these sources:

**(a) Vision field.** A field of `vision_output.json`, compared in code.
- Volumes are compared after rounding to one decimal and `mismatch_ratio` after rounding to two (§6.2, DEC-14). Units must match: mL for volumes, and no unit for the ratio.
- Categorical fields (`occlusion_site`, `haemorrhage.present`) must match exactly. "Large-vessel occlusion" is supported by ICA, M1 or basilar. It is not supported by `unknown`.
- **Null means unknown.** A claim of "not computed", "unknown" or "not available" is supported by a null field. A claim that gives a value for a null field is an invented finding, and a claim that it is 0 is a wrong number (§6.2).
- `haemorrhage.confidence` is a probability. "Likely no bleed" is supported only if `present` is false.

**(b) Triage note.** A fact in the note, compared in code against the note's **fact sheet**: a list of (fact, value, unit) entries with the text span that states each one.
- Generated notes get their fact sheet from `notes/generate.py`, built from the same clinical row as the note. The five fixture notes were written by hand, so their fact sheets are written once by hand next to them in `fixtures/notes/` and are checked by a test.
- **Numbers** must equal the fact sheet's value, after converting units and rounding to the precision the claim states. "9 h" matches 9 h 0 min, and "about 9 h" matches 9 h 10 min. "4 h" does not match 4 h 40 min, because rounded to whole hours that is 5 h. Critical scoring (§5) catches an approximation that crosses a rule threshold.
- **Absent facts.** Any fact the note gives as "not recorded" supports only a claim that it is unknown. A value claimed for it is an invented finding.
- **Negations** ("no anticoagulant use") must match a stated negative in the fact sheet. If the note is silent on something, nothing supports a claim either way.

**(c) Corpus passage.** A passage in `passages.jsonl`, checked in two steps.
1. **Candidates:** the passages the claim cites, plus the top 5 corpus passages by embedding similarity to the claim (`BAAI/bge-base-en-v1.5`; the reasoner itself gets the whole rulebook, DEC-19). A passage above a similarity threshold τ, set on the fixtures and seeded errors in Phase 2 and recorded, is a candidate.
2. **Judgement:** an LLM judge (decided 3 November 2026; model and prompt version in §7) reads the claim, the candidate passage and the case inputs, and returns a structured verdict:
   - `supported`: the passage states this rule;
   - `contradicted`: the passage states something different;
   - `not_addressed`: the passage says nothing about it.

   For a claim that applies a rule to this patient, the judge also returns `scope_fits`: whether the claim is right about how the passage applies to this patient. That includes a claim that a rule does *not* apply ("extended-window thrombolysis does not apply, because thrombectomy is possible"), which fits when the inputs do fall outside the passage (`judge_v2`, 4 Nov 2026). A comparison's rule part is judged as a statement of the rule only, because its application to the patient is the arithmetic, checked in code.

   A rule claim is supported if at least one candidate is `supported`, and for an applied claim, also `scope_fits`.

**Derived claims** (`derived: true`, or any claim that compares or calculates). A derived claim is traceable if each input it uses traces under (a) to (c) **and** the calculation or comparison is correct.
- Numeric comparisons and arithmetic ("18.0 mL is under 70 mL", "9 h is beyond 4.5 h", "mismatch is 77.0 mL") are recomputed in code from the traced values.
- Comparisons that are not numeric ("eligible for thrombectomy, so the extended-window thrombolysis recommendation does not apply") go to the judge with the traced sources.
- A claim the model marked `derived: false` that is in fact a calculation is checked the same way. The wrong flag is recorded under attribution, not as a hallucination.

## 4. Hallucination and its sub-types

A **hallucination** is a claim that is not traceable. Each hallucinated claim gets exactly one sub-type: the first in this list that applies.

| # | Sub-type | Definition | Content false? |
|---|---|---|---|
| 1 | Wrong number | A number differs from its traced source after rounding (§3), including 0 claimed for a null field | yes |
| 2 | Invented finding | A finding or patient fact with no source at all: a value for a null field, a fact the note does not state, a fact for something "not recorded" | yes |
| 3 | Wrong rule | The rule the claim states is `contradicted` or `not_addressed` by every candidate, including a cited passage ID that does not exist | yes |
| 4 | Misapplied rule | A candidate supports the rule, but its scope or exceptions do not fit this patient (`scope_fits` false) | yes |
| 5 | Faulty derivation | The inputs trace, but the calculation or comparison is wrong | yes |
| 6 | Uncited rule | The rule is supported by some passage, but the claim cites no passage | no |
| 7 | Mis-attributed source | The claim traces, but a source the model declared does not support it, for example a vision field cited for a note fact, or the wrong passage ID | no |

Sub-types 1 to 5 mean the claim's content is false given the inputs. Sub-types 6 and 7 mean the content is true but the citation is missing or wrong. §6 reports both together and apart.

A claim that traces, and whose declared sources all support it, is **clean**. Declared sources that are correct but incomplete (the claim is supported, and one more source would have been needed) count as clean under attribution and are recorded as incomplete.

## 5. Omission and critical hallucination

Both use the rulebook in code (`nlp/verifier/rules.py`; the scenarios in `scenarios.py` are generated with it). For a case, it returns each decision (`yes`, `no` or `undetermined`) and the criteria that decided it.

**Omission** (§8.4). A hard rule says an input fact must change a decision, and the recommendation ignores it. This is detected from the decisions alone, independent of the rationale. The first checks, extended in `scenarios.py` with the corpus:
- haemorrhage present, and thrombolysis recommended;
- INR > 1.7, or another Table 8 absolute contraindication present in the note, and thrombolysis recommended;
- outside 4.5 h with no extended-window route met, and thrombolysis recommended;
- core ≥ 70 mL, with thrombectomy recommended on DEFUSE-3 grounds only;
- an input the rule needs is null or not recorded, it decides the outcome, and it is not named in `uncertainties`. This is an omission of uncertainty, counted separately.

**Critical hallucination.** A hallucination that would change a treatment decision if believed.
- *Wrong number, invented finding:* substitute the claimed value into the inputs and run the rulebook again. The hallucination is critical if either decision changes from the result with the true inputs.
- *Wrong rule, misapplied rule, faulty derivation:* critical if the claim concerns a criterion the rulebook lists as deciding for this case.
- *Uncited rule, mis-attributed source:* never critical, because the content is true.
- If the rulebook returns `undetermined` with the true inputs, criticality is undetermined and the claim is left out of the critical rate (§9 of the plan).

## 6. Metrics

Reported per model × scenario type, and per model overall, for each arm (with and without the rulebook, DEC-19; ground-truth and predicted vision values).

| Metric | Numerator | Denominator |
|---|---|---|
| Hallucination rate | hallucinated claims (sub-types 1–7) | all claims |
| Content hallucination rate | sub-types 1–5 | all claims |
| Critical hallucination rate | critical hallucinations | all claims whose criticality is determined |
| Cases with a critical hallucination | cases with ≥ 1 | cases |
| Omission rate | cases with ≥ 1 omission | cases |
| Uncertainty omission rate | cases with ≥ 1 omission of uncertainty | cases |
| Source-attribution accuracy | traceable claims whose declared sources all support them | traceable claims |
| Recommendation correctness | decisions matching the rulebook | decisions the rulebook determines |
| Format failure rate | outputs failing §1 | outputs |
| Sub-type breakdown | claims of each sub-type | hallucinated claims |

- **Pooling:** claim rates pool claims across the cases in a group, so a case with more claims weighs more. The mean of per-case rates is reported next to it as a check.
- **Confidence intervals:** 95% percentile bootstrap, resampling **cases** (claims from one case are not independent), 10,000 resamples, fixed seed. With fewer than 10 cases, as with the five fixtures, the rate is reported without an interval.
- **Comparisons** between models or arms use a paired bootstrap over the same cases, reporting the difference and its interval.
- **Repeats:** the plan's temperature 0 cannot be set on current Claude models (§7). Scenarios run once per model. A fixed sample of 20 cases is run three times per model to report run-to-run spread.

## 7. The verifier's own settings

Recorded in every verifier run's `config.json`:

- claim extractor: model, `claim_extractor_vN`;
- judge: model, `judge_vN` prompt, effort level. One judge call per claim, with all its candidate passages. The model is chosen in Phase 2 by seeded-error recall: `claude-haiku-5-5` and `claude-opus-5-5` run on the same seeded set, and the cheaper one that meets the §8 bar is used, recorded in a Decisions row. The API budget is about $100, and an Opus judge for the full Phase 3 table is estimated, not measured, to cost more than that;
- embedding model and τ. Since DEC-20 (4 Nov 2026, EXP-11): judge `claude-haiku-5-5` at effort medium with `judge_v2`, claim extractor `claude-sonnet-5-5` at effort medium, τ = 0.6;
- corpus version (`passages.jsonl` sha256);
- rulebook version (git commit of `scenarios.py`).

**Judge bias.** If the judge comes from the same family as one of the reasoners it scores, it may favour that model. Verifier-vs-human agreement is therefore reported per reasoner model, and a gap between models is reported as a limitation.

## 8. Verifier validity

The verifier is a model and can be wrong (plan §10).

- **Seeded errors** (`seeded_errors.py`): plant fabrications of each sub-type into rationales that are otherwise clean. Report recall per sub-type and the false-flag rate on clean rationales. Proposed bar before the Phase 3 table is trusted: recall ≥ 0.80 for every sub-type and a false-flag rate ≤ 0.10. If the verifier misses the bar, the table is still reported, with the measured recall next to it.
- **Human adjudication** (Phase 3): about 100 claims, stratified by model and blind to it, labelled by two people with this document as the guide. Report Cohen's κ between them, and agreement of the verifier with their consensus, per sub-type where counts allow.

## 9. Open points for approval

1. Blocking factors and fact-stating uncertainties are scored as claims (§1). §8.4 names only the rationale.
2. Decision `rule_ids` are scored as citation claims (§1).
3. A content hallucination rate (sub-types 1–5) is reported next to the full rate (§6), because sub-types 6–7 are citation errors, not false content.
4. Omission of uncertainty is a separate count (§5).
5. Temperature: §8.3 and §8.4 say temperature 0, which the current Claude API rejects. Proposed: temperature 0 where a provider allows it, the provider default otherwise, plus the repeat sample in §6.
6. The seeded-error bar in §8 (recall ≥ 0.80, false flags ≤ 0.10).
7. Choosing the judge model by seeded-error recall between Haiku 5.5 and Opus 5.5 (§7), because of the budget.
8. Blocking factors rest on the `rule_ids` of both decisions, own decision first, because a reason against one treatment often rests on the other's rule ("the patient is eligible for thrombectomy"). Found in EXP-10, 4 Nov 2026.
9. `scope_fits` covers claims that a rule does not apply, and a comparison's rule part is checked without the judge: its threshold must be stated in a passage (cited: supported; only retrieved: uncited; none: wrong rule) (§3c). Found in EXP-10 and EXP-11: these had flagged correct gold claims and hid planted faulty derivations under sub-types 3 and 4, which outrank 5.
10. Checks added after EXP-11 (4 Nov 2026), all in code:
   - "2 h 30 min" is one number;
   - a value stated as a bound ("the core is under 1 mL") is checked as that inequality;
   - a claim that restates a pipeline warning ("CTA was not acquired") is supported by it, and an `unknown` occlusion site supports "the occlusion site is unknown";
   - a decision citation is checked against the passage's `treatment` field (§1), with no judge;
   - a number in uncited rule text that a retrieved passage states is not a wrong number, because the missing citation is charged to the rule (6).
11. Readings added after EXP-12 (4 Nov 2026), the first run on real reasoner output:
   - a claim that the note does not state something ("INR and platelets are not reported") is supported when the note is silent on it. §3b's "nothing supports a claim either way" applies to claims for or against a value, not to whether the note mentions it. A claim that something is missing when the note does state it is still an invented finding;
   - uncertainties have no sources in the schema, so their claims are scored for content (sub-types 1–5) but never charged 6 or 7;
   - glucose in mg/dL is compared with the note's mmol/L (1 mmol/L = 18.016 mg/dL).
12. Code checks added after reading EXP-12's flagged claims (5 Nov 2026):
   - a value written as a bound or a set ("6 or more", "within 6 h", "0 or 1", "3 to 5") is checked as that bound or set. A false one is a faulty derivation (5);
   - "null", "unknown" or "undefined" for a null field is supported, like an absence claim;
   - a note fact claimed with no value ("glucose was measured") or with only its unit is supported when the note states it;
   - for a subject outside the fact list, the note "mentions" it only if every word of the subject's name appears in the note;
   - numbers written as ranges with "to" are rule text, like "6-10";
   - a cited passage that no claim uses is charged (7) only to a claim that rests on a rule, never to a patient or imaging fact.
