# Error taxonomy

The eight error types the verifier flags, with plain-English definitions,
real examples from our outputs, how the verifier detects each, and when it
counts as critical. Companion to `docs/HALLUCINATION_METRIC.md`.

**Author:** Benny (NLP B). **Phase:** 3. **Last updated:** 7 Nov 2026.

Examples are drawn from the 28 claims the verifier flagged in EXP-12
(`runs/EXP-12_end_to_end_fixtures/flagged_rescore.json`), labelled
independently in `EXP-12_benny_labels.csv`, and from the planted seeds in
`fixtures/seeded_errors.json` where EXP-12 contains no real example.

---

## 1. Wrong number

**Definition.** The model states a number that does not match any of its
inputs, compared after rounding to the precision the claim states and after
unit conversion.

**Example.** Planted seed `fx01-1a`: the clean output says "2 h 30 min" from
onset to imaging; the seed changes it to "3 h 30 min." Neither the note nor
the vision output supports 3 h 30 min. (EXP-12's 28 flags hold no wrong
number. The regex number check does not filter wrong numbers out: a number it
finds that no input supports becomes a wrong-number flag.)

**How the verifier detects it.** Compares the claimed number in code against
the traced source — a vision field, the note's fact sheet, or the corrected
value from a passage — after unit conversion and rounding to the claim's
stated precision (§3 of the metric).

**Critical?** Yes, if believing the wrong number would change a treatment
decision. Determined by substituting the claimed value into the inputs and
re-running the rulebook in code (§5).

---

## 2. Invented finding

**Definition.** The model states a patient fact or imaging finding that none
of its inputs provided — a value for a null field, a fact the note does not
state, or a fact for something "not recorded."

**Example.** `fx03:rationale[1].0`: the claim says "Presentation included
sudden severe headache and vomiting before the weakness." The only cited
source is a note fragment about headache followed by weakness; the note does
not state vomiting before weakness. The vomiting detail is invented.

**How the verifier detects it.** Checks all three source types — vision
field, note fact sheet, corpus passage. No source supports the claimed value
or the claimed absence (§3).

**Critical?** Yes, if believing the invented finding would change a decision.
Determined by substitution and re-running the rulebook.

---

## 3. Wrong rule

**Definition.** The rule the model states is contradicted by every candidate
passage, or no candidate passage addresses it, or the cited passage ID does
not exist in the corpus.

**Example.** `fx02:blocking_factors[0].2`: the claim says "Treatment cannot
start before imaging." The relevant passage (`AHA2026-4.6.1-within-4.5h`)
says the opposite — that extra imaging "should not hold up" thrombolysis.
The rule is reversed.

**How the verifier detects it.** Retrieves candidate passages by embedding
similarity (`BAAI/bge-base-en-v1.5`), then an LLM judge (`claude-haiku-5-5`
with `judge_v3` since DEC-23, 12 Nov 2026) returns `supported`, `contradicted` or `not_addressed`. A
claim is a wrong rule if every candidate is `contradicted` or
`not_addressed`, or if a cited passage ID does not exist (§3c). Candidates
are the cited passages, the retriever's top 5 above τ = 0.6, the passages the
claim names (a trial, Table 3 or 8, an ASPECTS range) and, for an uncertainty,
the passages its output's decisions cite (§9 point 14).

**Critical?** Yes, if the wrong rule concerns a criterion that decided the
case's treatment. Determined by checking whether the claim touches a
deciding input or passage.

---

## 4. Misapplied rule

**Definition.** A passage states the rule correctly, but the rule's scope or
exceptions do not fit this patient.

**Example.** `fx03:rationale[7].4`: the claim says "Blood pressure would need
lowering before any thrombolysis." The BP rule
(`AHA2026-4.3-bp-before-ivt`) applies only to a patient "otherwise eligible
for IV thrombolysis." fx03 has a haemorrhage and is not eligible, so the rule
does not apply — the threshold is correct, the patient is wrong.

**How the verifier detects it.** The judge returns `supported: true` but
`scope_fits: false` for this patient. A claim that a rule does not apply is
also checked: it fits when the inputs do fall outside the passage (`judge_v3`).

**Critical?** Yes, if the misapplied rule bears on a deciding criterion.

---

## 5. Faulty derivation

**Definition.** The inputs trace correctly, but the calculation or
comparison over them is wrong.

**Example.** Planted seed `fx01-5a`: the time from onset to imaging is 2 h
30 min, which is *within* the 4.5 h window; the seed changes the comparison
to "is beyond the 4.5 h." Both numbers trace; the inequality is reversed.
(EXP-12 contains no faulty-derivation flags in the 28.)

**How the verifier detects it.** Recomputes numeric comparisons and
arithmetic in code from the traced values (§3, derived claims). For a
comparison's rule part, checks that each threshold it takes from a rule is
stated in a passage — cited: supported; only retrieved: uncited; none: wrong
rule.

**Critical?** Yes, if the claim concerns a criterion the rulebook lists as
deciding for this case (§5). The verifier does not recompute the decision for
a derivation; that substitution is only for wrong numbers and invented findings.

---

## 6. Uncited rule

**Definition.** The rule is true and supported by a passage in the corpus,
but the model cites no passage for it.

**Example.** `fx05:blocking_factors[1].0`: the claim says "Mismatch volume
14.5 mL is below the 15 mL DEFUSE-3 threshold." The content is true — 14.5
is below the threshold — but the DEFUSE-3 passage that states the threshold
(`DEFUSE3-inclusion-imaging`) is not cited. The model cites eight sources,
none of them the right one.

Another example in EXP-12: `fx05:thrombectomy.blocking_factors[1].1`
("DEFUSE-3 imaging criteria are not met"). Uncertainties are never charged
uncited (6) or mis-attributed (7): the schema gives them no sources, so a rule
named in an uncertainty is checked for content only (§9 point 11). The EXP-12
uncertainty flags were wrong rules, not uncited rules.

**How the verifier detects it.** Finds a passage that supports the rule, but
notices the model cited no passage for it (§4).

**Critical?** Never. The content is true, so a doctor reading the claim would
not be misled.

---

## 7. Mis-attributed source

**Definition.** The claim is true, but a source the model declared does not
support it.

**Example.** `fx04:rationale[3].0`: the claim says "Core 110.0 mL is not
under 70 mL." Most cited sources do support it — `core_ml` gives 110,
`AHA2026-T3-extend` and `AHA2026-T3-trace3` give the 70 mL threshold. But
the model also cited `AHA2026-4.6.3-lvo-4.5-24h`, an extended-window drug
passage that says nothing about core volumes. One declared source supports
nothing, so the claim is mis-attributed under the strict metric.

**How the verifier detects it.** The claim traces to at least one source,
but at least one declared source does not support it (§4). Declared sources
that are correct but incomplete count as clean.

**Critical?** Never. The content is true.

---

## 8. Omission

**Definition.** An input fact that should have changed the recommendation was
ignored. Detected from the decisions, not from the claims in the rationale —
a fully clean rationale can still commit an omission.

**Example.** The check set in `nlp/verifier/rules.py`: haemorrhage present
with thrombolysis recommended; INR above 1.7 with thrombolysis recommended;
outside 4.5 h with no extended route met and thrombolysis recommended; core
≥ 70 mL with thrombectomy recommended on DEFUSE-3 grounds only; no occlusion
with thrombectomy recommended. (EXP-12 shows zero omissions: all five
fixture decisions were correct.)

**How the verifier detects it.** Rule checks run against the recommendation
booleans, independent of the claims in the rationale (§5).

**Critical?** Not applicable. An omission is a wrong decision, not a claim:
it is counted in its own rate (omission rate, §6) and never enters the
critical hallucination rate, which counts claims.

---

## Criticality: summary

A hallucination is **critical** if believing it would change a treatment
decision.

| Type | Can it be critical? | How criticality is judged |
|---|---|---|
| 1. Wrong number | Yes | Substitute claimed value; re-run the rulebook |
| 2. Invented finding | Yes | Substitute claimed value; re-run the rulebook |
| 3. Wrong rule | Yes | Check whether it touches a deciding criterion |
| 4. Misapplied rule | Yes | Check whether it touches a deciding criterion |
| 5. Faulty derivation | Yes | Check whether it touches a deciding criterion |
| 6. Uncited rule | **Never** | Content is true |
| 7. Mis-attributed source | **Never** | Content is true |
| 8. Omission | Not a claim | Counted in the omission rate, not the critical rate |

---

## Notes on the boundary cases

Two patterns recur in EXP-12 and are worth recording for future reviewers:

- **Negative applications of a rule** ("BP is below 185/110, so no lowering is
  needed") are correctly traceable to the passage that states the rule in
  positive form. The verifier flagged these as wrong rule in EXP-12; judge
  prompt `judge_v3` (DEC-23, §9 point 15) now says how to judge them.
- **Uncertainty claims** carry no `sources` field in the schema, so a citation
  cannot be required there. This is settled (§9 point 11): their claims are
  scored for content only, never as uncited or mis-attributed, and since
  12 Nov 2026 their rule claims are judged against the passages their output's
  decisions cite (§9 point 14). A restatement of the output's own decision is
  not a claim (§9 point 13).