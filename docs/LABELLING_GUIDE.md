# Labelling guide — verifier adjudication (Phase 3)

This is the one-page guide for the two human labellers (Lorna and Mohamed)
checking the verifier's flags. You are not being tested; the verifier is.
Your job is to give an independent human reading so we can measure how often
the verifier is right.

## What you will do

You will receive a CSV and a folder `cases/`. Each row is one claim the
verifier flagged as a hallucination in a reasoner's output. The sheet shows
the claim, the field it came from, the sources the model declared, and a case
alias (`C001`, ...). It does not show which model wrote it or what the verifier
said, so your reading stays independent. You decide, on your own, whether the
claim is a real error.

You will see these columns:

- `sample_id`: an anonymous ID (S001, S002, ...). Does not reveal which model.
- `claim_id`: which fixture and which claim (e.g. `fx02:rationale[3].0`).
- `subtype`: the verifier's label — what type of error it thinks this is.
- `reason`: the verifier's one-line justification.
- `label_A` and `label_B`: the empty cells. You fill in one of them.

Fill only **your column** (`label_A` if you are labeller A, `label_B` if B).
Do not look at the other column. Do not look at each other's sheets.

## The two possible labels

Every row gets exactly one of these two:

- **`real_reasoner_error`** — the reasoner made a mistake. The verifier was
  correct to flag it. Use this if:
  - The claim's content is false (the number is wrong, the fact is invented,
    the rule is stated wrongly).
  - The claim's content is true but the citation is wrong or missing
    (uncited rule, mis-attributed source).

- **`verifier_mistake`** — the claim is fine. The verifier was wrong to flag
  it. Use this if:
  - The claim is true and correctly cited.
  - The claim only restates the output's own decision ("thrombolysis is not
    recommended", "the rulebook does not support thrombectomy"). That is not a
    claim: the decision is scored separately.

Correct medicine is not enough. The metric scores claims against the
reasoner's inputs, not against clinical truth: a true clinical rule that no
rulebook passage states is still an error.

## How to decide each row

You have two things to check against:

1. `corpus/RULEBOOK.md` — every guideline passage the reasoner can cite.
2. `cases/<case_alias>.md` — that case's imaging findings and triage note,
   exactly as the reasoner received them.

Rules the metric uses (`docs/HALLUCINATION_METRIC.md`):

- **null means unknown.** A claim that a null finding is unknown or not
  reported is fine; any value for it is an error.
- **Numbers** are compared after rounding the true value to the precision the
  claim states: "9 h" matches 9 h 0 min, "4 h" does not match 4 h 40 min.
- **Uncertainties** have no sources in the output's format: judge only whether
  their content is true, never their citation.
- **Blocking factors** may rest on the rules cited for either decision.

For each row:

1. Read the claim (`claim`) and the sources the model declared
   (`declared_sources`).
2. Ask: **is the content of this claim true** given the vision output, the
   note, and the rulebook?
3. Ask: **does the source the model cited actually support it**?
4. Pick `real_reasoner_error` if either is "no". Pick `verifier_mistake` if
   both are "yes".

When you are unsure, pick the label these rules give and write why in your
note column. Whether an error would change a decision is measured separately
(criticality); it does not decide whether the claim is an error.

## Examples (from EXP-12)

| Claim | Verifier said | Correct label | Why |
|---|---|---|---|
| "BP 168/92 does not need lowering before thrombolysis" | wrong_rule | verifier_mistake | Rule says lower only if BP ≥ 185/110; 168/92 is below, so no lowering needed. Claim correct. |
| "This patient is eligible for thrombectomy" | misattributed_source | real_reasoner_error | Content true, but cited thrombolysis passages, not thrombectomy ones. |
| "Treatment cannot start before imaging" | wrong_rule | real_reasoner_error | Rulebook says the opposite ("imaging should not hold it up"). |

## What happens after

Fill in your own label and note columns only, and return the sheet to Benny.
Do not ask for the key file: it maps rows to models and stays with Benny and
Parsa until both of you have finished. We compute:

- **Cohen's κ** — how often you and your co-labeller agree. A kappa near 1
  means you agree a lot; near 0 means you agree as often as chance.
- **Agreement with the verifier** — how often you and the verifier agree. This
  is the number the project reports: it tells us how much to trust the verifier.

If your two labels agree 90% of the time and agree with the verifier 75% of
the time, the verifier has ~75% precision on flagged claims. That's the number.

## Time estimate

About 3 minutes per row, so 100 rows is roughly 5 hours. Take a break halfway.
**Do not label more than 10 rows in a row without a break** — fatigue causes
drift, and drift shows up as a drop in kappa.