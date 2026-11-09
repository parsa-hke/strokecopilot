# Labelling guide — verifier adjudication (Phase 3)

This is the one-page guide for the two human labellers (Lorna and Mohamed)
checking the verifier's flags. You are not being tested; the verifier is.
Your job is to give an independent human reading so we can measure how often
the verifier is right.

## What you will do

You will receive a CSV. Each row is one claim the verifier flagged as a
hallucination in a reasoner's output. You decide, on your own, whether the
verifier was right or wrong.

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
  - The claim is a general statement about medical reasoning that doesn't
    need a specific rulebook passage.

## How to decide each row

You have three files to check against:

1. `corpus/RULEBOOK.md` — every guideline passage the reasoner can cite.
2. `fixtures/<fx>/` — the vision output for that case.
3. `fixtures/notes/<fx>` — the triage note for that case.

For each row:

1. Read the claim text (in the `claim_id` and any context the sheet gives).
2. Ask: **is the content of this claim true** given the vision output, the
   note, and the rulebook?
3. Ask: **does the source the model cited actually support it**?
4. Pick `real_reasoner_error` if either is "no". Pick `verifier_mistake` if
   both are "yes".

When you are unsure, pick `real_reasoner_error` if the error is material
(would change a decision or a clinical fact). Pick `verifier_mistake` if the
claim is technically fine but the verifier was pedantic.

## Examples (from EXP-12)

| Claim | Verifier said | Correct label | Why |
|---|---|---|---|
| "BP 168/92 does not need lowering before thrombolysis" | wrong_rule | verifier_mistake | Rule says lower only if BP ≥ 185/110; 168/92 is below, so no lowering needed. Claim correct. |
| "This patient is eligible for thrombectomy" | misattributed_source | real_reasoner_error | Content true, but cited thrombolysis passages, not thrombectomy ones. |
| "Treatment cannot start before imaging" | wrong_rule | real_reasoner_error | Rulebook says the opposite ("imaging should not hold it up"). |

## What happens after

Your labels go into `EXP-12_benny_labels.csv` (or its successor). We compute:

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