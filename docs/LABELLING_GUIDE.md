# Labelling guide — verifier adjudication (Phase 3)

This is the one-page guide for the two human labellers (Lorna and Mohamed)
checking the verifier. You are not being tested; the verifier is.
Your job is to give an independent human reading so we can measure how often
the verifier is right, both when it flags a claim and when it lets one pass.

## What you will do

You will receive a CSV and a folder `cases/`. Each row is one atomic claim
from a reasoner's output: about 100 rows, some flagged by the verifier and
most not, sampled across models (`nlp/verifier/adjudicate.py`). The sheet
shows the claim, the field it came from, the sources the model declared, and
a case alias (`C001`, ...). It does not show which model wrote it or whether
the verifier flagged it, so your reading stays independent. You decide, on
your own, whether the claim is an error.

You will see these columns:

- `sample_id`: an anonymous ID (S001, S002, ...). Does not reveal the model.
- `case_alias`: the case, as `cases/<case_alias>.md`.
- `field`: where the claim came from (`rationale`,
  `thrombolysis.blocking_factors`, `uncertainties`, ...).
- `claim`: the claim, as the claim extractor wrote it.
- `declared_sources`: the sources the model declared for it.
- `label_A`, `label_B`, `note_A`, `note_B`: you fill in your label and note.
- `consensus`, `note_consensus`: filled together afterwards (see below).

Fill only **your columns** (`label_A` and `note_A` if you are labeller A,
`label_B` and `note_B` if B). Do not look at the other column. Do not look at
each other's sheets.

## The two possible labels

Every row gets exactly one of these two:

- **`error`** — the reasoner made a mistake. Use this if:
  - The claim's content is false (the number is wrong, the fact is invented,
    the rule is stated wrongly).
  - The claim's content is true but the citation is wrong or missing
    (uncited rule, mis-attributed source).

- **`no_error`** — the claim is fine. Use this if:
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
4. Pick `error` if either is "no". Pick `no_error` if both are "yes".

When you are unsure, pick the label these rules give and write why in your
note column. Whether an error would change a decision is measured separately
(criticality); it does not decide whether the claim is an error.

## Examples (from EXP-12)

| Claim | Verifier said | Correct label | Why |
|---|---|---|---|
| "BP 168/92 does not need lowering before thrombolysis" | wrong_rule | no_error | Rule says lower only if BP ≥ 185/110; 168/92 is below, so no lowering needed. Claim correct. |
| "This patient is eligible for thrombectomy" | misattributed_source | error | Content true, but cited thrombolysis passages, not thrombectomy ones. |
| "Treatment cannot start before imaging" | wrong_rule | error | Rulebook says the opposite ("imaging should not hold it up"). |

## What happens after

Fill in your own label and note columns only, and return the sheet to Benny.
Do not ask for the key file: it maps rows to models and verdicts and stays with
Benny and Parsa until both of you have finished.

**Consensus.** Where your two labels differ, the two of you meet briefly, look
at the claim again with this guide, and write the agreed label in `consensus`
and why in `note_consensus`. Do not change your own columns. Where you agree,
leave `consensus` empty: your shared label is the consensus.

We compute:

- **Cohen's κ** — how often you and your co-labeller agree, beyond chance, on
  every row you both labelled (before the consensus step). A kappa near 1 means
  you agree a lot; near 0 means you agree as often as chance.
- **Agreement with the verifier** — how often the verifier (flagged = error,
  not flagged = no error) matches your consensus: overall, for flagged and for
  unflagged claims, per sub-type and per model. Flagged claims are
  over-sampled, so the overall figure is also re-weighted to all claims. This
  is the number the project reports: it tells us how much to trust the
  verifier, both its flags (precision) and its misses.

## Time estimate

About 3 minutes per row, so 100 rows is roughly 5 hours. Take a break halfway.
**Do not label more than 10 rows in a row without a break** — fatigue causes
drift, and drift shows up as a drop in kappa.