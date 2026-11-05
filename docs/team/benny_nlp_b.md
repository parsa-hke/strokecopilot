# Benny: NLP B (verifier and evaluation), Phase 3 brief

Read [README.md](README.md) first: access, rules and how work is merged.

## Where the verifier stands

The verifier splits the reasoner's rationale into small claims and traces each one:
- **numbers** against the vision output;
- **patient facts** against the triage note;
- **rule claims** against the cited passage, using an LLM judge.

Whatever it can't trace is a hallucination, and the definition is in `docs/HALLUCINATION_METRIC.md`. Code is in `src/strokecopilot/nlp/verifier/`.

Results so far:
- **EXP-11:** planted errors are caught with recall of at least 0.8 for every type, and a false-flag rate of 0.010.
- **EXP-12:** on real reasoner output, 28 of 351 claims were flagged (rate 0.080).

## Task 1: error taxonomy with examples (start now)

The metric has eight error types: wrong number, invented finding, wrong rule, misapplied rule, faulty derivation, uncited rule, mis-attributed source, and omission.

**Inputs:**
- `docs/HALLUCINATION_METRIC.md`;
- `fixtures/seeded_errors.json` (planted examples);
- on Drive, `runs/EXP-12_end_to_end_fixtures/flagged_rescore.json` (the 28 flagged claims, with text) and `traces/`.

**For each type, write:**
- a plain-English definition;
- one or two real examples from our outputs;
- how the verifier detects it;
- when it counts as critical (it would change a treatment decision).

Then go through the 28 EXP-12 flags yourself and mark each one as a real reasoner error or a verifier mistake. Do this independently: an earlier reading exists, but don't look at it until you've finished yours.

**Deliverable:** `docs/ERROR_TAXONOMY.md`, plus your 28 labels as a CSV on Drive in the EXP-12 folder.

## Task 2: the comparison notebook (start now)

`notebooks/41_nlpB_model_comparison.ipynb` will hold the project's main table. The rows are models. The columns are the source (fixtures, synthetic scenarios, real cases), each with and without the rulebook. The values are:
- hallucination rate;
- critical hallucination rate;
- omission rate;
- each with a bootstrap confidence interval (`nlp/verifier/metrics.py` already has `summarise` and `paired_difference`).

Build it now against the EXP-12 outputs as a stand-in for one model, so it's ready when the real runs land.
- Put the logic in `src/strokecopilot/pipeline/compare.py`, with tests.
- The notebook only reads saved runs. No API calls.

**Done when** it produces the table and a figure from EXP-12 alone, and adding a second run folder adds a row.

## Task 3: the labelling kit (when the model runs exist)

Fill in `nlp/verifier/adjudicate.py`:
- sample about 100 claims across models, stratified by model and claim type;
- strip anything that reveals the model;
- export a sheet for two labellers (Lorna and Mohamed);
- compute agreement between them (Cohen's kappa) and between each of them and the verifier.

Write the one-page labelling guide too. You won't be a labeller, so the kit stays unbiased.

## Not yours

Changes to the verifier's logic or thresholds (they define the metric, so Parsa owns them), and the paid runs.
