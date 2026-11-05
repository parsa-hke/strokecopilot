# Hana: NLP A (reasoner), Phase 3 brief

Read [README.md](README.md) first: access, rules and how work is merged.

## Where the reasoner stands

The reasoner reads three things and returns `recommendation.json` (`schemas/recommendation.json`): thrombolysis and thrombectomy decisions, where every claim cites its source. The three inputs are:
- the vision output (`schemas/vision_output.json`);
- a triage note;
- the whole rulebook: 47 passages in our own words, in `corpus/RULEBOOK.md` and `corpus/passages.jsonl`.

The code is in `src/strokecopilot/nlp/reasoner/` and the prompt is `prompts/reasoner_v1.md`. Phase 2 ran it on five hand-written fixtures:
- EXP-9: 18 of 18 decisions judged right.
- EXP-12: the verifier flagged 28 claims. On Claude's reading, about 9 were real reasoner errors and the rest verifier errors. No human has checked this reading yet.

## Task 1: shortlist the models (start now)

The comparison needs at least three models: one frontier API model, one open mid-size model and one small medical model. Parsa makes the final choice and records it as a decision.

For each candidate, give:
- licence (research use allowed?);
- context length (prompts are about 5–8k tokens in, and up to a few thousand out);
- how reliably it produces valid JSON;
- where it can run: an API, a hosted endpoint, or our rented RTX 4090 (24 GB, so roughly up to 30B parameters at 4-bit);
- an estimated cost for about 300 runs.

Propose 2–3 candidates per class. **Deliverable:** a table on your Notion task.

## Task 2: prompt v2 (start now)

**Inputs (on Drive):**
- `runs/EXP-12_end_to_end_fixtures/flagged_rescore.json`, every flagged claim with its text;
- `outputs/` and `traces/` in the same folder;
- `runs/EXP-9_reasoner_fixtures/`.

Read `docs/HALLUCINATION_METRIC.md` for what counts as an error.

**Patterns to address:**
- numbers not quoted exactly;
- rules applied when their conditions don't hold;
- rules named under uncertainties without a passage ID;
- statements about the answer itself written as claims.

**Deliverables:**
- `prompts/reasoner_v2.md`, with v1 left unchanged;
- tests that the prompt builds and that the fake model's output parses;
- a short note listing each change and the error it targets.

Parsa runs the paid comparison of v1 against v2.

## Task 3: `providers.py` adapters (after the model decision)

Add one adapter per chosen model to `nlp/reasoner/providers.py`, with the same call signature as the Claude one.
- Temperature 0 where the provider allows it.
- Log the full prompt and response to the run folder.
- Return the same usage and cost fields.

Test with fakes, without network calls in tests. Parsa runs the real calls.

## Task 4: real inputs (after Parsa's `contract/build.py`)

Switch the reasoner from fixtures to the real `vision_output.json` files plus the generated notes, which are on Drive in `artifacts/isles24_notes_v1/` (129 development cases). Use development cases only.

## Rules for this role

- Never paste guideline or trial paper text into AI tools. The rulebook is our own words.
- Rulebook changes go through Parsa: the verifier and the rules in code (`nlp/verifier/rules.py`) depend on it.
- No paid calls (see README).
