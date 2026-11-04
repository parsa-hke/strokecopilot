# CHANGELOG

## 2026-11-04 — Claude Code session (three verifier readings from EXP-12; free e2e rescore)
- **Three changes, approved by Parsa and recorded in `docs/HALLUCINATION_METRIC.md` §9 point 11:**
  - **Silent note:** a claim that the note does not state something is supported when the note is silent. If the note does state it, the claim is still an invented finding. For subjects outside the fact list, "silent" means the note never mentions the subject's words.
  - **Uncertainties:** their claims are scored for content (sub-types 1–5) but never charged attribution (6, 7), because the schema gives them no sources.
  - **Glucose:** mg/dL is now convertible to mmol/L; the regex extractor reads mg/dL.
- **`pipeline/end_to_end.py`:**
  - `rescore()` re-scores a finished fixture run with the current code. It uses the saved outputs and model claims, rebuilds the regex and citation claims, and takes judge verdicts from the cache only, with no API calls.
  - Both a run and a rescore now write a compact list of flagged claims (`flagged.json`, `flagged_rescore.json`) with text, kind, subject, value, reason and sources.
- **`50_end_to_end.ipynb`** has a `rescore` mode: free, no key, and it loads bge for retrievals not in the cache. Tested locally after a faked run: no model calls.
- **Tests:** silent-note absence (and a note that does mention the subject), uncertainties free of attribution charges, glucose units, and the e2e rescore.
- pytest: 339 passed.
- **Notion sync:**
  - pending: the EXP-12 row after the rescore.

## 2026-11-04 — Claude Code session (EXP-12 end to end: acceptance run)
- **EXP-12 ran on Colab** (commit `dc79189`): five fixtures through reasoner → verifier, about $0.62.
  - **Result:** 371 claims, hallucination rate 0.235 (content 0.148), attribution accuracy 0.899, decisions right 1.000, format failures 0.
  - **Sub-types:** wrong number 9, invented finding 27, wrong rule 15, misapplied 4, uncited 25, misattributed 7.
  - **The Phase 2 acceptance criterion is met mechanically:** the notebook runs the chain and prints a rate. §7 task "Run verifier on the five fixture outputs" ticked.
- **The rate is not a result yet.** Reading the flag reasons, most flags look like verifier issues on real output. The gold outputs were written in the verifier's style, so they never exposed these, and their 0.010 false-flag rate does not carry over to real outputs.
  1. **"X is not reported" against a silent note:** about 20 of the 27 invented findings. When the note is silent, the claim is true. The code followed "silence supports nothing either way", which the doc wrote for claims about a value, not about whether the note mentions it.
  2. **Rules mentioned in uncertainties:** most of the 25 uncited rules. The schema gives uncertainties no sources, so citing there is impossible.
  3. **Glucose units:** given in mg/dL to compare with the rulebook's 60 mg/dL; the verifier cannot convert from mmol/L.
  - The rest (wrong rules with no candidate passage, some wrong numbers) need the claim text to judge.
- **Notion sync:**
  - done: Experiments → EXP-12 row (Done, marked not yet adjudicated).

## 2026-11-04 — Claude Code session (end to end, fixture mode)
- **`pipeline/end_to_end.py`**, `run_fixtures()`: runs fixture → reasoner (`reasoner_v1`, whole rulebook) → recommendation → verifier (Sonnet extractor, code checks, bge retrieval, Haiku judge, τ = 0.6) → `metrics.summarise()`.
  - A reasoner format failure is counted and not verified.
  - It saves outputs, traces and calls, with caches so a stopped run resumes, and writes `config.json` and `metrics.json` with per-fixture flagged claims, usage and cost.
  - Criticality and omissions stay None until `scenarios.py` exists.
- **`50_end_to_end.ipynb`** (Colab, CPU, EXP-12): the Phase 2 acceptance notebook. It prints each flagged claim with its sub-type, then the hallucination, content and attribution rates, decision correctness and format failures.
  - **Estimate, from EXP-8 to EXP-11 rates:** about $0.40 and 10 minutes.
  - **Tested locally** with Colab, git, pip and the API faked (gold outputs as the reasoner), and real bge retrieval: it runs, refuses to overwrite a finished run, and keeps no key. The fake extractor's rates mean nothing.
- `tests/test_end_to_end.py`: every stage with fake models, and a reasoner format failure.
- pytest: 336 passed.
- **Notion sync:**
  - pending: Experiments row for EXP-12 after the run.

## 2026-11-04 — Claude Code session (EXP-11 Opus pass)
- **The Opus pass ran** (commit `3c07e9a`): 60 judge calls on the same extracted claims, about $1.30 ($0.022 per call). Every other verdict came from the caches.
- **Results with the final code, τ ≤ 0.7:**

  | | Haiku 5.5 | Opus 5.5 |
  |---|---|---|
  | Lowest strict recall | 0.80 (uncited rule) | 0.80 (uncited rule) |
  | Wrong rule | 0.9 | 0.9 |
  | Other five sub-types | 1.0 | 1.0 |
  | False flags on gold | 0.010 | 0.019 |

  - Opus's extra false flag: fx05's "extended-window thrombolysis does not apply" judged misapplied.
  - Both judges miss the same three seeds, fx04-3a, fx03-6b and fx04-6b, so the misses come from the verifier (candidate selection and retrieval), not the judge model.
- **DEC-20 (Parsa):** the judge is `claude-haiku-5-5` at effort medium with `judge_v2`, and τ = 0.6, following the protocol in `docs/HALLUCINATION_METRIC.md` §7 (the cheaper judge that meets the bar). Recall is flat from −1 to 0.7 and falls at 0.75, so 0.6 leaves a margin.
  - Recorded in plan §2, the metric doc §7, and code: `trace.JUDGE_MODEL`, `JUDGE_EFFORT` and `TAU`, the default τ of `Inputs`, and `claims.EXTRACTOR_MODEL` and `EXTRACTOR_EFFORT`.
  - §7 `trace.py` and `seeded_errors.py` tasks ticked.
- **Notion sync:**
  - done: Experiments → EXP-11 row (Done); Decisions → DEC-20 row.

## 2026-11-04 — Claude Code session (EXP-11 second rescore)
- **Second rescore** (commit `3c07e9a`, no API calls):
  - **Strict recall, Haiku judge, τ ≤ 0.7:** 1.0 for wrong number, invented finding, misapplied rule, faulty derivation and misattributed source; 0.9 for wrong rule; 0.8 for uncited rule.
  - **False flags:** 0.010, one gold claim of about 100: fx01's "the core is under 1 mL" in an uncertainty.
  - **Above τ 0.7:** uncited-rule recall falls, to 0.3 at τ 0.9.
- **Still missed:**
  - fx04-3a ("at any age"): the Haiku judge accepted a loosened rule.
  - fx03-6b and fx04-6b: both state the 0–6 h ICA/M1 rule with no passage and came out as wrong rule (3) rather than uncited (6). The likely cause, not checked, is EXP-7's weakness: the right ICA/M1 row may fall outside the top 5 retrieved.
- §7 `claims.py` task ticked: the real extractor ran on 75 outputs with no failures.
- **Notion sync:**
  - pending: the EXP-11 row and a Decisions row for the judge model and τ.

## 2026-11-04 — Claude Code session (EXP-11 rescore; four more verifier fixes)
- **EXP-11 rescore** (commit `984a884`, no API calls): the Haiku pass re-scored with comparison thresholds checked in code.
  - **Bar met at τ ≤ 0.7:** recall 1.0 for sub-types 1, 2, 4 and 7, 0.9 for 3 and 5, 0.8 for 6. False flags 0.095 (10 of 105 gold claims).
  - **Above τ 0.7**, uncited-rule recall falls, to 0.3 at τ 0.9.
  - Faulty-derivation recall rose from 0.1 to 0.9.
- **The remaining flags pointed to four generic bugs, now fixed in code:**
  - "2 h 30 min" was read as two numbers, behind fx01's four wrong-number flags and the missed fx01-5a.
  - A value stated as a bound ("the core is under 1 mL") was compared as an equality.
  - Restating a warning ("CTA was not acquired") and an `unknown` occlusion site were not accepted as absence. That accounts for fx03's three invented-finding flags.
  - Haiku rejected two valid `rule_ids` (DAWN and the late-window passage). Citations are now checked against the passage's `treatment` field, as `docs/HALLUCINATION_METRIC.md` §1 defines them.
  - Numbers inside an uncited rule were wrong numbers unless a cited passage held them. A retrieved passage now counts, and the citation gap goes to the rule (6).
  - All are recorded in `docs/HALLUCINATION_METRIC.md` §9 point 10 for Parsa's OK, with a test each.
- **Still missed:** fx04-3a ("at any age", a loosened rule), which the Haiku judge let pass.
- **Rescore changes:**
  - It now rebuilds the regex and citation claims with the current code, keeping the model's saved claims.
  - It loads the embedding model for retrievals the cache does not hold. It needs no API key, costs nothing, and may be repeated.
- pytest: 334 passed.
- **Notion sync:**
  - pending: the EXP-11 row after the second rescore and the Opus decision; a Decisions row for the judge model and τ.

## 2026-11-04 — Claude Code session (EXP-6 Vision B; EXP-11 Haiku pass; comparison thresholds in code)
- **EXP-6 finished on Kaggle** (ResNet-50, 10 epochs, commit `d7e274f`).
  - **Split:** the validation and test scan bleed rates are 0.4058 and 0.4119, against 0.4085 overall.
  - **Speed:** 156 slices per second, about 18.4 minutes per epoch.
  - **Validation (2,166 scans):** slice AUC 0.975 and scan AUC 0.977. ECE is 0.010. At the threshold set for 95% sensitivity, sensitivity is 0.951 and specificity 0.917. The threshold was set on this same split, so specificity is optimistic.
  - The test split is untouched.
  - The three §7 Vision B tasks are ticked.
  - **Files on Drive** (`runs/EXP-6_visionB_resnet50/`, DEC-17): `best.pt` (epoch 8), `last.pt`, `calibration.json`, `config.json`, `metrics.json`, `reliability.png`. They hold no IDs, only counts, metrics and the split fingerprint.
  - **Aggregation**, by validation scan AUC: the top-2 mean (0.9772) was chosen, against max 0.9771, top-3 0.9767 and top-5 0.9749.
  - **Calibration**, by out-of-half log loss and ECE:

    | Method | Log loss | ECE |
    |---|---|---|
    | None | 0.290 | 0.123 |
    | Temperature | 0.286 | 0.131 |
    | Platt (chosen) | 0.175 | 0.014 |

    The chosen temperature is 0.99 and the bias −2.07. Balanced epochs train on half bleeds, so raw scores overstate bleeds; the bias corrects that, and temperature scaling alone cannot.
  - **Training:** 3.6 hours. Validation slice AUC was flat from epoch 3 onwards (0.9746 to 0.9752).
  - **Operating point:** a threshold of 0.278 calls 43.5% of scans positive.
- **EXP-11, Haiku pass** (commit `d3666a3`): 75 outputs. Extraction cost about $1.06 (60 calls, 472 s), the judge about $0.10 (113 calls, 702 s).
  - **Strict recall at τ ≤ 0.7:**

    | Sub-type | Recall | Missed |
    |---|---|---|
    | Wrong number | 1.0 | |
    | Invented finding | 1.0 | |
    | Wrong rule | 0.9 | fx04-3a, "at any age" |
    | Misapplied rule | 1.0 | |
    | Faulty derivation | 0.1 (lenient 1.0) | |
    | Uncited rule | 0.8 | |
    | Misattributed source | 1.0 | |

  - **False flags:** 10 of 105 gold claims (0.095), 3 of 5 gold outputs.
  - **Higher τ:** uncited-rule recall falls above τ 0.7, to 0.3 at τ 0.9, so τ must stay low.
- **Diagnosis from the fx02-5a trace:** the comparison's arithmetic was rightly found false (5), but its rule part still went to the judge as the whole claim text. No passage states "9 h is within the 4.5 h window", so the judge returned wrong rule (3), which outranks 5.
- **Fix:** a comparison's rule part is now a code check that each threshold it takes from a rule is stated in a passage:
  - in a cited passage: supported;
  - only in a retrieved one: supported but uncited (6);
  - in none: wrong rule (3).
  - No judge call is made for comparisons.
  - **Known limit:** a threshold attributed to the wrong trial, whose number appears in the cited passage, is not caught.
  - New tests cover a wrong threshold (3), a reversed operator (5) and a threshold found only through retrieval (6).
- **Diagnosis output:** `score()` now lists the flagged gold claims and the sub-types a missed seed got instead. The notebook prints both.
- **`MODE='rescore'`:** re-scores a run from its saved claims and caches with the current code, with no API calls and no key. Tested locally with no new calls.
- pytest: 325 passed.
- **Notion sync:**
  - done: Experiments → EXP-6 (Done, scan AUC 0.977).
  - pending: EXP-11 row after the rescore and the Opus pass.

## 2026-11-04 — Claude Code session (stale Colab session guard)
- **The first EXP-11 attempt stopped before any API call:** `run_seeded() got an unexpected keyword argument 'metrics_name'`. It ran in the same Colab session as EXP-10, so the old package was still imported, and pip treats the same version (0.0.1) as already installed.
- **Fix in `31_nlpA_reasoner` and `40_nlpB_verifier`:**
  - The install cell now stops with a plain message if the session has already imported `strokecopilot`.
  - After the normal install, it force-reinstalls the pinned commit without dependencies.
  - Tested locally: the guard fires in a reused session; dummy, haiku, opus and resume all pass after a restart.
- **Workaround until this is pushed:** Runtime → Disconnect and delete runtime, then Run all.
- **Notion sync:** none.

## 2026-11-04 — Claude Code session (EXP-10 verifier dummy; two verifier fixes)
- **EXP-10 ran on Colab** (commit `3456287`): fx02's gold output and its 14 seeds, with a `claude-sonnet-5-5` extractor and a `claude-haiku-5-5` judge.
  - **Schemas and caches:** both schemas were accepted, there were no failures, and the caches worked. Extraction made 12 calls (the 3 source-only seeds reused the gold extraction) and the judge made 36.
  - **Cost and speed:** about $0.23 for extraction (9 s per call) and $0.03 for the judge (7.5 s per call).
  - **Recall at every τ:** strict recall was 1.0 for sub-types 1, 2, 3, 4, 6 and 7 (2 seeds each). For faulty derivation (5) it was 0 strict and 1.0 lenient. 4 of the 22 gold claims were flagged.
- **Diagnosis, from the traces:** the flags came from the verifier, not the gold output.
  1. Comparisons went to the judge as rules applied to the patient. "9 h exceeds the 4.5 h window" got `scope_fits` false: the rule's scope does not fit, which is the claim's own point. So it was marked misapplied (4), and because 4 outranks 5, the planted faulty derivations were reported as 4.
  2. A claim that a rule does not apply ("extended-window thrombolysis does not apply to patients eligible for thrombectomy") also got `scope_fits` false.
  3. "The patient is eligible for thrombectomy", a thrombolysis blocking factor, inherited only the thrombolysis `rule_ids`, so it was marked uncited (6).
- **Fixes:**
  - A comparison's rule part is now judged as a statement only, since the code checks the arithmetic.
  - `prompts/judge_v2.md` defines `scope_fits` as whether the claim is right about how the rule applies, including that it does not apply. `judge_v1` is kept for the record.
  - The judge cache key now includes the prompt version.
  - Blocking factors now rest on both decisions' `rule_ids`.
  - All three are recorded in `docs/HALLUCINATION_METRIC.md` (§3c, §9 points 8 and 9) for Parsa's OK. New tests cover each.
- **`40_nlpB_verifier.ipynb` is now staged:**
  - `MODE='haiku'` (EXP-11): all 75 outputs with the Haiku judge, about 35 minutes and $1.50 (estimated from EXP-10).
  - `MODE='opus'`: same folder, adding Opus. It reuses the extraction and Haiku verdicts and pays only for Opus, about $5–10 (unmeasured).
  - Each pass writes its own `config_<mode>.json` and `metrics_<mode>.json`. A local test with fakes confirmed the Opus pass makes no extraction or Haiku calls.
- pytest: 322 passed.
- **Notion sync:**
  - done: Experiments → EXP-10 row.

## 2026-11-04 — Claude Code session (EXP-9 reasoner on the fixtures)
- **EXP-9 ran on Colab** (`claude-sonnet-5-5`, effort high, commit `592fceb`): five fixtures, twice each.
  - **Result:** 0 format failures, and all 18 judged decisions match the expected ones. fx01's thrombectomy accepts either answer; it was no in run 1.
  - **Cost and speed:** every call read the cached system prompt (105,590 tokens read, none written). Each call used about 433 uncached input and 3,391 output tokens and took 20 s. The run cost about $0.37.
  - **Claude read run 1 of fx01, fx04 and fx05;** the other seven outputs have not been read.
    - fx04 recommends thrombectomy on ASPECTS 3–5 despite the 110 mL core, so it passes the pre-2026 trap, and it names mass effect as unknown.
    - fx05 rests thrombectomy on the DAWN group for age 80 and over, and flags the DEFUSE-3 near-miss.
    - **A likely real hallucination:** fx05 lists the ASPECTS 6–10 recommendation among its thrombectomy `rule_ids` while saying ASPECTS is null, so the rule is misapplied. The verifier should catch it.
  - Entries bundle up to six sources.
- §7 `reasoner/run.py` task ticked.
- **Notion sync:**
  - done: Experiments → EXP-9 row (Done).

## 2026-11-04 — Claude Code session (NLP B: verifier run and judge bake-off notebook)
- **`nlp/verifier/run.py`:**
  - `verify()` runs extraction then tracing.
  - `run_seeded()` traces the 5 gold outputs and the 70 seeds once per judge model, on the same extracted claims.
  - **Caches** are saved to the run folder after every new entry, so a stopped run resumes without paying twice:
    - extraction: per distinct text, so seeds that change only sources or a rule ID reuse the gold extraction;
    - judge verdicts: per claim and passage, so an unchanged claim in 14 seeded copies is judged once;
    - retrieval: per claim text.
  - **τ:** the run judges every top-5 retrieved passage (τ = −1). `replay()` then scores τ from −1 to 0.9 from the caches alone, with no API calls.
- **`40_nlpB_verifier.ipynb`** (Colab, CPU):
  - `MODE='dummy'` (EXP-10): fx02's gold output and its 14 seeds, with a Haiku judge.
  - `MODE='run'` (EXP-11): all 75 outputs, with `claude-haiku-5-5` then `claude-opus-5-5` as judge. The extractor is `claude-sonnet-5-5` at effort medium; that is my choice, held fixed so only the judge varies.
  - It prints strict recall per sub-type and the false-flag rate for each τ, and the τ with the best lowest recall among those with false flags ≤ 0.10.
  - **Estimates, unmeasured:** dummy about 10 minutes and $0.50; run about 2 hours and $7–20.
- **Notebook test:** run locally with Colab, git, pip and the API faked, and real bge retrieval.
  - In dummy and run modes, it made 12 and 60 extractor calls and 25 and 119 unique judge calls per judge model.
  - A resume after a stop made no new calls, and no key or token was printed or kept.
  - The fake extractor's recall numbers mean nothing.
- **Not measured yet:** real recall, false flags, τ and cost.
- pytest: 319 passed.
- **Notion sync:**
  - pending: Experiments rows for EXP-10 and EXP-11 after the runs; a Decisions row for the judge model and τ after EXP-11.

## 2026-11-04 — Claude Code session (NLP B: seeded errors)
- **Gold outputs** (`fixtures/gold/fx0N_*.json`): one clean recommendation per fixture, written by hand in the verifier's style, with one fact or one comparison per entry and every source cited.
  - A test checks each in code: schema; every note quote verbatim; every passage ID in the corpus; every number matches the vision JSON, the fact sheet or a cited passage; decisions match the expected ones.
  - The test cannot check whether a rule claim matches its passage. Only the judge, or a person, can.
- **Seeds** (`fixtures/seeded_errors.json`): 70 hand-written seeds, 2 per sub-type per fixture. Each is one edit to a gold output, labelled with its sub-type and the entry it touches. By sub-type:
  - 1: a changed vision or note number.
  - 2: a fact no input gives (INR, platelets, a warning, ASPECTS for a null field).
  - 3: a reversed or loosened rule, or a cited passage ID that does not exist.
  - 4: a rule applied outside its scope.
  - 5: a reversed comparison.
  - 6: a true rule stated with only a non-passage source.
  - 7: a true claim cited to the wrong field or quote.
  - A test checks that each seed changes exactly the labelled entry and leaves the decisions alone.
- **`nlp/verifier/seeded_errors.py`:**
  - `apply()` and `seeded_set()` build the seeded outputs.
  - `score()` reports recall per sub-type, both strict (that sub-type in the labelled entry) and lenient (any sub-type there), with missed seed IDs.
  - It also reports the false-flag rate on the gold outputs and flags outside the labelled entry, and checks them against the bar in `docs/HALLUCINATION_METRIC.md` §8 (recall ≥ 0.80 for every sub-type, false flags ≤ 0.10).
- **Not measured yet:** it needs the real extractor and judge. That is the verifier notebook, next.
- pytest: 313 passed.
- **Notion sync:** none.

## 2026-11-04 — Claude Code session (EXP-8 reasoner dummy)
- **EXP-8 ran on Colab** (`claude-sonnet-5-5`, effort high, commit `592fceb`): fx02 twice.
  - **Result:** 0 format failures, and both runs got the decisions right (thrombolysis no, thrombectomy yes). The API accepted the output schema.
  - **Cost and speed:** the system prompt is 10,559 tokens, against an estimate of about 6,000; the first call wrote it to the cache and the second read it. Each call used about 426 uncached input and 3,144 output tokens and took 20 s. The two calls cost about $0.09 together, about half the estimate.
  - **Read through by Claude:**
    - Every number matches the fixture, every note quote checked is verbatim, and every passage ID exists.
    - Entries bundle up to four facts, despite the prompt asking for one or two.
    - One run's uncertainty says the rulebook counts time from last known well; it is the prompt that says so.
    - One run lists the blood-pressure passage among the thrombectomy `rule_ids`.
    - These are what the verifier is for; no prompt change yet.
- **EXP-9 estimate:** 10 calls at this rate is about $0.37 and 4 minutes (estimated from EXP-8).
- **Notion sync:**
  - done: Experiments → EXP-8 row (Done).

## 2026-11-04 — Claude Code session (NLP A: reasoner v1)
- **`prompts/reasoner_v1.md`:**
  - Explains each vision field: null means unknown, and `penumbra_ml` includes the core.
  - Treats the note as the only source of patient facts, and the rulebook as the only source of rules, with no outside knowledge.
  - Decides thrombolysis and thrombectomy separately. Counts time windows from onset, using last known well when onset was not witnessed.
  - Names the unobservable inputs as uncertainties.
  - Asks for one or two statements per rationale entry with every source, verbatim note quotes, numbers exactly as given, and the `derived` flag.
  - The whole rulebook fills `{{RULEBOOK}}` (DEC-19).
- **`nlp/reasoner/prompt.py`:**
  - The system prompt holds the instructions and the rulebook, and is the same for every case, so it is cached.
  - The user message holds the vision JSON (without its bookkeeping fields) and the note.
  - `api_schema()` is `recommendation.json` without `case_id`, `model` and `prompt_version`, with `oneOf` turned into `anyOf` and with the length and pattern limits the API rejects removed. The full schema is still checked on the assembled output.
  - Passage IDs are not limited to the corpus, so a model can still invent one. That stays a measurable hallucination (sub-type 3).
- **`nlp/reasoner/run.py`:**
  - `recommend()` adds the run fields and validates. A refusal, a reply that is not JSON, or a schema failure is recorded as a failure and never repaired.
  - `run_fixtures()` writes `config.json` (model, prompt and corpus hashes, passages still draft, commit), `metrics.json` (failures, decisions against `fixtures/expected_rules.json`, usage, estimated cost), `system_prompt.md`, and one file per call holding the user message, raw reply, error and usage.
- **`providers.py`:** `PRICES` and `cost_usd()`.
- **`fixtures/expected_rules.json`** gains the expected decisions. fx01's thrombectomy accepts either answer, because it depends on M2 dominance, which is not reported.
- New extra: `llm` (`anthropic`).
- **`31_nlpA_reasoner.ipynb`** (Colab, CPU):
  - `MODE='dummy'` (EXP-8) runs fx02 twice to check that the API accepts the schema, the reply validates, and the second call reads the cache.
  - `MODE='run'` (EXP-9) runs the five fixtures twice each.
  - The default model is `claude-sonnet-5-5` at effort high. Cost is estimated, unmeasured, at about $0.10 per call.
  - The notebook was tested locally with Colab, git, pip and the API faked, in dummy mode, run mode and with one bad reply. No key or token is printed or kept in a variable, and it refuses to overwrite a finished run.
- **Not yet run against the real API:** schema acceptance, output quality, tokens and cost are all unmeasured.
- pytest: 302 passed.
- **Notion sync:**
  - pending: Experiments rows for EXP-8 and EXP-9, after the runs.

## 2026-11-04 — Claude Code session (corpus: retrieval test, DEC-19)
- **EXP-7, the retrieval check, ran locally** with `BAAI/bge-base-en-v1.5` and sentence-transformers 6.2.0 on the Mac CPU, in about 50 s including the model download.
  - **Setup:** two decision queries, each the vision JSON in words plus the note, top 8 each.
  - **Result:** 9 of the 15 rules the fixtures rest on (`fixtures/expected_rules.json`) were found. 5 were never returned, including the main ICA/M1 thrombectomy rows for fx02 and fx04.
  - **Variants:** the best of 12 query and passage formats found 11; four focused queries found 11; whole-section expansion found 13, but only by returning 26–35 of the 47 passages.
  - **Why:** bge ranks the right family of rules first but cannot order rows that differ only in their numbers.
  - Files are in `runs/EXP-7_retrieval_check/` (not committed).
- **DEC-19 (Parsa):** the reasoner gets the whole rulebook, about 4,600 tokens, in the cached part of its prompt (`ingest.prompt_block()`).
  - Retrieval stays only for the verifier's candidate passages.
  - Experiment arm 1 becomes with vs without the rulebook.
  - Updated: plan §1, §2, §3, §5.3, §7, §8.3, §8.4, §9 and §12; `docs/HALLUCINATION_METRIC.md`; README.
- **`nlp/corpus/index.py` now serves only the verifier:** `Embedder` (loaded on first use, bge query instruction) and `Index.search(text, k)`, which fits `trace.Inputs.retriever`. The per-case queries and the must-have logic were removed. The `must_have` flag stays in the rulebook but has no role while the whole rulebook is sent.
- New extra: `retrieval` (`sentence-transformers`), installed in `.venv`.
- §7 `index.py` ticked.
- pytest: 294 passed.
- **Notion sync:**
  - done: Experiments → EXP-7 row (Done); Decisions → DEC-19 row, which also records the embedding model.
  - pending: Decisions rows for the LLM judge, RSNA split and temperature wording.

## 2026-11-04 — Claude Code session (corpus: AHA passages checked)
- **AHA passages checked against the guideline** at Parsa's request. Claude read the text Drive extracted from `aha_asa_2026_ais.pdf`; the text was kept outside the repo, and no wording was copied.
  - All 43 AHA passages are now `checked`, with page, recommendation number, COR and LOE.
  - Corrections:
    - The thrombectomy windows count from symptom onset.
    - Tenecteplase 0.4 mg/kg is not recommended, and the alteplase dose is now stated.
    - "May be considered" or "not well established" wording now matches each COR 2b.
    - ASPECTS 0–2 and 3–5 apply to "selected" patients.
- **18 new passages** (now 47 in all), so retrieval has to choose:
  - more of §4.6.1: early ischaemic change, glucose, antiplatelets, not waiting for labs;
  - §4.6.5: sickle cell disease, retinal artery occlusion;
  - §4.3: blood pressure before and after each treatment;
  - §4.5: glucose;
  - §4.8: aspirin and antiplatelets after IVT;
  - the definition of a dominant M2.
- **Copy check against the guideline text:** no passage shares 12 or more consecutive words. Table 8 has no text layer, so the check cannot cover its 8 passages. They were reworded by hand while reading the page images. The longest runs are 8–10 words: the dose, vessel and glucose-range phrases.
- **Points checked against plan §8.3** (TRACE-III and onset time are recorded there and added to `docs/CLINICIAN_REVIEW.md` as questions 3 and 7):
  - Table 3 gives TRACE-III and TIMELESS as "penumbra volume > 15 mL", not "mismatch > 15 mL".
  - Table 8 is an image, so Parsa downloaded the PDF and Claude read its four pages. Its last group is titled absolute contraindications, so the plan's list stands. The six passages now match the table; for example, the trauma criteria are now defined, and the ARIA entry now includes amyloid immunotherapy. Two passages were added, one for the relative contraindications and one for the conditions where benefit outweighs the risk.
  - The thrombectomy windows count from symptom onset. We use last known well when onset was not witnessed.
- Still `draft`: 2 DAWN and 2 DEFUSE-3 (to be checked by a person; NEJM content stays out of AI tools).
- pytest: 290 passed.
- **Notion sync:**
  - pending: Tasks → `ingest.py` in progress (DAWN and DEFUSE-3 checks left).

## 2026-11-04 — Claude Code session (corpus: own-words rulebook, DEC-18)
- **Licence check** (`docs/DATA_USE.md` → Guideline documents):
  - AHA allows no reproduction without written permission.
  - NEJM reserves text and data mining and AI use for DAWN and DEFUSE-3.
  - Only the DEFUSE-3 PMC author manuscript allows text mining.
- **DEC-18 (Parsa):** the corpus is a rulebook written in our own words.
  - One passage per criterion. Numbers are kept as facts and no wording is copied.
  - Each passage carries a pointer to its source and a status: `draft`, `checked` or `reviewed`.
  - The PDFs are never sent to an LLM.
  - Plan §2, §3, §4.2, §5.3, §8.3 and §10 updated.
  - Open question added for the professors.
- **`nlp/corpus/ingest.py`:**
  - Parses `corpus/RULEBOOK.md` into `corpus/passages.jsonl`. Both are committed and ship in the wheel (`config.CORPUS_DIR`).
  - Fails on bad or duplicate IDs, missing or unknown fields, passages over 120 words, and `checked` passages with no page.
  - `load()` gives the verifier's `{id: record}`. `version()` is the corpus sha256.
  - The copy check reads the PDFs locally with pypdf (optional `corpus` extra). It reports passages that share 12 or more consecutive words with their source, printing only IDs and counts.
- **`corpus/RULEBOOK.md`:** 30 draft passages drafted from plan §8.3 (our text, checked against the guideline on 30 Oct). No PDF was opened.
  - 14 cover thrombolysis, 11 thrombectomy, 4 DAWN or DEFUSE-3 criteria and 1 bridging.
  - 6 are must-have.
  - None has a page, and recommendation numbers are not filled in yet.
  - The DAWN core range for the third group is written as "31 mL up to under 51 mL". The plan says 31–51; check against the paper.
  - The DEFUSE-3 clinical criteria (age 18–90, pre-stroke mRS 0–2, NIHSS ≥ 6) are not in plan §8.3. They come from general knowledge and need checking against the PMC manuscript.
- **Still to do:**
  - Check every passage against the PDFs and fill in pages and pointers.
  - Add off-topic guideline passages (blood pressure, glucose, imaging) so retrieval has to choose.
  - Run the copy check.
- pytest: 290 passed.
- **Notion sync:**
  - done: Decisions → DEC-18 row.
  - pending: Tasks → `ingest.py` in progress; Decisions rows for the LLM judge, embedding model, RSNA split and temperature wording.

## 2026-11-04 — Claude Code session (NLP B: claims, tracing, metrics)
- **Claim extraction** (`nlp/verifier/claims.py`, `prompts/claim_extractor_v1.md`):
  - The LLM splits rationale entries, blocking factors and fact-stating uncertainties into atomic claims. It labels each claim's kind and subject from fixed vocabularies of vision fields, note facts and inputs the pipeline cannot observe (plan §8.3).
  - It is never shown the inputs' values, and it returns numbers as written, so stated precision is kept.
  - A regex adds a claim for any number with a unit, or after a score name, that the LLM missed. It skips passage IDs, section numbers and ranges.
  - Each `rule_ids` entry becomes a citation claim.
- **Tracing** (`nlp/verifier/trace.py`, `prompts/judge_v1.md`):
  - Vision and note facts are checked in code. The true value is converted to the claim's unit and rounded to the claim's stated precision. Null means unknown, and silent notes support nothing.
  - Comparisons are recomputed from the true values.
  - Rules go to the LLM judge with cited and retrieved passages, returning a verdict plus scope.
  - Sub-types are assigned in the §4 order.
- **Attribution is settled per rationale entry**, because the claims of one entry share its source list. A part left uncovered while the entry declares a source that backs nothing is mis-attributed (7). Otherwise an uncovered rule is uncited (6), and an uncovered vision or note part is incomplete but clean.
- **Two interpretations of `docs/HALLUCINATION_METRIC.md` (for Parsa's OK):**
  - Blocking factors have no `sources` in the schema, so they inherit their decision's `rule_ids`.
  - A claim that a known value is unknown ("ASPECTS is not available" when it is 8) counts as an invented finding (2).
- **Metrics** (`nlp/verifier/metrics.py`): every §6 rate pooled over claims, the mean per-case rate, the sub-type breakdown, 95% case-bootstrap intervals from 10 cases up, and paired differences. Criticality and omissions stay None until `scenarios.py` exists. §7 task ticked.
- **Fact sheets** for the five fixture notes (`fixtures/notes/*.facts.json`, written by hand), with a format in `nlp/notes/facts.py` that `notes/generate.py` will reuse. Every span is checked verbatim against its note.
- **Provider:** `nlp/reasoner/providers.py` has `ClaudeJSON` (structured JSON output, cached system prompt, effort set, no temperature). Refusals raise rather than fall back to another model, so the judge model stays fixed. The key comes from Colab or Kaggle Secrets, or the environment.
- `prompts/` now ships in the wheel; `config.PROMPTS_DIR` added.
- **Tests:** `test_claims.py`, `test_trace.py`, `test_verifier_metrics.py`, `test_providers.py`. They cover a clean fx02 output that traces fully, one seeded error per sub-type caught as that sub-type, the doc's rounding examples, and the extractor never seeing the note. All use a fake extractor, a fake judge and a synthetic test corpus.
- **Not yet run with a real model:** extractor quality, judge verdicts, whether the API accepts both JSON schemas, cost, and τ (needs the corpus) are all untested. The `claims.py` and `trace.py` tasks stay open until then.
- pytest: 274 passed.
- **Notion sync:**
  - pending: Tasks → `metrics.py` done; Decisions rows for the LLM judge, embedding model, RSNA split and temperature wording.

## 2026-11-04 — Claude Code session (EXP-5 Vision B dummy; EXP-6 sized)
- **EXP-5 ran on Kaggle** (T4, one GPU used, 4 CPUs, commit `0e8740a`):
  - **Split:** the scan bleed rate is 0.4058 in validation and 0.4119 in test, against 0.4085 overall, well inside the 1.5-point check.
  - **Speed:** the data loader gives 196, 724 and 895 slices per second with 1, 2 and 4 workers. Training on random images runs at 171 per second for ResNet-50 and 124 for EfficientNet-B3. Real training ran at 158 per second, so the GPU is the limit, not data loading.
  - **Epoch:** a balanced epoch is 172,618 slices, about 18.2 minutes of training.
  - The AUCs (slice 0.805, scan 0.730, on a 300-scan sample after 150 steps) only show the pipeline runs; they are not results.
- **EXP-6 set up in `20_visionB_train.ipynb`, which now opens in train mode:** ResNet-50, 10 epochs, about 3.5 hours (estimate: 18.2 minutes per epoch plus validation, which is unmeasured at full size). ResNet-50 is 38% faster than EfficientNet-B3 on this GPU.
- **Fix before any real run:** when resuming from an earlier version's attached output, the notebook copied only `last.pt`. If an earlier epoch had been the best, `best.pt` would have been missing and scoring would fail. It now copies both. The resume path was tested on synthetic shards: one version stops after one epoch, its output is attached and the next version resumes.
- pytest: 239 passed.
- **Notion sync:**
  - done: EXP-5 → Done, with speeds; EXP-6 row (Planned).
  - pending: Decisions rows for the LLM judge, embedding model, RSNA split and temperature wording.

## 2026-11-04 — Claude Code session (Phase 2: RSNA split and Vision B code)
- **RSNA split** (`data/rsna_split.py`): seed 20261103, patients ranked by a SHA-256 hash of seed and ID and cut 80/10/10, as agreed. The cut is made separately for patients with and without a bleed. In a simulation at the real counts (18,938 patients, 21,744 scans), a plain hash cut missed the 1.5-point bleed-rate check in 35% of seeds, and the stratified cut never did. The split is rebuilt each Kaggle session and never saved. A fingerprint hash confirms it is unchanged. The test split raises unless `ALLOW_RSNA_TEST=1`, so it stays closed until the final Vision B model is chosen (recommendation, needs Parsa's OK).
- **Vision B code:**
  - `model.py`: timm ResNet-50 or EfficientNet-B3, one logit, normalised on the GPU.
  - `train.py`: balanced epochs (every training positive plus as many fresh negatives), flip, rotation, scale and brightness augmentation, AMP, `last.pt` and `best.pt` checkpoints, resume, and a time budget so a Kaggle session ends on a checkpoint. It also has speed probes for the data loader and the GPU.
  - `aggregate.py`: scan score by max or top-k mean, picked by validation scan AUC. Calibration fits temperature and temperature plus bias; the one with the lower out-of-half log loss is kept. Temperature alone cannot undo the balanced training prior. Also reliability bins and diagram, ECE, and a threshold for 95% scan sensitivity.
  - `infer.py`: slice probabilities → the contract's `{present, confidence}`.
  - `data/rsna.py`: `ShardReader` can now be pickled, so spawned workers open their own file handle.
- **Notebook** `20_visionB_train.ipynb` (Kaggle). Dummy mode is EXP-5: split check, loader and GPU speed for both models, 150 real training steps, and aggregation and calibration on a 300-scan validation sample. Train mode is set from the dummy's numbers. Both modes ran on synthetic shards with Kaggle, git, pip and nvidia-smi stubbed, on CPU: no token or ID in any output, expected files only. The harness loosened the bleed-rate tolerance, because it had 12 validation patients.
- **Not yet run on Kaggle or a GPU:** real speeds, GPU memory, timm weight download and resuming from an attached earlier version are untested.
- pytest: 239 passed (CPU torch and timm installed in the local venv; the Vision B training tests skip without them).
- **Notion sync:**
  - done: EXP-5 row (Planned).
  - pending: Decisions rows for the LLM judge, embedding model, RSNA split (now with stratification and the closed test split) and temperature wording.

## 2026-11-04 — Claude Code session (Phase 2, task 3: EXP-2 dummy result)
- **EXP-2 dummy ran to the end** at `703871e` on a Tesla T4 runtime with 2 CPUs and 12.7 GB RAM.
  - Colab's PyTorch 2.11.0+cu130 was kept, not replaced.
  - Fingerprinting and preprocessing fit in memory.
  - torch.compile ran, our `splits_final.json` was used (103 training and 26 validation cases), and resuming from the checkpoint worked.
  - Prediction and scoring worked.
  - Plan: patch 96 × 128 × 112, batch 4.
- **Epochs took 664, 509 and 570 s.** The first and third include compiling. Timings: fetch 86 s, convert 69 s, preprocess 143 s, training plus validation 1,920 s.
- **Most likely limited by data augmentation on 2 CPUs**, not by the GPU. That is an inference, not measured. At about 510 s, 250 epochs would take about 35 hours on a T4, too long for one fold and impossible for five.
- Fold-0 Dice was 0.247 after 3 epochs. It is not a result, but it is already above the threshold baseline's fold-0 Dice of 0.123.
- **Change:** data-augmentation workers are now CPUs minus 1, from 2 to 12. The dummy run can be repeated under a new experiment ID: EXP-4, on an L4 or A100, will size EXP-3. Tested on synthetic data only.
- **EXP-4 dummy on an L4** at `64974be`: 12 CPUs, 53 GB RAM and 11 data workers.
  - Epochs took 191, 119 and 140 s, the first and third including compile. That is about 4.3 times faster than the T4, which fits the CPU explanation.
  - Fold-0 Dice was 0.224 after 3 epochs.
- **EXP-3 sized at 150 epochs:** about 5 hours of training on an L4. The 250 epochs first planned would take about 8.3 hours per fold, and five folds would probably exceed the remaining compute units.
  - The L4 compute-unit rate has not been checked against Colab yet.
  - The notebook now defaults to train mode. Tested on synthetic data only.
- **Notion sync:**
  - done: EXP-2 and EXP-4 → Done, with their timings; EXP-3 row updated with 150 epochs.

## 2026-11-04 — Claude Code session (Phase 2, task 3: nnU-Net set-up)
- **Code:** added `vision/perfusion/nnunet_wrapper.py`, pinned to nnU-Net v2 2.8.1 after reading its trainer and command-line code. It:
  - converts development cases only to six channels: NCCT, CBF, CBV, MTT, Tmax with negative values set to 0, and a Tmax-valid mask. The label is the whole final infarct.
  - writes our five folds as `splits_final.json`;
  - adds a trainer, `nnUNetTrainer_StrokeCopilot`, whose epochs, iterations and checkpoint interval come from the environment so a run can resume across Colab sessions;
  - parses epoch times from nnU-Net's logs, and scores the validation predictions with `evaluate.py`.
- **Notebook:** added `notebooks/10_visionA_nnunet.ipynb`.
  - The dummy mode (EXP-2) trains 2 epochs, resumes for a third, then predicts and scores fold 0, to measure epoch time and check that checkpoints load on Colab's PyTorch.
  - The train mode (EXP-3) needs its epoch count set from EXP-2. It checkpoints to Drive and refuses to resume if a session plans a different network.
- **Tested on synthetic data only**, with Colab, git, pip, nvidia-smi and nnU-Net's command line faked: both modes, a train session that stops after preprocessing and is rerun, a refused rerun of a finished run, and no token or held-out ID in any output.
- **Not tested at all:** real nnU-Net, real PyTorch and GPU behaviour, and real epoch time. nnU-Net 2.8.1 excludes PyTorch 2.9, so pip may replace Colab's PyTorch; the notebook prints the version before and after.
- Tests: `test_nnunet_wrapper.py`. pytest: 221 passed.
- **First EXP-2 attempt on an L4 failed:** the system killed planning with exit code -9 after 6 of 129 cases of fingerprint extraction, almost certainly from running out of memory. nnU-Net starts 8 fresh Python processes by default, and each imports PyTorch. Colab now runs Python 3.13.
  - Fix: fingerprinting and preprocessing use 2 workers (`N_PROC`), and training's data-augmentation workers are capped at 4 (`N_PROC_DA`, up to 12 by default). The notebook prints RAM, and a -9 exit now says it was a memory kill. Tested on synthetic data only.
  - pytest: 222 passed.
- **Second attempt failed the same way,** on a T4 runtime (12.7 GB RAM; the runtime was not an L4), with 2 workers. Our progress filter had hidden the tqdm lines once they passed 1 it/s, so the output stopped at 4/129, though the kill came later.
- **Cause, measured locally** with real nnU-Net 2.8.1 (CPU, Python 3.14) on 129 synthetic cases of our size:
  - nnU-Net's fingerprint step pools 1e8 foreground samples per channel, which peaked at 6.4 GiB.
  - With our `StrokeCopilotFingerprintExtractor` (1e7 samples), fingerprinting, planning and preprocessing ran to the end, peaking at 4.4 GiB summed over all processes, most of it short-lived worker startup.
  - The planned 3d_fullres network is patch 96 × 112 × 96, batch 4, at 1.5 mm.
  - Training setup and the first batch with 4 data workers stayed at 0.6 GiB in the main process.
- **Fixes:**
  - `install_extensions` writes the trainer and the extractor into nnU-Net before planning;
  - progress bars now print every 10% instead of being dropped.
  - Real GPU training memory and epoch time are still unmeasured.
- pytest: 223 passed.
- **Notion sync:**
  - done: EXP-2 and EXP-3 rows (Planned).

## 2026-11-04 — Claude Code session (Phase 2: EXP-1 results)
- **EXP-1 ran on Colab** at `bf86e15` with all 129 development cases, at 0.21 s per case. Results are in `runs/EXP-1_thresholds_baseline/`.
- **Threshold core vs final infarct:**
  - Dice 0.111 ± 0.031 over folds; the pooled median is 0.027.
  - Absolute volume error 25.2 ± 5.0 mL. The median error is -4.2 mL, and the core underestimates large infarcts (p5 -106 mL).
  - The predicted core falls on the other side of a cut-off from the reference in 35, 29, 23 and 17 of 129 cases at 21, 31, 51 and 70 mL.
  - Dropping the Tmax > 6 s restriction changes little (Dice 0.117).
  - Penumbra matches native Tmax > 6 s to a median of 2.8 mL, as in Phase 1.
- **Not accepted yet:** a Dice this low needs checking for a bug before it stands. Added `thresholds.diagnose_case`, with a test, and `notebooks/11b_visionA_threshold_check.ipynb`. The notebook measures how much infarct lies inside Tmax > 6 s and on the side called affected, the rCBF inside the infarct, and Dice at rCBF 20–45%. Tested on synthetic data only.
- **Occlusion distance limit set to 5 mm** from the EXP-1 distances:
  - ICA-nearest occlusions sit at a median 0 mm (p95 6.0 mm);
  - basilar-nearest ones sit 17–51 mm away, except one under 3 mm;
  - 5 mm gives 13 ICA, 1 basilar and 115 `unknown` of 129, the `unknown` cases including 72 nearest an MCA label, 7 ambiguous and 2 empty.
- **Consequence:** about 89% of real cases will have `unknown` occlusion site, so most real-case thrombectomy decisions will be undetermined (§9).
- **11b ran at `869e47f` and EXP-1 is accepted.** The low Dice comes from the data, not from a bug that the checks can see:
  - the infarct is on the affected side in almost every case (median share 1.0);
  - only half of it lies inside Tmax > 6 s, and its median rCBF is 0.65;
  - small infarcts score near zero, while the largest third has a median Dice of 0.24.
  - The 30% cut-off is kept. Details are in `docs/DATA.md` under "Perfusion and the final infarct".
- **MCA-nearest occlusions:** 20 of 72 overlap the M1 label. They stay `unknown` under the approved rule.
- §7: the `thresholds.py` and `volumes.py` tasks are ticked.
- pytest: 214 passed.
- **Notion sync:**
  - done: EXP-1 row → Done, with headline numbers and the check result.
  - pending: the Decisions rows listed in the entry below.

## 2026-11-03 — Claude Code session (Phase 2 start: metric definition, threshold baseline code)
- **Phase 2 set-up, agreed with Parsa:**
  - **LLMs:** the frontier model is Claude, with about $100 of API credits. The open and medical models are to be chosen in Phase 3.
  - **Rule-claim check:** an LLM judge, not NLI.
  - **Retrieval embeddings:** `BAAI/bge-base-en-v1.5`.
  - **RSNA split:** a seeded hash of patient ID into 80/10/10 train/validation/test, checked to keep the natural bleed rate. No IDs leave Kaggle.
  - **Compute:** 180 Colab compute units and 30 Kaggle GPU hours this week.
  - **Notion Experiments** gained `Fold` and `Git commit` columns. Rows are created first, and their ID names the run folder.
- **Hallucination metric:** `docs/HALLUCINATION_METRIC.md` was written before any verifier code and approved with its seven open points, then mirrored to Notion. Additions to §8.4:
  - claims also come from blocking factors, fact-stating uncertainties and decision `rule_ids`;
  - a content hallucination rate (sub-types 1–5) is reported next to the full rate;
  - omission of uncertainty is a separate count;
  - the judge model is chosen by seeded-error recall between Haiku 5.5 and Opus 5.5, because of the budget.
  - §7 task ticked.
- **Temperature:** the current Claude API rejects `temperature`, so §8.3 and §8.4 now say "temperature 0 where the provider allows it", with a repeat sample to show run-to-run spread. The §10 cost risk was updated.
- **Vision A code:** added `vision/perfusion/volumes.py` (mask to mL, contract volumes, the occlusion-site rule), `thresholds.py` (rCBF < 30% core, Tmax > 6 s penumbra, Tmax < 0 is no data), and a new `evaluate.py`. `evaluate.py` holds Dice, Dice on reliable Tmax, volume error against `native_volumes.csv` and cut-off crossings, and is shared with nnU-Net. Tests: `test_volumes.py`, `test_thresholds.py`.
- **Notebook:** added `notebooks/11_visionA_thresholds.ipynb`, EXP-1, which also measures occlusion-to-label distances. It ran on synthetic data with Colab, git and pip stubbed: full run, refusal to replace a finished run, and no token or held-out ID in any output.
- **Not yet run on real data:** real Dice, volume errors, timing and the distance distribution are unknown. The occlusion distance limit stays provisional at 5 mm until the run, so the §7 `thresholds.py` and `volumes.py` tasks stay open.
- **Label numbering:** the TopBrain 2025 data page confirms that TopCoW labels 1–12 and 15 are kept, and that 5 and 7 are the M1 segments of the MCAs. MCA-nearest occlusions stay `unknown` under the approved rule.
- pytest: 213 passed.
- **Notion sync:**
  - done: Hallucination metric page; Experiments columns; EXP-1 row (Planned).
  - pending: Decisions rows for the LLM judge, embedding model, RSNA split and temperature wording; Tasks → hallucination metric done.

## 2026-11-03 — Claude Code session (Phase 1 closed)
- Shards 1–3 re-stored with `02c`. Final `02b` on the four `rsna-slim-v1-file-*` datasets: every shard is one file, 752,802 readable slices all match their indexes, copying all four takes 199 s and listing them 35 s. Local random reads: 373 slices per second on 1 thread, 1,227 on 4, about 10 minutes per pass reading only. Lower than the one-shard test, probably because 17.5 GiB no longer fits in memory.
- §7: loader task ticked. README badges ticked: rows for unwritten Phase 2+ notebooks no longer link to missing files, and `02_prep_rsna` is marked Kaggle-only.
- **Phase 1 acceptance (as changed under DEC-6) is met:** the ISLES'24 tarball and RSNA shards are ready, both loaders work on real data, and RSNA set-up and read speed fit training. Still deferred, not done: `00_template`, onboarding and Phase 2 assignment. Phase 0's open items are still Parsa's: professors' deliverables and deadline, the Kaggle no-copies question, the shared Drive folder and clinician review.
- Notion sync: pending. Tasks → `isles24.py` and `rsna.py` loaders done; RSNA preprocessing done, with datasets now `rsna-slim-v1-file-s0` to `-s3`. The old folder datasets and `dummy-unpack-test` can be deleted.

## 2026-11-03 — Claude Code session (Phase 1: re-store test passed on shard 0)
- Parsa suggested a quick test with dummy files first: Kaggle unpacked a `.tar` and kept a `.shard` whole, so the choice goes by file name. Shard 0 then packed in 748 s with 16 threads, and `rsna-slim-v1-file-s0` kept the 4.371 GiB file whole.
- `02b` on it: copy 103 s, list 17.5 s, local reads 877–2,193 slices per second without repeating slices, best-case. Random reads from the attached file are 15–17 per second, so training must copy first. All four shards should set up in about 8 minutes per session, projected from one shard.
- Next: `02c` for shards 1–3, then a final `02b` with all four files.
- Notion sync: pending until all four are done.

## 2026-11-03 — Claude Code session (Phase 1: re-storing RSNA shards)
- Parsa approved re-storing the shards. Added `pack_shard` to `rsna.py`, which packs a shard's folder into one tar named `.shard`, and `notebooks/02c_rsna_repack.ipynb`, run for shard 0 first as a test. A single file now wins over a folder if both are attached.
- Fixed `02b`: every measurement uses different slices. A single-file shard is copied locally and listed there, because listing it over Kaggle's storage may be very slow; it also reports copy time, the set-up cost of a session.
- Tested by running both notebooks' real cells on synthetic data, with Kaggle and pip stubbed. **Not yet run on Kaggle**: whether Kaggle keeps a `.shard` file whole is an assumption, and copy speed and multi-thread first-read speed are unknown.
- Notion sync: pending until the test shard has passed.

## 2026-11-03 — Claude Code session (Phase 1: RSNA read speed)
- Ran `02b_rsna_read_speed.ipynb` on Kaggle. All four datasets exist and match their indexes: 752,802 readable slices. All four are folders, because Kaggle unpacked the tars. Full table in `docs/DATA.md`.
- **First read from a folder is 25 slices per second**, the same as raw DICOM. One pass or packing all four shards would take about 8.4 hours. Packing one shard took 2.1 hours. A local tar reads at 847–1,856 per second, best-case.
- **My mistake:** `02b` measured 4 threads on the same files as 1 thread, so its 1,726 per second from a folder is cached and means nothing. To be fixed in the next notebook.
- Recommended: re-store the shards once in a form Kaggle does not unpack, starting with one shard as a test. Waiting for Parsa's go-ahead. The §7 loader task stays open.
- Notion sync: pending until the storage decision. RSNA preprocessing → all four datasets exist.

## 2026-11-02 — Claude Code session (DEC-6 changed: onboarding deferred)
- **Parsa changed DEC-6:** no fixed onboarding date. Parsa sends teammates instructions when assigning them work. Nobody has been onboarded yet; this is deferred, not done.
- **Phase 1 acceptance changed:** tarballs ready, loaders working on real data for both datasets, and RSNA read speed measured and fit for training. It was every teammate running `00_template`.
- `00_template`, onboarding and Phase 2 assignment moved to a deferred list in §7, unticked. `00_template` gets written when the first teammate is brought in. Week 1 in §7 now starts when the first Phase 2 tasks are assigned.
- Still needed before anyone joins: the shared Drive folder, and their acceptance of the RSNA competition rules.
- Notion sync: pending. Decisions → update DEC-6. Tasks → mark the onboarding task deferred.

## 2026-11-02 — Claude Code session (Phase 1: RSNA shards are unpacked on Kaggle)
- The first run of `02b_rsna_read_speed.ipynb` showed all four datasets attached, so `-s2` and `-s3` exist, but `-s0` holds a folder instead of a `.tar`: Kaggle unpacks archives when a dataset is made. The other three are assumed the same, because no shard was found.
- `rsna.py` now reads both layouts: `FolderReader`, `open_shard`, and `copy_shard` packs a folder into one local tar. `02b` measures speed from the attached folder and from a local tar, and the time to pack, which is the set-up cost of a session.
- **Not yet run on real data.** Whether the shards can stay as folders, or need re-storing in a form Kaggle does not unpack, depends on these numbers. The earlier `.tar` sizes in `docs/DATA.md` describe the files as built, not the datasets.
- Notion sync: none pending until the notebook has run.

## 2026-11-02 — Claude Code session (Phase 1: slim check results)
- Ran `01d_check_slim.ipynb` on Colab. All 129 development cases load through the loader at 0.13 s each with no problems, all 20 held-out cases are refused, and copy and unpack takes 54 s. The checksum on Drive matches the build.
- **Label loss explained:** in 4 development cases one tiny label vanished in resampling, as expected. One held-out case lost 8 labels, all outside the crop, very likely the one off-grid mask. Not looked at further. v1 stands, and occlusion site must return `unknown` when labels are absent or too far away.
- **MTT:** non-finite voxels are in 21 cases, median 250, worst about 1% of a volume.
- **Negative Tmax:** typically -28 to -0.7, median -14 s, and only about half lie on soft tissue or CSF-like values. Cause unknown. Infarct with Tmax of 0 or more is a median of 95.3%, p5 65.1%, lower than the 99.4% and 80.9% that counted only -30 as no data. Recommendation for Vision A: treat all negative Tmax as no reliable data.
- The §7 loader task stays open until `02b_rsna_read_speed.ipynb` has run.
- Notion sync: pending. Tell Vision A about negative Tmax when Phase 2 tasks are assigned.

## 2026-11-02 — Claude Code session (Phase 1, Task 7: loaders)
- Implemented `data/isles24.py` (`fetch_slim`, `list_cases`, `load_case`, `load_native_ncct`) and `data/rsna.py` (`find_shards`, `load_index`, `scan_table`, `slices_of_scan`, `ShardReader`, `decode`, `window_hu`, `copy_shard`, `measure_read`), with `tests/test_isles24_loader.py` and `tests/test_rsna_loader.py`. `load_case` and `load_native_ncct` refuse held-out cases, and `list_cases` leaves them out. `Case.note_fields` filters by note tag, because `clinical` also holds forbidden columns.
- Added `cow_label_check`, `mtt_non_finite_check` and `tmax_negative_check` to `data/isles24_survey.py`, and notebooks `01d_check_slim.ipynb` (Colab) and `02b_rsna_read_speed.ipynb` (Kaggle).
- Tested by running both notebooks' real cells on synthetic data with Colab, Kaggle and pip stubbed. **Not yet run on real data**: the real read speed, whether Kaggle kept each shard as a `.tar`, whether `-s2` and `-s3` exist, and the answers to the label, MTT and Tmax questions are unknown. The §7 loader task stays open until both notebooks have run.
- RSNA patient-level splits are not in `rsna.py`; they belong to Vision B in Phase 2.
- Notion sync: none pending until the notebooks have run.

## 2026-11-02 — Claude Code session (Phase 1: slim tarball built)
- Ran `01_prep_isles24.ipynb` on Colab: `slim/isles24_slim_v1.tar`, 149 cases, 2.86 GiB, sha256 `b0afd16e…4728`, read back from Drive and matching, no problems in the final check. 7 s per case, 990 s in all, well under my 30–90 minute estimate. §7 grid and prep tasks ticked; 1.5 mm is in `manifest.json`. Full results are in `docs/DATA.md`.
- **Results:** infarct volume error 0.8% median, 6.2% p95. Tmax > 6 s error 2.8 mL median, 9.6 mL p95, against 7.5 and 17.4 mL if averaged, so not averaging maps was right on real data. Reference volumes should come from `native_volumes.csv`.
- **New findings:** in 15 of 127 development cases over 10% of the infarct has no Tmax data, and one has none. A median of 16% of non-background Tmax voxels are negative, which is unexplained. MTT has 417,749 non-finite voxels, now background. Source orientation is 100 LAS and 49 LPS. Both `sub-stroke0180` rows are duplicates. Time text is `hh:mm:ss`, not hrs:min.
- **Corrected:** `01_prep`'s figure caption said black on Tmax is tissue without data; it can also be a negative value. Only -30 counts as no data in the coverage figures.
- **Open:** 5 cases lost a Circle of Willis label, including what looks like one case losing most of them, possibly outside the crop. Not diagnosed. v1 stands; occlusion site must return `unknown` when labels are absent. `01_prep` has not been rerun since the caption change, and a rerun refuses to replace the tarball.
- Notion sync: pending. Tasks → grid and ISLES'24 slim are done. Tell Vision A about Tmax coverage, negative Tmax and `native_volumes.csv`, and NLP A about time text format, when Phase 2 tasks are assigned.

## 2026-11-02 — Claude Code session (Phase 1: split in the repo)
- The `01c` section 9 `OK:` line confirms the results were read back from Drive, the archive is unchanged, and there are 129 development cases.
- Added `splits/folds.json`. Its sha256 matches the one the notebook recorded (`023813f2…d291`). §7 task ticked. The split was not regenerated.
- **Held-out exposure:** attaching `folds.json` to the chat loaded the whole file, held-out IDs included, into this Claude Code session's context and local transcript. That was the file's ID list only, with no images, volumes or clinical data for those cases. They were not repeated or used in any analysis or code, and DEC-5 is unchanged. From now on `folds.json` is checked by checksum and never opened or attached in a session.
- Notion sync: pending. Tasks → split generated and in the repo.

## 2026-11-02 — Claude Code session (Phase 1: survey results and slim preparation)
- Ran `01c_unpack_split_survey.ipynb` on Colab. Working copy on Drive (10.32 GiB) in 9,845 s, 6,089 s of that decoding. The split was generated and all 149 cases are complete. Full results are in `docs/DATA.md`. The `OK:` line from section 9 has not been seen yet, though the results are on Drive.
- **Alignment passes,** so no registration. **The Circle of Willis labels cannot separate M1 from M2**; §5.1 and §8.1 now use the nearest label with a distance limit, and MCA is `unknown`. That is about 57% of development masks. The schema is unchanged. Parsa approved both. §7 tasks ticked.
- **Grid: 1.5 mm isotropic, cropped to the brain.** Parsa approved. The §7 task stays open until `manifest.json` exists.
- **New findings:** Tmax has gaps inside the brain and some infarct lies in them, not yet quantified; CBV and MTT have negative values and MTT has non-finite voxels; one Circle of Willis mask, in a held-out case, is off the NCCT grid; 2 development infarct masks and 2 occlusion masks are empty; held-out has no very large infarcts.
- **Fixed a bug:** `extra_rows` compared a row with itself when two rows had equal counts, so the survey's "no conflicts" for `sub-stroke0180` means nothing. `docs/DATA.md` was also wrong to call the extra row nearly empty.
- Added `data/prep_isles24.py`, `tests/test_prep_isles24.py`, `notebooks/01_prep_isles24.ipynb` and `sha256_of` in `data/archive_listing.py`. Slim v1 gains `native_volumes.csv`, because resampling moves Tmax > 6 s volumes by 2–7 mL and native Tmax is not in the tarball.
- **Changed my own suggestion:** on irregular synthetic shapes, averaging before shrinking lost more mask volume than plain sampling (about -40% vs -20% at 0.03 mL, -8% vs -3% at 0.3 mL), so only NCCT is averaged. The report also measures Tmax error with averaging, for comparison.
- Tested by running the notebook's real cells on a synthetic 30-case working copy with Drive and pip stubbed: full run, refusal to replace a finished tarball, resuming from saved batches, and no held-out ID in the output or report. A full-size synthetic case took 1.6 s on a Mac. **Not yet run on Colab or real data**: real time per case, Tmax coverage, resampled volume errors, time text format and what the `sub-stroke0180` rows hold are all unknown. `splits/folds.json` is not in the repo yet, so that §7 task stays open.
- Notion sync: pending. Tasks → spot-check and M1/M2 check are done, and the grid is decided. Decisions → occlusion site rule (MCA is `unknown`), 1.5 mm grid, `native_volumes.csv` in slim v1. Tell Vision A about Tmax gaps and NLP A about `unknown` occlusion site when Phase 2 tasks are assigned.

## 2026-11-02 — Claude Code session (Phase 1: split, spot-check and grid notebook)
- Added `data/splits.py`, `tests/test_splits.py`, `data/isles24_layout.py`, `data/isles24_survey.py`, `tests/test_isles24_survey.py`, `splits/README.md` and `notebooks/01c_unpack_split_survey.ipynb`. The `[isles]` extra gains numpy, nibabel, scipy and matplotlib, and the wheel now ships `splits/`.
- The notebook makes a working copy of the files we use and saves it to Drive, checks the known anomalies before any split exists, generates the split once, checks headers for all cases, spot-checks alignment and labels on development cases only, and measures candidate grids.
- **Split:** §5.5 now stratifies by centre as well as infarct volume tertile, seed 20261102. DEC-5 is unchanged. Parsa approved this on 2 November 2026.
- Tested by running its real cells on a synthetic 30-case archive with Drive and pip stubbed, with 7-Zip 26.05 and p7zip 17.05: first run, rerun from the saved copy, a changed split being refused, and no held-out ID in the output or report. **Not yet run on Colab or real data**: real timing, disk, Drive upload of the tar, image contents and the survey's reading of real labels are untested. `folds.json` does not exist yet, and no §7 task is ticked.
- Notion sync: none pending until the notebook has run.

## 2026-11-02 — Claude Code session (Phase 1, Task 2 results)
- Ran `01b_clinical_isles24.ipynb` on Colab: extraction took 129 s, results saved to Drive and read back, archive unchanged. The printed PDF cut off long lines, so the full results were read from the saved JSON files on Drive through the Google Drive connector, read-only.
- Added `data/clinical_columns.py` and `tests/test_clinical_columns.py`: all 33 real columns are tagged allowed or forbidden for notes, and an untagged column raises an error. `mRS at admission` and `Center` are forbidden by our choice; the rest follow §8.3. Full table in `docs/DATA.md`. §7 task ticked.
- **Findings:** at least 61 of 149 cases have no onset or alert time, and there is no last-known-well column. `INR` is missing in 61 rows and `mRS premorbid` in 60. Of 99 rows with a TICI value, 2 are below 2c and the rest of the rows have none, so §5.1 and §8.1 now carry that caveat. One baseline and one outcome workbook have an extra row; which cases is unchecked. `Hyperlipidemia` has a 99 code. Actual column names differ from the dictionary, and `Center` is undocumented.
- The extra lesion mask belongs to `sub-stroke0142`. Its contents are unchecked.
- Fixed a small bug: `workbooks_holding_their_case_id` reached the JSON as a string. The value, 0, was right.
- Time columns are text and their format has not been parsed yet, so the `hrs:min` format is only what the dictionary states.
- Notion sync: done in this session for Tasks → column tagging. Still pending: tell NLP A about the missing-data findings when Phase 2 tasks are assigned.

## 2026-11-02 — Claude Code session (Phase 1, Task 2: clinical survey)
- Added `data/clinical_survey.py`, `tests/test_clinical_survey.py`, `notebooks/01b_clinical_isles24.ipynb`, `extract_members` in `data/archive_listing.py`, and an `[isles]` extra. The notebook extracts only the data dictionary and clinical workbooks, prints the dictionary, and surveys each column in aggregate: counts for columns with few distinct values, percentiles for numbers, and no case IDs or single-patient rows. It also names the case holding the extra lesion mask.
- Tested by running its real cells on a synthetic archive with Drive and pip stubbed, with 7-Zip 26.05 and p7zip 17.05, including a rerun and an archive with no clinical files. **Not yet run on Colab or the real files**: the real workbook structure, column names, extraction time and Excel value formats are untested. Columns are not tagged yet, so the §7 task stays open.
- Notion sync: none pending until the notebook has run.

## 2026-11-02 — Claude Code session (Phase 1, Task 1 results)
- Ran `01a_list_isles24.ipynb` on Colab: file list read in 3 s, saved to Drive and read back, archive unchanged. The Colab install and Drive read both worked. 2,982 files, 95.840 GiB unpacked, 149 cases with IDs from 0001 to 0189. All four perfusion maps exist for every case. Exact paths and the mapping to slim files are in `docs/DATA.md`. §7 task ticked.
- Corrected the plan: clinical data is two Excel files per case, not CSVs (§5.1).
- The archive is solid: any `derivatives/` file type costs about 79 GiB of decoding, including lesion masks, while clinical files cost about 1.7 GiB. So the slim build and the infarct volumes for the split both need what is effectively a full unpack.
- **Order change:** the split (Task 6) moves ahead of the spot-check (Task 3), so nobody looks at images of cases that later become held-out. The split, spot-check and grid survey share one unpack session. Task 2 stays next because it is cheap.
- Unexplained: one case has an extra `ses-02_lesion-msk.nii` without `space-ncct`. To be checked in Task 3. Nothing has been extracted, so shapes, spacing and contents are unconfirmed.
- Notion sync: pending. A search of Tasks found no row for the archive inspection, so no row was changed. Data guide → point to `docs/DATA.md` for the layout.

## 2026-11-02 — Claude Code session (Phase 1, Task 1: archive listing)
- Added `data/archive_listing.py`, `tests/test_archive_listing.py` and `notebooks/01a_list_isles24.ipynb`. The notebook reads the file list of `raw/isles24_train.7z` without extracting anything and reports layout, perfusion maps, unpacked size, whether the archive is solid, and how much must be decoded to extract each file type. It saves to `artifacts/isles24_listing/`, reads the result back from Drive, and checks the archive's size and modified time did not change.
- Tested by running its real cells on small synthetic archives with Drive and pip stubbed, with 7-Zip 26.05 and p7zip 17.05: solid, non-solid, rerun, wrong size, truncated archive and missing file. **Not yet run on Colab or the real archive**: the private install, the `apt-get` step (Colab's 7z version), reading the list through the Drive mount and the real layout are untested. The §7 task stays open.
- Added a local `.venv` (git-ignored). pytest: all pass.
- Notion sync: none pending until the notebook has run.

## 2026-11-02 — Claude Code session (Phase 0 wrap-up)
- Read the Kaggle competition rules for RSNA. They apply to our copy and are stricter than the plan said, which had relied on the looser challenge terms: academic use is allowed, but no redistribution and no access for anyone who has not accepted the rules. Corrected §5.2, and added `docs/DATA_USE.md` with what this means for teammates, Drive, the repo, the report and local copies.
- **Open question:** the rules' clause against making copies conflicts, taken literally, with storing private shards (DEC-17). DEC-17 is unchanged; this needs asking the professors. Guideline licence terms are not checked yet.
- Added `docs/CLINICIAN_REVIEW.md`: rules, the five cases and questions, for sending to a reviewer. Not sent. DAWN and DEFUSE-3 numbers are still unchecked against the trial papers.
- Replaced `<org>` in the README. Confirmed the repo is private. Ticked the terms and GitHub tasks in §7.
- Phase 0 acceptance criteria are met. Three tasks stay open and are Parsa's: record deliverables and deadline, finish and share the Drive folder, and get the clinician review. Datasets `rsna-slim-v1-s2` and `-s3` were still uploading and their contents are unchecked.
- Notion sync: done in this session (Data-use terms, Data guide, and the terms, ISLES'24, RSNA and clinician tasks). The calendar was not touched.

## 2026-11-02 — Claude Code session (RSNA shards complete)
- All four `rsna_slim_v1` shards built on Kaggle: 752,803 slices match the labels, 752,802 usable, and each tar matches its index. One slice in shard 2 failed with a `ValueError`; its ID is in that shard's index with `ok` false. Total size is 17.47 GiB including tar overhead, under the 19 GB limit.
- Datasets `rsna-slim-v1-s0` and `-s1` exist. `-s2` and `-s3` are not created yet: on those versions Kaggle's Output menu offers "New dataset version" on `rsna-slim-v1-s0`, which must not be used.
- Speed varied from 35 slices per second (shard 0) to 10 (shard 3), so a shard took 1.5–5.4 hours. The notebook's 2–3 hour estimate is not reliable. Reading speed from an attached dataset is still unmeasured.
- Shard 2's verify step printed a harmless pandas `DtypeWarning`, caused by the failed slice's empty fields.
- Notion sync: pending. Tasks → ISLES'24 download is done; RSNA preprocessing is built, with two datasets to create.

## 2026-10-31 — Claude Code session (data download records)
- ISLES'24 download finished on Colab: 99,022,114,670 bytes on Drive, md5 matches Zenodo. Resuming across sessions worked; one attempt stalled for about 15 minutes and the notebook's retry recovered. The archive has not been opened yet. Phase 0 task ticked.
- RSNA shard 0 built on Kaggle: 188,251 slices, 0 failed, 4.371 GiB, about 92 minutes, with the tar matching its index. Projected total is 16.94 GiB at 320 px, under the 19 GB limit. Shards 1–3 are not recorded yet. Reading speed from an attached dataset is still unmeasured; the 8,043 images per second in the report came from local disk and decoded only a sample.
- Added `docs/DATA.md` with both records.
- Notion sync: pending. Tasks → ISLES'24 download is done; RSNA preprocessing has shard 0 done.

## 2026-10-31 — Claude Code session (RSNA preprocessing)
- Implemented `data/prep_rsna.py`: one pass over the raw DICOM writes 320 px three-window JPEGs into hash-assigned tar shards, plus an index with patient, study and series IDs, slice position, spacing and all six labels. Added `tests/test_prep_rsna.py` and an `[rsna]` extra.
- Added `notebooks/02_prep_rsna.ipynb`, run once per shard with Save Version. It installs the package with a read-only GitHub token from Kaggle Secrets and stops early if the projected size would exceed 19 GB.
- Tested on synthetic DICOMs with the Kaggle secret client and pip stubbed. **Not yet run on Kaggle or real data**: the private `pip install`, real speed, real size, and tar reading from an attached dataset are untested.
- Notion sync: Tasks → RSNA preprocessing points to this notebook.

## 2026-10-31 — Claude Code session (RSNA timing results, DEC-17)
- First Kaggle timing run: 752,803 labelled slices, 14.3% with a bleed, no missing files or failed reads, patient/study/series IDs in all 2,000 sampled files, `SliceThickness` absent. Raw reading is about 20–28 files per second, or about 7.5 h per epoch.
- **Test flaw:** the 384 px loader speed (206 images per second) is inflated because it re-read files the earlier steps had cached. The report's 384 px epoch times should be ignored, and the notebook still has this ordering flaw; it is kept as the record of this run.
- DEC-17 decided: preprocess once to 320 px JPEG tar shards, then train on Kaggle; 256 px is the fallback. Plan §2, §4.2, §5.2 and §7 updated. Reading speed from tar shards is not measured yet.
- `02a_rsna_kaggle_timing.ipynb`: silence pydicom's warning about RSNA's non-standard IDs.
- Notion sync: done in this session (DEC-17, RSNA task, Data guide).

## 2026-10-31 — Claude Code session (Kaggle notebook fix)
- Fixed `02a_rsna_kaggle_timing.ipynb`: the first Kaggle run could not find `stage_2_train.csv`, because the search only went two folders deep. It now walks the input folder and skips image folders. Tested on five folder layouts; not yet rerun on Kaggle.
- Notion sync: none pending.

## 2026-10-31 — Claude Code session (run calendar)
- Created the Notion "Run Calendar" database under the hub, covering every planned run and its prerequisites from 31 Oct to 7 Nov 2026, split across Parsa and four teammates. Teammates are placeholder roles until they join the workspace. Estimates are unmeasured.
- This compresses the plan's 8-week phases into 8 days; §7 has not been changed. 7 Nov needs confirming as the real deadline before the plan is rewritten around it.
- Notion sync: done in this session.

## 2026-10-31 — Claude Code session (contract freeze)
- Parsa froze `schemas/vision_output.json` at v0.1 with no new fields (DEC-4). Mass effect, M2 dominance and extensive hypodensity stay out of the contract; the reasoner must name them as uncertainties. Later changes need a Decisions row and a version bump.
- Notion sync: done in this session (contract task marked Done).

## 2026-10-31 — Claude Code session (RSNA Kaggle timing notebook)
- Added `notebooks/02a_rsna_kaggle_timing.ipynb`. It runs on Kaggle, trains nothing, and saves only `rsna_timing_report.json`, with counts and timings and no IDs or images. It estimates epoch time per model and image size, preprocessing time and output size, to settle DEC-17. Self-contained, because the package cannot be installed there yet.
- Tested by running its real cells on synthetic DICOMs on a Mac CPU, including corrupt, missing and duplicate entries. Testing caught and fixed a JSON crash in the final cell. **Not yet run on Kaggle, on a GPU or on real RSNA files**, so the GPU training path, real file reading and Kaggle's limits are untested.
- Notion sync: Tasks → RSNA timing test now points to this notebook.

## 2026-10-30 — Claude Code session (RSNA full data)
- DEC-16: Parsa chose all labelled RSNA data, no subset. DEC-17 proposed: preprocess and train Vision B in Kaggle notebooks so no 180 GB transfer is needed, pending a timing test.
- Plan §2, §5.2, §7, §8.2 and §10 updated. Phase 0 acceptance now requires full RSNA attached and timed, not downloaded. The `rsna_slim` tarball is no longer assumed.
- Notion sync: done in this session (DEC-16, DEC-17, RSNA task, Data guide).

## 2026-10-30 — Claude Code session (guideline verification)
- Parsa confirmed DEC-14 and DEC-15, and supplied the 2026 guideline PDF and the RSNA challenge page. The PDF stays out of the repo.
- §8.3 rulebook rewritten from the primary text. **Corrections to the secondary summaries:** extended-window thrombolysis (§4.6.3 rec 2) applies only to patients not eligible for thrombectomy; M2 is not a single COR 2b, but COR 2a for dominant proximal M2 and COR 3 no benefit otherwise; ASPECTS ≥ 6 at 6–24 h is confirmed as COR 1.
- Fixture expectations revised: fx02 and fx05 thrombolysis return to no, reversing the borderline expectations added earlier today; fx01 thrombectomy depends on M2 dominance; fx04 stays a thrombectomy yes. Vision values and notes unchanged.
- DEC-7 supported: guideline Table 3 defines penumbra as Tmax > 6 s and uses penumbra − core as the mismatch volume.
- Salvageable-penumbra rule: the recommendations give no numbers, so we use the trial criteria in Table 3 (EXTEND for 4.5–9 h, TRACE-III for 4.5–24 h). Our choice; needs clinician review.
- RSNA terms: the challenge Terms of Use allow sharing with attribution, but the AWS listing forbids it. Plan §5.2 now prefers the challenge copy on Kaggle; the Kaggle rules are still unread. Earlier note that the licence bars sharing was incomplete. RSNA states 5 mm slices.
- No contract change. Mass effect, M2 dominance and extensive hypodensity stay out of the contract as named uncertainties, pending the freeze decision.
- Notion sync: done in this session (DEC-8, DEC-14, DEC-15, guideline task, RSNA tasks, Data guide).

## 2026-10-30 — Claude Code session (Notion sync and RSNA review)
- Notion synced by this session: DEC-1 Decided; DEC-7 to DEC-13 added as Decided; DEC-14 and DEC-15 added as Proposed. Nine tasks added, stale Phase 0 rows updated, Data guide and hub links updated. Plan §2 now matches.
- RSNA review: the AWS listing's licence bars sharing or redistributing the data, which affects third-party preprocessed copies and possibly the shared Drive. Source not chosen; plan §5.2 marks this and asks for whole-scan sampling.
- Notion sync: done in this session.

## 2026-10-30 — Claude Code session (git attribution)
- Parsa's rule: commits and PRs never credit Claude. Added to CLAUDE.md.
- Rewrote the three existing commits to remove the `Co-Authored-By` line and force-pushed `main` and `lead/phase0-scaffold`. File contents and dates are unchanged; commit hashes changed. Anyone with an earlier clone must re-clone or reset to `origin/main`.
- Notion sync: none pending.

## 2026-10-30 — Claude Code session (Phase 0 download notebook)
- Added `notebooks/00_download_isles24.ipynb`: downloads `train.7z` from Zenodo record 17652035 (99,022,114,670 bytes, md5 `4959a5dd2438d53e3c86d6858484e781`) to `raw/`. It resumes after interruptions, stops if the Drive folder is missing or too full, and keeps the file as `.part` until the md5 from Drive passes. It writes `raw/isles24_download.json` as the provenance record.
- The notebook is self-contained, an exception to "logic lives in src/": it must work before the repo is pushed, and it runs once. Plan §3 and the README table updated.
- Tested by running its real cells with a stubbed Drive against a local server: dropped connection and resume, rerun, recovery of a lost record, checksum mismatch, missing folder, changed Zenodo metadata. **Not yet run on Colab or against the full file**; only Zenodo metadata and a 1-byte range request were checked live.
- The download has not been started. The Phase 0 task stays open.
- Notion sync: Tasks → "Start ISLES'24 download" now points to this notebook. Once it has run, copy `isles24_download.json` into `docs/DATA.md`.

## 2026-10-30 — Claude Code session (Phase 0 architecture review)
- Plan v0.2. Professors approved the project (reported by Parsa), so DEC-1 is Decided. The Phase 0 confirmation task stays open until deliverables and the deadline are recorded.
- **Clinical rules:** the corpus baseline is now the 2026 AHA/ASA guideline, which replaced the 2018/2019 documents (DEC-8). §8.3 rulebook rewritten. The primary text returned 403, so rules come from secondary summaries and are marked for verification; single-source items and the numeric definition of salvageable penumbra are flagged as unconfirmed.
- **Contract (still draft v0.1):** added required, nullable `aspects` (DEC-9), in `schemas/vision_output.json`, the pydantic mirror and all fixtures. No ASPECTS model is planned.
- **`schemas/recommendation.json`:** a claim now has `sources` (one or more) and a required `derived` flag instead of a single `source` (DEC-10). §6.4 example updated.
- **Fixtures:** new expectations in §6.3 and `fixtures/README.md`. fx04 thrombectomy is now yes (large-core tier, ASPECTS 4), fx02 and fx05 thrombolysis are now borderline and must cite. Scores: fx01 9, fx02 8, fx03 null, fx04 4, fx05 null. All expectations still need primary-text and clinician review. Vision values and notes unchanged.
- **Data (§5):** ISLES'24 moves to Zenodo record 17652035, which includes the data dictionary (DEC-12). Recorded the licence, the mTICI 2c/3 cohort, tabular fields, ±5% noise on times and labs, and masks already in NCCT space. Slim format gains `ncct_native` and `cow`. The registration task became a spot-check; occlusion site now uses Circle of Willis labels.
- **Notes (§8.3):** treatment-revealing and post-treatment columns are forbidden, including door to groin and time of intervention.
- **Vision (§8):** the core that feeds the contract is chosen by comparison in Phase 3, because trial cut-offs were defined on perfusion core. Vision B runs on native-thickness NCCT.
- **Evaluation (§8.4, §9, DEC-11):** "exactly one source" replaced; added misapplied-rule and faulty-derivation sub-types; synthetic scenarios are the primary evaluation; added seeded errors, no-retrieval arm, ground-truth vs predicted vision arm, case-level CIs and recommendation correctness. New stubs: `nlp/verifier/scenarios.py`, `nlp/verifier/seeded_errors.py`.
- `tests/test_contract.py` updated and extended; all pass.
- Still open: full guideline text, clinician review, `mismatch_ratio` two decimals and `haemorrhage.confidence` = P(haemorrhage). The contract is **not** frozen.
- Notion sync:
  - Decisions → DEC-1 Decided. New rows: DEC-7 (`penumbra_ml`), DEC-8 (2026 guideline), DEC-9 (`aspects`), DEC-10 (multi-source claims, hallucination definition), DEC-11 (evaluation design), DEC-12 (Zenodo record). §5.4 slim format change needs a row too.
  - Tasks → new in Phase 0: verify rulebook against primary text; clinician review; confirm provisional contract choices and freeze. New in Phase 1: spot-check masks, tag clinical columns, check M1/M2 labels. New in Phase 2: scenarios, seeded errors.
  - Docs → update the ISLES'24 download source to record 17652035.

## 2026-10-29 — Claude Code session (Phase 0 scaffolding)
- Package skeleton per §3 under `src/strokecopilot/`: every module exists as a docstring stub naming its phase. `config.py` holds Drive paths and slim versions; `schemas/` and `fixtures/` resolve from the repo root (editable/source) or from inside the installed package (wheel).
- `pyproject.toml` (hatchling): deps `pydantic>=2.5`, `jsonschema>=4.18`; extra `dev` = pytest. The wheel ships `schemas/` and `fixtures/` so `pip install git+...` can validate on Colab without a checkout. Checked with a non-editable install into a fresh Python 3.11 venv, run from outside the repo.
- `schemas/vision_output.json` v0.1 (JSON Schema 2020-12) per §6.1, plus `contract/schema.py` (pydantic mirror) and `contract/validate.py` (JSON Schema first, then pydantic rules; raises `ContractError`). All fields required (null must be explicit); undeclared fields rejected; strict typing, so null is never coerced to 0.
- `contract.schema.derive_mismatch` is the single definition of the mismatch arithmetic; `vision/perfusion/volumes.py` should call it in Phase 2.
- Five fixtures + triage notes per §6.3, with `fixtures/README.md` holding the expected recommendations (kept out of the notes so they never reach the reasoner). fx01 has core 0.6 mL → `mismatch_ratio` null (null-handling case); fx03 has occlusion `unknown`; fx05 is a deliberate DEFUSE-3 near-miss (mismatch 14.5 mL) that passes DAWN.
- `schemas/recommendation.json` per §6.4. Added a `note` source type (`{"type": "note", "quote": ...}`), which §8.4(b) needs but §6.4's example does not show. Vision `field` is an enum of dotted vision_output paths.
- `tests/test_contract.py`: 59 tests, all passing — fixtures validate, derived fields, null handling, rejections, pydantic mirror in sync with the JSON Schema, recommendation schema.
- README.md with placeholder Open-in-Colab badges (`<org>` until the repo is pushed; notebooks do not exist yet, so the Phase 1 README task stays open). `data/README.md` added.
- `.gitignore` extended. The old `data/` pattern also matched `src/strokecopilot/data/`; patterns are now anchored to the repo root (`/data/*`, `!/data/README.md`, `/runs/`, `/artifacts/`).
- Decisions made here (contract-level, confirm before freezing v0.1):
  - `mismatch_ratio` rounded to **two** decimals and computed from the already-rounded volumes. §6.2 only fixes rounding for volumes; one decimal would push e.g. 1.76 over the DEFUSE-3 1.8 cut-off.
  - `haemorrhage.confidence` read as P(haemorrhage present), not confidence in the `present` call.
- **Decided by Parsa: `penumbra_ml` is the total Tmax > 6 s volume, core included.** §6.1 and §8.1 disagreed: §6.1's formulas (`mismatch_ratio = penumbra_ml / core_ml`, `mismatch_volume_ml = penumbra_ml - core_ml`) only match DEFUSE-3 under this reading, while §8.1 said "Tmax > 6 s minus core". §6.1 stands. Reworded §6.1 (meaning column), §8.1 (penumbra bullet), the §13 glossary, the `penumbra_ml` description in `schemas/vision_output.json`, and `fixtures/README.md`. No field, type or formula changed, and the fixtures are unchanged. fx05 therefore stays a DEFUSE-3 near-miss (mismatch 14.5 mL < 15 mL).
- Not done in this session: professor confirmation, Drive folder, downloads, licence notes, GitHub push.
- Notion sync:
  - Tasks → Done: "Scaffold repo", "Write vision_output schema + pydantic mirror", "Write five fixtures and notes", "tests/test_contract.py".
  - Decisions → new rows (Proposed): mismatch_ratio rounded to 2 dp; haemorrhage.confidence = P(haemorrhage); `note` source type added to recommendation.json.
  - Decisions → new row (Decided, Parsa): `penumbra_ml` = total Tmax > 6 s volume including core; salvageable tissue = `mismatch_volume_ml`. Plan §6.1/§8.1/§13 aligned.

## 2026-10-29 — Parsa
- Repo created. Added STROKE_COPILOT_PLAN.md v0.1 and CLAUDE.md.
- Notion hub created under Project HQ → Stroke Decision Copilot (tasks, decisions, experiments, docs).
- Notion sync: none pending.
