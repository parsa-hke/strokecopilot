# CHANGELOG

## 2026-11-20 — Claude Code session (Phase 5 task B: SHAP rows 1 and 5 ready to run)
- **Row 1, Vision A channels (EXP-75):** `analysis.visiona_channel_shap` and `notebooks/15_visionA_channel_shap.ipynb` (Colab A100).
  - Five players (NCCT, CBF, CBV, MTT, Tmax with its mask), 32 combinations per development case, each predicted by its validating fold model with nnU-Net's validation settings.
  - Absent inputs follow DEC-35 and are filled only inside the case's nonzero region, so the crop box never changes (checked per combination).
  - The full combination is checked against EXP-27's recorded volume. Rows go to Drive as they come, so a disconnect loses nothing.
  - Dummy first: one case.
- **Row 5, Vision B windows (EXP-76 Kaggle, EXP-77 Mac):** `analysis.visionb_window_shap` and `notebooks/26_visionB_window_shap.ipynb`.
  - Every RSNA validation slice under the 8 window combinations, with absent windows at their RSNA training means (a seeded sample of 3,000 training slices).
  - The full combination must reproduce EXP-6's scan AUC.
  - Only aggregates and the three means leave Kaggle. The ISLES'24 half runs on the Mac with those means.
  - Dummy first: 200 validation scans.
- **Not run yet:** both need a push, then Parsa's Colab and Kaggle runs. The Kaggle `metrics.json` must be downloaded into Drive `runs/EXP-76_visionB_window_shap/`.
- **Tests:** `tests/test_analysis.py` (24). pytest: 903 passed. **Spend:** $0.
- **Notion sync:** Experiments: EXP-75, EXP-76, EXP-77 (Planned).

## 2026-11-20 — Claude Code session (Phase 5 task B: SHAP row 9 ready for Colab)
- **DEC-35 amended (Parsa, 20 Nov 2026), row 9:** the bleed stays fixed at its real value, because the frozen contract cannot say "unknown" for `haemorrhage.present`. The reasoner game has six groups (perfusion, occlusion site, ASPECTS, time, patient, contraindications), and the rules are scored on the same six. Recorded in plan §2 and the survey.
- **`analysis.reasoner_shap`:**
  - picks 24 scenarios by seed (one per key route, then 8 more);
  - builds all 64 versions of each, with the absent groups null in the contract and "not recorded" in a note re-rendered with the first phrasing throughout;
  - writes the Colab requests and scores model against rules (`python -m strokecopilot.analysis.reasoner_shap`).

  1,280 distinct requests per model. Every full version reproduces its scenario's rule decision.
- **Notebook 55** gains `MODE = 'shap_dummy'` (one scenario) and `'shap'` (EXP-74); three lines of the notebook changed. Checked end to end on the Mac with rule-derived gold answers (the model's values then equal the rules', as they must). Not yet run on Colab.
- **Tests:** `tests/test_analysis.py` (21). pytest: 900 passed. **Spend:** $0.
- **Notion sync:** Decisions: DEC-35 (amended: row 9 bleed fixed). Experiments: EXP-74 (Planned: reasoner SHAP, notebook 55).

## 2026-11-20 — Claude Code session (Phase 5 task B: SHAP rows 3, 6, 10 and 11 done)
- **`shap` 0.48.0** installed in the project venv and pinned in a new `analysis` extra (`pyproject.toml`; it built from source on Python 3.14).
- **EXP-70, row 3, `confidence.core` v2** (`analysis.confidence_shap`; seconds, $0): exact SHAP of v2's logistic mapping, identical to `shap.LinearExplainer` with all 129 cases as background. The distance to the nearest cut-off carries **73%** of the attribution, core size 14%, missing Tmax 12%, disagreement 1%. In-sample; it backs DEC-27.
- **EXP-71, row 10, the retriever** (`analysis.retrieval_shap`; Mac CPU, 45 s, $0): Partition SHAP over passage tokens of the bge similarity, for the 12 key fixture passages with a citing gold claim (EXP-7's queries were not saved).
  - Number tokens carry at least their share in 10 of 12 (median 13% of the attribution on 10% of the tokens).
  - So bge does not ignore numbers; sibling rows share the same numbers. This refines EXP-7's diagnosis and still supports DEC-19.
- **EXP-73, row 11, a surrogate of the verifier's flags** (`analysis.surrogate_flags`; 18 s, $0), labelled a surrogate:
  - gradient boosting on claim features known before verification, 27,472 claims; out-of-fold AUC **0.92 [0.92, 0.93]**, grouped by case, so SHAP is reported;
  - rule claims (44% flagged) and non-numeric comparisons (26%) push towards a flag; imaging values (2%) away;
  - model and claim kind carry 60% of the attribution.
- **EXP-72, row 6, Vision B pixels** (`analysis.visionb_pixel_shap`; dummy 2 scans then full; Mac MPS, 10.4 min, $0; no case ID saved). Expected gradients on the top 2 slabs of the 68 flagged ISLES'24 development scans, beside Grad-CAM on the same slabs.
  - Reproduces EXP-50: 68 flags; Grad-CAM 76% in brain.
  - SHAP: median **50%** in brain, 26% in bone (Grad-CAM 9%); brain over bone in **127 of 136** slabs; median Spearman with Grad-CAM 0.50.
  - A second method supports Item 2: the false alarms come from brain tissue.
- **Figures** in Drive `artifacts/figures/` with README lines: `fig_confidence_v2_shap`, `fig_retrieval_token_shap`, `fig_surrogate_flags_shap`, `fig_visionB_pixel_shap`. Results in `docs/PHASE5_SHAP_SURVEY.md` §5; runs on the Mac and Drive.
- **Fixed:** the figures module's entry point ran before its later functions were defined.
- **Tests:** `tests/test_analysis.py` (19). pytest: 898 passed. **Spend:** $0.
- **Notion sync:** Experiments: EXP-70, EXP-71, EXP-72, EXP-73 (Done, figures above).

## 2026-11-20 — Claude Code session (Phase 5 task C: EXP-69, the t-tests)
- **`analysis.ttest_plan`** loads every approved test from the saved runs. It first recomputes each reported bootstrap result from the same files and seed and stops on any mismatch (all matched). It then runs `ttests.paired` / `ttests.welch` and applies Holm. T7's held-out side comes from a new `phase4 rates` command: per-case rates only, no ID or alias, ID-checked; post hoc, as Parsa allowed. One `ALLOW_HELDOUT=1 … phase4 rates` run.
- **EXP-69** (`runs/EXP-69_ttests/` on the Mac and Drive; Mac, seconds, $0): **34 of 35 tests**; T9 waits for task D.
  - **Clear after Holm (11):** T1 (nnU-Net Dice +0.104); T3a–d (rulebook effect, −9 to −15 points); T5b–e (Gemma 27B, MedGemma, 4B, tuned 4B above Sonnet); T6a–b (tuned below base, −29 and −24 points).
  - **Not clear after Holm:** T2 (volume −1.6 mL, p 0.29); all of T4 (vision arm; Gemma 27B p 0.046 unadjusted); T5a (Qwen vs Sonnet); T6c (tuned critical rate); all of T7 (post hoc; the 4B pair nominally worse on held-out cases); all of T8 (subgroups; nominal Dice by age and volume error by sex, skewed).
  - **t-test and bootstrap disagree in 2 of the 18 with a reported interval:**
    - T5a: t +1.3 [+0.2, +2.4] against +1.0 [−0.1, +2.1]; not established after Holm.
    - T6c: t +0.9 [−0.8, +2.5] against +1.9 [+0.4, +3.4]. This weakens EXP-48's higher critical rate for the tuned model: it holds for the pooled rate, not the per-case mean. Both are reported.
- **Figures:** `fig_ttests_forest` (T3–T6, t interval beside the reported bootstrap) in Drive `artifacts/figures/` with its README line; `qq_plots.png` in the run folder. Results in `docs/PHASE5_TTESTS.md` §7.
- **Fixed on the way:** a numpy bool saved as 0.0 in the run's JSON (`agrees_with_bootstrap` now returns a Python bool).
- **Tests:** `tests/test_analysis.py` (15). pytest: 894 passed. **Spend:** $0.
- **Notion sync:** Experiments: EXP-69 (Done: 34 of 35 t-tests, figures above). Tasks: Phase 5 task C (done except T9, which waits for task D).

## 2026-11-20 — Claude Code session (Phase 5 additions: owner, and SHAP row 8 done, EXP-68)
- **Owner:** Parsa owns and runs every Phase 5 addition (his decision, 20 Nov 2026); recorded in both docs.
- **EXP-68, the rule engine's inputs** (DEC-35, SHAP row 8; `analysis.rule_shap`; Mac, 6 s, $0; `runs/EXP-68_rule_shap/` on the Mac and Drive). Exact Shapley values, 13 input groups, 8,192 coalitions per scenario, on the 150 synthetic scenarios.
  - **Thrombectomy** turns mostly on the occlusion site (mean |value| 0.57); time, NIHSS and pre-stroke mRS give 0.14 each, ASPECTS 0.06, the core 0.007 and Tmax > 6 s 0.003.
  - **Thrombolysis** turns on time (0.41), then the occlusion site (0.14) and NIHSS (0.13); the core and Tmax > 6 s give 0.06 and 0.05.
  - **Where the rules say yes,** vision inputs carry 28% of thrombolysis's attribution (40 scenarios) and 50% of thrombectomy's (32).
  - **With every input unknown,** the rules already reach the decision in 55/150 (thrombolysis, undetermined) and 86/150 (thrombectomy, no). There the values sum to 0 and some are negative, so the magnitudes are the summary. This agrees with EXP-63 (core ×0.8/×1.2 change 9 and 10 of 300 decisions).
  - **Limits:** a synthetic scenario set with even route coverage, not real patients; it measures the rules, not any model.
- **Figure** `fig_rule_engine_shap` (PNG and SVG) in Drive `artifacts/figures/`, with its README line (`analysis.figures`, which adds or replaces only its own lines). Palette pair checked with the dataviz validator.
- **Tests:** a README-line test added (`tests/test_analysis.py`, 12). pytest: 891 passed. **Spend:** $0.
- **Notion sync:** Experiments: EXP-68 (Done: rule engine SHAP, figures above). Tasks: Phase 5 task B (in progress: row 8 done).

## 2026-11-20 — Claude Code session (Phase 5 additions, tasks B and C: the SHAP survey and the t-test list, approved)
- **`docs/PHASE5_SHAP_SURVEY.md`** (draft): 19 places checked against the code and saved outputs. Each row gives the method (exact enumeration or which `shap` explainer), the question, the data, the cost and whether it is feasible.
  - **Proposed to run:** Vision A input channels; Vision A `confidence.core` v2; Vision B windows and pixels; the rule engine; the Gemma 3 4B pair; the retriever; three surrogates. Optional: Vision A spatial. Sonnet only in a small form (about $9, estimate).
  - **Said not to apply or to tell us nothing:** DEC-27's one-input confidence; the top-2 slab score; `confidence.penumbra`; calibration; the probes' embeddings; the LLM judge; the threshold baseline; a surrogate on EXP-41's 21 human-judged errors.
  - **Found:** nnU-Net reads six channels (NCCT, CBF, CBV, MTT, Tmax, a Tmax-valid mask), not four maps. The proposal has five players, with Tmax and its mask as one. The proposed baseline for absent Tmax is the trained "no reliable data" form (Tmax 0, mask 0); the other maps are mean-filled (a constant, all zeros after nnU-Net's per-image z-score; to be checked in the dummy run, since nnU-Net is not installed on the Mac).
  - **Found:** EXP-7's retrieval queries are not in `src/`; only its config and metrics are on Drive.
- **`docs/PHASE5_TTESTS.md`** (draft): 35 Student's t-tests (paired, or Welch where unpaired) beside the existing results, at the case level. It says plainly that the data has already been seen. Each test reports a rank test beside it, Holm over the whole list, and its effect size and assumption checks. T7 (development against held-out) is post hoc on saved held-out results and needs Parsa's OK.
- **Parsa's choices (20 Nov 2026):** both docs approved with the proposed defaults. T7 (post hoc, held-out, aliases only) is allowed. **DEC-35** (the SHAP analyses) and **DEC-36** (the t-tests) are added to plan §2. Owners are decided later: the code is built and tested now, and nothing runs on data until owners are assigned. EXP numbers start at EXP-68, in order as each analysis runs.
- **Code, tested without data** (new package `strokecopilot.analysis`):
  - `shapley.py`: exact Shapley values by enumeration (up to 16 players; number or vector outputs). Tested on an additive game, the glove game and efficiency.
  - `rule_shap.py`: SHAP row 8, the rule engine's 13 input groups (every `rules.Case` field exactly once; ratio and mismatch kept only with both volumes), with the value 1 when the decision matches the full-input one. Also vision against note as a two-player game. Its `main` refuses to write without the run's `config.json`.
  - `ttests.py`: paired and Welch t-tests with the interval, d_z or Hedges' g, skewness, Shapiro–Wilk and the rank test beside each; Holm; the check against a bootstrap interval. Checked against scipy and a worked Holm example.
  - `tests/test_analysis.py` (11).
- No analysis has been computed yet. pytest: 890 passed. **Spend:** $0.
- **Notion sync:** Decisions: DEC-35 and DEC-36 (new, Decided, 20 Nov 2026, Parsa). Tasks: Phase 5 additions, task B (SHAP: survey approved, code in progress) and task C (t-tests: list approved, code in progress).

## 2026-11-20 — Claude Code session (Phase 5 additions, task A: the clinician pack, approved and fixed)
- **Scope of this side session:** Phase 5 additions A to J (clinician pack, SHAP, t-tests, BiomedCLIP and probes, subgroups, a non-Claude judge, retrieval, ONNX, reproducibility), report only under DEC-31's freeze. Other Phase 5 sessions own the reports, paper, deck and Space page.
- **`docs/CLINICIAN_PACK.md`** (approved by Parsa, 20 Nov 2026): two instruments for two clinicians, answered independently; delivery; the case-selection rule; the fixed questions; consent; the analysis plan; the rule-error policy; and the sha256 of each built file. It says what may be claimed (expert opinion, never clinical validation). It supersedes `docs/CLINICIAN_REVIEW.md` as the thing that gets sent; that file gets a pointer line.
- **`pipeline/clinician_pack.py`** builds the pack from the Space's bank and the corpus. Nothing is run or paid for.
  - **Instrument 1:** 72 items, each agree / disagree / unsure with a comment: the 47 passages, the 5 fixtures' rule answers (10), 12 rule conventions, and 3 open questions.
  - **Instrument 2:** 9 cases, seeded draw (seed 20261120) within strata: right answers, flagged ones, wrong-and-caught, wrong-and-missed. Cases: fx04, fx05, scenario-025/032/056/057, D-03, D-05, D-12. D-05 is the only development case with a vision error the verifier missed (a false bleed flag). One case is Gemma 3 27B's. Per-case questions Q1–Q8 and R1, plus general questions G1–G6.
  - The output goes to `runs/EXP-67_clinician_review/pack/` (gitignored: real-case notes by alias). The build refuses to finish on a development case ID, an RSNA ID or a key pattern; it was clean. The bank loader refuses held-out cases.
- **Found:** `docs/CLINICIAN_REVIEW.md` gave TRACE-III as "mismatch > 15 mL", but `rules._trace3` and the rulebook use total Tmax > 6 s volume > 15 mL. The pack states what the code does (convention C09) and asks the clinicians. Nothing in the code changed.
- **Parsa's answers (20 Nov 2026):**
  - §7, the rule-error policy, is approved: nothing is re-run; the affected scenario and development decisions are counted in code; held-out decisions only with his OK.
  - Ethics clearance for the expert interviews is confirmed.
  - The consent has no withdrawal date or address, because the clinicians are known to the team.
  - Roles are self-reported: instrument 1 now opens with the consent and P1–P4 (name, role and specialty, years in practice, acute stroke decisions). That makes 76 items, 69 of them scored.
  - The pack was rebuilt; the case selection is unchanged, and the checksums in `docs/CLINICIAN_PACK.md` §9 are updated.
- **Delivery: Google Forms, generated** (Parsa's choice, after a form inside the public Space was advised against: it would change DEC-32 and publish, the Space has no lasting storage, and anyone could submit).
  - The pack now also writes `forms_spec.json`, `build_forms.gs` (an Apps Script that builds both forms with private response Sheets in Parsa's Drive and sends nothing), and `interview_cases.md` / `.docx` (the nine case pages for the interview form, through pandoc, reproducible; no questions or expected answers).
  - Checked: `node --check` passes, and a run against a mock FormApp builds both forms (171 and 122 items) with only the consent questions required.
  - Two tests added: every response-sheet item appears once in the forms under its ID, only consent is required, and the expected answers come after every case; the case document holds no questions or expected answers.
- **Pack approved and fixed by Parsa (20 Nov 2026):** the cases, questions and analysis plan. A rebuild reproduces every recorded checksum (`docs/CLINICIAN_PACK.md` §9); any change from here needs a dated note there. The pack is copied to Drive `runs/EXP-67_clinician_review/pack/` (10 files, checksums match).
- **Next, Parsa:** run `build_forms.gs` at script.google.com, check both forms (published, no e-mail collection, link works in a private window), and send each clinician the two links and `interview_cases.docx`.
- **Tests:** `tests/test_clinician_pack.py` (6). pytest: 879 passed. **Spend:** $0.
- **DEC-34 (Parsa, 20 Nov 2026)** added to plan §2: Phase 5 additions are report only under DEC-31's freeze, and the clinician review is EXP-67 under `docs/CLINICIAN_PACK.md`. DEC-33 stays reserved for Item 1.
- **Notion sync:** Decisions: DEC-34 (new, Decided, 20 Nov 2026, Parsa). Experiments: EXP-67, clinician review (Planned: pack fixed, waiting for answers). Tasks: Phase 5 additions, task A, clinician pack (Done: fixed and ready to send; sending is Parsa's).

## 2026-11-20 — Claude Code session (Phase 4 done: the Space is public)
- **Public** on Parsa's word: https://huggingface.co/spaces/parsa-hke/stroke-copilot (visibility switched from private; same build, nothing re-uploaded). Checked without a token: health 200 (195 cases, key present, live on, cap $5), page with the banner and CSP, held-out cases served; in the browser, the story images, nine results cards and the held-out tab load with no console errors.
- **Plan §7 Phase 4:** the Space line and the acceptance line are ticked. Phase 4 is done.
- **Phase 4 spend (estimates):** Anthropic about $5.29 (EXP-65 $4.73 with Sonnet's $0.61 dummy cell inside it, the Space's dummy EXP-64 $0.45, one live run on the Space $0.11); OpenRouter $0.28; Kaggle free; Colab A100 for notebooks 53, 55 and 60 (compute units not reported).
- **Still open, not blocking:** the Colab compute units for the spend record; a monthly limit on the key's Anthropic workspace (Parsa, recommended now that the Space is public).
- **Notion sync:** Tasks: the Space (Done). Decisions: DEC-32 (public, 20 Nov 2026).

## 2026-11-20 — Claude Code session (the Space's key and daily cap)
- **Key:** Parsa added `ANTHROPIC_API_KEY` as a secret on the private Space; the health check reports `key_present: true`, `live_enabled: true`, nothing spent.
- **Daily cap $5** (Parsa's choice, a change to DEC-32 §7, recorded there): Space variable `DAILY_CAP_USD = 5`; the code's default stays $1. The cap is soft (in memory, reset by a restart or sleep); the Anthropic workspace's monthly limit is the hard cap.
- **One live run on the Space** (Parsa's go; builder defaults: witnessed onset, 5 h, M1, core 20 mL, Tmax > 6 s 90 mL): live, 51.6 s (reasoner 28 s, extractor 11 s, judge 11 s, its calls concurrent); thrombolysis no, thrombectomy yes; 24 claims, 1 flagged, 0 critical; **$0.108**, matching the health check's spend. A first request was refused (422, nothing spent) because the script sent `builder.validate`'s output, whose not-applicable fields are null; the page drops those fields, so visitors are unaffected.
- **Spend:** $0.11.
- **Notion sync:** Decisions: DEC-32 (note the $5 daily cap).

## 2026-11-20 — Claude Code session (Phase 4, step 5: the Space filled with held-out results, not yet published)
- **Bank with held-out cases** (`ALLOW_HELDOUT=1 … phase4 bank`, under the existing permission rule): 195 cases (5 fixtures, 150 scenarios, 20 development `D-xx`, 20 held-out `H-xx`, the same aliases as EXP-66's report), each with the six models' saved answers on both vision arms (EXP-65 cells). `demo.bank` now has `real_cases`, `heldout_cases` and `build`; held-out cases enter only through `phase4`.
- **Chart data:** `charts.heldout(report)` adds EXP-66's numbers (correctness per model and arm, error propagation with case aliases, the ±20% core check, held-out vision, the realism check). Numbers and aliases only.
- **Page:** a "Held-out cases" tab; three results cards (the final test with a caution note on the all-"no" thrombectomy truth; where wrong decisions came from, with links to the case; development vs held-out hallucination rates); held-out sentences on the core step and the sensitivity card; the case count is read from the bank. Checked in the browser on desktop and at 375 px (no overflow, no console errors).
- **Server:** `Cache-Control: no-cache` unless a response sets its own, so an update reaches returning visitors (found when the local browser kept an old script).
- **Space tree** (`phase4 space`): 346 files, 23.3 MB; scan clean for every case ID (held-out included), RSNA IDs, keys, e-mail, file types, no split. Local load test (fake live runner): 50/50 results, 0 errors, median 0.13 s, worst 5.45 s. Space README: 195 cases.
- **Parsa's choice (20 Nov 2026):** the Space stays **private for now**, under Parsa's own Hugging Face account, named `stroke-copilot`, shared only with invited people. The source names Parsa in code comments and four rulebook pointers ("checked by Parsa"), plus the GitHub username in install links; it ships unchanged. DEC-32 is not changed: going public stays pending, so Phase 4's acceptance line ("the public Space is live") stays open until Parsa says so or amends DEC-32.
- **Private Space live** on Parsa's word: `parsa-hke/stroke-copilot` (Docker, cpu-basic, private: 404 without a token), created and uploaded from the Mac's Hugging Face sign-in. First upload refused before sending anything (`short_description` over 60 characters); shortened, and a test now checks the README metadata. Built in about 2 minutes; serves 195 cases, the held-out charts, the banner and the cache header.
- **Load test on the Space** (`loadtest --url … --hf-auth`, live runs off, no key set): 50/50 visitors got a result, 0 errors, all precomputed, median 1.6 s, worst 1.8 s. $0.
- **Not done:** the key secret (Parsa sets `ANTHROPIC_API_KEY` in the Space's settings; until then the builder falls back to the closest preset); a visual check of the Space in a signed-in browser; making it public (DEC-32, on Parsa's word).
- **pytest:** 873 passed. **Spend:** $0.
- **Notion sync:** Tasks: the Space (still in progress; note updated).

## 2026-11-20 — Claude Code session (Phase 4, steps 3 and 4 done: EXP-65 runs, EXP-66 D3 analyses)
- **EXP-65, all 14 held-out cells done** (`runs/EXP-65_heldout_runs/` on the Mac; Colab answers on Drive):
  - Sonnet: Batch API, 4 cells;
  - Qwen: OpenRouter, Alibaba route, system prompt merged, 2 cells;
  - Gemma 3 27B and MedGemma 27B: notebook 53 `heldout`, Colab A100, 4 cells;
  - Gemma 3 4B and tuned: notebook 55 `heldout`, 4 cells;
  - every cell verified (Sonnet extractor, Haiku `judge_v3`, Batch API).

  | Model | Arm | Hallucination rate | Format failures |
  |---|---|---|---|
  | Sonnet 5.5 | predicted / ground truth | 3.2% / 5.3% | 0, 0 |
  | Sonnet 5.5 | core ×0.8 / ×1.2 | 3.3% / 4.6% | 0, 0 |
  | Qwen3.5-27B | predicted / ground truth | 6.3% / 3.6% | **9** / 2 |
  | Gemma 3 27B | predicted / ground truth | 11.6% / 11.3% | 0, 0 |
  | MedGemma 27B | predicted / ground truth | 10.0% / 9.0% | 2 / 6 |

- **Measured spend (estimates at batch or list prices):** Anthropic **$4.73** (DEC-31 estimate about $5.5 with contingency); OpenRouter **$0.28**; Colab A100 for notebooks 53 and 55 (compute units not reported yet).
- **EXP-66, DEC-31 D3** (`phase4 analyse`; `runs/EXP-66_heldout_analysis/report.json` and `report.md` on the Mac and Drive; aliases `H-01`…, ID-checked):
  - **Truth:** thrombolysis undetermined 17, yes 2, no 1; thrombectomy no 20/20. Rules on the predicted contracts get 22/23 right.
  - **Correctness, predicted vision (scored decisions):** Sonnet 22/23; Qwen 13/13 (9 answers missing); Gemma 3 27B 22/23; MedGemma 21/21; Gemma 3 4B **3/23** (it recommends thrombectomy where there is no established occlusion); tuned 4B 21/23.
    - The tuned model's bias towards "no" (EXP-48) happens to match a held-out truth that is almost all "no", so the comparison says little.
    - Ground-truth vision: Sonnet 23/23, Qwen 20/20, Gemma 22/23, MedGemma 16/16, 4B 5/23, tuned 22/23.
  - **Error propagation (predicted arm):**
    - Sonnet: 1 wrong, a vision error (false bleed flag), missed by the verifier.
    - Gemma 3 27B: 1 reasoning error, caught.
    - Tuned 4B: 1 vision error (bleed flag), caught; 1 reasoning error, caught.
    - Base 4B: 20 reasoning errors, all caught.
    - Qwen and MedGemma: 0.
    - The verifier cannot see a vision error whose claims faithfully restate the contract, as EXP-63 found on development cases.
  - **Sensitivity:** core ×0.8 and ×1.2 change 0/40 rule decisions and 0/40 of Sonnet's own answers. With the DEC-31 §4 supplement (scenarios 9/300 and 10/300), the reading is: a 20% core error can matter in principle, but in this cohort the unknown occlusion site decides first.
  - **Held-out vision (ensemble):** Dice 0.182 (median 0.095); absolute volume error 13.4 mL (median 7.9; signed median −6.5); wrong side of 21/31/51/70 mL in 4/4/3/0 of 20. Bleed flags 10/20 (Wilson 29.9–70.1%).
  - **Realism check, hallucination rate held-out vs development real cases (unpaired, predicted vision):** Sonnet 3.2% vs 3.3%; Qwen 6.3% vs 3.3% (11 answers); Gemma 11.6% vs 10.9%; MedGemma 10.0% vs 7.9%; 4B 40.4% vs 35.1%; tuned 4B 21.5% vs 12.3%. The larger models hold; the tuned 4B's gain over its base shrinks on real held-out cases.
- **Figure:** `fig_heldout_realism_check` added to `artifacts/figures/` (`figures.heldout_figures`; README updated).
- **Plan §7 Phase 4:** the end-to-end, error-propagation, sensitivity and figures lines are ticked. Still open: the Space, filled with held-out results and published on Parsa's word; then the acceptance line.
- **pytest:** 870 passed.
- **Notion sync:** Experiments: EXP-65 (Done, spend), EXP-66 (Done, figures above). Tasks: reasoner runs, error propagation, sensitivity, figures (Done).

## 2026-11-19 — Claude Code session (Notion synced with Phase 4 so far)
- **Decisions:**
  - **DEC-32** added (Decided, Infra, 19 Nov, decided by Parsa), with its rationale and alternatives.
  - **DEC-31**'s page gains its three dated amendments: DEC-32's demo change; the scenario sensitivity supplement; and Phase 4 code paths on the Mac (Option A).
- **Experiments:**
  - EXP-60, EXP-61, EXP-62, EXP-63, EXP-64 added as Done, with the figures above and their commits;
  - EXP-65 as Running (dummy cell result);
  - EXP-66 as Planned.
- **Tasks:** nine Phase 4 tasks from plan §7, owned by Parsa and linked to DEC-31 (the Space to DEC-32):
  - Done: RSNA test split, contract path, D3 code, unlock;
  - In progress: reasoner runs, the Space, figures;
  - Next up: error propagation, sensitivity.
- Run Calendar not touched. No case IDs in Notion.

## 2026-11-19 — Claude Code session (Phase 4: report figures exported)
- **`pipeline/figures.py`** draws the report's figures from the same numbers as the Space (`demo.charts`; `charts.default_sources` now shared with the bank), so the report and the page cannot disagree.
  - **Style:** the reference palette's categorical slots in fixed order (one per model), thin marks, hairline grid, one axis per panel, whiskers for intervals, and direct labels where they carry the point.
- **Exported to Drive `artifacts/figures/`** (PNG at 300 dpi and SVG, with a `README.md` of captions and sources). Numbers only, no IDs, no RSNA images:
  - `fig_c9_hallucination_by_model`: all, content and critical rates, small multiples;
  - `fig_rulebook_effect`;
  - `fig_human_check`;
  - `fig_small_reasoner_tradeoff`;
  - `fig_visionB_rsna_test` (reliability and subtypes, EXP-60);
  - `fig_visionA_cutoff_crossings`;
  - `fig_core_sensitivity_scenarios`.
- Checked by eye. Fixed before export: label overlap and a 105% tick on the Vision B panel, misaligned rows in the small-model panels, and the human-check bar order.
- **Held-out figures** (EXP-66) are added with the same module after the analysis. Plan §7's figures line stays open until then.
- Test added. pytest: 870 passed.

## 2026-11-19 — Claude Code session (Phase 4: the analysis command, and ISLES'24 images for the Space, while the runs go)
- **`phase4 analyse`** (EXP-66, once the cells are done): DEC-31 D3 on the finished held-out cells, through EXP-63's tested `heldout_analysis.report`. It covers:
  - correctness per model and arm;
  - C9 next to EXP-37's and EXP-47's development real cells;
  - error propagation;
  - the ±20% core at rule level, plus Sonnet's two perturbed cells (merged);
  - held-out Dice, volume and crossings from EXP-62's `per_case.csv` (case column dropped);
  - the bleed flags;
  - DEC-31 §4's scenario supplement.

  A model whose cells are unfinished is listed and left out. Output goes to `runs/EXP-66_heldout_analysis/` (`report.json` and `report.md`, aliases `H-01`…, ID-checked). One test.
- **ISLES'24 images for the story** (`demo/images.py`, `python -m strokecopilot.demo.images --bank …`):
  - One **development** case, shown as its bank alias **D-10**, never by ID. It is chosen in code: a sizeable reference infarct that nnU-Net also found, and no bleed flag.
  - Four panels at the slice where the reference infarct is largest: plain CT, CBF, Tmax, and an overlay (predicted core red, Tmax > 6 s amber, reference final infarct outlined white). The caption gives the volumes: predicted core 96.2 mL against a final infarct of 108.5 mL.
  - Written into the gitignored bank (`img/`). The server serves only `story_*.png` and `story.json`; other names get a 404. The Space build now allows `.png`. Credit on the page: "ISLES'24, CC BY-NC-SA 4.0".
  - Story steps 1 and 2 show them; the schematic stays as the fallback. On screens under 900 px, each step's visual now sits under its own text.
  - Checked in the browser: images load, no console errors. Tests: the image route's whitelist; slice choice. pytest: 869 passed.

## 2026-11-19 — Claude Code session (Phase 4, step 3: EXP-65 dummy cell, then the Mac's runs)
- **Dummy cell** (Sonnet 5.5, `heldout_predicted`, Batch API, `9dd4db0` plus the `--poll` option): 20 of 20 answered; reasoner, extraction and judge batches done in 22 minutes; 0 errors and 0 format failures.
  - **Cost (estimate, batch prices):** $0.61 (reasoner $0.32, extractor $0.26, judge $0.03; 76 judge calls), as estimated.
  - **Verifier rates:** hallucination **3.2%** (19/589 claims; development real cases 3.3%), content 3.1%, critical 2/578, omissions 0. Its decisions agree with the rules applied to the contract it was given, 32/32; D3's correctness against the ground-truth rules comes in the analysis.
- **Started:** Sonnet's other three cells (`heldout_groundtruth`, `heldout_core_x0.8`, `heldout_core_x1.2`) and Qwen's two, polling every 3 minutes.
- `phase4 advance --poll N` repeats `advance` until every cell is done or waits for Colab (one command, so Parsa's permission rule covers it).
- **Spend so far in Phase 4:** Anthropic about $1.06 ($0.45 for the dummy on the Space workspace; $0.61 for this on the project workspace).

## 2026-11-19 — Claude Code session (Phase 4, step 3 prepared: the held-out reasoner runs, EXP-65)
- **Parsa's go (19 Nov)** for DEC-31 D2's runs at the estimate (Anthropic about $5.5 with contingency, OpenRouter about $0.40, Colab about 1 A100 hour).
- **`pipeline/phase4.py`, held-out run layer:**
  - **Cells:** `heldout_predicted` and `heldout_groundtruth` for all six models, plus Sonnet's `heldout_core_x0.8` and `heldout_core_x1.2` (D3's 40 extra requests; contracts perturbed by `heldout_analysis.perturbed_cases`). 14 cells.
  - **Commands:** `status`, `advance` (EXP-37's `batched.run_batched`: Sonnet, the extractor and the judge through the Batch API; Qwen through OpenRouter; open models wait for their Colab answers), and `import` (Colab's answers from Drive). Output is ID-checked.
  - **Run folder:** `runs/EXP-65_heldout_runs`. EXP-37's `paid_runs` is unchanged.
- **Notebooks 53 and 55** gain `MODE = 'heldout'` (EXP-65). It sets `ALLOW_HELDOUT=1` for the notebook only, copies the three held-out folders, answers both arms for the Gemma 27B pair and the 4B pair, saves each answer to Drive as it arrives, and prints counts per cell.
- **Tests:** `tests/test_phase4.py` (5). They cover: the cells and sensitivity arms; the Colab pass, import (idempotent) and a full advance to `metrics.json` with fake models; and open models waiting for Colab. pytest: 866 passed.

## 2026-11-19 — Claude Code session (Phase 4: where held-out code runs, DEC-31 §4)
- **Decision (Parsa, Option A):** Phase 4 code paths on the Mac may set `ALLOW_HELDOUT=1` per command, in addition to Phase 4 notebooks. It is recorded as a dated note in `docs/PHASE4_PREREGISTRATION.md` §4, with its reason and conditions, and pointed to from plan §7.
  - **Reason:** the paid-run pipeline that produced every development result runs from the Mac. Rewriting it as notebook code after unlocking would add untested code where mistakes cost money.
  - **Conditions:** set per command only; only commands that start with it (Parsa's permission rule); held-out IDs on the Mac only as file names in gitignored `runs/`; summaries ID-checked; nothing retrained or re-tuned. What is measured is unchanged.
  - **Pros and cons were weighed with Parsa before deciding:** reuse of tested code and robustness to Colab disconnects, against a small widening of the rule and IDs as file names on the laptop.
- **`pipeline/phase4.py`:** the one module for held-out work, so the permission rule can name exactly it.
  - It checks the split by sha256 and refuses without the flag.
  - Its first command, `check`, validates the 20 held-out contracts and ground-truth contracts, fact-checks the notes, and counts what the rules say on each. Counts only, ID-checked.
  - Tests: `tests/test_phase4.py` (2, fixtures posing as held-out cases).
- **Permission rule added by Parsa** (`.claude/settings.local.json`, gitignored): only `ALLOW_HELDOUT=1 .venv/bin/python -m strokecopilot.pipeline.phase4 …`.
- **`phase4 check`** (free, counts only): 20 contracts, 20 ground-truth contracts and 20 notes valid.
  - **Rules on the ground truth (D3's truth):** thrombolysis undetermined 17, yes 2, no 1; **thrombectomy no in 20/20** (occlusion site unknown in every case, so no large-vessel occlusion is established).
  - **Rules on the predicted contracts:** thrombolysis no 11, undetermined 8, yes 1; thrombectomy no 20.
  - **Rules on predicted vs truth:** 22/23 scored decisions right.
  - So held-out correctness rests on 23 scored decisions (3 thrombolysis, 20 thrombectomy "no"). The report states it plainly; the main held-out result is C9's realism check, which does not depend on it.
- **Cost estimate given to Parsa** (batch prices, from the development real-case cells): Anthropic about $4.6 (about $5.5 with 20% contingency) for 240 verified requests plus Sonnet's 40 perturbed-core ones; OpenRouter about $0.40; Colab about 1 A100 hour. Waiting for his go.

## 2026-11-19 — Claude Code session (Phase 4, step 2: held-out cases unlocked, EXP-62)
- **EXP-62** (notebook 60 in `heldout` mode, `UNLOCK = True`; Colab A100; `e9bb941`). **4 of 4 checks passed**, including "the dev check passed with these same plans and checkpoints" (plans `71b6a8cdca4d…`, all five folds at epoch 300).
  - A first attempt stopped at the guard that refuses to overwrite EXP-61: section 1 had run with `MODE = 'dev_check'`. `ALLOW_HELDOUT` was never set, and nothing was changed.
- **Written on Drive** (case IDs stay there):
  - `artifacts/vision_outputs_heldout_v1/` (20 contracts, nnU-Net five-fold ensemble, 2.3 s per case);
  - `artifacts/vision_outputs_gt_heldout_v1/` (20, the EXP-34 way);
  - `artifacts/isles24_notes_heldout_v1/` (20 notes, same generator and seed 20261105);
  - `runs/EXP-62_heldout_contracts/` (ensemble masks, `per_case.csv` with D3's vision scores, and `metrics.json` with counts only).
- **Counts (from `metrics.json`):**

  | | Predicted contracts | Ground-truth contracts |
  |---|---|---|
  | Bleed flagged | **10/20** (development 68/129) | 0/20 |
  | Occlusion site | **unknown in 20/20** | unknown in 20/20 |
  | ASPECTS | null in 20/20 | null in 20/20 |
  | Core, median | **1.85 mL** | 11.0 mL |
  | Near-empty core (< 1 mL, with the DEC-27 warning) | 7 | none |
  | Mismatch ratio null | 7 | 2 |

  The Tmax > 6 s median is 157.7 mL in both.
  - Notes: onset time known in 8, wake-up 6; not recorded: onset 6, last known well 6, pre-stroke mRS 7, INR 7.
- **What this implies for D3 (expected, not yet computed):** with the occlusion site unknown everywhere and ASPECTS null, thrombectomy truth will be mostly "no" or undetermined, as on development cases. Correctness will rest on few determined decisions, and the bleed flag (10/20) will drive most vision errors.
- **Not done:** the held-out contracts and notes were not reloaded on the Mac. Setting `ALLOW_HELDOUT=1` in a Mac shell was refused by the session's permission check, and DEC-31's text says "Phase 4 notebooks only". Parsa to decide where Phase 4's held-out code runs.
- **Plan §7 Phase 4:** the unlock line is ticked.
- **Notion sync:** Experiments: EXP-62 (done, counts above). Tasks: Phase 4 unlock (done).

## 2026-11-19 — Claude Code session (Phase 4, Task 4: the Space image checked locally)
- **Docker image** (`docker build` of `demo/space/build/space`, 2.52 GB, python 3.12-slim, CPU PyTorch).
  - The first build crashed at start-up: `nibabel` and `pandas` were missing (`builder` → `confidence` → `isles24_survey`). Both are now in the `space` extra.
  - The rebuilt image:
    - starts in about 4 s with live off and about 8 s with live on (the bge retrieval index loads);
    - `/healthz` lists 175 bank cases and 154 fallback cases;
    - serves the page (200) with the CSP and the not-for-clinical-use header;
    - **contains no `splits/` folder and no `folds.json` anywhere** (searched the whole filesystem).
- **Load test against the container** (live off): 50 of 50 visitors got a result, 0 errors, at most 0.64 s.
- **Live failure path with an invalid key** (no spend: the API refuses it): the run stopped at the reasoner step and fell back to the closest precomputed case with its reason. Spend counted $0.00, the failure time was recorded in `/healthz`, and the key does not appear in the logs.
- **Fixed:** "note" and "rules" progress events were sent twice (by the server and by the live run); the server now sends them once. Test updated. pytest: 861 passed.
- Not run against a real key in the container. EXP-64 ran the same live code from the Mac; the Space itself gets a short live check after the push, on Parsa's word.

## 2026-11-19 — Claude Code session (Phase 4, Task 2 done: EXP-61, the held-out contract path checked on development cases)
- **EXP-61** (Colab A100-SXM4-40GB, `5b5594b`; torch 2.14.1, nnU-Net 2.8.1; `runs/EXP-61_heldout_path_dev_check/` on Drive, counts only). **7 of 7 checks passed:**
  1. the five checkpoints agree with each other, their `debug.json` and their recorded `plan_3d_fullres.json`;
  2. fold 0's recorded validation masks (EXP-22) came from this fold 0 checkpoint (sha256 match);
  3. **DEC-31's rebuild matches:** `nnUNetv2_plan_and_preprocess --no_pp` on the 129 development cases gives the same 3d_fullres configuration, intensity properties and transposes (no difference);
  4. **single-fold reproduction:** 10 development cases (2 per fold, seed 20261119), re-predicted by their validating fold, reproduce the recorded masks: 8 identical, at most 3 voxels different, Dice ≥ 0.9998, volume within 0.01 mL. The differences are GPU arithmetic;
  5. **the five-fold ensemble** runs at 3.0 s per case. That was on development cases, where 4 of 5 folds trained on each case, so it is a path test only;
  6. **the held-out code rebuilds all 129 EXP-33 contracts** with 0 differences in `haemorrhage.present`, `core_ml`, `penumbra_ml`, mismatch, occlusion site, ASPECTS and warnings. `confidence.core` differs by design: the all-129 mapping is in-sample on development cases;
  7. **Vision B reproduces EXP-16:** 68 flags; largest confidence difference 0.0037.
- **Times:** fetch 62 s, assembling the results folder 86 s, rebuild 90 s, predictions 153 s, 129 contracts 245 s.
- **What this means for unlock day:** notebook 60 in `heldout` mode runs the same path, with the ensemble, on the 20 held-out cases. It refuses to run unless this dev check passed with the same plans and checkpoint hashes.
- **Plan §7 Phase 4:** ticked lines for the held-out contract path (EXP-61) and the D3 code (EXP-63).
- **Before unlocking (DEC-31 §1):** EXP-41 consensus done; freeze list in force; RSNA test check done (EXP-60); D1 code done (EXP-61); D3 code done (EXP-63). Still open: the Space (Task 4: the page is built, the image is being checked locally).
- **Notion sync:** Experiments: EXP-61 (done, 7/7). Tasks: Phase 4 held-out contract path (done), D3 code (done).

## 2026-11-19 — Claude Code session (EXP-61's first try stopped; runs now named exactly)
- **EXP-61, first try** (Colab A100): stopped in section 4 with "too many values to unpack". The pattern `EXP-*_nnunet_fold0` matched two folders on Drive: `EXP-14_nnunet_fold0` (the 300-epoch model) and the older `EXP-3_nnunet_fold0` (150 epochs, no final checkpoint).
  - The same loose pattern would have picked fold 0's reference masks from the wrong place.
  - My local check hadn't caught it, because it matched on `checkpoint_final.pth`, which only EXP-14 has.
  - Nothing was read past the split check, and nothing was written.
- **Fix:** `contract/heldout.py` names the runs exactly.
  - `CHECKPOINT_RUNS` = EXP-14 and EXP-18 to EXP-21.
  - `VALIDATION_RUNS` = the same, except fold 0 → `EXP-22_nnunet_fold0_masks`. EXP-14 kept no validation masks; EXP-22 wrote them from EXP-14's `checkpoint_final.pth`.
  - `fold_path`, `validation_path` and `recorded_plan_path` raise if a folder is missing.
  - Notebook 60 uses these everywhere (no pattern remains). It also checks that EXP-22's recorded checkpoint sha256 equals fold 0's checkpoint.
  - **Checked on Drive:** each fold has its checkpoint, `debug.json`, recorded plan and validation masks (26/26/26/26/25).
  - Test added. pytest: 861 passed.

## 2026-11-19 — Claude Code session (Phase 4, Task 4: the Space's page, chart data and build)
- **The page** (`demo/space/static/`: `index.html`, `styles.css`, `app.js`, `charts.js`). Plain HTML/CSS/JS, no third-party code or CDN, matching the server's CSP. Every server value is set as text, never as HTML.
  - **Banner:** "Research prototype — not for clinical use." is sticky at the top of every screen, cannot be dismissed, and is repeated in the footer.
  - **Story:** six scroll-driven steps (perfusion, core and penumbra, bleed check, rulebook, reasoner, verifier), each with measured numbers (EXP-25/63, EXP-60, EXP-50) and a small visual. Step 1's visual is a labelled schematic for now; the ISLES'24 images are not yet in.
  - **Case explorer:** the 5 fixtures, 150 scenarios and 20 real development cases (by alias), with the six-model switch and, for real cases, the predicted or ground-truth vision arm.
    - Each claim is clickable, and its sources light up: contract rows, note lines (marked), and rulebook passages with ID and pointer.
    - Flagged claims show their sub-type, whether they would change a decision, and the reason.
  - **Results:** six chart cards, each with a hover tooltip and a table view:
    - C9 dot-and-interval chart, with all / content / critical;
    - the rulebook effect (dumbbell);
    - the human check (tiles and agreement by verdict);
    - the tuned 4B (rate on held-out routes, and thrombectomy yes caught);
    - Vision B on the RSNA test split (tiles, reliability, subtypes), with Item 2;
    - the ±20% core sensitivity.

    Model colours are the reference palette's categorical slots in fixed order. Light and dark themes are each chosen, plus a toggle.
  - **Builder:** 16 controls built from `/api/meta`'s bounds and choices; 155 presets to start from; live progress per step with timers; and the result with the reason when it fell back.
- **Chart data** (`demo/charts.py` → `charts.json`): counts only, gathered from EXP-41, EXP-42, EXP-48, EXP-49, EXP-50, EXP-60 and EXP-63. **`passages.json`:** the 47 own-words passages. The server gains `/api/bank/charts` and `/api/bank/passages`.
- **Space build** (`demo/space/Dockerfile`, `demo/space/README.md` with Space metadata and `license: cc-by-nc-sa-4.0`, and `demo/space_build.py`):
  - It assembles the exact tree to push: the page, the bank, and a package copy **without `splits/`** (the copied pyproject drops it from the wheel).
  - It refuses the tree on: any of the 149 case IDs (compared in code, never printed); RSNA IDs; key patterns; e-mail addresses; disallowed file types; or a split file.
  - Built locally: 317 files, 19.1 MB, clean.
- **Found and fixed (privacy):** two source files hard-coded the Drive-for-desktop path, which contains Parsa's e-mail address (`isles_false_alarms.py`, EXP-50's default; the new `bank.py`). A new `config.local_drive_root()` finds the folder by pattern (as `compute/pod.py` already did) or from `STROKECOPILOT_DRIVE`. No address remains in `src/`, and the Space build now refuses any address.
- **Checked in the browser** (local preview, live off, no key): no console errors; the hero, story, explorer and results render.
  - Claim clicks highlight contract rows, note lines and passages.
  - The builder's run streams its steps and falls back with its reason.
  - Phone width (375 px) has no sideways scroll; it was 531 px before `fieldset`/`select` min-width fixes.
  - Light and dark themes both checked.
- **Tests:** server endpoints for charts and passages; the Space build's pyproject edit and its scan (split folder and file, e-mail, case ID, file type). pytest: 860 passed.
- **Not done yet:**
  - the Docker image has not been built yet (building it locally next);
  - the load test;
  - ISLES'24 map images for the story;
  - the Space itself: nothing has been pushed.
- **For Parsa:** the package's code comments and docstrings name team members (first names, as decision credits) and the GitHub repository. A public Space shows its source. Keep it (normal for a course project), strip them in the build, or use "protected" visibility.

## 2026-11-19 — Claude Code session (Phase 4, Task 4: EXP-64, the Space's live dummy)
- **EXP-64** (Mac, Parsa's shell with the Space's dedicated key; `runs/EXP-64_space_live_dummy/` on the Mac and Drive; synthetic inputs, no IDs). 5 of 5 live runs succeeded. **$0.4485 in all**, Anthropic, Space workspace.

  | Run | Cache | Time | Cost | Claims flagged | Judge calls |
  |---|---|---|---|---|---|
  | 1 | cold | 77 s | $0.100 | 2/29 | 7 |
  | 2 | warm | 45 s | $0.058 | 0/31 | 3 |
  | 3 | warm | 55 s | $0.052 | 1/28 | 6 |
  | 4 | warm | 68 s | $0.085 | 0/37 | 3 |
  | 5 | cold (after 330 s) | 153 s | $0.154 | 2/53 | 12 |

  - **By step:** reasoner 13–36 s (2,100–5,600 output tokens); extractor 12–21 s; judge 14–97 s.
  - **The prompt cache works:** the 11,977 rulebook tokens are written on cold runs and read on warm ones.
  - **Decisions:** Sonnet agreed with the rules on every determined decision (run 2's thrombectomy is undetermined: M2).
  - **Against DEC-32's estimate** (warm $0.06–0.08, cold $0.10–0.12): warm as estimated; one cold run at $0.15 was above it, because it had a longer answer.
- **Changes from it:**
  - **Concurrent judge calls in live runs.** `trace.trace_all` gains `map_fn` (default `map`, so every existing run is unchanged), and `demo/live.py` passes a 6-worker pool. Each claim's trace depends only on that claim and read-only inputs; a test checks the traces are identical to sequential ones. Expected to cut a 12-call judge step from about 97 s to about 20 s; to be confirmed on the Space.
  - `ClaudeJSON` updates its usage under a lock, so overlapping calls can't drop counts that the spend cap relies on. A test with 400 overlapping calls covers it.
  - The server's cold-run reservation goes from $0.12 to **$0.16** (the dearest measured).
  - Per-call timeouts (reasoner 150 s, extractor 120 s, judge 45 s) and the 300 s deadline stay. The largest measured step was well under each.
  - `docs/DEC-32_PUBLIC_SPACE.md` §7 records the measured figures.
- pytest: 858 passed.
- **Spend so far in Phase 4:** Anthropic **$0.45** (the dummy, Space workspace); Kaggle free (EXP-60); Colab EXP-61 still to run.
- **Notion sync:** Experiments: EXP-64 (done, figures above).

## 2026-11-19 — Claude Code session (Phase 4, Task 4: the Space's bank and server)
- **`demo/bank.py`:** the precomputed bank. `python -m strokecopilot.demo.bank --out demo/space/build/bank` builds it from `runs/` and Drive.
  - **Presets:** the 5 fixtures and the 150 scenarios, each with the saved, verified answers of all six models (EXP-37 rulebook arms rescored with points 19–20; EXP-47), shaped by `live.view`. Each preset carries the builder inputs it corresponds to (all 155 map), and the features used for fallback matching.
  - **Development cases:** EXP-37's 20 real cases as `D-01`…`D-20`, with the predicted and ground-truth arms. The case ID is replaced everywhere by its alias.
  - **Leak scan:** case IDs, RSNA IDs and key patterns, run on the whole folder.
  - **Built locally:** 175 cases, 17 MB, scan clean. Sonnet's answer is available for 154 presets (one scenario has no verified Sonnet answer).
  - `demo/space/build/` is gitignored: it holds real-case notes by alias.
- **`demo/server.py`** (FastAPI):
  - Endpoints: `POST /api/run`; `GET /api/run/{id}/events` (server-sent events with progress per step, then exactly one result); `/api/meta`; `/api/bank/index`; `/api/bank/case/{id}` (IDs whitelisted); `/healthz`; and the static page.
  - **Flow:** validate the inputs (422 with the field named, never echoing input), then the cache, then a live slot, then the spend reservation, then the visitor's limit, then a live run in a worker thread.
  - **Every other path ends in the nearest precomputed scenario, with its reason:** live switched off, slots busy, budget spent, visitor limit reached, a failure at any step, or the watchdog (deadline + 30 s).
  - A failed run's spend is still counted, and the slot is always released.
  - **Headers:** CSP `default-src 'self'` (no third-party anything), `nosniff`, `no-referrer`, and a not-for-clinical-use header.
  - Settings come from Space variables: `LIVE_ENABLED`, `DAILY_CAP_USD`, `PER_HOUR`, `PER_DAY`, `MAX_CONCURRENT`.
- **Bug found and fixed in the smoke test:** features read back from JSON hold lists, so `nearest` reported the decisions as "different" for a correct match. It now normalises both sides; a test covers it.
- **Smoke test (local, live off, no key):** `/healthz` lists 175 cases and 154 fallback cases; a builder run streams note → rules → result and falls back to a preset with the same decisions (yes/yes).
- `pyproject.toml` gains a `space` extra (fastapi, uvicorn, anthropic, numpy, sentence-transformers).
- **Tests:** `tests/test_demo_server.py` (8: live then cached; fallback per failing step with spend and slot handling; switched off; visitor limit per address; spend cap; busy slots; watchdog; refused inputs; bank, meta, health and headers) and `tests/test_demo_bank.py` (4). pytest: 856 passed.
- **Not done yet:** the page (story, explorer, charts, builder UI), the Dockerfile and Space README, the chart data, ISLES'24 map images, and the load test.

## 2026-11-19 — Claude Code session (Phase 4, Task 1 done: EXP-60, Vision B on the RSNA test split, opened once)
- **EXP-60** (Kaggle T4 ×2, one GPU used, `63b3390`, 8.8 min, free). Run 1 was a dry run on validation (`689d616`, 8.2 min). Run 2 opened the test split once, after the validation gate passed.
- **Checks:**
  - split fingerprint as EXP-6;
  - 2,180 test scans, as EXP-6 recorded;
  - test bleed rate 0.4119, as DEC-22;
  - validation pass: scan AUC 0.9772, sensitivity 0.959, specificity 0.8819.

  Validation sensitivity is one scan short of 0.96 (843 of 879): the threshold sits on a bleed scan's score and is stored rounded (0.174984), so that scan falls just below. The contracts apply the same stored value.
- **Test figures (DEC-31's pre-registered set; patient bootstrap, 2,000, seed 20261108):**

  | Measure | Test | 95% interval |
  |---|---|---|
  | Scan AUC | **0.9805** | 0.9743–0.9859 |
  | Sensitivity at 0.175 | **0.9621** (864/898) | 0.9476–0.9752 |
  | Specificity at 0.175 | **0.8853** (1,135/1,282) | 0.8666–0.9038 |
  | ECE (10 bins) | **0.0151** | |

  - Per 100: 3.8 missed bleeds per 100 bleeds; 11.5 false alarms per 100 non-bleeds.
  - By subtype (caught / bleeds, bootstrap interval): epidural 27/28 (0.964, 0.88–1.0); intraparenchymal 521/534 (0.976, 0.961–0.988); intraventricular 368/370 (0.995, 0.986–1.0); subarachnoid 341/353 (0.966, 0.940–0.986); subdural 400/409 (0.978, 0.961–0.993).
  - Extras: slice AUC 0.9776; log loss 0.1655.
  - The reliability diagram's middle bins hold 28–73 scans each, so they are noisy. The largest gaps are 0.65 → 0.82 (28 scans) and 0.86 → 0.74 (73 scans).
- **Reading:** the test figures match EXP-17's cross-fitted validation estimate (sensitivity 0.961, specificity 0.878), so DEC-28's operating point holds on unseen patients. Vision B's problem is ISLES'24's domain shift (68 of 129 flagged), not RSNA performance.
- **Files:** `runs/EXP-60_visionB_rsna_test/` (`config.json`, `metrics.json`, `reliability_test.png`) on the Mac and Drive. Checked: no `ID_` strings. Plan §7 Phase 4 gains a ticked line. **The test split is not to be opened again.**
- **Notion sync:** Experiments: EXP-60 (done, figures above). Tasks: Phase 4, Vision B RSNA test split (done).

## 2026-11-19 — Claude Code session (Phase 4, Task 4 begins: the Space's live core, DEC-32)
- **`demo/builder.py`:** the scenario builder.
  - Structured inputs only: numbers in fixed ranges with fixed steps, and fixed choices. Unknown fields and any text are refused (`InputError` names the field and never echoes input).
  - Bounds are the evaluated ranges: the scenario grids, the fixtures and the development volumes.
  - Consistency rules: basilar has no ASPECTS, penumbra ≥ core (DEC-7), INR only with warfarin, sleep time only for a wake-up stroke.
  - `build` makes the contract and note the way `scenarios._draw` and `notes.render` did. Hidden fields are seeded from the inputs' hash, so the same inputs always give the same case (the cache key).
  - `nearest` finds the closest precomputed case for a fallback: same rule decisions first, then reasons, then categories, then normalised numbers.
- **`demo/limits.py`** (in memory; the workspace limit is the hard cap):
  - per-visitor limits by salted hash: 3 runs an hour, 10 a day;
  - a $1 daily spend cap, reserving a cold run's estimate of $0.12;
  - 3 concurrent slots;
  - an LRU result cache.
- **`demo/live.py`:** one live run with EXP-37's settings, sent live instead of batched.
  - Sonnet: effort high, max_tokens 20,000. Extractor: Sonnet, medium. Judge: Haiku `judge_v3`, τ 0.6.
  - Timeouts per call: reasoner 150 s, extractor 120 s, judge 45 s. A 300 s deadline for the run. The SDK retries twice.
  - Progress per step goes to a callback. Any failure raises `LiveFailure` naming its step.
  - `view` is the page's shape for one verified answer, shared by live runs and the precomputed bank.
- **`demo/dummy.py`:** DEC-32 §7's dummy (EXP-64): five live builder runs covering five routes (late M1 thrombectomy, early thrombolysis with M2 undetermined, a bleed, a large core, a wake-up DAWN case).
  - Runs 1–4 run back to back (1 cold, 2–4 warm). It then waits 330 s for the prompt cache to expire, and run 5 is cold.
  - It records the cost and seconds per step. Synthetic inputs, so no IDs.
  - Not run yet: it needs the Space's dedicated key in Parsa's shell. About $0.50 (estimate).
- The Gradio stub `demo/app.py` was removed (DEC-32).
- **Tests:** `tests/test_demo.py` (15), with fake models (gold answers, a supporting judge). They cover bounds, refusals and determinism; that every scenario fits the bounds; matching; the limits with a fake clock; and a live run's steps, failures per step and its deadline. pytest: 844 passed.
- **Notion sync:** Experiments: EXP-64 (planned: Space live dummy, five runs).

## 2026-11-19 — Claude Code session (Phase 4, Task 3: the D3 analyses, code and development check)
- **`pipeline/heldout_analysis.py`:** DEC-31 D3, written before unlocking.
  - **Truth:** `rules.decide` on the ground-truth contract and the fact sheet. Undetermined truths and missing answers are excluded and counted.
  - **`correctness`:** per decision and pooled, with a case bootstrap (10,000, seed 20261104).
  - **`c9_side_by_side`:** C9's metrics, held-out next to development, unpaired.
  - **`error_propagation`:**
    - a vision error if the rules on the predicted contract already miss the truth, otherwise a reasoning error;
    - "caught" means a flagged claim in the decision's own fields (`rule_ids`, `blocking_factors`) or in the shared rationale and uncertainties, because the schema ties rationale to neither decision. The own-fields count is reported too;
    - for each vision error, the contract fields that differ: core across a cut-off, bleed flag, occlusion site.
  - **`perturb` / `rule_sensitivity`:** core ×0.8 and ×1.2, with mismatch and the near-empty warning rederived and the result schema-validated.
  - **`perturbed_cases` / `model_sensitivity`:** Sonnet's 40 requests.
  - **`vision_summary`:** Dice, volume error, cut-off crossings, bleed flags.
  - **`report` / `markdown`:** the whole report, aliases only (`check_no_ids` before saving).
- **Tests:** `tests/test_heldout_analysis.py` (11), on the five fixtures with hand-made answers and traces. pytest: 828 passed.
- **EXP-63 (Mac, free; `runs/EXP-63_d3_dev_check/`, Mac and Drive, counts and aliases only):** D3 on EXP-37's and EXP-47's 20 development real cases, plus all 129 development contracts for rule sensitivity and vision. This is a code check, not a result.
  - **Reproduces the development numbers exactly:** Dice 0.215, absolute volume error 23.5 mL, wrong side of 21/31/51/70 mL in 30/28/20/17 cases (EXP-25), and 68/129 bleed flags (EXP-16).
  - **Truth on real cases is mostly undetermined or "no":**
    - thrombolysis is undetermined in 18 of 20 cases (no ASPECTS, mostly unknown occlusion site);
    - thrombectomy is "no" in 17 of 20.

    Held-out cases will look the same, so correctness will rest on few decisions, and on thrombectomy "no".
  - **Rule sensitivity:** ×0.8 and ×1.2 change **0 of 258** rule decisions across the 129 development contracts. The core does not decide any of them; the unknown occlusion site, ASPECTS and the bleed flag do. The pre-registered ±20% analysis is therefore likely to show no flips on held-out cases either, and Sonnet's 40 extra requests (about $1.2) would probably add little.
  - **Error propagation on development (predicted arm):**
    - Sonnet, Qwen and the tuned 4B each make 1 wrong decision. Each is a vision error (a false bleed flag), and the verifier misses all three, as expected: the claims are faithful to a wrong contract.
    - The base Gemma 3 4B makes 16 reasoning errors (thrombectomy 1/17), all caught.
  - Qwen's predicted arm has 11 of 20 format failures (schema: extra properties), as already in C9 (0.55). This is likely to recur on held-out cases.
- **DEC-31 amended before unlocking (Parsa, 19 Nov):** the pre-registered real-case ±20% analysis is kept unchanged: the rules on the 20 held-out contracts, and Sonnet's 40 requests, which test whether the model reacts to a change the rules ignore. A supplementary rule-level ±20% on the 150 synthetic scenarios is added, reported apart (`docs/PHASE4_PREREGISTRATION.md` D3 and §4; `heldout_analysis.scenario_sensitivity`, one test).
  - **Result (free, in EXP-63's report):**

    | Core | Decisions changed | Yes/no flips | Details |
    |---|---|---|---|
    | ×0.8 | 9 of 300 (3.0%) | 4, all thrombolysis no → yes | 3 to undetermined, 2 from it |
    | ×1.2 | 10 of 300 (3.3%) | 0 | 5 to undetermined, 5 from it; 3 of them thrombectomy yes → undetermined |

  - The share depends on the scenario grid, which has only six core/penumbra profiles, so it shows *that* a core error can decide a case, not how often in practice.
- **Notion sync:**
  - Experiments: EXP-63 (done: D3 code check on development cases, with the scenario supplement).
  - Decisions: DEC-31 note (supplementary scenario sensitivity, added before unlocking).

## 2026-11-19 — Claude Code session (Phase 4, Task 2: the held-out contract path, D1, code)
- **Plans from the checkpoints (Parsa, 19 Nov):** each fold's `checkpoint_final.pth` carries the plans and `dataset.json` it was trained with. These are the authority, and DEC-31's rebuild is a cross-check.
  - **Checked on the Mac from Drive (no data, no IDs):** all five agree (plans and `dataset.json` hashes identical; trainer `nnUNetTrainer_StrokeCopilot`; epoch 300; mirroring on all three axes).
  - For every fold, the checkpoint's plans, 3d_fullres configuration and `dataset.json` equal that fold's `debug.json`, and the configuration equals the `plan_3d_fullres.json` the run recorded.
- **`contract/heldout.py`:** every contract field computed case by case, as the development runs did:
  - **core:** nnU-Net (the ensemble with `-f 0 1 2 3 4` and `checkpoint_final.pth`; a single fold for the dev check);
  - **penumbra and occlusion evidence:** `thresholds.threshold_masks` and `volumes.occlusion_evidence`, as EXP-1;
  - **haemorrhage:** EXP-16's 5 mm slabs, EXP-6 and DEC-28;
  - **confidence:** `confidence.case_confidence` with EXP-32's mapping (DEC-27).

  It also provides:
  - ground-truth contracts the EXP-34 way;
  - the results folder assembled from the checkpoints, with `check_folds` and `plan_differences` for the rebuild;
  - the case-set guard: development, or exactly the 20 held-out cases with `ALLOW_HELDOUT=1`;
  - a field-by-field comparison with EXP-33, counts only;
  - output that never overwrites.

  If the threshold core cannot be computed, DEC-27's distance mapping goes on without it, because it does not use it.
- **`notebooks/60_heldout_contracts.ipynb` (Colab A100), one code path with two modes:**
  - **`dev_check` (EXP-61, before unlocking):** six checks:
    1. the five checkpoints agree, with their `debug.json` and recorded plans;
    2. DEC-31's rebuild (`nnUNetv2_plan_and_preprocess --no_pp`) matches;
    3. 2 cases per fold, re-predicted by their validating fold, reproduce the recorded validation masks;
    4. the ensemble runs on those 10 and is timed;
    5. the held-out code rebuilds all 129 EXP-33 contracts field by field (all but `confidence.core`, which is in-sample here);
    6. Vision B reproduces EXP-16's 68 flags.
  - **`heldout` (EXP-62):** needs `UNLOCK = True` and a passed dev check with the same plans and checkpoint hashes. It writes `vision_outputs_heldout_v1` and `vision_outputs_gt_heldout_v1` on Drive.
- **Tests:** `tests/test_heldout_contracts.py` (18) on synthetic data. Among them, `vision_a_inputs` equals EXP-1's own `baseline_row` on a synthetic case. pytest: 817 passed.
- **Blocked, rightly:** a local check that counted held-out cases in the tarball read the held-out list, and the permission check stopped it. Nothing was printed, and it isn't needed.
- **Not run yet:** the code must be pushed, then Parsa runs notebook 60 in `dev_check` mode. Spend so far: $0 (Colab: the dev check's A100 time, to be measured).
- **Notion sync:** Experiments: EXP-61 (planned: held-out path dev check), EXP-62 (planned: held-out contracts, after unlocking).

## 2026-11-19 — Claude Code session (Phase 4, Task 1: Vision B's RSNA test-split check, code)
- **`vision/haemorrhage/final_check.py`:** DEC-31 §1 item 3's figures for one split, with nothing chosen. It uses EXP-6's top-2 aggregation and Platt calibration with DEC-28's 0.175 (`infer.contract_calibrator`). It reports:
  - the scan AUC;
  - sensitivity and specificity, with patient-bootstrap intervals (2,000 replicates, seed 20261108, whole patients; one set of resamples serves every figure);
  - sensitivity per subtype, as counts with Wilson and bootstrap intervals;
  - ECE with 10 bins.

  Extras, labelled as such: slice AUC, Wilson intervals, counts, log loss and reliability bins. `validation_check` gates the opening: on validation, scan AUC within 0.005 of EXP-6's 0.977 and sensitivity at 0.175 of at least 0.955.
- **`notebooks/25_visionB_rsna_test.ipynb` (Kaggle T4, free):**
  - It runs the same report on validation first. The test split is opened only if that check passes and `OPEN_TEST_SPLIT = True`, and `ALLOW_RSNA_TEST` is set only for the moment the rows are taken.
  - Run 1 is a dry run (validation only); Run 2 opens the test split, once.
  - Guards: the split fingerprint must match EXP-6, the test scan count must match EXP-6's record if it has one, and the ID check runs before saving.
  - The experiment ID is set to EXP-60, pending Parsa's confirmation of the numbering.
- **Tests:** `tests/test_final_check.py` (9), on synthetic slices. They check the weighted bootstrap AUC against the rank AUC (with ties and with duplicated rows), the counts and subtypes, the threshold on the calibrated scale, determinism, that no IDs are saved, refusals, and the validation gate. pytest: 799 passed.
- **Not run yet:** the notebook installs the package from `main`, so the code must be committed and pushed before Run 1.
- **Notion sync:** Experiments: EXP-60 (planned: Vision B RSNA test split, once), pending the numbering decision.

## 2026-11-19 — Claude Code session (Phase 4, Task 0: DEC-32, the public Space)
- **Phase 4 started** in its own session, from DEC-31. `splits/folds.json` checked by sha256 only (matches). No held-out case is unlocked.
- **DEC-32 approved as drafted by Parsa (19 Nov):** the demo is a public Hugging Face Space (`docs/DEC-32_PUBLIC_SPACE.md`), replacing the Gradio app run from Colab.
  - **Design:** a Docker Space on CPU basic (FastAPI server, static page) with a story, a case explorer with clickable claims, results charts, and a live scenario builder. The builder takes structured inputs only, runs Sonnet 5.5 (prompt v3, rulebook) and the verifier with EXP-37's settings, and falls back to the nearest precomputed scenario.
  - **Choices a–f as recommended:** public; a private staging Space first; HF sign-in off at launch, behind a switch; continuous controls within the evaluated ranges; a six-model switch in the explorer; limits in memory, with the workspace limit as the hard cap.
  - **Spend limit:** kept small. Parsa expects fewer than 20 live runs, so about $5 a month is suggested for the workspace limit (not $20), with a $1 server daily cap.
- **Cost per live run, measured from EXP-37's logged usage** (Sonnet `scenarios_rulebook`, live prices): about $0.08 with a warm cache (reasoner $0.044, extractor $0.031, judge $0.003), about $0.11–0.12 cold. Latency is not measured yet (batch runs); the dummy measures it.
- **ISLES'24 licence rechecked on Zenodo (record 17652035, v7):** CC BY-NC-SA 4.0, so images may appear with attribution. The Zenodo page gives the dataset paper as arXiv:2408.09259, which is an unrelated paper; `docs/DATA_USE.md`'s 2408.11142 is right.
- **Amended to match:**
  - plan §2 (DEC-32 row), §4.4 (the Space's secret), §7 Phase 4 (the demo task and the acceptance line) and §8.5;
  - DEC-31 (`docs/PHASE4_PREREGISTRATION.md`): §1 item 4, D4 step 9, §3, and a new §4 for amendments, dated before unlocking;
  - CLAUDE.md's secrets line.
- **Open questions for Parsa:**
  - DEC and EXP numbering between the two sessions (Item 1 to take DEC-33; Phase 4 to use EXP-60 onward);
  - D1: may the plans saved in each `checkpoint_final.pth` be the authority, with DEC-31's rebuild as the cross-check?
  - the actual Anthropic credit, from the console.
- No code changed, so pytest was not rerun. Spend this session: $0.
- **Notion sync:** Decisions: DEC-32 (new, decided 19 Nov 2026); DEC-31 (note: demo step amended by DEC-32). Tasks: Phase 4 demo task reworded, acceptance line changed.

## 2026-11-19 — Claude Code session (EXP-41 done; Phase 3 closed)
- **EXP-41 consensus:** Parsa settled the 28 disagreements as third rater on `consensus_parsa.html` (`labels_parsa_2026-11-19.csv`; he sided with labeller A 12 times and B 16 times). `label_page.merge_consensus` wrote them into `runs/EXP-41_labelling/labels/final.csv` (Mac only, with the key). **S035 changed to no error** at Parsa's request, after we checked that passage AHA2026-T3-trace3 is about the TRACE-III and TIMELESS thrombolysis trials; the reason is in `note_consensus`.
- **Final agreement** (`agreement_final.json`):
  - human–human κ **0.27** (Lorna + Benny 0.05, Mohamed + Hana 0.33);
  - verifier–human agreement on all 100: **0.61** with the rescored verifier (points 19 and 20), 0.48 at the first scoring; weighted 0.87;
  - verifier-clean claims agree **43/46**;
  - the verifier caught **18 of the 21** human-judged errors, and humans confirmed **18 of its 54** flags (content 14/33, citation 4/21). The verifier's rates are upper bounds; the comparison between models stands.
  - **Not reported:** population human error rates per model. Weighted by the strata they come out at Sonnet 0.9%, Gemma 21.5%, MedGemma 17.3% and Qwen 24.5%, but with stratified-bootstrap intervals as wide as 0.5–49% (Qwen), because each clean stratum of thousands of claims is represented by about 8.
- **Plan §7:** the adjudication line is ticked. "Phase 3 closed 19 Nov 2026 (Parsa)" is under the acceptance criteria, all three met.
  - The optional per-subtype outputs and **Item 1 (BiomedCLIP, minimal)** are recorded as report-only extras that continue in this session after the close. They never touch held-out cases or the RSNA test split, and do not feed the pipeline (DEC-31).
  - Phase 4 runs in its own session from DEC-31.
- **Notion sync:** Experiments: EXP-41 (done, final agreement), EXP-42 (human check added to Notes).

## 2026-11-19 — Claude Code session (Phase 3 wrap-up check)
- **Plan §7 NLP B, the C9 line:** the rescored numbers (points 19 and 20) replace the first scoring, which is kept in the same line.
- **Phase 3 status:**
  - Open: human adjudication, waiting for Parsa's third-rater consensus on the 28 disagreements; and the optional Vision B per-subtype outputs.
  - Acceptance: the hallucination table with ≥3 models and CIs is met (EXP-42, rescored), and so are the validated contracts for all development cases (EXP-33). Adjudication agreement is pending.
- **Still unticked from Phases 0–2, for Parsa to confirm:**
  - the professors' written confirmation (DEC-1);
  - the shared Drive folder for the team;
  - a clinician's review of the rulebook;
  - `00_template`'s first T4 run;
  - onboarding.
- **Found for Phase 4, before the held-out cases are unlocked:**
  - DEC-25 leaves open how the five fold models are combined for held-out cases.
  - The plan says "× 3 models", while there are now four paid models plus DEC-30's two small ones.
  - Notes exist only for the 129 development cases.
  - `demo/` does not exist yet.
  - Nothing implements the error-propagation or ±20% core analyses yet.
- **`docs/PHASE4_PREREGISTRATION.md`, approved as drafted by Parsa (19 Nov) as DEC-31.** It is in plan §2, and the §7 Phase 4 task links to it. It was written before any held-out case is opened. Held-out set: 20 cases, counted by sha256-checked split only.
  - **To close first:** the EXP-41 consensus; the freeze list; Vision B's one RSNA test-split check (DEC-22, now that EXP-6 + DEC-28 is final); and the code, tested on development cases only.
  - **D1:** a five-fold nnU-Net ensemble for held-out cores, with everything else as for development cases. `plans.json` and `dataset.json` are not on Drive (only `fold_N/` was kept) and must be rebuilt and matched to each fold's `debug.json` first.
  - **D2:** all six reasoner models, with the rulebook, in predicted and ground-truth vision arms: 240 requests.
  - **D3:** correctness; C9's metrics next to EXP-37's development real cases; error propagation defined mechanically by `rules.decide` on the predicted contract (vision against reasoning error, verifier caught against miss); the ±20% core sensitivity check, rule-level and for Sonnet; held-out Dice, volume and cut-off crossings.
  - **D4:** the order of work.
  - **Estimates:** Anthropic about $5–6, OpenRouter under $1, Colab A100 about 1–2 h, Kaggle free.

## 2026-11-19 — Claude Code session (Item 2: EXP-51 and EXP-50, the ISLES'24 read-out)
- **EXP-51** (Kaggle T4, 1f0849f, 11.6 min, free): EXP-49 rerun with the threshold export, and its selection is identical field by field.
  - "none" full-split threshold 0.174984 on EXP-6's calibration, **exactly DEC-28**.
  - gate(0.55): the same raw cut (0.6268).
  - Files in `runs/EXP-51_visionB_false_alarms/` (Mac and Drive).
- **EXP-50** (Mac MPS, 1f0849f, 1.4 min; `runs/EXP-50_visionB_isles_false_alarms/` on Drive only, because `per_case.csv` holds case IDs):
  - All 129 development scans scored; "none" **reproduces the 68 flags** (52.7%).
  - **gate(0.55) flags 64:** it clears 4 and adds none, McNemar exact p = 0.125. Three of the 4 are skull-base slabs (height 0.17–0.19, relative brain fraction 0.47–0.54).
  - Other candidates: gate(0.25) and gate(0.40) 68; gate(0.70) 65; the adjacency rules 63–64 but swap 9–14 flags each way; both(0.70) 77.
  - **The flagged scans' top slab is mostly in brain tissue:** median relative brain fraction 0.87 (none below 0.40), median height 0.38. Grad-CAM puts 76% of the heat in brain tissue (median of 136 slabs); 2 of 136 slabs have more heat on bone than brain. 16% of flagged scans' top two slabs are neighbours, against 25% of unflagged.
  - **Reading:** skull-base slabs explain only a few of ISLES'24's false alarms. Most come from single hot slabs in the brain tissue of acute-stroke scans, a domain shift no slab rule fixes. EXP-35's 12 of 23 described the borderline band only.
- `isles_false_alarms.gradcam_figure`: row spacing fixed (the second row's captions were hidden), and the figure was redrawn from the same scans.
- `docs/ITEM2_FALSE_ALARMS.md`: results section added.
- **Decision (Parsa, 19 Nov): DEC-28 stands.** gate(0.55) is not adopted. Item 2 is reported as an explanation, with the negative result that no slab rule fixes the false alarms. Recorded in `docs/ITEM2_FALSE_ALARMS.md` and as a ticked line in plan §7 (Vision B). The contracts, the paid runs and C9 are unchanged.
- **Notion sync:** Experiments: EXP-51 (done), EXP-50 (done).

## 2026-11-19 — Claude Code session (Item 2: EXP-49's result)
- **EXP-49** (Kaggle T4, 218faf8, 12.0 min, free): the scan AUC (0.9772) reproduces EXP-6. All 10 candidates are eligible under the pre-registered margins.

  | Candidate | Sensitivity | Specificity |
  |---|---|---|
  | none | 0.9597 | 0.8812 |
  | gate(0.25 / 0.40 / **0.55** / 0.70) | 0.9597 / 0.9595 / **0.9595** / 0.9584 | 0.8813 / 0.8827 / **0.8853** / 0.8726 |
  | adjacent | 0.9605 | 0.8702 |
  | both(0.25–0.70) | 0.9585–0.9610 | 0.8637–0.8707 |

  These are means over the 20 seeds, at the 96% target.
- **Proposal: gate(0.55).** Paired specificity gain on the main seed **+0.0047 [+0.0015, +0.0086]**: 6 fewer false alarms of 1,287 bleed-free scans. It meets the adoption rule. Subtype sensitivities are unchanged within 0.001 (epidural 0.921 on 31 bleeds).
- **Reading:**
  - Requiring the top two slices to be neighbours (adjacent, both) costs specificity on RSNA.
  - The Grad-CAM reference shows why a gate can do little on RSNA. RSNA's false positives put 73% of their heat in brain tissue at mid-head (median height 0.52), as true positives do (85%). So RSNA's false alarms are not skull-base ones.
  - EXP-35 traced ISLES'24's borderline flags to skull-base slabs. Whether the gate helps there is EXP-50's question.
  - The gain is also the best of 10 candidates on the same data, so it is slightly optimistic.
- The run predates the threshold export (1f0849f), so gate(0.55)'s threshold for EXP-50 needs a rerun of notebook 24 (EXP-51).
- Files: `runs/EXP-49_visionB_false_alarms/` on the Mac and Drive, numbers only (checked for IDs).
- **Notion sync:** Experiments: EXP-49 (done).
- **EXP-41 consensus: Parsa settles the 28 disagreements as third rater** (his decision, 19 Nov), instead of each pair meeting. The report will say so.
  - `label_page.build_consensus_page`: the labelling page in a third-rater mode. It shows both labellers' labels and notes under each claim and fills `consensus` and `note_consensus`. It is still blind: no model, run or verifier verdict.
  - `label_page.merge_consensus` puts the download back into the combined sheet. It refuses a consensus on a claim the pair agreed on.
  - The page is `runs/EXP-41_labelling/consensus_parsa.html`, built by `build_consensus.py` next to it. It is local only, with the key.
  - Checked in the browser: it renders, shows the two labels, and marks a verdict.
  - Test added to `tests/test_label_page.py`; pytest 790 passed.

## 2026-11-19 — Claude Code session (Item 2: the threshold export and EXP-50's ISLES'24 read-out)
- **Gap found in EXP-49 (running from 218faf8):** `select()` did not export the thresholds a candidate needs to be applied to ISLES'24, and those can only be computed on Kaggle.
  - Fixed: `false_alarms.full_split` now gives each candidate's full-split raw threshold for 96% sensitivity, its own Platt calibration, and the threshold on that scale (numbers only). This is how DEC-28's threshold was set for "none".
  - The running EXP-49 lacks it. If its proposal is "none", nothing is lost. Otherwise one rerun of notebook 24 (about 20 free Kaggle minutes) exports it.
- **`vision/haemorrhage/isles_false_alarms.py`** (EXP-50, on the Mac):
  - Each development scan goes through EXP-16's preparation (`isles_check.prepare`: 5 mm slabs, windows, JPEG), then EXP-6's model, then the per-slab brain fraction and height.
  - Flags under "none" use DEC-28 (EXP-6's calibration and `CONTRACT_THRESHOLD`, as the contracts were built). Flags under every other candidate use the threshold EXP-49 exported.
  - The proposal against "none" is tested with an exact McNemar test.
  - Where the top slab sits: height, brain fraction relative to its scan, whether the top 2 are neighbours, and the share of top slabs at the skull base.
  - Grad-CAM on the top 2 slabs of each flagged scan: medians of heat in brain, bone, other tissue and outside, plus a figure of 8 drawn by seed.
  - `contract_check` expects the 68 flags of EXP-16 and EXP-33 under "none".
  - Run with `python -m strokecopilot.vision.haemorrhage.isles_false_alarms --out <Drive folder> [--exp49 metrics.json]`. It checks the split's sha256 and that there are 129 development cases, uses the repo's `data/local` copy (or fetches the Drive tarball by checksum), and refuses an `--out` outside Drive, because `per_case.csv` holds case IDs. It checks `metrics.json` for IDs before saving. It runs on MPS.
- **Smoke test on 3 real development scans** (nothing saved, counts only): it runs end to end (28–30 slabs per scan, Grad-CAM on 6 slabs) in 38 s including the model load, so the full 129 should take about 15–25 min (estimate). DEC-28's 0.175 is 0.627 on the raw score.
- **Tests:** `tests/test_isles_false_alarms.py` (5), on synthetic NIfTI heads through the real slab preparation.
  - A scan with a hot skull-base slab is flagged under "none" and cleared by the gate; a scan with two hot mid-brain slabs is flagged under both.
  - Also tested: the threshold inversion and the EXP-49 parsing, McNemar, the summary with no IDs, the figure, and a failed case kept with its error. Test for `full_split` added to `test_false_alarms.py`.
- pytest: 789 passed.
- **Notion sync:** Experiments: EXP-50 (planned).

## 2026-11-19 — Claude Code session (EXP-41: the four labellers' sheets, preliminary agreement)
- **Labels in from all four:** Lorna (17 Nov), Hana (17 Nov), Mohamed (18 Nov), Benny (18 Nov). Each pair labelled its 50 claims completely.
  - Merged per pair (`label_page.merge_sheets`) and combined, then joined with the key (`adjudicate.read_labels`).
  - The files are in `runs/EXP-41_labelling/labels/` on the Mac only, next to the key, which never goes to Drive.
- **Agreement between labellers, preliminary:**
  - Cohen's kappa pooled **0.27** over 100 claims: Lorna + Benny **0.05**, Mohamed + Hana **0.33**.
  - 72 claims agreed; **28 to settle** by each pair's consensus, as the guide sets out (12 and 16). `pair*_to_settle.csv` lists them; the sheets are blind, with aliases only.
  - "Error" calls out of 50: Lorna 11, Benny 3, Mohamed 23, Hana 11. Most disagreements are true-but-uncited claims, which the guide counts as errors and the more lenient labeller of each pair passed.
- **Verifier against the 72 agreed claims:**
  - Verifier-clean claims: 23/23 agree.
  - Verifier content errors: 8/19 confirmed.
  - Verifier citation errors (all "misattributed source"): **2/30 confirmed**.
  - Weighted by the sampling strata: 0.94, dominated by the clean stratum.
- **Finding to check: the verifier's misattributed-source flags look unreliable.** Several are plain mistakes. For example, the quote "Time of symptom onset: not recorded." was judged to support nothing for the claim "Time of symptom onset is not recorded", and S014, S025, S028 and S031 are the same. Citation flags are 20–31% of each model's C9 hallucination rate (EXP-42), so **C9's content rate (sub-types 1–5) is the safer headline** until the cause is found.
- **Cause, investigated the same day.** `trace.attribute` charges a declared source that backs no claim of its entry, and for a vision field or note quote "backs" means matching an input the claim's trace recorded as used. Some claims are checked without recording any input, so their correct sources cannot match:
  - **Pattern A, absence claims.** For example "Time of symptom onset is not recorded" (`used: []`): the note quote that says so is charged.
  - **Pattern B, rule or number claims checked without traced inputs.** For example "meets the basilar criteria" citing `occlusion_site`.
  - **Pattern C, inputs traced but a source matches none of them.** These are mostly real wrong citations, such as `core_ml` or `haemorrhage.present` cited for a time. A few are partial spans, for example a "Last seen well at 07:37" quote when only the time-to-imaging line was traced.
  - **In the sample's 30 flags:** A 7, B 7, C 16. A and B are the verifier's mistakes. In C the labellers mostly passed claims the guide counts as errors.
  - **Across EXP-37's headline cells**, A + B make up this share of all flags: Sonnet 26 of 260 (10%), Qwen 64 of 281 (23%), Gemma 3 27B 48 of 681 (7%), MedGemma 83 of 1,162 (7%).
  - **Without A + B, the rates would be about:** Sonnet 4.0% (from 4.4), Qwen 4.8% (from 6.2), Gemma 17.3% (from 18.6), MedGemma 17.0% (from 18.3). These are estimates; no rescore has been run.
  - **Proposed fix:** two new metric points for Parsa. Not applied at this point, because it changes the metric.
- **Points 19 and 20 approved by Parsa (19 Nov) and applied** (`docs/HALLUCINATION_METRIC.md` §9):
  - Point 19 (`trace._backs`): a confirmed absence claim is backed by the declared note quote that states the absence.
  - Point 20 (`trace.attribute`): imaging and note sources are not charged when a claim of the entry is about the patient but was checked without tracing an input.
  - Still charged: passages, general rule statements, and pattern C.
  - Tests: two new ones in `tests/test_trace.py`. One existing no-rulebook test, a general rule citing an imaging field, keeps its charge, which is why point 20 is limited to claims about the patient.
- **`end_to_end.rescore` fixed and extended:**
  - It now takes the run's `rulebook` arm and `judge_prompt` from its config. Before, a without-rulebook cell was rescored as if it had the rulebook (wrong-rule flags 35 → 82 on the first try; caught, and those files were regenerated).
  - It accepts preloaded `cases` for real-case cells.
  - It writes `traces_rescore/` after scoring, so criticality is kept; a first version wrote them before scoring and lost it (caught and fixed). A new test checks criticality matches the original traces.
  - `compare.case_records` reads the traces folder that matches its metrics file.
- **Rescore (no API calls: 0 judge calls in all 36 cells):** EXP-37's 30 cells and EXP-47's 6. Only misattributed-source flags changed; a few became uncited-rule flags where the claim still lacks its passage. Per-cell record in `runs/rescore_points_19_20.json`.
- **C9 rebuilt** in `runs/EXP-42_c9_table/rescore_points_19_20/`. Content and critical rates are unchanged.
  - Hallucination rate: Sonnet 4.4 → **4.0%**, Qwen 6.2 → **5.0%**, Gemma 3 27B 18.6 → **17.1%**, MedGemma 18.3 → **17.7%**.
  - **Qwen minus Sonnet: +1.0 pts [−0.1, +2.1]**, which now includes 0 (it was +1.7 and excluded 0). Gemma +13.2 and MedGemma +13.7 are unchanged in substance.
- **DEC-30 report rebuilt** in `runs/EXP-48_small_reasoner_report/rescore_points_19_20/`.
  - Tuned minus base: −27.3 pts over all 175 (was −27.2); **−23.5 on held-out routes** (was −24.6).
  - The tuned model is level with the 27B Gemmas on held-out routes (+2.7 and +1.6, intervals spanning 0).
  - The critical-rate rise is unchanged (+1.9 [+0.4, +3.4]). The treatment-bias finding stands.
- **EXP-41 against the rescored verifier:** 13 of the 30 sampled citation flags are now clean. On the 72 agreed claims, agreement rises **0.46 → 0.61** (weighted 0.94 → 0.95). Verifier citation errors are confirmed 2/19; most of the rest are real wrong citations (pattern C) that the labellers passed. Content errors are confirmed 8/19, and clean 34/34.
- `compare._diff` prints "n/a" for an undefined difference.
- Copied to Drive: the rescored cells, C9 and the DEC-30 report.
- pytest: 784 passed.
- **Notion sync:** Experiments: EXP-41 (preliminary agreement in Notes, still Running until consensus).

## 2026-11-19 — Claude Code session (Item 2: EXP-49's selection code and Kaggle notebook)
- **Inputs found:**
  - EXP-6's `best.pt` (94 MB) and `calibration.json` are on Drive (`runs/EXP-6_visionB_resnet50/`).
  - No per-slice probabilities were saved for either dataset. EXP-16 kept ISLES'24 scan scores and top-slab positions only. EXP-17 recomputed RSNA validation on Kaggle (75,536 slices in 202 s; 6.4 min in all on the free T4) and kept metrics only.
  - So EXP-49 and EXP-50 both recompute them.
- **`vision/haemorrhage/false_alarms.py`:** the pre-registered selection (docs/ITEM2_FALSE_ALARMS.md), numbers only:
  - `select`: for each of the 10 candidates, scan scores; the 96% threshold cross-fitted on patient halves over EXP-17's 20 seeds; mean sensitivity, specificity and sensitivity per bleed subtype; patient-bootstrap intervals at the main seed.
  - `eligible`: the 1-point and 3-point margins.
  - The proposal is the eligible candidate with the highest specificity. `paired_specificity_gain` (a paired patient bootstrap) tests it against "none"; `meets_adoption_rule` says whether it passes.
  - On Kaggle: `measure` computes brain fraction and head area per slice through the shard readers, on the loader's workers. `heights` gives each slice's height in its scan. `reference_slices` picks the hottest slice of 20 true-positive and 20 false-positive scans at DEC-28's threshold, by seed, and `reference_summary` gives Grad-CAM medians per group.
- **`notebooks/24_visionB_false_alarms.ipynb`** (EXP-49, Kaggle, free T4).
  - It reuses notebook 22's install, shard, split and prediction cells unchanged, including its scan-AUC check against EXP-6.
  - It then measures every validation slice, runs `select`, runs Grad-CAM on the 40 reference slices, and saves `metrics.json` after the ID check.
  - Dry-run locally on synthetic slices, fake shard readers and a random ResNet-18 (sections 7–9 end to end); not yet run on Kaggle.
- **Tests:** `tests/test_false_alarms.py` (6), on synthetic validation slices with skull-base false alarms and low subarachnoid bleeds.
  - A mild gate gains specificity and is proposed, and the gain's interval is above 0.
  - A gate that drops the subarachnoid slices is ruled out by the subtype rule, even though overall sensitivity holds: at a fixed target the threshold falls to catch them, so the loss shows up per subtype or in specificity, which is why the subtype rule is needed.
  - Also tested: the margins, the paired bootstrap, heights and the reference sample, and `measure` through fake readers.
- pytest: 780 passed.
- **Notion sync:** Experiments: EXP-49 (planned).

## 2026-11-19 — Claude Code session (Item 2 started: Vision B's ISLES'24 false alarms, code and pre-registration)
- **`vision/haemorrhage/explain.py`:**
  - Grad-CAM on the last convolutional block (ResNet-50's `layer4`, EfficientNet's `conv_head`) for the single logit, in batches (`grad_cam`, `explain_slices`).
  - Density masks from the windowed uint8 image both datasets share (`tissue_masks`): head above −500 HU, bone above 300 HU, and brain-density tissue (HU 0–80) at least 8 px inside the head and 2 px from bone.
  - Heat shares in brain, bone, other head tissue and outside (`heat_shares`).
  - Per slice, the brain fraction and the height in the head (`slice_measures`).
  - The masks are density rules, not a segmentation: they are measured the same way on both datasets.
- **`vision/haemorrhage/slab_rules.py`:** the candidate scan-score rules, fixed before any result: "none" (EXP-6's top-2 mean), "gate" (slices below `ratio` × the scan's largest brain fraction left out, falling back to "none"), "adjacent" (top 2 must be neighbours), and "both" (falling back to "adjacent"). The grid is ratio ∈ {0.25, 0.40, 0.55, 0.70}, 10 candidates. `scan_table` orders slices by `z_mm`.
- **`docs/ITEM2_FALSE_ALARMS.md`:** the pre-registration, approved by Parsa on 19 Nov.
  - Selection on RSNA validation only, as in EXP-17: 96% target cross-fitted on patient halves, 20 seeds.
  - **Eligibility:** sensitivity no more than 1 point below "none", and no subtype more than 3 points below.
  - **Proposal:** the eligible candidate with the highest specificity.
  - **Adoption** needs the specificity gain's paired interval above 0, and Parsa's decision (it amends DEC-28 and rebuilds the contracts before Phase 4).
  - ISLES'24 flags are read out, never optimised.
  - Runs planned: EXP-49 (Kaggle, RSNA) and EXP-50 (Mac, ISLES'24).
- **Tests:** `tests/test_explain.py` (6), on a synthetic head phantom (air, scalp, skull ring, brain) windowed as in training, and a small random ResNet-18 on CPU.
  - The masks find the brain, bone and outside; no brain is found in or beyond the skull.
  - The heat shares follow the map and sum to 1.
  - A skull-base phantom slice has a low brain fraction; heights run 0 to 1.
  - The Grad-CAM maps are in [0, 1], and its probabilities equal `infer.slice_probs`.
  - Each rule gives its hand-worked score, including both fallbacks.
- pytest: 774 passed.
- **Notion sync:** none yet (EXP-49 and EXP-50 rows when they are set up).

## 2026-11-19 — Claude Code session (Item 4 step 7: EXP-47 scored, EXP-48 the DEC-30 report)
- **EXP-47 scored.** All 6 cells' extraction and judge batches finished, from 13:53 to 14:24 (the auto-collector stopped itself). **Cost $3.15 at batch prices** ($1.90 for the base, $1.25 for the tuned model), worked out from the logged token usage. The estimate was $3–6. Scored cells copied to Drive.
- **EXP-48** (`small_reasoner_report`, no API): `runs/EXP-48_small_reasoner_report/` on the Mac and Drive. Verifier rates, not yet checked against the human labels.
  - **Hallucination rate, tuned against base** (paired):
    - all 175: 41.6% → **14.3%** (−27.2 pts [−29.5, −25.0]);
    - held-out routes (51): 42.0% → **17.4%** (−24.6 [−28.6, −20.7]);
    - seen routes (99): 42.7% → 12.3% (−30.4);
    - real development cases (20): 36.3% → 15.5% (−20.8).
    - The content rate falls by about the same amount (−21.9 pts over all 175).
  - **Against the EXP-37 models (paired):**
    - The tuned 4B is lower than Gemma 3 27B and MedGemma 27B over all 175 (−4.3 and −4.0 pts) and on seen routes (−9.1 and −9.4).
    - On held-out routes it is level with both (+0.6 and +0.9, intervals spanning 0).
    - It stays above Sonnet (+9.9) and Qwen (+8.2).
  - **Decisions right:** 83 of 249 for the base, 193 of 251 tuned.
  - **But the tuned model under-recommends treatment.**
    - The base model says yes to nearly everything: thrombolysis yes caught 42/42 but no kept 0/68; thrombectomy 33/34 and 8/105.
    - The tuned model swung the other way. Thrombolysis yes caught **24/42**, no kept 57/69. Thrombectomy yes caught **10/35**, no kept 102/105.
    - On held-out routes it catches thrombectomy yes **3/18**. The held-out "beyond 4.5 h and eligible for thrombectomy" route is a thrombectomy-yes route, and it got 8 of 30 decisions right against the base's 15.
    - On seen routes it catches 7/14.
    - Sonnet catches 34/35, and Gemma 3 27B 33/33.
    - Its **critical rate rose**: +1.9 pts over all 175 [+0.4, +3.4], +3.4 on held-out routes [+0.9, +5.8].
    - Likely cause, unverified: the training set's thrombectomy answers were 20% yes (246 of 1,200), with the eligible-beyond-4.5 h route held out. The model may have learned the base rate.
    - The report gives this as its own section ("Treatment recommended when the rulebook says yes, withheld when it says no"), because the pooled "decisions right" hides it.
  - **Real cases:** decisions right 2/29 for the base against 29/29 tuned. Most real cases carry a bleed flag (DEC-28), so the right answer there is mostly no, which suits the tuned model's bias.
- **`small_reasoner_report.decision_counts`**: treatment-yes caught and treatment-no kept per model and group, as report section 4. Test extended.
- **Parsa's decision (19 Nov):** the treatment bias is reported as a finding, and no rebalanced variant is trained. Plan §7 (NLP A) has a ticked line for Item 4 with the result. Item 4 is done, apart from the human-label check of the verifier rates (EXP-41).
- pytest: 768 passed.
- **Notion sync:** Experiments: EXP-47 (done), EXP-48 (done).

## 2026-11-19 — Claude Code session (Item 4 step 7 started: EXP-47, the small reasoner on DEC-30's arms)
- **EXP-47 reasoner pass** (Colab, notebook 55 at 451556a; Parsa's go at the $3–6 estimate):
  - Colab gave an **A100 80 GB** after all, so the G4 path was not used. vLLM 0.34.0.
  - All 350 requests answered: `gemma-3-4b` in 270 s, `gemma-3-4b-tuned` in 136 s. That is much faster than the 30–40 min estimate.
  - The tuned folder was filled from the base snapshot with 7 files, among them `preprocessor_config.json`, `chat_template.json` and `tokenizer.model`.
- **Imported to the Mac** (`scripts/paid_runs import`): 5 + 150 + 20 per model. **349 of 350 valid.** One base-model fixture hit the 8,000-token limit before finishing its JSON, and counts as a format failure, the same rule as every model. The tuned model: 175 of 175.
- **Extraction and judge** through the Batch API as in EXP-37: the first extraction batches were submitted 19 Nov. `runs/EXP-47_small_reasoner/autocollect.sh` (not committed) advances every 15 min under `caffeinate` and stops when all 6 cells are done.
- **`pipeline/small_reasoner_report.py`** (DEC-30's report, to run as EXP-48 once EXP-47 is scored):
  - **Case groups.** The 150 scenarios are split by route: **held-out** (51 scenarios take at least one of the 4 held-out routes; the main result) and **seen** (99). Fixtures (5), real development cases (20) and all 175 are reported too.
  - **Comparisons.** Each group has the summary rates for the small pair and the four EXP-37 models. Tuned minus base is paired on the same cases (hallucination, content, critical, decisions right). Tuned and base are each compared with every EXP-37 model on the scenario groups and on all 175.
  - **Per route.** A table for each key route (counts only: the groups are small).
  - **Output.** `report.json`, `report.md`, and one figure (`small_reasoner.png`): base and tuned per group, hallucination rate and decisions right, with 95% intervals, in the C9 style and palette.
  - `compare.paired` and `compare._pooled` take a `keep(case_id)` filter for the groups. Display names were added for the 4B pair.
  - **`notebooks/42_nlpB_small_reasoner.ipynb`:** a thin Colab wrapper, like notebook 41.
  - **Tests:** `tests/test_small_reasoner_report.py` (2): the split is DEC-30's 4 routes, 51 and 99; the report on fake cells gives the planted differences per group.
- pytest: 768 passed.
- **Notion sync:** Experiments: EXP-47 (running).

## 2026-11-18 — Claude Code session (Item 4 step 6: serving the small reasoner pair, notebook 55)
- **`open_models.py`:** two new entries.
  - `gemma-3-4b` (DEC-30's prompted small reasoner, `google/gemma-3-4b-it`, gated).
  - `gemma-3-4b-tuned` (EXP-45's merged model, served from a local copy at `/content/models/EXP-45_merged`, no token).
  - Both are served in bf16 (`--dtype bfloat16`, stated rather than read from the merged config), max_model_len 16,384, and the same structured-output backend. A test checks the two serve commands differ only in the checkpoint and the served name.
  - `SMALL_MODELS` and `fill_from_base()`: copies into the tuned folder every base-snapshot file the merge did not write (weights and `config.json` excepted), so the tuned model serves with the base model's tokenizer, chat template and processor files.
- **`paid_runs.py`:** the small pair answers only DEC-30's arms (`SMALL_ARMS`): fixtures and scenarios with the rulebook, and the 20 development cases with predicted vision. That is 175 requests per model, the same 20 cases as every other model. The paid runs' own matrix is unchanged (tested).
- **`notebooks/55_small_reasoner_colab.ipynb`** (notebook 53's pattern):
  - Copies the inputs and EXP-45's merged model to local disk, downloads the base once and fills the tuned folder from it.
  - Serves each model in turn and saves each answer to Drive as it arrives.
  - The Hugging Face token goes only to the base model's download and server.
  - `MODE='test'` (EXP-46): one request per model. `MODE='run'` (EXP-47): all six cells.
  - The Mac then imports the answers and runs extraction and judge as for EXP-37.
- **Held-out routes in the evaluation set:** 51 of the 150 scenarios take at least one of the 4 held-out routes, and 99 take only seen routes. The per-route counts are 15, 6, 9 and 22, as `scenarios.routes` matches them by prefix. Step 7 reports the two groups apart.
- **EXP-46** (19 Nov, A100, c6594a7): both models valid on fx02, so vLLM loads the merged model (written by transformers 5, filled from the base snapshot).
  - `gemma-3-4b`: started in 300 s (download and load), answered in 11.6 s with 695 output tokens.
  - `gemma-3-4b-tuned`: started in 165 s, answered in 8.8 s with 557 output tokens.
- **EXP-47 estimate** (from EXP-46 and EXP-37's matching cells; unmeasured):
  - Colab: about 30–40 min of A100.
  - Anthropic: extraction and judge at batch prices about $3–6. EXP-37's same three cells cost $1.64–2.61 per model. This is below DEC-30's $12–15, which was made before EXP-37 ran.
- **Notebook 55 accepts a G4** (RTX PRO 6000 Blackwell, 96 GB) as well as an A100. On 19 Nov Colab had no A100 and gave an L4. For serving a 4B model the G4's memory bandwidth (about 1.8 TB/s) is in the A100's range (1.5–2 TB/s), and it runs bf16, so the settings are unchanged. An L4 would be several times slower, and a T4 has no bf16. Both models of a run always share one GPU, recorded in `config_colab.json`.
- **A vLLM server that exits on start-up now stops the wait at once**, with the end of its log, instead of after 30 minutes.
- pytest: 766 passed.
- **Notion sync:** Experiments: EXP-46 (done).

## 2026-11-18 — Claude Code session (Item 4 step 5: the dummy run EXP-44 and the fine-tune EXP-45)
- **Fixes found by the runs:**
  - **torchao:** Colab ships torchao 0.10, and peft 0.21 refuses to build a LoRA adapter alongside it (it needs above 0.16). LoRA here does not use torchao, so notebook 54 uninstalls it after the package install (3de3499). It is removed rather than upgraded because an upgrade could pull in a different torch.
  - **Generation stops on `<end_of_turn>` explicitly** (`stop_ids`). The model's generation config already lists it (`[1, 106]`), so this was not the cause of the dummy's long answer.
  - **The dummy keeps its full answer** (`generated.txt`) and reports whether it stopped by itself (247e892).
  - **`generation_check()`:** after the full run, the merged (best) model answers 3 validation rows with a 1,500-token cap (the longest gold answer is about 900 tokens). It reports whether each answer stops, validates, and gives the gold verdicts. Validation loss cannot see a looping model. The notebook prints OK only if all 3 stop and validate (247e892).
  - **The merge needs no token.** EXP-45's first merge failed because Colab would not hand over `HF_TOKEN` three hours into the run (`TimeoutException`; the tab was in the background). The notebook now downloads the whole base repo into the cache when it loads the model (`prefetch`), and `merge(..., local_files_only=True)` reads it from there. This change is not yet run on Colab.
- **EXP-44** (dummy, A100, at 3de3499): 30 steps on 20 examples in 156 s. Peak GPU 13.7 GB; 1,615 tokens/s; 8,229 tokens per example (mean), answers 646 tokens. Validation loss 0.200, 0.130, 0.119. Projected full run 3.06 h.
  - The generated answer started as schema JSON with real passage IDs, then repeated one claim ("The occlusion site is basilar.") until the 3,000-token cap: a model trained only 30 steps.
- **EXP-45** (full run, A100, 247e892): **270 steps** (1,080 examples × 2 epochs ÷ 8), about 40 s per step at about 1,604 tokens/s, so about 3 h of training. Early stopping never triggered.
  - Validation loss fell at every evaluation, from 0.092 at step 25 to **0.0045 at step 270** (the best). The gold answers are templated, so a low loss is expected and says little by itself.
  - Merged model on Drive (`runs/EXP-45_finetune/merged`, with `processor_config.json` and the chat template). The merge was rerun by hand in the same session after the token timeout.
  - **Generation check: 3 of 3 answers stopped by themselves (375, 672 and 668 tokens) and validated; 2 of 3 gave the gold verdicts.** Checked locally against the gold answers and `gold.check()`:
    - train-0734 (basilar, NIHSS 3) and train-0533 (extended-window thrombolysis, M2): identical to the gold verdicts, `rule_ids` and blocking factors; `gold.check()` finds no problem.
    - train-0005 (recent head injury; M1, 4 h 25 min, ASPECTS 5, mRS 1): thrombolysis right. **Thrombectomy wrongly not recommended** (gold: recommended, COR 1), with no blocking factor. It also adds two wrong uncertainties: that a COR 1 recommendation "leaves the decision to judgement", and that ASPECTS is not reported, though it cites ASPECTS 5.
  - These 3 are validation rows from routes seen in training. The real evaluation is step 7, with the held-out routes as the main result.
- **Cost:** Colab Pro compute only (A100, about 4 h for both runs; compute units not measured). No API.
- pytest: 763 passed.
- **Notion sync:** Experiments: EXP-44 (done), EXP-45 (done).

## 2026-11-17 — Claude Code session (Item 4 step 4: the fine-tuning code and notebook 54)
- **`nlp/reasoner/finetune.py`:** bf16 LoRA fine-tuning of Gemma 3 4B instruction-tuned on EXP-43 (DEC-30).
  - **LoRA targets** only the language model's attention and MLP projections (`TARGET_MODULES` excludes the vision tower and projector): r 16, alpha 32, dropout 0.05.
  - **Loss on the answer only.** The chat template's prompt render must be a prefix of the full render, or the example is refused. Logits are computed only for the last `answer_len + 1` positions (`logits_to_keep`): Gemma's 262k-token vocabulary would otherwise need several GB for a ~7,500-token request.
  - **Training loop:** a minimal loop (no Trainer), batch of one with gradient accumulation 8, AdamW at lr 1e-4 with a cosine schedule and 20 warm-up steps, 2 epochs. Validation loss every 25 steps, early stopping after 3 evaluations without improvement; the best adapter is kept.
  - **Resuming:** a checkpoint after each evaluation (adapter, optimiser, scheduler, step) on Drive lets a disconnected Colab resume.
  - **`merge()`** writes base + best adapter as one full bf16 model for vLLM, served with the same settings as the base.
  - **`dummy()`** reports peak GPU memory, tokens per second and validation loss, and checks that one generated answer is valid schema JSON.
- **`notebooks/54_finetune_colab.ipynb`**, a thin Colab A100 notebook:
  - `MODE='dummy'` (EXP-44): 30 steps on 20 examples, then one generated answer, and the projected full-run time.
  - `MODE='run'` (EXP-45): the full fine-tune, then the merge.
  - It checks the training set is EXP-43, built against the unchanged evaluation set.
  - It needs `HF_TOKEN` with access to `google/gemma-3-4b-it`, whose terms must be accepted on Hugging Face.
- **`pyproject.toml`:** a `finetune` extra (transformers 5.22.1, peft 0.21.3, accelerate 1.16.0: the versions the CPU tests ran).
- **Tests:** `tests/test_finetune.py` (7) on CPU, with a tiny random Gemma 3 text model and a character tokenizer with a Gemma-style chat template.
  - The answer-only loss equals cross-entropy on full logits.
  - Training lowers the validation loss, keeps the best adapter and resumes.
  - LoRA never targets the vision tower.
  - The merged model loads.
  - `dummy()` reports.
- **Not yet measured:** memory and speed on the A100. The dummy run measures them. A rough estimate (unmeasured) is 30–40 minutes per epoch at about 7,500 tokens per example.
- pytest: 761 passed.
- **Notion sync:** Experiments: EXP-44 (planned).

## 2026-11-17 — Claude Code session (DEC-30 approved; Item 4 step 3: training scenarios and the fine-tuning dataset, EXP-43)
- **DEC-30 approved by Parsa:** the fine-tuned small reasoner (Gemma 3 4B, prompted and LoRA-tuned). Recorded in plan §2 and in Notion's Decisions.
  - **Amended the same day:** the 4 held-out routes are drawn by seed from the 14 routes other than "within 4.5 h with a disabling deficit" and "acute intracranial haemorrhage", which training always keeps.
  - Drawn from all 16, the seed held out both. That left extended-window thrombolysis as the only "yes" in training.
  - **Held out:** beyond 4.5 h and eligible for thrombectomy; LVO 4.5–24 h without thrombectomy; INR above 1.7; no occlusion identified.
- **`scenarios.py`:**
  - The value grids are a parameter of `_draw`. `TEST_GRIDS` keeps the 150 evaluation scenarios byte-identical (sha256 `ea5b5977…`, checked by a test).
  - `TRAIN_GRIDS` shares no decision value with them and puts values just either side of the rules' cut-offs.
  - `generate_training`: new seed (20261117), held-out routes skipped, no case repeating an evaluation or training case (`signature`), caps per decision path and per route.
  - The route cap rejects a draw only when every route it takes is full. A first version rejected a draw when any route was full, and left 4 DAWN and 2 DEFUSE-3 scenarios in 1,200, because a late thrombectomy "yes" mostly comes paired with a thrombolysis contraindication.
- **`nlp/reasoner/train_data.py`:** each training scenario's gold answer in prompt v3's request format (system prompt with the rulebook, user message as in evaluation). A gold answer that fails `gold.check()` stops the build. The validation slice is every tenth example of a fixed shuffle.
- **EXP-43** (`runs/EXP-43_train_set/`, on Drive; no API, no patient data): **1,200 examples, 1,080 train and 120 validation**, every gold answer passing its checks.
  - Thrombolysis yes / no / undetermined: 280 / 393 / 527. Thrombectomy: 246 / 570 / 384.
  - Routes: DAWN 36, DEFUSE-3 15, 6–24 h 67 and 59, basilar 113, M2 58, haemorrhage 40, within 4.5 h 160; the held-out routes 0.
  - About 27,700 characters per example (the system prompt is about 25,200 of them); answers about 1,600 characters.
- Tests: `tests/test_train_data.py` (7). pytest: 754 passed.
- **Notion sync:** Decisions: DEC-30 (with the amendment); Experiments: EXP-43.

## 2026-11-17 — Claude Code session (C9: the hallucination table, EXP-42)
- **EXP-42: C9, the hallucination table** (`runs/EXP-42_c9_table/`, on Drive; `notebooks/41_nlpB_model_comparison.ipynb`; no API calls). It reads EXP-37's 30 cells.
  - `compare.case_records` rebuilds each cell's per-case records from its saved traces, and **reproduces all 30 cells' saved counts exactly**.
  - Intervals are 95% percentile bootstraps over cases (10,000 resamples).
- **Headline** (with the rulebook, predicted vision, each case once: 5 fixtures + 150 scenarios + 20 real cases):

  | Model | Hallucination rate | Content rate (1–5) | Critical | Decisions right | Format failures |
  |---|---|---|---|---|---|
  | Sonnet 5.5 | 4.4% [3.7, 5.2] | 3.6% [2.8, 4.4] | 0.8% [0.4, 1.3] | 231 / 249 | 1 / 175 |
  | Qwen3.5-27B | 6.2% [5.3, 7.1] | 4.2% [3.5, 5.0] | 1.0% [0.7, 1.5] | 165 / 180 | 50 / 175 |
  | Gemma 3 27B | 18.6% [17.1, 20.0] | 13.7% [12.3, 15.2] | 5.1% [4.2, 6.1] | 150 / 243 | 5 / 175 |
  | MedGemma 27B | 18.3% [16.5, 20.1] | 14.0% [12.4, 15.7] | 4.5% [3.6, 5.5] | 156 / 208 | 29 / 175 |

- **Paired comparisons** (same cases on both sides, paired case bootstrap):
  - **Rulebook effect.** The content rate, with minus without, on 100 matched cases. It uses the content rate because the without-rulebook arm is not charged for citations (DEC-29). The rulebook lowers content errors for every model: Sonnet −14.9 pts [−16.6, −13.2], Qwen −12.9 [−14.7, −11.1], Gemma −13.0 [−15.3, −10.7], MedGemma −8.0 [−10.4, −5.4].
  - **Vision effect.** Predicted minus ground-truth vision on 20 real cases. No difference except for Gemma: −5.0 pts [−8.9, −1.3], fewer errors with predicted vision.
  - **Against Sonnet** (hallucination rate): Qwen +1.7 pts [+0.6, +2.9]. On the content rate Qwen is +0.7 [−0.3, +1.7], no clear difference, so its gap is in citations. Gemma +14.2 [+12.6, +15.8]; MedGemma +13.9 [+12.0, +15.9].
  - **Medical training** (MedGemma against its base, Gemma) shows no lower hallucination rate.
  - **Sonnet run to run** (same 20 real cases): 4.8%, 7.6%, 4.9%.
- **Code** (`pipeline/compare.py`):
  - `_arm_label` tells the paid runs' arms apart: rulebook on or off, predicted or ground-truth vision, repeats.
  - `case_records`, `paired` (each side's interval too), and `c9_report` (`table.csv`, `report.json`, `report.md`, two figures).
  - Flat rows also carry the content rate, decisions, format failures, attribution and named unknowns.
  - Notebook 41 is now a thin wrapper around `c9_report`. Tests: `tests/test_compare.py`, 3 new.
- **Limits to report:**
  - These are verifier rates; EXP-41's labels will give the verifier's agreement with people.
  - Without-rulebook rates count content only.
  - The real cases number 20, so their intervals are wide.
- EXP-37's run folders are copied to Drive (`runs/EXP-37_paid_runs/`, logs left out).
- **Plan §7 ticked:** notebook 41, the hallucination table.
- pytest: 747 passed.
- **Notion sync:** Experiments: EXP-42; EXP-37's Drive copy is done.

## 2026-11-17 — Claude Code session (EXP-41: the human-labelling sample and webpages)
- **EXP-41** (`runs/EXP-41_labelling/build.py`; no API calls): **100 claims** sampled from the 23,245 the verifier scored in EXP-37's arms with the rulebook (4 models × fixtures, scenarios, real cases with predicted and with ground-truth vision).
  - Stratified by model × verifier verdict (clean, content error, citation error), 8–9 claims per group, seed 20261109 (`adjudicate.sample_claims`, Benny's kit). They come from 97 cases: 78 scenario, 21 real, 1 fixture.
  - `sheet.csv` is what labellers may see. `key.csv` maps each claim to its model, run, case and verdict, and stays on Parsa's Mac, not on Drive.
- **Four labellers in two mixed pairs (Parsa, 17 Nov 2026): Lorna (A) + Benny (B), Mohamed (A) + Hana (B).**
  - Each pair labels one half of 50, split evenly by model × verdict (`label_page.split_halves`, seed 20261117; `halves.json`). Every claim gets two independent labels, and each person labels 50.
  - Mixed pairs (vision + NLP) so shared background does not inflate kappa.
  - Benny's brief said he would not label; he built the sampling code but has not seen this sample's key.
- **`nlp/verifier/label_page.py`:** one self-contained HTML page per labeller. It runs offline in any browser, with no server and nothing sent anywhere.
  - **It shows** the claim, the answer entry it came from, the declared sources (imaging values, note quotes highlighted, full passage text), the case's imaging and note as the model received them, and the searchable rulebook.
  - **It never shows** the model, the run, the case ID or the verdict.
  - **It flags** a note quote that is not in the note, and a passage that does not exist.
  - **Saving:** labels save in the browser, with a warning when storage is off. A download writes the sheet's own CSV, and loading it back restores the labels (rows outside the person's half are ignored).
  - **Merging:** `merge_sheets` (per pair) and `combine_sheets` (both halves) feed `adjudicate.read_labels`; `kappa_by_pair` gives κ per pair. Tests: `tests/test_label_page.py`.
- **Guides:** `docs/team/labelling_{lorna,benny,mohamed,hana}.md`, one per person (file, partner, letter). `docs/LABELLING_GUIDE.md` is updated for the pairs, and labels go to Parsa directly.
- **On Drive** in `StrokeCopilot/labelling/`: the four pages, the four guides and the shared guide.
- **Plan §7 ticked:** `providers.py` (four models) and the model runs (EXP-37). The adjudication line records EXP-41.
- pytest: 744 passed.
- **Notion sync:** Experiments: EXP-41; Tasks: the labelling assignment (Lorna, Benny, Mohamed, Hana, 50 each).

## 2026-11-16 — Claude Code session (EXP-37 finished; Item 4 steps 1–2: the gold-answer builder, EXP-40)
- **EXP-37 finished:** 4 models × 7 arms, plus Sonnet's two repeats, all verified. Pooled verifier numbers, not adjudicated and without intervals (C9 builds the table):

  | Model | With rulebook | Without | Critical (with) | Format failures |
  |---|---|---|---|---|
  | Sonnet 5.5 | 4.5% (294 / 6,493) | 19.0% | 0.7% | 1 / 295 |
  | Qwen3.5-27B | 6.2% (332 / 5,377) | 18.2% | 0.9% | 55 / 295 |
  | Gemma 3 27B | 18.4% (761 / 4,137) | 29.7% | 4.6% | 5 / 295 |
  | MedGemma 27B | 17.1% (1,241 / 7,238) | 22.8% | 4.0% | 35 / 295 |

  - Sonnet's three runs on the same 20 real cases: 4.8%, 7.6%, 4.9%.
  - **Correction to the entry below:** Qwen's 47 one-item lists in the redo were not answers. 44 are the JSON schema itself, echoed back, and 3 are broken otherwise. Point 17 unwrapped them, and each still failed validation, so they count as format failures. Point 17 rescued no Qwen answer.
  - **Fix:** `run.recommend` changed a shared saved answer in place, so the scoring pass counted 0 stripped IDs. It now works on a copy. Every cell's `unwrapped_lists` and `stripped_ids` were recounted from `reasoner_answers.jsonl`, with no API calls (MedGemma 3,321, others 0); scores are unchanged.
  - **Cost (estimates at batch prices):** Anthropic $26.91, including $2.67 for the invalid Qwen pass; OpenRouter about $2.95; Colab about 2 A100 hours.
- **Item 4, step 1: `nlp/reasoner/gold.py`**, the gold-answer builder. A scenario's inputs and `rules.decide` give the reply prompt v3 asks for:
  - `recommend` is true only for a "yes" verdict;
  - `rule_ids` are about the decision's own treatment, plus the passages the blocking factors rest on;
  - each deciding condition is stated as one comparison citing its input and passage;
  - unknowns go in `uncertainties` with the passage ID in brackets.

  `gold.check()` runs the code checks (schema, decisions, own-treatment IDs, verbatim quotes, omissions). All 150 scenarios and 5 fixtures pass. Tests: `tests/test_gold.py`.
- **Item 4, step 2: EXP-40, the live verifier check of 30 gold answers** (`runs/EXP-40_gold_check/run.py`), one or more from each pair of decision routes:
  - **Round 1:** 500 claims, 15 flagged (3.0%), 4 critical, all from three template faults: over-citing, blocking factors resting on passages not in `rule_ids`, and a computed sleep-midpoint time.
  - **Round 2, after the fixes:** 507 claims, **0 flagged**, 45 of 45 decisions right. It reused round 1's judge cache for unchanged claims, so it does not measure the judge's run-to-run variance.
  - Cost $0.56.
- pytest: 739 passed.
- **Notion sync:** Experiments: EXP-37 (results), EXP-40. DEC-30 (the fine-tuned reasoner) is still awaiting Parsa's approval.

## 2026-11-16 — Claude Code session (EXP-37, the main paid run: fixes made while it ran)
- **EXP-37 started** (C8, the go given 16 Nov 2026): prompt v3, commit `6162cd2` on Colab.
  - **Colab A100**, notebook 53 in `run` mode, about 2 h: Gemma 3 27B and MedGemma 27B text answered all 295 requests each.
  - **The Mac:** Sonnet's 315 answers through the Batch API, no errors; Qwen3.5-27B live through OpenRouter.
  - Checking runs through the Batch API, with an auto-collector advancing every 15 min.
  - **Sonnet's and Gemma's cells are done.** MedGemma's and Qwen's checking is still running.
- **Qwen's first 295 answers were invalid and are set aside** (`runs/EXP-37_paid_runs/_set_aside/`; their open check batches were cancelled).
  - OpenRouter's Alibaba route had started dropping the system message. A probe was billed 95 prompt tokens for an 11k-character system prompt, where EXP-36's smoke the day before was billed 7,758.
  - So Qwen answered without the instructions or the rulebook: 0 of its 833 cited IDs exist in the rulebook.
  - It was found from Qwen's first finished cell: 42% of claims were flagged, mostly citations of non-existent passages.
  - **Fix:** the route now puts the system prompt at the top of the user message (`ApiRoute.merge_system`), so the same text arrives (8,513 tokens per request).
  - **Guard:** `VLLMJSON` raises `ProviderError` when a provider bills fewer than one prompt token per 8 characters sent (`MIN_CHARS_PER_BILLED_TOKEN`), so the run stops instead of scoring the answer.
  - Qwen was redone: 295 answers, 287 valid.
  - **Cost of the mistake (estimates):** about $1.03 of OpenRouter credit and about $2.67 of Anthropic batch spend.
- **Other fixes to the live path:**
  - A reply with no choices (OpenRouter's upstream error sent as a normal reply) is retried after 10, 30 and 90 s, counted in `usage["provider_errors"]`, then raises `ProviderError`.
  - `ProviderError` is not a `RuntimeError`, so it stops the run and is never scored as a format failure.
  - A reply that is not a JSON object is a format failure instead of a crash.
- **Two metric rules, approved by Parsa on 16 Nov 2026** (`docs/HALLUCINATION_METRIC.md` points 17 and 18; both apply to every model and are counted per cell in `metrics.json`):
  - **17.** A list holding exactly one object is scored as that object (`unwrapped_lists`). Qwen on Alibaba wraps answers as `[ {...} ]` although the request asks for a strict object: 47 of its 295 redone answers. Any other non-object reply stays a format failure.
  - **18.** One pair of square brackets around a passage ID is removed in `rule_ids` and passage sources (`stripped_ids`). The prompt's rulebook shows IDs as `[ID]`, and MedGemma copied the brackets into 3,321 of its 4,578 IDs. With them removed, all but one name a rulebook passage or `OWN-KNOWLEDGE`.
- **MedGemma:** 25 of its 295 answers hit the 8,000-token output limit (the server's 16k context with a prompt of about 7.5k) and are format failures. Gemma had none.
- **Spend so far (estimates from the run's token counts, batch prices):**
  - Anthropic about $18.40, including the $2.67 for the invalid Qwen pass. The whole run is projected at about $25, against the $37–53 estimate.
  - OpenRouter about $2.95.
  - Colab about 2 A100 hours.
- pytest: 731 passed.
- **Notion sync:** Experiments: EXP-37 (in progress), with the Qwen set-aside and the two metric rules.

## 2026-11-16 — Claude Code session (v2's critical flags checked; EXP-39: prompt v3)
- **v2's 9 critical flags (EXP-38) checked: all 9 are verifier mis-scores, not reasoner errors.** The judge was re-asked for each flagged claim with the same inputs to get its reasons ($0.008).
  - **3 come from a clash between v2 and the citation rule (§1).** v2 told the model to list limit-setting passages under a decision against a treatment, "under either treatment". The model listed thrombectomy passages (6–24 h ASPECTS 6–10 on fx02, the two DAWN passages on fx05) under thrombolysis's `rule_ids`. Those become citation claims "Passage X is a rule about thrombolysis", flagged as wrong rule. The reasoning is right, and blocking factors are already checked against both decisions' `rule_ids` (metric point 8).
  - **6 come from the judge** on "the patient meets the thrombectomy criteria, so extended-window thrombolysis does not apply":
    - 3 did not come back when the judge was asked again.
    - 2 (fx04) hinge on imaging at exactly 6 h 0 min. One run read it as outside "within 6 hours" and then charged the unreported mass effect of the 6–24 h rule. The other run accepted it. The fixture expects thrombectomy yes on both rules.
    - 1 (fx05) misread the claim as "the rule does not apply to that group", while a near-identical claim in the same case was supported.
- **Prompt v3** (`prompts/reasoner_v3.md`): v2 with one sentence changed. `rule_ids` list only passages about their own treatment; when the reason against a treatment is that the patient meets the other's criteria, those passages go under the other decision. `reasoner_v3_norulebook.md` is a byte copy of v2's (DEC-29's prompt, unchanged: there `rule_ids` is only `OWN-KNOWLEDGE`). Tests: `tests/test_reasoner_v3.py`.
- **EXP-39** (`runs/EXP-39_prompt_v3/run.py`; on Drive): v3 on the 5 fixtures with EXP-38's settings. Cost **$0.35** (reasoner $0.20, extractor $0.13, judge $0.02).

  | | v2 (EXP-38) | v3 (EXP-39) |
  |---|---|---|
  | Decisions right | 10/10 | 10/10 |
  | Format failures, omissions | 0 | 0 |
  | Claims | 154 | 142 |
  | Flagged | 13 (8.4%) | 5 (3.5%) |
  | `rule_ids` citation flags | 4 | 0 |
  | Wrong numbers, invented findings, faulty derivations | 0 | 0 |
  | Critical | 9 (3 cases) | 4 (3 cases) |

  - **v3's 4 critical flags, judge reasons kept ($0.004; Claude's reading, not adjudicated):** none is a reasoner error.
    - 2 did not come back when the judge was asked again (fx03 "ASPECTS range cannot be checked", fx05 rationale).
    - fx01 "Imaging is within 6 hours of onset" (2 h 30 min): the judge says the passage's window counts to treatment, not imaging.
    - fx05 "Meeting DAWN criteria puts the patient outside the extended-window thrombolysis recommendations" chains two passages (DAWN for eligibility, the wake-up passage for the exclusion), and the judge wants one passage that says both.
  - The one non-critical flag (fx04, "fits the criteria for thrombectomy 6 to 24 hours after onset" without naming the unreported mass effect) is a fair call.
- **judge_v3 is not changed.** The remaining mis-scores are judge variance on borderline "meets the other treatment's criteria" claims. Changing the judge is a DEC-23 change with an EXP-11 re-check, and the labelled sample measures the judge's error on the main run anyway. Known limit to report: claims that chain two passages.
- **`paid_runs.PROMPT` is now `reasoner_v3`.** Colab notebook 53 installs the package at `main`'s latest commit, so this must be pushed before the Colab run, or Colab answers with v2.
- pytest: 726 passed.
- **Notion sync:** Experiments: EXP-39; the EXP-38 page gains the flag check.

## 2026-11-16 — Claude Code session (EXP-38: reasoner prompt v1 vs v2)
- **EXP-38** (`runs/EXP-38_prompt_v1_v2/run.py`; on Drive): the 5 fixtures with the rulebook, Claude Sonnet 5.5 (effort high), verified live with the paid runs' settings (extractor Sonnet, judge Haiku `judge_v3`, τ 0.6). Cost **$0.95** (v1 $0.60, v2 $0.36).

  | | v1 | v2 |
  |---|---|---|
  | Decisions right | 10/10 | 10/10 |
  | Format failures, omissions | 0 | 0 |
  | Claims | 339 | 154 |
  | Flagged | 27 (8.0%) | 13 (8.4%) |
  | Wrong numbers, invented findings, faulty derivations | 9 | 0 |
  | Uncited or mis-attributed | 6 | 0 |
  | Wrong or misapplied rules | 12 | 13 |
  | Critical | 8 (3 cases) | 9 (3 cases) |

- **v2 stays the paid runs' prompt.** Plan §7 ticked.
- **To check before the main run (Claude's reading, not adjudicated):** 8 of v2's 9 critical flags fall on the step v2 asks for: "eligible for thrombectomy, so extended-window thrombolysis does not apply". That step is correct on fx04 and fx05, which both expect thrombectomy yes.
  - The step is flagged as misapplied or wrong rule.
  - Thrombectomy passages listed under thrombolysis's `rule_ids` (as v2 instructs, for limit-setting passages) become citation claims "Passage DAWN-… is a rule about thrombolysis", flagged as wrong rule.
- **Notion sync:** Experiments: EXP-38.

## 2026-11-16 — Claude Code session (C8: Parsa's three decisions, the without-rulebook arm, first test requests)
- **Decisions (Parsa, 16 Nov 2026):**
  - **DEC-21 amended:** a Hugging Face token may go to the serve pod for the gated Gemma weights.
    - It must be a fine-grained, read-only token for the two repositories.
    - It travels on SSH's stdin into a root-only file and is never logged.
    - It is **removed from the pod as soon as the last model is being served** (new: `remote.FORGET_TOKEN`, run on errors too).
    - Parsa revokes it after the run.
  - **DEC-24:** MedGemma's HAI-DEF terms cleared, so the Gemma pair stays.
  - `jobs.SERVE_POLICY` turned on.
  - **DEC-29 (new): the without-rulebook arm.**
    - The prompt is `prompts/reasoner_v2_norulebook.md`, v2 with the rulebook removed. Rules come from the model's own knowledge, named in the claim and cited as `OWN-KNOWLEDGE`, so the frozen schema still validates.
    - The verifier (`trace.py`) drops the placeholder and checks rule content against the corpus like an uncited rule. It does not charge passage attribution (6 and 7). Imaging and note citations stay checked.
    - `HALLUCINATION_METRIC.md` gains point 16, with the limits.
  - Plan §2 rows DEC-21, DEC-24 and DEC-29 updated.
- **Code:**
  - `prompt.system_prompt(..., rulebook=False)`.
  - `Inputs.rulebook`; `attribute(passages=False)`.
  - `run_cases`, `run_batched` and `collect_judge_requests` take `rulebook`, and the batched judge stays byte-identical in this arm.
  - `paid_runs` runs all 7 arms.
  - **Fixed:** `run_batched` did not pass the prompt version on to `run_cases`, so a run with any prompt but the default could not find its own answers.
  - **Qwen pinned to Alibaba on OpenRouter**, with no fallback, structured outputs and no data collection. Of six providers, the only one listed at 16-bit (Novita) lacks structured outputs, the other four run 8-bit, and Alibaba lists no precision. The full-precision rule could not be met, so it was dropped.
  - New `paid_runs smoke-pod` command for the Gemma test.
- **Tests, 6 added:** the no-rulebook prompt, the placeholder in both arms, attribution in this arm, a batched no-rulebook run against a live one, token removal, the pod test.
- **Test requests, EXP-36** (`runs/EXP-36_paid_runs_test/smoke.jsonl`, fixture fx02, prompt v2 with the rulebook). Both returned valid recommendations:

  | Model | Time | Output tokens | Cost |
  |---|---|---|---|
  | Sonnet 5.5, live at the batch settings | 20 s | 3,042 | $0.032 |
  | Qwen3.5-27B | 99 s | 9,098 (7,950 thinking) | $0.016 |

  - An earlier Qwen attempt hit OpenRouter's "no endpoint" error under the first provider settings. Sonnet's result from that attempt was lost, which is why `smoke` now saves each result as it arrives.
  - Three routing probes cost under a cent.
- **The Gemma pair moves to Colab (Parsa: keep the Vast credit for Phase 4).** DEC-24 note added.
  - `notebooks/53_paid_runs_colab.ipynb`, with `MODE` set to `'test'` (EXP-36: one request per model, with the start time) or `'run'` (EXP-37: every cell, 295 requests per model):
    - installs vLLM and the package at the pushed commit;
    - copies the contracts and notes to local disk;
    - serves each model in turn on the A100 at `--quantization fp8`;
    - saves each answer to Drive as it arrives, so Run all again resumes.
  - The Hugging Face token comes from Colab Secrets into the server's environment only. No Anthropic or OpenRouter key is used on Colab.
  - `paid_runs.reasoner_pass` sends only the reasoner requests, with no API key.
  - `paid_runs.import_answers` / `scripts/paid_runs import --from <Drive run> --out <Mac run>` copy the answers in. The Mac then runs extraction and judge as batches (`advance`).
  - The paid runs' prompt is v2 (`paid_runs.PROMPT`), the only one with a without-rulebook variant. The v1 vs v2 check is still to run.
  - **Tests:** 1 added, covering Colab's answers imported on the Mac and every cell finished without asking the model again. Also a local run of the notebook's input, run-record and summary cells against the real Drive files, which wrote nothing to Drive.
  - The Vast serve-pod code stays for Phase 4.
- `pytest`: 723 passed. **Spend:** about $0.10 of API (Sonnet $0.065, Qwen $0.016, probes under $0.01); no GPU.
- **Notion sync:**
  - Decisions: DEC-29 (new), DEC-21 and DEC-24 amended.

## 2026-11-16 — Claude Code session (C8 prep: batched judge, batches that survive a closed laptop, the run plan)
- **DEC-24 amended (Parsa): the judge runs through the Batch API too.** Plan §2 row updated.
- **`pipeline/batched.py`:**
  - **Batched judge.** `collect_judge_requests` walks every claim as the live path does and records each judge request the live path would send next. A second round exists only where a request depends on an earlier one. `CollectingJudge` keeps the live rule that a passage already sent is not sent again.
  - **Staged batches.** `run_batched(..., wait=False)` submits the batch that is due and returns `{"waiting": ...}`. The next call with the same folder fetches it and moves on.
    - `batches.json` records each batch by role and request set, so none is submitted twice.
    - `<role>_answers.jsonl` keeps every answer, so none is paid for twice.
    - Usage is added up over sessions, so the cost estimate covers the whole run.
  - A direct judge is still called live, and `wait=True` (the default) behaves as before.
- **`providers.ClaudeBatch`:** `submit()` and `collect(wait=False)`; `run()` is the two together.
- **`pipeline/paid_runs.py`:** the DEC-24 matrix as cells (model × arm), with the settings of earlier runs.
  - `advance()` moves one cell a stage; `status()` reads the folders.
  - Arms ready: fixtures, 150 scenarios, 20 real cases with predicted and with ground-truth vision, and Sonnet's two repeats.
  - **The three without-rulebook arms are refused for now** (`NotReady`). What the reasoner is told and how its uncitable rule claims are scored is not defined anywhere, and the recommendation schema requires a source for every claim. Parsa's decision.
- **Tests, 9 added:**
  - the batched judge sends byte for byte the live requests (system prompt, user message, schema), and every trace, flag and summary matches a live run;
  - the passage-on-its-way corner case;
  - a run across closed sessions submits nothing twice and costs the same;
  - `ClaudeBatch` collects in a later session;
  - the run plan's cells, 150/75 scenarios and settings.
- **Dry run with fake models on the real inputs** (no API, no cost): all 18 ready cells for the four models finished in 4 sittings: 820 cases, 42 batches, 4,116 judge requests. The no-rulebook arms would add 400 cases.
- **Local setup:** the `openai` client (a declared dependency, used for OpenRouter and vLLM) was missing from the Mac's venv and is installed.
- **The 48 GB serve pod (code only; no pod was created):**
  - `compute/jobs.py`: EXP-36 (one test request per model) and EXP-37 (the reasoner pass) are `serve` runs.
    - They carry their own offer filters: one GPU with at least 44 GB, at most $0.90/h, 200 GB of disk, at least 500 Mbps down.
    - They serve `SERVE_MODELS`, the Gemma pair.
    - `SERVE_POLICY` keeps `allow_gated` and `allow_medgemma` **off**. A commit that turns one on records Parsa's decision.
  - `compute/pod.py`:
    - `search`, `find_offer` and `offers` use a run's own filters.
    - `up` for a serve run sends no data and none of our code: bootstrap, watchdog, `pip install vllm`.
    - Only when `allow_gated` is on, the Hugging Face token is written to a root-only file on the pod. It travels on SSH's stdin and is never logged.
    - `Tunnel` forwards a local port to the pod's 127.0.0.1.
  - `compute/remote.py`: `SERVE_SETUP`, `hf_env`, `start_serve`, `STOP_SERVE`.
  - `pipeline/paid_runs.py`:
    - `run_pod_models` handles each served model in turn: start vLLM, wait until it lists the model, answer every cell that lacks answers (8 at a time, each saved on the Mac), then stop the server. It keeps the server's last log lines and **always destroys the pod**, on errors too.
    - `advance_all` skips a pod model's cell until its answers are saved.
    - `smoke` sends one request and projects the cost.
    - `scripts/paid_runs status|advance|serve|smoke` is the launcher.
  - **Tests, 8 added:** the serve spec and policy, a run's own filters, the scripts and the tunnel command, the driver's order of actions, the pod destroyed after a server failure and after a refused gated model, the readiness wait, and the smoke request.
  - **Unverified until the test requests on a real pod:**
    - that `pip install vllm` works in the PyTorch image;
    - that each model fits and loads with `--quantization fp8`;
    - that the JSON-schema backend accepts our schema;
    - the download time and the speed.
- **Not done yet:**
  - the without-rulebook arm's definition;
  - the v1 vs v2 prompt check;
  - test requests.
- `pytest`: 717 passed. **Spend:** none.
- **Notion sync:** Decisions: DEC-24 amended (judge batching).

## 2026-11-16 — Claude Code session (Phase 3: teammates' open tasks done from the Mac — Hana's real inputs, Lorna's reruns, Mohamed's checks)
- **Setup:** the slim tarball, checksum matched, was unpacked to `data/local/` (git-ignored). The 129 contracts and the notes were copied there too, because Google Drive stalled on one note file.
- **Hana: real inputs.**
  - `pipeline/end_to_end.py`:
    - `load_cases("real", ids, vision_dir, notes_dir)` loads a development case's contract, note and fact sheet;
    - the fact sheet is checked against the note;
    - expected decisions come from the rule checks on the given inputs, as for the scenarios;
    - held-out cases are refused.
  - `sample_real_ids(vision_dir, n=20)` takes the same number per fold, seeded.
  - `contract/build.py` gains ground-truth contracts (`build_ground_truth_all`, `--ground-truth`):
    - the reference final infarct as `core_ml`;
    - no bleed (`present` false, `confidence` 0.0);
    - the same penumbra and site as the predicted contracts;
    - confidence 1.0, and no warnings.
  - **EXP-34** wrote them to Drive `artifacts/vision_outputs_gt_v1/`, 129 validated. 8 have a negative mismatch volume, where the final infarct exceeds the Tmax > 6 s area; the schema allows it.
  - **Checked:** all 129 notes load and pass their fact check.
  - **The 20-case sample, predicted vs ground truth:** thrombolysis expected "no" in 10 vs 0 cases (Vision B's false bleed flags). Thrombectomy expected "no" in 18 vs 17 (the occlusion site is mostly unknown).
  - **Tests:** 2 added.
- **Lorna: notebooks 13 and 14 re-run** at `6596e3b`. Her own cells ran on the Mac, with the Colab install and Drive mount skipped (a scratch runner). Her `config.json` files were kept, and the earlier outputs were saved in `before_rerun_2026-11-15/` in each folder.
  - **EXP-27:** 37 Dice-0 misses under the merged no-infarct rule; 0 of 51 EXP-25 numbers differ.
  - **EXP-31:** reproduces her per-case values exactly, and v2's weights. It adds the `distance` version, `baseline_check` and `complete_miss_check`. Its `mapping_distance.json` equals EXP-32's.
  - Each `metrics.json` carries a `rerun_note`.
- **Mohamed: the two open Vision B checks (EXP-35**, scripts in `runs/EXP-35_visionB_crop_and_review/`):
  - **Crop:** on the 33 scans over 250 mm, no head or brain voxel falls outside the centred 250 mm square. Heads span about 18 × 15 cm, inside the window.
  - **The 23 scans flagged at 0.175 but not at 0.278:** their top two slabs were read by Claude, not a radiologist and not Mohamed, at low resolution.
    - skull-base slab 12;
    - normal calcification 4;
    - dense falx 3;
    - dense vessels 1;
    - nothing obvious 3.
    - None was read as a bleed.

    Labels only; the label key stays on Drive.
- **Plan §7 ticked:** domain shift check, real inputs, error taxonomy (Benny's PR #5).
- `pytest`: 701 passed. **Spend:** no GPU, no API.
- **Notion sync:**
  - Experiments: EXP-34, EXP-35.
  - Tasks: Hana's real-inputs task, Mohamed's domain-shift task and Benny's error-taxonomy task set to Done.
  - Notes added on Lorna's error-analysis and confidence tasks.

## 2026-11-15 — Claude Code session (Phase 3: PRs #5–#9 reviewed, fixed and merged; DEC-27 confidence; DEC-28 Vision B threshold; C7 contract build; DEC-24 hosting amended; C8 code prep)
- **Five teammate PRs reviewed, fixed on their branches and squash-merged** (one review agent per PR, findings re-checked by hand; the review entries for each PR are below):
  - #7 `0920766`, #9 `e17999b`, #6 `3305288`, #8 `6b10cd0`, #5 `3847cf6`.
  - Each branch was updated from main before its merge, with the CHANGELOG entries combined; `main` passes all tests after each one.
  - Still for the authors:
    - Lorna reruns notebooks 13 and 14 so the Drive outputs match the merged code.
    - Mohamed's two image checks: the scans between 0.175 and 0.278, and the 33 cropped scans.
- **DEC-27 (Parsa): `confidence.core` is the cut-off distance alone.**
  - **EXP-32** (free, on the Mac; `runs/EXP-32_visionA_confidence_dec27/run.py`, results on Drive) cross-fitted it by fold on EXP-31's 129 development cases.

    | | AUC | Brier |
    |---|---|---|
    | Distance alone | 0.736 | 0.188 |
    | v2 | 0.722 | 0.193 |
    | No signal | 0.5 | 0.222 |

    v2 minus distance: AUC −0.013 (95% interval −0.048 to +0.019). The two are equivalent, and distance needs one input, so it is chosen for simplicity.
  - **Correction:** the 0.755 quoted at review was the raw, unfitted ranking.
  - **Near-empty cores:** 30 cases are under 1 mL. They get median confidence 0.90, but only 25 are right. `core_warnings` adds the contract warning "Little or no core predicted: a missed infarct cannot be ruled out".
  - **Code** (`vision/perfusion/confidence.py`):
    - version `distance` is the default;
    - `NEAR_EMPTY_CORE_ML` and `core_warnings` are added;
    - the docstrings now say which file `build.py` reads: EXP-32's `per_case.csv` for development cases, `mapping_distance.json` for held-out cases.
  - **Tests:** updated for the new default, and a test added for the warning.
  - **Docs:** `docs/CONFIDENCE_VISION_A.md` gets a DEC-27 section, and Lorna's v2 sections stay for the record. Notebook 14's markdown names the fourth version.
  - **Plan:** §2 row DEC-27 added; the §7 confidence note updated.
- **DEC-28 (Parsa): the Vision B threshold is 0.175**, replacing EXP-6's provisional 0.278.
  - **Evidence (EXP-17, on RSNA validation):** sensitivity 0.961 and specificity 0.878, cross-fitted. That is 3.9 missed bleeds per 100, against about 5.5 at 0.278.
  - **Cost on ISLES'24:** 68 of 129 presumed bleed-free scans are flagged, against 45 at 0.278. This is reported as domain shift, not tuned away, and the predicted-vision arm will show its downstream cost.
  - **Code:** `vision/haemorrhage/infer.py` has `CONTRACT_THRESHOLD = 0.174984` and `contract_calibrator(calibration_json)`, which applies it over EXP-6's calibration and leaves the Drive file unchanged. A test was added.
  - **Plan:** §2 row DEC-28 added; the §7 "Tune aggregation and threshold" item ticked. The domain-shift item stays open for Mohamed's image checks.
- **C7 done: `contract/build.py`** (commit `5acea14`) assembles `vision_output.json` for the 129 development cases. No model is run; each field is read from a finished run and joined by case:
  - core from the nnU-Net fold runs (DEC-25);
  - penumbra and occlusion evidence from EXP-1, with the contract's site rule applied (DEC-7, DEC-26);
  - haemorrhage from EXP-16's calibrated scan confidence at 0.175 (DEC-28);
  - confidence from EXP-32 (DEC-27), and ASPECTS null (DEC-9).
  - **Safeguards:** held-out cases, repeated or missing cases, fold mismatches and failed Vision B scans are refused. Every object is validated, and nothing is written unless all 129 build. 3 tests on synthetic tables.
  - **EXP-33** wrote the 129 files to Drive `artifacts/vision_outputs_v1/`; all re-validated. Counts are in `runs/EXP-33_contract_build/metrics.json`:
    - bleed flagged 68;
    - core under 1 mL 30;
    - core 70 mL or more 9;
    - occlusion site ICA 13, basilar 1, unknown 115.
    - Median core 4.5 mL, median penumbra 143.1 mL.
  - **Note for the paid runs:** 115 unknown sites follow the §8.1 rule. The labels cannot separate M1 from M2, so real-case thrombectomy reasoning will often meet "site not reported".
  - Plan §7 C7 ticked.
- **DEC-24 amended (Parsa): hosting of the open models.** The RTX 4090 cannot hold three of the four at 4-bit in a form vLLM loads.
  - **Qwen3.5-27B** runs through OpenRouter at full precision.
  - **Gemma 3 27B and MedGemma 27B text** run together on one rented 48 GB GPU (L40S or RTX A6000, about $0.40–0.60/h on Vast, searched today) at the same 8-bit setting.
  - **Estimated** open-model cost: about $5–7, not measured.
  - **Pending before the runs:**
    - Parsa's OpenRouter key in his shell environment;
    - a Hugging Face token on the pod for the gated Gemma models, which needs a DEC-21 change;
    - Parsa's reading of the HAI-DEF terms §3.2.
  - **Fallback:** Baichuan-M2-32B and Qwen2.5-32B-Instruct replace the Gemma pair.
  - **Code still to do (C8 prep):** concurrent requests for direct llms, the 8-bit and OpenRouter entries in `open_models.py`, and one test request per model.
- **C8 prep (code only; nothing run, no spend):**
  - **Concurrent requests:** `pipeline/batched.gather(..., workers=N)` sends N direct requests at once with a thread pool, so vLLM can batch them on the GPU and an API answers in parallel. `run_batched(..., reasoner_workers=N)` passes it on, and the config records it.
    - Answers are still saved one line each as they finish, so a crash loses at most the requests in flight.
    - A server error still stops the run, and requests not yet started are cancelled.
    - `VLLMJSON` updates its usage counters under a lock.
  - **`nlp/reasoner/open_models.py`, for the amended DEC-24:**
    - Gemma 3 27B and MedGemma 27B text are served from their BF16 weights with vLLM's online 8-bit (`--quantization fp8`). Their serve commands are identical apart from the model. The gated and HAI-DEF refusals stay off by default.
    - New `API_ROUTES` and `api_llm()` route Qwen3.5-27B through OpenRouter:
      - the same JSON-schema request as the pod models;
      - temperature 0 and `max_tokens` 20000;
      - provider routing that requires every parameter to be honoured, 16-bit weights, and no data collection;
      - the key read from `OPENROUTER_API_KEY` at call time, never stored.
  - **`providers.py`:**
    - `VLLMJSON` takes `provider` and `billing`, so an API run is not labelled GPU-billed.
    - The OpenRouter URL may be recorded; a pod's address still may not.
    - The Qwen3.5-27B list price is added to `PRICES` (an estimate; providers vary).
  - **`compute/keyscan.py`:** watches `OPENROUTER_API_KEY`.
  - **Tests:** 5 added (concurrency, a server error mid-run, the identical Gemma pair at 8-bit, the OpenRouter route, keyscan).
  - **Still before the runs:**
    - Parsa's OpenRouter key, the Hugging Face token decision and the HAI-DEF reading;
    - one test request per model;
    - wiring the 48 GB pod job into `compute/jobs.py`;
    - whether Qwen thinks, a run-plan choice;
    - the measured cost estimate for Parsa's go.
- `pytest`: 699 passed. **Spend:** no GPU, no API.
- **Notion sync:**
  - Decisions: DEC-27, DEC-28; DEC-24 amended (hosting).
  - Tasks: the Vision B operating-point task set to Done; contract/build.py set to Done; a note on Hana's real-inputs task.
  - Experiments: EXP-33.
  - Experiments: EXP-32.
  - Tasks: a note on Lorna's confidence task.

## 2026-11-16, Claude Code session (Lorna, Vision A confidence: EXP-31 results, v2 chosen, Task 2 done)
- **EXP-31 at `217bc59`** (numbers from Lorna's Colab run; full tables in `metrics.json` on Drive):
  - **All three versions pass the done check.**
  - **Missing-data share:** median 11.5% over the measurement area, against 32.2% over all plain-CT tissue (EXP-30).
  - **Penumbra:** median 0.885, lowest 0.596, 94 of 129 under 0.9.
  - **Signals alone (weight / Spearman with volume error):**

    | Signal | Weight / Spearman |
    |---|---|
    | cut-off distance | −0.911 / −0.324 |
    | predicted size | 0.703 / 0.353 |
    | disagreement | −0.437 / −0.08 |
    | no-data | 0.015 / 0.074 |
    | Tmax > 6 s volume | 0.17 / 0.11 |

  - **Versions:**

    | Version | Wrong side, low / high | 10 mL errors, low / high | Under-calls in high band | Spearman |
    |---|---|---|---|---|
    | v1 | 24 / 7 | 33 / 17 | 7 of 34 | −0.33 |
    | v2 | 26 / 7 | 32 / 16 | 7 of 34 | −0.287 |
    | v3 | 26 / 7 | 31 / 17 | 7 of 34 | −0.267 |

- **Lorna chose v2.**
  - Her reasons: it puts the most cut-off errors in the low band, and distance to a cut-off is the clearest clinical reason for doubt; v3 adds nothing.
  - Recorded honestly: the choice was made after seeing all three, v1 and v2 differ by 2 cases, and v1 ranks the overall volume error slightly better.
- **`confidence.py`:**
  - `CHOSEN_VERSION = "v2"`, and `FEATURES` (the default for fitting) is v2's signals.
  - `load_default_mapping(run_dir)` loads `mapping_v2.json` and refuses a file with other signals.
  - `confidence_core` scores one case from it, and v2 needs no Tmax > 6 s volume.
  - The saved mapping is EXP-31's on Drive. It is not copied into the repo.
- **`docs/CONFIDENCE_VISION_A.md` filled in:**
  - status done;
  - the before/after missing-data share;
  - the signals-alone table;
  - the three versions side by side;
  - penumbra;
  - the version chosen and why.

  Its limits:
  - all 7 high-band cut-off errors are under-calls in every version, so complete misses like Miss 1 still get high confidence;
  - the no-data signal adds little;
  - penumbra is a data-quality score only;
  - the choice was made on the same cases.
- **Plan §7, "Confidence estimates", is ticked** and reworded to the method used: v2's signals, a cross-fitted logistic map, penumbra as data quality, and no fold-ensemble disagreement or MC dropout.
- **Tests:** 25 in `test_confidence.py`. The new one covers the v2 default, single-case scoring from `mapping_v2.json`, and refusing a mapping file with the wrong version.
- **Notion sync:**
  - Experiments → EXP-30: Done (first run at `aac4cb0`, before the fix; results above in the 2026-11-16 slab-fix entry).
  - Experiments → EXP-31: Done, Git commit `217bc59`, with the results above and "v2 chosen".
  - Tasks → Vision A confidence (Task 2): Done. Link `docs/CONFIDENCE_VISION_A.md`.

## 2026-11-16, Claude Code session (Lorna, Vision A confidence: slab fix, checks, three versions)
- **First run, EXP-30 at `aac4cb0` (v1, numbers from Lorna):**
  - **The done check passed:** wrong side of a cut-off, low band against high, 24 against 7; errors of 10 mL or more, 33 against 17; Spearman −0.345.
  - **Final weights:** at-risk no-data 0.239, log predicted core 0.667, log disagreement −0.212.
  - **Penumbra:** median 0.678, all 129 cases under 0.9.
  - **Brain no-data share:** median 0.322, range 0.172 to 0.700.
- **Two problems, and what was done:**
  - **The no-data share was inflated** by the plain CT covering the whole head (scalp included) while the perfusion scan covers a slab. `confidence.measurement_area` now counts missing data only in plain-CT tissue inside the perfusion region (`perfusion_region`): the slices from the first to the last with perfusion values, and in each the maps' area with holes filled. A covered slice with no values takes the slab's outline, so a missing slice still counts. This applies to the at-risk signal and to confidence.penumbra. The old whole-tissue share is kept as `tissue_no_data_share` for a before/after `slab_check`.
  - **Two signals pointed against the design:** a bigger core gave lower confidence, and more disagreement gave higher confidence. The likely reasons, written in `docs/CONFIDENCE_VISION_A.md`: small predictions rarely cross a cut-off, errors grow with size, and disagreement overlaps with size.
- **Checks added:**
  - `signals_alone` gives each signal's weight when fitted alone, and its Spearman correlation with the volume error.
  - `add_targets` marks `undercall` (smaller than true and on the wrong side of a cut-off), and the band table counts under-calls per band.
- **Two candidate signals (Lorna's decision, with Parsa's go-ahead):**
  - `log_cutoff_distance`: |log(1 + nnU-Net mL) − log(1 + nearest cut-off)|, nearest on the log scale;
  - `log_tmax6_ml`: log(1 + mL of reliable Tmax > 6 s).
- **Three versions:**
  - v1: the 3 approved signals;
  - v2: v1 plus the cut-off distance;
  - v3: v2 plus the Tmax > 6 s volume.

  `compare_versions` cross-fits every version on the same cases and reports each band table, done check (now with under-calls in the high band) and final mapping. All versions are kept. `fit_mapping`, `cross_fitted_confidence` and `confidence_core` take any version's signals.
- **Notebook 14:** slab check (section 5), signals alone (6), the three versions side by side with band tables and final weights (7), `mapping_v1/v2/v3.json` and penumbra (8), save (9).
  - **The notebook stops if the run folder's `config.json` is from the first run,** because that file predates the changes. The rerun is EXP-31 (Lorna's new Notion row), set in a follow-up commit; its folder is `runs/EXP-31_visionA_confidence`.
- **`docs/CONFIDENCE_VISION_A.md`** now has:
  - the measurement area and the five signals;
  - the three versions;
  - the first run and its two problems;
  - empty tables for the slab check, signals alone, the three versions, and the version chosen.
- **Tests:** `tests/test_confidence.py`, 24 tests (from 15). New ones cover:
  - the slab and footprint fix (scalp and uncovered slices not counted; an enclosed gap and a missing slice inside the slab still counted);
  - the Tmax > 6 s volume and the cut-off distance;
  - under-calls;
  - every version's leak check and single-case scoring;
  - the signals-alone check, the side-by-side versions and the slab check.
- pytest on Windows: 513 passed, 5 skipped, 9 failed. These are the same 9 as before (8 Windows, 1 missing `cv2`).
- **Notion sync:**
  - Experiments → EXP-30: record the first run's results, and that the method changed after it.
  - Experiments → EXP-31 added by Lorna for the rerun (Running); fill in the commit and results after the run.
  - Tasks → Vision A confidence: In progress.

## 2026-11-15 — Parsa's review of PR #7 (Lorna's Vision A error-analysis follow-ups, EXP-27)
- **Notebook 13:** `REPO_REF` is `main`, for reruns after merge. The per-fold table shows `no_infarct`. Still output-free.
- **Code, cases with no final infarct (0 mL):** `error_analysis.has_no_infarct`, `is_miss` (Dice 0 on a case with a final infarct) and `miss_kind` (the `miss` column, now with "no infarct"). These cases score Dice 0 whenever anything is marked, but miss nothing. They are now counted apart (`no_infarct`) and left out of the Dice-0 miss counts, the miss rate by size, both sides of the miss comparisons, the onset groups, `group_summary`'s `dice0`, `miss_profile` and the figure picks. They stay in the over-call and under-call counts. The follow-up `config.json` definitions say so. 5 new tests (84 in `test_error_analysis.py`). No fold-0 (EXP-15) number changes: fold 0 has no such case.
- **Write-up** (`docs/ERROR_ANALYSIS_VISION_A.md`):
  1. Artery groups: "does not seem to change how often the AI misses" (3/13 ICA, 12/45 MCA) now says no difference was seen, but the groups are too small to rule one out.
  2. "Missed by both" explained: Dice < 0.05 for both methods, a looser line than Dice 0, so it is larger than the Dice-0 misses (restated in 4).
  3. Added: 10 of 129 predictions are empty, all of them misses; how many misses have under 1 mL predicted (restated in 4).
  4. 2 of the 39 have no final infarct (0 mL; the AI marked 0.35 and 3.9 mL, both fold 2). With the code change above, **every five-fold count that involves misses is restated**, recomputed at review from EXP-27's `per_case.csv` with the new rule, not by a notebook rerun (`metrics.json` on Drive still has the old counts): 37 misses of 127 cases with a final infarct; 32 of 82 under 21 mL against 5 of 45; fold 2 has 5 misses, not 7; 24 of the 37 have under 1 mL predicted; missed by both 50 and AI alone under 0.05 in 53; MCA 11 Dice 0, "no occlusion found" 1, front 17. Visible share, no-data, onset, over-call and under-call counts are unchanged. The write-up's section intro and Limits say this.
  5. The intro's "this draft" is now "this write-up". Figure legends unchanged.
  6. After "no significance test was run": the size and visible-share gaps are large at 129 cases, but size and visibility go together, and the split by size (7) cannot fully separate them. No p-values added.
  7. Added the visible share split by size: under 21 mL, median 35% for misses (31 with data) against 53% (50); at 21 mL and over, 22% (5) against 69% (40). The gap shows in both groups, but size still varies inside each, so it hints, not proves.
- **EXP-27 config exception accepted:** the rerun at `0cd342c` wrote `per_case.csv` and `metrics.json` into the folder whose `config.json` is from `f493883`. It is documented in the write-up's Limits, and `metrics.json` records both commits.
- **CHANGELOG:** Lorna's 7 working entries are folded into one (2026-11-14 below). The stale "artery numbers need a rerun" line is dropped; the rerun happened.
- **Lorna's own entry below** keeps the counts as her run gave them (39 Dice 0); the restated counts are in this entry and the write-up.
- **Back to Lorna:** only rerun notebook 13 after merge, to refresh `metrics.json` and the figures with the new rule.
- `pytest` (Mac): 594 passed.
- **Notion sync:** none from this review.

## 2026-11-15 — Parsa's review of PR #9
- **Development cases now read their cross-fitted confidence.** `mapping_v2.json` was fitted on all 129 development cases, so scoring one of them with it is in-sample (AUC 0.757 against 0.722 cross-fitted). New `development_confidence(per_case_csv, version="v2")` reads `confidence_core_v2` and `confidence_penumbra` from EXP-31's `per_case.csv` as {case: {"core", "penumbra"}}. It refuses a missing column, a repeated case, a value that is missing or outside 0 to 1, and held-out cases (`splits.check_ids`).
- **One call for held-out cases:** `case_confidence(case, mapping, nnunet_ml, threshold_ml)` returns {"core", "penumbra"}, so `contract/build.py` does not stitch `case_signals` and `confidence_core` itself. The module text and docstrings say `mapping_v2` is for held-out (Phase 4) cases only.
- **A NaN core volume now raises `ValueError`** instead of silently becoming 0 mL (`max(0.0, nan)` is 0.0). A NaN Tmax > 6 s volume counts as missing, like `None`.
- **`docs/CONFIDENCE_VISION_A.md`:**
  - the slab-check range after the fix, 6.1% to 40.4% (from EXP-31's `metrics.json`);
  - v2's final weights: cut-off distance −1.188, missing data 0.248, core size −0.249, disagreement −0.017;
  - the per-signal story corrected to match them: inside v2 a bigger core gives slightly higher confidence, and missing data carries a real weight;
  - a "Which values build.py uses" note.
- **Notebook 14:** confidence.penumbra described as the share of the measurement area, not of the brain; the save step now also records the two checks below in `metrics.json`.
- **Cut-off distance alone against the versions** (`baseline_check`, run at review on EXP-31's `per_case.csv`): AUC for the wrong side of any cut-off, distance alone 0.755, v2 0.722, v3 0.703, v1 0.695, penumbra 0.538. v2 minus the distance alone −0.033, paired bootstrap 95% interval −0.066 to −0.002 (seed 0, 5000 resamples). v2 was chosen after seeing the results, and the simpler signal does at least as well.
- **Complete misses** (`complete_miss_check`): 25 cases predicted under 0.5 mL, 5 on the wrong side of a cut-off, median v2 confidence 0.90. A cross-fitted flag (under 1 mL and a large Tmax > 6 s volume) marks 5 cases and catches 1 of the 5 wrong-side ones, so it does not work. Written up as options; `confidence_core` and the default method are unchanged.
- **"What confidence.core means"** added to the write-up for the reasoner: roughly the chance the volume is on the right side of every cut-off; not steady in volume (0 mL 0.91, 21 mL 0.36, 100 mL 0.53 at average inputs).
- **Remaining, for Parsa (a DEC):** the method choice (cut-off distance alone or v2) and how complete misses are handled. Nothing goes back to Lorna.
- **Tests:** 593 passed.
- **Notion sync:** none from the review.

## 2026-11-15, Claude Code session (Lorna, Vision A Task 2: confidence, built on fake data)
- **Branch `lorna/confidence` from `main` at `dfc832f`.** Design approved by Parsa (15 Nov 2026). Choices agreed with Lorna:
  - the main error measure is the wrong side of at least one cut-off; a volume error of 10 mL or more is the second;
  - the no-data signal counts the Tmax > 6 s area plus a 10 mm margin;
  - log(1 + mL) is used for the volumes;
  - the cases are split into three equal bands.
- **`vision/perfusion/confidence.py`:**
  - **Signals from the scan, never the infarct:**
    - `at_risk_no_data_share`: no or unreliable Tmax in the Tmax > 6 s area plus 10 mm, inside plain-CT tissue; the whole brain if there is no such area;
    - `brain_no_data_share`.
  - **Volume signals:** `add_volume_signals` gives `log_pred_ml` and `log_disagreement`.
  - **Targets:** `add_targets` gives `large_error` and `error_10ml`.
  - **The mapping:** `fit_mapping` is a logistic regression on standardised signals, with ridge 1.0 and missing values filled with the training mean, written with scipy so no new dependency is needed. `predict_confidence` gives confidence.core = 1 − P(large error).
  - **Cross-fitting:** `cross_fitted_confidence` fits on 4 folds and scores the 5th.
  - **For inference:** `confidence_core` scores one case from a saved mapping.
  - **Bands and the check:** `bands` gives three equal bands, `band_table` the per-band errors and crossings, and `done_check` whether low beats high, plus the Spearman correlation.
  - **Penumbra:** `penumbra_confidence` = 1 − the brain's no-data share. It is a data-quality score only.
  - **Run record:** `write_config` refuses a placeholder experiment ID. `build_metrics` refuses case IDs.
- **`notebooks/14_visionA_confidence.ipynb`** (CPU). Install and mount are copied from notebook 12. It:
  1. writes `config.json`;
  2. reads EXP-27's `per_case.csv` and checks it against the split;
  3. measures the scan signals from the slim tar (Tmax, CBF and NCCT only);
  4. computes the cross-fitted confidence and the band table and check;
  5. fits the final mapping (`mapping.json`) and summarises confidence.penumbra;
  6. saves `per_case.csv` (Drive only, held-out check first), then `metrics.json`, last.

  `EXPERIMENT` is `EXP-30`, Lorna's Notion row, set in a follow-up commit; the run folder is `runs/EXP-30_visionA_confidence`. Reading EXP-27's table means this branch does not need the unmerged follow-ups code.
- **`docs/CONFIDENCE_VISION_A.md`:** the method in plain English, with empty results tables for after the run.
- `schemas/vision_output.json` and `contract/build.py` are unchanged.
- **Tests:** `tests/test_confidence.py`, 15 tests on fake data. They include:
  - a leak check: flipping a case's own label leaves its own score unchanged;
  - an end-to-end metrics test.
- pytest on Windows: 504 passed, 5 skipped, 9 failed. These are the same 9 as before (8 Windows, 1 missing `cv2`).
- Plan §7 "Confidence estimates" is not ticked yet: no results. The item's wording ("fold ensemble disagreement or MC dropout") will be updated to the approved method when it is ticked.
- **Notion sync:** Experiments → EXP-30 added by Lorna (Running); fill in the commit and results after the run. Tasks → Vision A confidence: In progress.

## 2026-11-15 — Parsa's review of PR #6 (Vision B ISLES'24 follow-ups: EXP-28 RSNA geometry, EXP-29 field-of-view rerun)
- **Merged `origin/main` into the branch** (a merge commit; Mohamed's history kept). The branch was built on the pre-squash PR #3 branch, 8 commits behind main, with 6 conflicts, each resolved by taking main's version and adding the branch's new work:
  - `isles_check.py`: main's, so `wilson` comes from `vision.haemorrhage.stats` (PR #2), plus `RSNA_FOV_MM`, `FOV_VARIANTS`, `variant_spec` and the field-of-view plumbing. `tests/test_isles_check.py`: main's plus the four field-of-view tests.
  - `README.md`: main's `22_visionB_operating_point` row kept, the `23_visionB_rsna_geometry` row added after it. `STROKE_COPILOT_PLAN.md`: main's §7 ticks (A1, B4/DEC-26, B5/DEC-25), DEC-25/DEC-26 rows and EXP-17 note kept; the `rsna_geometry.py` and notebook 23 layout lines and the EXP-28/29 text added.
  - `CHANGELOG.md`: main's entries unchanged; the 14 Nov entry placed above the 13 Nov ones. Heading lists checked against both sides: nothing lost or duplicated.
  - Notebook 21: main's notebook was identical to the branch's base, so the branch's version holds both main's cleanups and the follow-up. It keeps section 8's save cell, which main lacks: the PR #3 review (`173b6ca`) dropped its code by mistake, so a rerun on main saved nothing. Output-free and valid nbformat; the restored cell and notebook 23's cells were given cell IDs.
- **Corrections:**
  - `REPO_REF = 'main'` in notebooks 21 and 23.
  - 14 Nov entry: the stale "dry-run only, not run on the real data / not run on Colab, every number unmeasured" lines removed (both runs happened: `0172bce`, `fea2c15`). RSNA's "13,201 scans at 250 mm" is after rounding to 10 mm.
  - ISLES'24's field of view was given as "near 210 mm" from the spacing in `docs/DATA.md`. Measured (EXP-29's `per_case.csv`): median 229.0 mm (p25 215.6, p75 250.2); 95 of 129 under 250 mm, 1 at 250, and 33 over, so matching to 250 mm crops about a quarter of the scans rather than padding them. The crop is about the image centre; whether any head was cut is unverified. Notebook 21's "heads look larger" now says this instead of claiming it for every scan.
  - EXP-29 at EXP-17's proposed 0.175, counted from `per_case.csv` (counts only): 68 of 129 unmatched (52.7%, Wilson 95% CI 44.1–61.1%) against 64 of 129 matched (49.6%, 41.1–58.1%); 54 under both, 14 only unmatched, 10 only matched; exact McNemar p ≈ 0.54. At 0.278 re-checked: 45 against 42, 34 under both, flips 11 and 8, p ≈ 0.65. Added to the 14 Nov entry and the plan's Vision B line.
  - Item 1, top-scan review, partly recorded from Mohamed's Drive sheet (read at review, not in `per_case.csv`; counts only): the 8 highest-scoring scans at 5 mm are calcification 3, hyperdense vessel 2, artefact 2, nothing obvious 1, none read as a bleed. All 8 score 0.66–0.99, above both thresholds, so they do not separate 0.175 from 0.278.
- **Still open, needs the images:** the top-scan review by cause for the scans between 0.175 and 0.278 (the part that bears on the threshold choice), with the result moved into `per_case.csv`'s `review` column or this CHANGELOG rather than a separate sheet; and a look at the 33 cropped scans for a cut head.
- pytest: 565 passed.
- **Notion sync:** none from the review (the 14 Nov entry's list stands, with item 1 now partly done).

## 2026-11-15 — Parsa's review of PR #8 (Hana's vLLM adapter, DEC-24)
- **Fixes to `providers.VLLMJSON`:**
  1. Default `max_tokens` 8000 (`VLLM_MAX_TOKENS`), not the Claude default 16000: vLLM rejects a request whose prompt plus `max_tokens` exceeds `--max-model-len`, and the system prompt is about 6k–7k tokens on the Qwen and Gemma tokenizers. Server errors still stop the run loudly.
  2. openai client: `timeout=1800`, `max_retries=1` (was the defaults, 600 s and 2 retries).
  3. A cut-off answer (`finish_reason` length) or unparseable JSON keeps the first 500 characters of the reply in the error, plus the length and first 200 characters of the thinking when vLLM splits it out (`reasoning_content` or `reasoning`).
  4. Cost: `describe()` records `"billing": "gpu-time (pod), not token-priced"`. Notebook 31 prints a None cost as "cost not token-priced (GPU time)" instead of crashing on `:.2f`; notebook 50 still sums `v or 0` but now names the GPU-billed roles, so None never reads as free.
  5. `compute/keyscan.ENV_KEYS` adds `VLLM_API_KEY` (VLLMJSON's conventional `key_name`, `providers.VLLM_KEY_NAME`), `OPENAI_API_KEY` and `HUGGING_FACE_HUB_TOKEN` (`HF_TOKEN` was there), so `scripts/pod`'s refusal covers them.
  6. `describe()` keeps `base_url` only for `http://localhost` or `http://127.0.0.1`, else records `"remote"`: a pod's public IP and port never land in a run's `config.json`.
  7. `tests/test_providers.py` uses `monkeypatch.syspath_prepend` instead of leaving `tests/` on `sys.path`; the module docstring names the open-model key lookup.
  8. Plan §7: the `providers.py` line is unticked until the first GPU call.
- **Done in the review instead of going back to Hana (no GPU, no policy decision):**
  - `pipeline/batched.gather(..., answers_path=)`: a directly called reasoner's answers go to `out_dir/reasoner_answers.jsonl` as they arrive, and a rerun into the same `out_dir` asks only for the rest (a line cut off by a crash is skipped; another model's answers stop the run). Batch llms unchanged; `config.json` records `resumed_reasoner_answers`. A direct Claude reasoner rerun into the same `out_dir` now reuses its answers too.
  - `nlp/reasoner/open_models.py`: the DEC-24 models and the fallback pair (HF id, 4-bit checkpoint, `max_model_len` 16384, gated, `merge_system`, thinking default and switch, reasoning parser), every id checked against its model card on 15 Nov 2026. Unchecked fields are None and listed in `verify`. Only Baichuan-M2-32B (official GPTQ-Int4) and Qwen2.5-32B-Instruct (official AWQ) are complete. Qwen3.5-27B has no 4-bit checkpoint that fits 24 GB: the official GPTQ-Int4 is about 30 GB. Gemma 3 and MedGemma have no official vLLM-ready 4-bit checkpoint. DEC-24 says "Qwen2.5-32B"; the table uses the instruct model (to confirm).
  - `open_models.serve_command()`: the `vllm serve` argv (checkpoint, served-model-name, max-model-len, `--host 127.0.0.1`, port, `--structured-outputs-config.backend xgrammar`, reasoning parser, optional fp8 KV cache). It refuses unverified entries, gated models without `allow_gated` and MedGemma without `allow_medgemma`. Not wired into `compute/jobs.py` or `pod.py`.
  - How the pipeline reaches vLLM: over an SSH tunnel from the Mac to the pod's 127.0.0.1, so every API key stays on the Mac (the current DEC-21 rule). Gated models (Gemma 3, MedGemma) wait on Parsa's decision about an HF token on the pod.
- **Remaining:** Parsa's decisions (HF token on the pod for gated models; HAI-DEF terms for MedGemma; thinking on or off per model; a 4-bit Qwen3.5-27B that fits 24 GB; Qwen2.5-32B base vs instruct), then a one-request GPU dummy per model (does weights plus a 16k context fit 24 GB, and does vLLM's structured-output backend accept every keyword in `prompt.api_schema()`), then wiring `serve_command` into a pod job.
- pytest: 578 passed (567 before the review).
- **Notion sync:** none from the review.

## 2026-11-15 — Parsa's review of PR #5 (second round)
- **Merged `origin/main`** into the branch (15 commits behind); CHANGELOG kept main's entries and placed Benny's 12 Nov entry first among the 12 Nov ones.
- **What Benny's second round got right:** sampling now includes clean claims, a consensus column, and the models × (source × arm) shape. **What it broke:** 12 tests failed. It deleted `export_case_inputs`, `read_labels` and label validation, agreement per sub-type and per-model kappa, and `source` in the claims; it removed `experiment`, `metrics_file` and `run_folder` from `compare.load_run`, so `build_table` returned a pivot that `plot_table` could not read (every bar n/a); notebook 41 built its own table, installed from his branch, dropped the token redaction and `del token`, and was committed with outputs (about 45 KB, a PNG) and a GPU runtime.
- **Correctness bugs fixed:**
  - **Kappa** was computed only on rows with a consensus, so unresolved disagreements were dropped and kappa inflated. It now uses every row both labellers filled.
  - **Claims were rebuilt from whole output entries and looked up by exact text**, but the verifier's claims are atomic, reworded by the extractor, with IDs like `rationale[0].2`. On EXP-12 at `judge_v3` it gave 125 entry-level "claims" (the metric counts 350) and matched none of the 19 flags, so every claim was clean. `collect_all_claims` now reads each run's `traces/<case>.json` and joins it to the flagged file by (case, claim ID); verdict, sub-type and criticality come from the flagged file, every other claim is clean.
  - **A missing flagged file silently counted every claim as clean.** It now raises, unless `allow_missing_flagged=True`.
  - **A rescore scores a different claim set** than the one saved in `traces/` (restatements dropped, regex and citation claims rebuilt). The kit takes, per case, the claim set whose count matches the metrics file of the same scoring, and raises if neither does; a flagged claim ID not among the claims also raises. EXP-12 with `flagged_phase3_judge_v3.json`: 350 claims, as its metrics count, 19 flagged (11 content, 8 citation). The older `flagged_rescore.json` is refused, because it was scored with older code.
- **`adjudicate.py` restored and rebuilt:** `export_case_inputs`, `read_labels` (now for `label_A`, `label_B` and `consensus`, validated), `_field`, `source` and the docstrings are back; the "NEW CODE" marker is gone. Labels are `error` / `no_error` (the old `real_reasoner_error` / `verifier_mistake` still read). Sampling: about 100 claims in total by default (was 30 per model), stratified by model × verdict (clean, content error 1–5, citation error 6–7), equal allocation across strata with a fixed seed, because flagged claims are a few percent of all claims and a proportional sample would hold too few for agreement per sub-type. Each sampled claim carries its stratum size and weight, and the key file keeps them. `summarise_agreement`: kappa on every row both labelled; consensus = the consensus column, else A when A = B; agreement of the verifier with the consensus overall (also re-weighted to all claims), per verdict, per sub-type and per model (with per-model kappa); unresolved disagreements counted.
- **`compare.py`:** `load_run` rows have `experiment`, `metrics_file` and `run_folder` again; `build_table` stays flat (one row per run, what `plot_table` draws); new `pivot_table` gives models × (source × arm) with the rates, CIs, experiment and metrics file per cell, merges runs of the same model, and raises on two runs for the same model, source and arm; `format_pivot` gives display text ("0.054 [0.030, 0.080]", "n/a", "not run").
- **Notebook 41** is thin again: it calls `build_table`, `pivot_table`, `format_pivot` and `plot_table` only; installs `REPO_REF = 'main'` pinned to its commit, with the token redacted from errors and deleted (the notebook 51 pattern); outputs stripped, CPU runtime, valid nbformat 4.5. Run locally against EXP-12 at `judge_v3`: one row, 0.054 / 0.009 / 0.000.
- **`docs/LABELLING_GUIDE.md`:** the sheet's real columns (it listed `claim_id`, `subtype` and `reason`, which the blind sheet does not show), all claims rather than flagged ones, the new labels, and the consensus step.
- **Tests:** `test_adjudicate.py` and `test_compare.py` updated to the API, with new tests for the join by claim ID with reworded text, a missing flagged file, a rescore's claim set, clean claims labelled `no_error`, kappa on all rows (an unresolved disagreement lowers it), the consensus fallback, agreement per sub-type and per model, old labels, the pivot for 2 models × 2 arms and its duplicate check, held-out refusal, and sample size and strata.
- pytest: 587 passed.
- **Back to Benny:** nothing to redo; read the changes, and run the kit once the paid runs exist.
- **Notion sync:** none.

## 2026-11-14 — Lorna (Vision A error-analysis follow-ups, EXP-27; 7 working sessions folded into one entry at review)
- **Branch `lorna/error-analysis-followups`**, from `main` at `dfc832f`: the four follow-ups from Parsa's PR #1 review (all five folds, artery groups and onset, AI outlines on the pictures), in one notebook and one Colab run.
- **Code, `error_analysis`:**
  - `five_fold_table` joins the five nnU-Net folds (EXP-14, EXP-18 to 21) with EXP-1 (`within_tmax6`). It stops, with counts only, on a repeated fold or case, a fold mismatch or a wrong total.
  - `is_overcall` / `is_undercall`: the AI's volume is larger / smaller than the true volume and on the wrong side of at least one cut-off (21, 31, 51, 70 mL), at any Dice. `five_fold_counts` gives the misses, the miss rate under and over 21 mL, over- and under-calls, and the visible and no-data comparisons, overall and per fold.
  - `exp25_check` compares with Parsa's EXP-25 numbers (`EXP25_REFERENCE`, from the CHANGELOG); `compare_with_per_case` does a case-by-case check.
  - `artery_group` / `add_artery_groups`: the nearest artery within 5 mm (`volumes.DISTANCE_LIMIT_MM`), else "too far", "no occlusion found" or "unknown". `onset_summary` gives onset to scan, misses against the rest.
  - **Bug found by Lorna, fixed in `0cd342c`:** `artery_group` first used only the reason "ok", so every artery except ICA and basilar fell into "unknown". It now uses "ok", "too_far", "ambiguous" and any "nearest_*" reason.
  - `case_figure(..., prediction=...)` draws the AI's outline in cyan next to the red infarct, with a legend; `load_prediction` and `find_mask_dir` read the EXP-22 masks. `write_followup_config` and `followup_metrics` refuse placeholder IDs and case IDs.
  - **Tests:** 79 in `test_error_analysis.py`, including the notebook's steps end to end on fake tables and the artery reasons through `case_features()`.
- **Notebook 13** (`notebooks/13_visionA_error_analysis_followups.ipynb`, CPU) checks the inputs, writes `config.json`, builds the five-fold table, runs `case_features` on the 129 development cases, prints the counts, artery groups and onset, runs the EXP-25 check, draws the fold-0 pictures with the EXP-22 outlines, and saves `per_case.csv` (Drive only) and `metrics.json`. Counts only, never case IDs.
- **EXP-27** (`runs/EXP-27_visionA_error_followups/`): `config.json` written at `f493883`, results from the rerun at `0cd342c` after the artery fix, into the same folder. `metrics.json` records both commits. Accepted by Parsa as a documented exception. All numbers except the artery groups were the same in both runs.
- **Key results (129 cases):** 39 Dice 0, 34 of 84 under 21 mL against 5 of 45 at 21 mL and over. 9 over-calls (1 of them Dice 0, Miss 8) against 34 under-calls (5 Dice 0). Median visible share 35% for misses against 62%. At least 5% with no data: 13 of 39 misses against 14 of 90. Onset known for 59, no clear difference. Artery groups within 5 mm: MCA 45 / 12 Dice 0, ICA 13 / 3, too far 61 / 17; ACA, posterior and no occlusion are too small to conclude. EXP-25 check: 0 of 51 numbers differ.
- **Fold-0 pictures:** Miss 8's AI outline sits in a different, deeper place (an over-call); Miss 1 and 2 have none; Miss 4 has marks only on 7 other slices. Lorna confirmed the Miss 1, 2 and 8 readings.
- **Plan §7** "Error analysis" is ticked; the write-up is `docs/ERROR_ANALYSIS_VISION_A.md`.
- **Open points:** the EXP-25 case-by-case check was skipped (EXP-25 has no per-case csv); the TopCoW label numbering is unconfirmed; no significance tests on the five-fold groups.
- pytest on Windows: 521 passed, 5 skipped, 9 failed, the same 9 as before (8 Windows-only, 1 missing `cv2`).
- **Notion sync:**
  - Experiments → EXP-27: Done; Git commit `0cd342c` (first run `f493883`, which wrote `config.json`); add the results above.
  - Experiments → EXP-15: point to EXP-27 for the five-fold follow-ups.
  - Tasks → Vision A error analysis (Task 1): Done; link the write-up.

## 2026-11-14 — Claude Code session (Vision B: RSNA slice spacing and field of view, PR #3 review items 2 and 3)
- **Scope:** Parsa's PR #3 review, open items 2 (RSNA's real slice thickness) and 3 (the ISLES'24 rerun with the field of view matched). The runs are Mohamed's. Branch `mohamed/isles-followup`, from `mohamed/isles-check`.
- **`data/rsna_geometry.py`** (tested): from the shard index only, no images, per RSNA training scan:
  - the gap between neighbouring slice centres (`z_mm`, ImagePositionPatient z): the slice interval, not the slice thickness, which RSNA's files do not carry (plan §5.2). Uneven spacing, repeated positions and missing z are counted;
  - field of view = 512 × `pixel_spacing_mm`. The image size is not in the index, so RSNA's documented 512 is assumed;
  - the distribution across scans, the share within 0.25 mm of the documented 5 mm, and the most common values.
- **`notebooks/23_visionB_rsna_geometry.ipynb`** (Kaggle, CPU): reads the four index files, rebuilds the split (checked against EXP-6's fingerprint when its `metrics.json` is attached), keeps training rows only, refuses to run if `ALLOW_RSNA_TEST` is set, and saves `config.json` and `metrics.json` after an ID check.
- **EXP-28 (Mohamed, Kaggle CPU, `0172bce`):** 17,398 RSNA training scans; split matches EXP-6.
  - Slice spacing (gap between slice centres): median 5.037 mm, p5 5.0, p95 5.473, range 1.946–6.928 mm; 11,475 scans (66.0%) within 0.25 mm of the documented 5 mm. Most common (rounded to 0.25 mm): 5 mm 9,410 scans, 5.25 mm 3,602, 5.5 mm 3,150, 2.75 mm 347, 2.5 mm 198. So RSNA is mostly 5–5.5 mm, with about 3% near 2.5 mm. A table step of 5.25–5.5 mm is what a 5 mm interval gives with a 18–25° gantry tilt (5 / cos θ); that is a possible reading, not checked, since tilt is not in the index. 3,579 scans (20.6%) have uneven spacing (largest gap over 1.1 times the smallest) and 116 a repeated position. Head extent median 162.8 mm (p5 138.7, p95 190.0). The 5 mm slab target stands.
  - Field of view (512 × pixel spacing): median 250.0 mm, p5–p95 216.0–253.3 mm, range 150–500 mm; rounded to 10 mm, 13,201 scans (75.9%) are at 250 mm, then 220 (1,236), 230 (1,029) and 240 (663). Pixel spacing median 0.4883 mm, constant within every scan. ISLES'24's measured field of view (EXP-29's `per_case.csv`, corrected at review) is median 229.0 mm (p25 215.6, p75 250.2): 95 of 129 scans are under 250 mm, where heads fill more of the image than in RSNA and matching pads with air, but 33 are over 250 mm, so matching to 250 mm crops about a quarter of the scans rather than padding them. Whether any head was cut is unverified.
- **Item 3, code:** `isles_check.py` variants can now carry a field of view (`variant_spec`; a number is a slab thickness, a dict may add `fov_mm`). `FOV_VARIANTS` scores 5 mm slabs as in EXP-16 and 5 mm slabs cropped or padded with air to `RSNA_FOV_MM` = 250 mm (EXP-28's median), so the change in positive calls is paired scan by scan. `VARIANTS` and EXP-16's behaviour are unchanged (tested).
- **Notebook 21:** section 8's save cell restored (it had lost its code, so a rerun saved nothing). `VARIANT_SET` picks `'fov'` (the follow-up) or `'standard'` (EXP-16); the run folder is `<EXP>_visionB_isles_fov` for the follow-up. Section 5 prints whether the unchanged reading reproduces EXP-16's 45 calls. `REPO_REF` is `main` (set at review).
- **EXP-29 (Mohamed, Colab T4, `fea2c15`):** notebook 21 with `VARIANT_SET = 'fov'`, 129 of 129 development scans, none failed, 8.0 min.
  - 5 mm slabs as in EXP-16: 45 of 129 called positive at the provisional 0.278 (34.9%, 95% CI 27.2–43.4%). **EXP-16 reproduced exactly.**
  - 5 mm slabs with the field of view matched to RSNA's 250 mm: 42 of 129 (32.6%, 95% CI 25.1–41.0%).
  - Paired: 34 positive under both, 11 only as in EXP-16, 8 only when matched. Exact McNemar test on the 19 discordant scans, two-sided p = 0.65.
  - At EXP-17's proposed 0.175 (counted at review from `per_case.csv`): 68 of 129 as in EXP-16 (52.7%, 95% CI 44.1–61.1%) against 64 of 129 matched (49.6%, 41.1–58.1%); 54 positive under both, 14 only as in EXP-16, 10 only when matched; exact McNemar p ≈ 0.54.
  - **Reading:** matching the field of view barely changes the rate (3 fewer, not significant), so the mismatch does not explain the ~35%. It does move 19 of 129 calls (14.7%) in one direction or the other, so many ISLES'24 scores sit close to the threshold and depend on image scale. The same holds at 0.175 (4 fewer, 24 moved).
- **Item 1, top-scan review (partial; read from Mohamed's Drive sheet at review, not in `per_case.csv`):** the 8 highest-scoring scans on 5 mm slabs, by eye: calcification 3, hyperdense vessel 2, artefact 2, nothing obvious 1; none read as a bleed. All 8 have 5 mm confidence 0.66–0.99, above both 0.175 and 0.278, so they do not separate the two thresholds. The scans between 0.175 and 0.278 were not reviewed.
- pytest: 491 passed (8 new: 4 geometry, 4 field of view; `test_real_repo_is_clean` needs a git checkout and was run separately).
- **Notion sync** (not done; for Mohamed):
  - Experiments → EXP-28: median slice spacing 5.037 mm (66.0% within 0.25 mm of 5 mm), median field of view 250.0 mm (p5–p95 216.0–253.3), commit `0172bce`, Done.
  - Experiments → EXP-29: 42/129 (32.6%) with the field of view matched, against 45/129 as in EXP-16 (reproduced); paired flips 11 and 8, p = 0.65; commit `fea2c15`, Done.
  - Tasks → Vision B "Domain shift check": items 2 and 3 done; item 1 (top-scan review) partly done: top 8 by cause recorded, the scans between 0.175 and 0.278 still open.

## 2026-11-14 — Hana (NLP A: providers.py adapter for the open models, DEC-24)
- **`providers.VLLMJSON`** (new): one adapter for every open model in DEC-24 (Qwen3.5-27B, MedGemma 27B text, Gemma 3 27B, and the Baichuan-M2-32B / Qwen2.5-32B fallback), served by vLLM's OpenAI-compatible server on the rented 4090 (DEC-21). Only `model` and `base_url` change between models.
  - Same call signature and attributes as `ClaudeJSON` (`llm(system, user, schema) -> dict`, `.model`, `.usage`, `describe()`), so `run.py`, `end_to_end.py` and `batched.py` use it unchanged. No `run()` method, so `batched.gather` calls it one request at a time.
  - Temperature 0 by default. The schema goes in `response_format` as `json_schema`, which vLLM enforces by constrained decoding; every model gets the same `prompt.api_schema()`.
  - Usage keys match `ClaudeJSON` (cached prompt tokens split into `cache_read_input_tokens`), plus `reasoning_tokens`. `cost_usd()` returns None: GPU cost comes from the pod's billing, not tokens.
  - Errors the pipeline already records: refusal → `RefusalError`; `finish_reason` length, empty content, or JSON that does not parse → `RuntimeError` (the last keeps the start of the raw reply). Server errors (unreachable, 4xx) are not caught, as in `ClaudeJSON`, so a broken server stops the run loudly.
  - `extra_body` passes vLLM-only options and is recorded in `describe()`, e.g. `{"chat_template_kwargs": {"enable_thinking": False}}` to turn Qwen3.5's thinking off. `merge_system` puts the system prompt in the user message for a chat template without a system role. `quantization` is recorded only.
  - Key: none for a local server; otherwise `key_name`, read through `secret()`.
- **`providers.secret(name)`** (new): Colab Secrets, then Kaggle Secrets, then the environment, skipping a source that is missing or empty. `api_key()` now calls `secret("ANTHROPIC_API_KEY")`, with the same behaviour.
- **`pyproject.toml`:** `openai>=1.40` added to the optional `llm` extra (imported only when `VLLMJSON` builds its own client).
- **`tests/test_providers.py`:** 10 new tests with a stub client and fake keys, no network: secrets order and fall-through; request shape (model, temperature 0, messages, `response_format`); usage over two calls; `extra_body`; `merge_system`; each failure raises what the pipeline catches; `batched.gather` calls it directly; a fixture runs through `reasoner.run.recommend` and `run_fixtures` and writes a normal run folder; the client is built with no key or a key from `secret()`.
- **For Parsa (the first real call, not done here):**
  - Context: the reasoner's system prompt is about 10.6k tokens (EXP-8) and its answers about 3.1k–3.4k (EXP-8, EXP-9). `--max-model-len` must hold the prompt plus `max_tokens`: about 19k with `max_tokens=8000`, about 31k with the 20,000 used for Claude, which may not fit a 27B model at 4-bit on 24 GB.
  - Not checked against a real server: that vLLM's structured-output backend accepts every keyword in `prompt.api_schema()` (`anyOf`, `const`, `enum`). A one-request dummy, like EXP-8 for Claude, would show it before a full run.
  - Model IDs, quantised checkpoints, the thinking setting for Qwen3.5 and the MedGemma terms (HAI-DEF §3.2) are yours to set at the paid-run step.
- pytest: on Windows (Python 3.14), the NLP tests (`test_providers`, `test_reasoner`, `test_reasoner_v2`, `test_batched`, `test_end_to_end`, `test_verifier_run`, `test_claims`, `test_trace`, `test_verifier_phase3_fixes`, `test_contract`): 193 passed. The full suite stops at collection on this laptop because `nibabel` (vision) is not installed; not related to this change.
- **Notion sync:**
  - pending: Tasks → "providers.py adapters for the chosen models": In review once the PR is open, with a link to it and a note that the first real call on the 4090 is Parsa's.

## 2026-11-13 — Parsa's review of PR #1 (Lorna's Vision A error analysis, EXP-15)
- **Write-up corrected** (`docs/ERROR_ANALYSIS_VISION_A.md`), every number re-checked against EXP-15's `per_case.csv` on Drive (counts and anonymous labels only):
  - **New failure group, no perfusion data.** 3 of the 4 cases with ≥ 5% of the infarct at Tmax −30 are misses. Miss 2 has 51% with no data, so its "62% visible" covers only the third with reliable data. The Limits line that said the slice disagreed with the whole-case number is replaced.
  - **Miss 8 is an over-call:** 32.6 mL predicted for a 1.1 mL infarct, crossing 21 and 31 mL. It has its own group. A table gives the predicted volume for every miss: 4 empty, one 0.04 mL.
  - **Base rates added:** 7 of 17 small cases are missed against 2 of 9 larger ones; 4 of 9 misses and 5 of 17 others are under 25% visible (Mann-Whitney p ≈ 0.28).
  - **The "missed by both, so the scan is the problem" argument** now carries its caveat: the baseline looks only inside Tmax > 6 s.
  - **Volume table:** each group's size and the overall error (AI 19.6, threshold 20.5 mL) are added.
  - **Stale and causal lines updated:** the outlines are EXP-22 on Drive; Miss 1 "wasn't showing yet" is now "may not have been showing yet, or died after the scan"; noise was judged by eye.
  - **Groupings not reported yet** are listed under Limits: occlusion site, front/back, onset.
  - The Miss 1, 2 and 8 visual readings, reworded to match the numbers, are marked "pending Lorna's check".
- **Notebook 12:** the install cell is split from the Drive mount and split check, and `REPO_REF` is `main` for reruns after merge.
- **CHANGELOG:** Lorna's 14 working entries are folded into one (2026-11-07 below), with nothing lost that the code or write-up does not hold.
- **Back to Lorna** (PR comment):
  - group by the nearest artery label with a distance limit;
  - report onset for the 9 cases that have it;
  - draw the EXP-22 outlines on the figures;
  - repeat the miss count over all five folds.
- `pytest` (Mac, after merging main into the branch): 557 passed.
- **Notion sync:** none from this review. EXP-15 and the task row are Lorna's to update after merge.

## 2026-11-13 — Claude Code session (Phase 3: A1 done — nnU-Net folds 1–4 collected, 5-fold table; B5 decided, DEC-25; B4 decided, DEC-26)
- **Phase 3, Parsa's task A1.** Nothing below used a held-out case; the 129 cases are the development folds 0–4 only.
- **Folds 1–3 (EXP-18/19/20) finished overnight 12 Nov** and were collected by their auto-collectors, copied to Drive, and their pods destroyed and confirmed gone:

  | Fold | Run | Dice | Abs volume error | Wrong side of 21/31/51/70 mL | Uptime cost |
  |---|---|---|---|---|---|
  | 1 | EXP-18 | 0.211 ± 0.246 | 22.0 ± 42.2 mL | 3/4/4/3 of 26 | 4.89 h, ≈ $1.84 |
  | 2 | EXP-19 | 0.173 ± 0.165 | 27.0 ± 46.1 mL | 9/6/3/3 of 26 | 3.86 h, ≈ $1.57 |
  | 3 | EXP-20 | 0.290 ± 0.281 | 20.9 ± 27.2 mL | 5/5/2/3 of 26 | 4.90 h, ≈ $2.04 |

  - **EXP-22's fold-0 masks** came back with EXP-18 and are on Drive `runs/EXP-22_nnunet_fold0_masks/` for Lorna's error analysis.
  - **Measured:** the Vast credit fell $6.62 from 17:09 UTC 12 Nov ($9.97 + the $5 top-up → $8.35), covering the three folds, EXP-19's first pod and EXP-21's failed setup. About $0.67 of it is not explained by uptime (storage and bandwidth, most likely; not checked).
- **Fold 4 (EXP-21), 13 Nov:** Parsa approved offer 59572201 (New Jersey RTX 4090, $0.402/h, 5.5 h limit), the machine that ran EXP-18 cleanly.
  - Upload 13 min; epochs 39.4 s; training 3.56 h; collected and destroyed at 16:11 UTC.
  - Dice 0.184 ± 0.240 over 25 cases; abs volume error 28.3 ± 39.9 mL; wrong side of 21/31/51/70 mL in 8/8/8/5.
  - **Measured cost $1.76** (credit $8.35 → $6.59; uptime 3.88 h ≈ $1.56).
- **5-fold table (EXP-25, analysis on the Mac, no cost):** `vision/perfusion/evaluate.py` gains `load_per_case` and `paired_comparison` (bootstrap CI over cases, Wilcoxon signed-rank, exact McNemar on crossings), with a test. Same 129 cases and the same fold per case in both sources (checked).

  | | nnU-Net | Threshold, within Tmax > 6 s | Threshold, unrestricted |
  |---|---|---|---|
  | Dice, mean ± std over folds | **0.215 ± 0.046** | 0.111 ± 0.031 | 0.117 ± 0.034 |
  | Abs volume error, mean ± std over folds | **23.5 ± 3.9 mL** | 25.2 ± 5.0 mL | 25.9 ± 6.1 mL |
  | Wrong side of 21/31/51/70 mL (of 129) | **30**/**28**/**20**/17 | 35/29/23/17 | 48/28/23/16 |

  - **Paired against within-Tmax6:** Dice +0.104 (95% CI +0.078 to +0.133; p < 0.0001). Abs volume error −1.6 mL (CI −4.5 to +1.3; p 0.35): not a clear difference. No crossing difference is significant.
  - **Against unrestricted:** nnU-Net crosses 21 mL far less often (30 vs 48, p 0.003), all of it in lesions under 21 mL, where the unrestricted threshold over-calls.
  - **Lesions ≥ 21 mL (45 cases):** both methods are far off, mostly under-calling (nnU-Net 52.8 vs 55.7 mL abs error).
- **B5 decided — DEC-25 (Parsa):** nnU-Net feeds `core_ml`, not the rCBF < 30% threshold. Each development case uses the validation prediction of the fold model that did not train on it; combining the five fold models for held-out cases is left to Phase 4.
  - Recorded with its limit: no worse on any contract measure and clearly better on Dice, but the volume is not measurably better, and the report keeps the threshold as the comparison.
  - The under-calling of lesions ≥ 21 mL is the open weakness for B4 and calibration. The penumbra stays Tmax > 6 s.
  - Plan §2 row DEC-25 added; §7 "Decide which core feeds the contract" ticked.
- **B4 decided — DEC-26 (Parsa), from EXP-26** (free; `runs/EXP-26_core_volume_checks/analysis.py`, results on Drive):
  - Vision A is two-stage: the nnU-Net final-infarct prediction is the core proxy (DEC-25), and the Tmax > 6 s rule is the penumbra (DEC-7). There are no core or penumbra heads, because ISLES'24 labels only the final infarct, and no volume calibration.
  - **Consistency:** the nnU-Net core never exceeds the Tmax > 6 s volume (0 of 129), so the mismatch fields need no clipping.
  - **Large infarcts are under-called by both methods.** For infarcts ≥ 70 mL the medians are: reference 118 mL, nnU-Net 35, threshold 25, Tmax > 6 s 188. In 7 of 22 the final infarct exceeds the whole Tmax > 6 s region. Missing Tmax does not explain it (median 92% reliable).
  - **Corrections, cross-fitted by fold:** linear ones made things worse (+2.4 to +2.6 mL; 54–61 cases on the wrong side of 21 mL against 30). The best robust one gained 0.6 mL, with a CI crossing zero. Averaging nnU-Net and threshold volumes did best at 31 mL, but it was picked after the fact from 16 variants and is not adopted.
  - The gap is reported as a limit (final infarct ≠ acute core; reperfused cohort) and is input to `confidence.core`. A cross-fitted lower probability threshold stays an optional later test.
  - Plan §2 row DEC-26 added; §7 core/penumbra item ticked.
- **GPU spend for A1:** about $8.8 measured from the credit (12 Nov from 17:09: $6.62; the three failed attempts before it: about $0.45; fold 4: $1.76). With the $2.56 before Phase 3, the project's GPU total is about $11.4 of the 8–15 USD budget (DEC-21). Credit left: $6.59. **API spend this session: none.**
- **Housekeeping:** the keep-awake and the pod guard were stopped after fold 4; the combined monitor is still serving `localhost:8717`.
- `pytest`: 510 passed.
- **Notion sync** (done this session):
  - Experiments EXP-18, EXP-19, EXP-20, EXP-21 set to Done with results, hosts and costs; EXP-22's notes point to the masks on Drive.
  - Experiments EXP-25 created (5-fold summary and paired comparison).
  - Tasks: "nnU-Net folds 1–4 … then the 5-fold table" → Done; "Decide which core feeds the contract" notes updated with EXP-25.
  - Decisions: DEC-25 created and linked to both tasks; the task "Decide which core feeds the contract" → Done.
  - Decisions: DEC-26 created; Tasks: "Decide the core/penumbra approach (B4)" created as Done and linked; Experiments: EXP-26 created.

## 2026-11-12 — Claude Code session (Phase 3: fold jobs, offer watcher, A3 approvals, verifier fixes, model list, PR review)
- **Phase 3, Parsa's tasks A1–A3.** Nothing below used a held-out case; `folds.json` was checked by sha256 only.
- **A1, nnU-Net folds 1–4 (commit `7c67c5b`):**
  - `compute/jobs.py`: EXP-18 to EXP-21 are folds 1–4, 300 epochs each. EXP-22 regenerates fold 0's validation masks from EXP-14's `checkpoint_final.pth`, on EXP-18's pod before fold 1 trains. A run can carry others (`with`), each with its own folder, log and exit code; `collect` brings all of them back.
  - `pod_job.py --mode predict`: nnU-Net's own validation (`--val`) on a finished checkpoint. It refuses a checkpoint whose plan or fold differs, deletes the copied checkpoint afterwards, and leak-checks the mask file names too.
  - **Dummy, free (Mac, MPS, 3 fold-0 cases):** per-case Dice matched EXP-14 within 0.0001 and volume within 0.004 mL.
  - **Launch, not done.** Parsa approved 4 offers at $0.363/h in parallel, and topped up the credit to $10.44.
    - By launch time, all four offers had been rented, and two replacements were taken within minutes.
    - One pod (Poland, $0.363/h) was created, but its tarball upload stalled at 0 kB/s for 18 min. I stopped it, and `up` destroyed the pod. **Measured cost $0.13** (credit $10.44 → $10.31; $10.29 an hour later).
    - Parsa chose to wait for offers at $0.363/h or less rather than pay up to $0.50/h.
    - `runs/offerwatch.py` (not committed) checks every 10 min and launches each fold on the cheapest US or EU offer that passes, skipping the stalled host. It then starts the run's monitor page and auto-collector, and stops after 14 h or 2 failed setups.
  - **`pod.py` fixes (not committed):**
    - an upload that moves no data for 5 min aborts (`rsync --timeout=300`), so `up` destroys the pod;
    - `up` searches up to 25 rounds for the offer, because a rentable offer appeared in only 2 of 10 searches.
- **A3, approved by Parsa on 12 Nov 2026:**
  - `docs/HALLUCINATION_METRIC.md` §9 points 8–12;
  - the RSNA split, now DEC-22 in plan §2 and in Notion Decisions.
- **Free EXP-12 rescore** with the critical-rate denominator fix, at `7c67c5b`, before today's verifier changes. Run locally from the caches.
  - Results: rate 0.080 (28/351), content 0.060, **critical 0.014 (5/350)**, cases with a critical claim 3/5, attribution 0.979, no omissions.
  - The 0.185 recorded earlier divided by the wrong denominator. Saved to Drive as `metrics_rescore_7c67c5b.json` and `flagged_rescore_7c67c5b.json`.
- **A2, the three verifier fixes from EXP-12** (tests in `tests/test_verifier_phase3_fixes.py`; metric doc §9 points 13–15, approved 12 Nov 2026):
  - **Rules under uncertainties:** an uncertainty's rule claim also gets both decisions' cited passages as candidates, like blocking factors (point 8). Every rule claim also gets lexical candidates for a trial, Table 3/8 or an ASPECTS range it names (`trace.lexical_candidates`). τ stays 0.6 (DEC-20).
  - **Judge errors on conditional and negative applications:** `prompts/judge_v3.md` adds rules for a requirement that is not needed, criteria that are not met, conditional and counterfactual claims, and a reason for a decision. It became the default with DEC-23 (below). Its examples come from the EXP-12 flags, so EXP-12 is an in-sample check.
  - **Meta-statements:** a pure restatement of the output's own decision is not a claim (`claims.restatements`), because `recommend` is scored by correctness. A reason given with it stays a claim, and so does a statement that contradicts the output.
  - **Rescores take a judge prompt** and can call the judge for missing verdicts only (`run.rescore`, `end_to_end.rescore`). `nlp/verifier/recheck.py` and `notebooks/51_verifier_recheck.ipynb` re-check EXP-11 and EXP-12 under v2 and v3 (EXP-23).
  - **Measured so far, free:**
    - Fix 3 alone removes 1 of the 28 EXP-12 flags.
    - Fixes 1 and 2 need new verdicts: v2 7 + 31 calls, v3 60 + 82 calls (counted in a dry run). That is about $0.20, an estimate.
    - There is no API key on this Mac, so EXP-23 runs on Colab once the code is pushed. **The EXP-11 bar is not yet re-checked.**
- **EXP-23** (Parsa ran notebook 51 on Colab at `a9dcfe2`, about $0.15): the re-check of the three fixes.

  | | EXP-11 min recall | EXP-11 false flags | EXP-12 flags | Rate | Critical |
  |---|---|---|---|---|---|
  | Before (rescore at `7c67c5b`) | 0.80 | 0.010 | 28 / 351 | 0.080 | 0.014 |
  | Fixes 1 + 3, judge_v2 | 0.80 | 0.010 | 23 / 350 | 0.066 | 0.009 |
  | Fixes 1 + 3, judge_v3 | 0.80 | 0.010 | 19 / 350 | 0.054 | 0.009 |

  - EXP-11 misses the same seeded items as before (fx04-3a, fx03-6b, fx04-6b), so the bar holds under both prompts.
  - All 9 reasoner errors from the earlier reading are still flagged under both.
  - **v3 against v2:** it removes 5 verifier false flags (no BP lowering needed; M2 dominance definition; "not recommended because eligible for thrombectomy"; "meets ASPECTS 3 to 5 except mass effect"; the within-6-h counterfactual). It adds 1 citation-only flag (fx04 "appears eligible for thrombectomy" as uncited). v3's examples come from EXP-12, so that run is in-sample.
  - **The 3 remaining critical flags look like verifier errors** on Claude's reading (not human adjudication): a truism ("treatment cannot start before imaging"), an imaging fact the extractor labelled a rule, and a correct negative application the judge still misreads.
- **DEC-23, approved by Parsa:** the judge prompt is `judge_v3`, amending DEC-20; model, effort and τ are unchanged.
  - `trace.JUDGE_PROMPT` is now `judge_v3`.
  - Metric §9 points 13–15 are approved.
  - Plan §2 has the DEC-23 row, and DEC-20 notes the amendment, in the plan and in Notion.
- **B6, the model list (DEC-24, approved by Parsa)**, from Hana's shortlist:
  - The models: Claude Sonnet 5.5, Qwen3.5-27B, MedGemma 27B text, and Gemma 3 27B as MedGemma's base, for a paired comparison. Open models run on a rented 4090 with vLLM at 4-bit and constrained JSON.
  - **No non-Claude frontier model for now.** Three of the four are non-Claude, and verifier-vs-human agreement is reported per model.
  - **MedGemma waits for Parsa's reading of HAI-DEF terms §3.2.** Their "Clinical Use" includes research studies; the fallback is Baichuan-M2-32B with Qwen2.5-32B. It is a gated download, so the Hugging Face token question is decided at the paid-run step.
  - **Budget finding.** Verifying an output costs about $0.08 with the Sonnet extractor (EXP-12: extractor $0.064, judge about $0.015), more than the reasoner call ($0.046). Four models at about 410 outputs each would need about $150 of API against about $96 left.
  - **The fix depends on EXP-24.** With a Haiku 5.5 extractor (about $0.003 per output), the main runs cost about $52. The project would then total about $65–77 of API and $12.5–15 of GPU (estimates). Haiku becomes the extractor only if EXP-24 keeps the seeded-error bar.
  - Notion: the DEC-24 row; Hana's shortlist task Done, with her questions answered and a link to DEC-24; her adapters task Next up, with the models.
- **EXP-24, Haiku as claim extractor** (Parsa ran notebook 52 on Colab, about $0.27): **bar NOT met.**
  - Strict recall for uncited rules was 0.30; every other sub-type 1.0; false flags 0.010.
  - Haiku's claims still flag 9 of 10 planted uncited rules (lenient recall 0.9), mostly as mis-attributed source.
  - On EXP-12 it flags 17 of 321 claims, but misses 3 of the 9 reasoner errors (both fx05 DEFUSE-3 citation errors and the last-known-well rule) and adds about 4 verifier errors of its own.
  - **DEC-24 amended (Parsa):**
    - the extractor stays Sonnet 5.5; the extractor and the Sonnet reasoner go through the Message Batches API (half price);
    - the matrix shrinks: per model, the 5 fixtures with and without the rulebook, 150 scenarios with it and 75 without, and 20 real cases three ways; the triple repeat is for Sonnet only;
    - estimated about $65 for the main runs and about $85 for the project's API.
- **Batch support** (tested with fakes only; the first paid use is a 2-request dummy batch):
  - `providers.ClaudeBatch` sends ClaudeJSON's exact requests through the Message Batches API.
  - `pipeline/batched.run_batched` works in two passes:
    - it gathers every reasoner answer, by batch for Sonnet or by direct calls for open models;
    - it batches the extractions into the run's extraction cache under the key `run_cases` already uses;
    - then it runs `run_cases` unchanged, with only the judge called live.
  - `run_cases` takes the reasoner prompt version and applies a batched llm's price factor to the cost estimate.
  - Tests are in `tests/test_batched.py`.
- **`providers.py`:** Sonnet 5.5 cache reads are now priced at $0.10/M, not $0.20/M (Hana's finding), so earlier logged Sonnet costs were slightly high.
  - `CachedExtractor` now keys the cache by model for any extractor but Sonnet, so another extractor never reuses Sonnet's claims. Sonnet keeps its old key.
- **EXP-24 prepared:**
  - `end_to_end.reverify` re-extracts a run's saved outputs with another extractor.
  - `recheck.extractor_check` runs EXP-11 and EXP-12 that way in `runs/EXP-24_haiku_extractor/`, seeded with copies of the original caches. It resumes without losing paid verdicts.
  - `notebooks/52_haiku_extractor_check.ipynb`: about $0.50–1.00, estimated.
- **D10, review of PRs #2–#5** (Parsa approved the fixes and the merges). Each PR's review entry is below.
  - **Rule checks:** no held-out IDs, no `folds.json` reads, no RSNA IDs, no secrets and no paid calls in any of them. Two independent reviewers covered #2/#3 and #5, and their key numbers were re-checked by hand.
  - **Fixes** went onto each branch as new commits, with a comment on each PR.
  - **Merged by squash:**
    - #3, Mohamed's ISLES'24 check, as `60bd2ca`;
    - #2, Mohamed's operating point, as `0c6436f`;
    - #4, Hana's prompt v2, as `a3af4e6`.
  - Each branch first merged main and resolved the CHANGELOG by date. Notebooks 21 and 22 now point at `main`.
  - **#5 (Benny) stays open.** Its bugs are fixed (kappa, stale metrics file, old run layout, blinding), but the sampling redesign and the table shape are back with him.
  - **The Vision B threshold is Parsa's to decide**, after Mohamed's ISLES'24 follow-ups. On ISLES'24, the proposed 0.175 would flag 68 of 129 presumed bleed-free scans.
- **A1, the folds launched (watcher cap raised by Parsa within the 8–15 USD budget):**
  - The watcher now launches US/EU offers up to $0.47/h. A credit guard launches a run only if the credit covers every running pod's worst case plus the new one, with $0.50 to spare.
  - It skips machines that failed setup. The tarball upload now has a 30-min limit (`pod.UPLOAD_LIMIT_S`), and pod state records the machine ID.
  - **Two more failed attempts, both destroyed:**
    - Oregon (machine 26487, $0.456/h): the upload crawled at 0.1–4 MB/s; 1.9 of 3.1 GB in 38 min. **$0.30 measured.**
    - Poland offer 54489655: the same host as the first stall, stopped before upload, about $0.02.
  - The home uplink measures about 50 Mbps. There is no VPN; the old SSH block was the university Wi-Fi (Parsa).
  - **EXP-18** runs on New Jersey (instance 60383981, $0.376/h) from 17:42 UTC: 38.6 s per epoch, due about 20:55 UTC, watchdog at 23:15.
  - **EXP-22, done on EXP-18's pod:** fold 0 Dice 0.218 ± 0.249 over 26 cases, the same as EXP-14. The masks come back with EXP-18's collect.
  - **EXP-19 on the same machine as EXP-18.** Its first attempt (New Jersey, $0.376/h) landed on EXP-18's machine (155712, two GPUs in one box).
    - Both runs' data workers starved each other: GPU at 0%, load average 34, and epochs went from 39 s to about 290 s, so both would have hit their watchdogs half done.
    - EXP-19 was destroyed at 18:59 after 1.08 h (about $0.41, 7 epochs), and EXP-18 recovered to 39 s.
    - The watcher now never puts two folds on one machine, and set fold 2 up again on an Estonia host ($0.405/h).
  - EXP-20 runs on Michigan ($0.416/h) from 18:51 UTC, at about 50 s per epoch.
  - **EXP-21 (fold 4):**
    - It waited on the credit guard until Parsa topped up $5 (credit $12.87).
    - Its first pod (Michigan, $0.402/h) never left "loading" in 15 min. `up` destroyed it (0.26 h, about $0.10), and the watcher retries.
    - `runs/podguard.py` (not committed) destroys a pod still without a job 45 min after creation, once no `up` is working on it, so an offline moment during setup cannot leave a pod idle until its watchdog.
    - **Postponed to 13 Nov by Parsa.** The offer watcher was stopped at about 19:48 UTC with no fold-4 pod up.
  - **Folds 1–3 left to finish overnight.** Each has one auto-collector (EXP-19's duplicate, left over from its first pod, was stopped). The guard and the combined monitor keep running, and `caffeinate -i` keeps the Mac from idle-sleeping for 5 h.
  - **`runs/monitor_all.py`** (not committed): one page for all fold pods at `localhost:8717`, served on 127.0.0.1 only.
    - It shows progress, the median and last epoch time, with an epoch over twice the median flagged; the ETA, watchdog, cost, Vast credit and the watcher's last lines.
    - The per-pod pages show only the median of the last 20 epochs, which hid the slowdown.
- pytest: 509 passed.
- **Spend so far:**
  - GPU: $3.01 measured before EXP-18 ($2.56 before, plus $0.45 of failed attempts today); EXP-18 and EXP-19 are running.
  - API: about $0.15 (EXP-23).
- **Notion sync:**
  - done: Experiments → EXP-18 to EXP-22 rows, all Planned (EXP-18 notes the failed attempt and its $0.13);
  - done: Decisions → DEC-22 (RSNA split);
  - done: "Hallucination metric" page → points 8–12 approved, arms wording (rulebook, DEC-19);
  - done: EXP-12 row → corrected critical rate 0.014;
  - done: EXP-23 row;
  - done: EXP-18 Running (offer, price, commit `a9dcfe2`), EXP-22 Done;
  - done: Decisions → DEC-23 (judge_v3); DEC-20 notes the amendment; metric page → points 8–15;
  - done: Tasks → "Verifier fixes" Done (commits `502a769`, `6bf370b`);
  - done: Decisions → DEC-24; Tasks → shortlist Done, adapters Next up;
  - done: Tasks → Hana's prompt v2 Done; Mohamed's two tasks and Benny's taxonomy In review; Benny's notebook and labelling kit In progress;

## 2026-11-12 — Parsa's review of PR #3 (EXP-16 recorded; notebook and test fixes)
- **EXP-16 result**, from Drive `runs/EXP-16_visionB_isles_check/metrics.json` (Mohamed, Colab, 8 Nov 2026, commit `23e2f92`):
  - 129 of 129 development cases scored, 0 failed.
  - At the provisional threshold 0.278, 45 of 129 were called positive on 5 mm slabs: a presumed false-positive rate of 34.9% (95% Wilson CI 27.2–43.4%).
  - On native thin slices: 61 of 129.
  - Slice thickness 0.8–4.0 mm (median 2.0); 86 scans LAS, 43 LPS.
  - The previous entry's "not run, unmeasured" is superseded.
- **Fixes in review:**
  - Notebook 21 was committed with outputs (8 cells, including a 755 KB figure of ISLES scans); the outputs are cleared.
  - The stray "flush and remount" cell before the install cell is removed.
  - `EXPERIMENT` is `None` with a guard, so a rerun cannot reuse EXP-16's ID.
  - `tests/test_isles_check.py` used 4 real development case IDs; they are now synthetic IDs that exist in no split, checked by count only.
- **Still open:**
  - the top-scan review (the `review` column is empty on Drive);
  - one check of RSNA's real slice thickness (`z_mm` in the shards, on Kaggle), since 5 mm is the documented value, not a measured one;
  - field of view is not matched to RSNA (`set_field_of_view` exists but is off). A rerun with it on is cheap, and worth doing before putting a 35% rate down to domain shift alone.
- pytest: `tests/test_isles_check.py` 19 passed.
- **Notion sync:** Experiments → EXP-16 row already Done; Tasks → "Vision B false-positive check": In review, with the result.

## 2026-11-12 — Parsa's review of PR #2 (EXP-17 recorded; one wilson; proposal rule documented)
- **EXP-17 result**, from Drive `runs/EXP-17_visionB_operating_point/metrics.json` (Mohamed, Kaggle, commit `ff522bd`).
  - 2,166 validation scans, 1,894 patients, 879 bleed scans. Scan AUC 0.9772, and EXP-6's 0.278 reproduces 0.951/0.917 on the same split.
  - **Cross-fitted at a 95% target:** sensitivity 0.945, specificity 0.914. It fell below 95% in 11 of 20 seeds.
  - **Proposed 0.175** (96% target): sensitivity 0.961 (95% CI 0.946–0.972), specificity 0.878 (0.859–0.895).
  - **Strict alternative 0.125:** sensitivity 0.968, specificity 0.858.
  - The previous entry's "not run" is superseded.
- **ISLES'24 cost of each threshold, measured in review** from EXP-16's `per_case.csv` (counts only). Of 129 presumed bleed-free development scans, Vision B would flag:
  - 45 at 0.278 (34.9%);
  - 68 at 0.175 (52.7%);
  - 87 at 0.125 (67.4%).

  This is added as a dated note in `reasoning_EXP-17.md` on Drive, since that doc only estimated it. It matters for Parsa's threshold decision more than the RSNA specificity does.
- **Fixes in review:**
  - `wilson()` was defined twice; it now lives in `vision/haemorrhage/stats.py` (same constant, so recorded intervals reproduce), with a test.
  - `propose()` documents that the median rule in practice steps up one candidate, so read 0.96 as a chosen safety margin, slightly optimistic until the RSNA test split.
  - `seeds_below_target` is commented as counting against each row's own target.
  - Notebook 22 notes that `REPO_REF` becomes `main` after merging.
- **Merge:** PR #3's branch, with its review fixes, is merged into this one. The plan and CHANGELOG conflicts are resolved, and the plan's two Vision B items carry the EXP-16 and EXP-17 results, both unticked until the threshold decision.
- pytest: 469 passed.
- **Notion sync:** Experiments → EXP-17 row already Done; Decisions → the threshold, once Parsa decides.

## 2026-11-12 — Parsa's review of PR #4 (reasoner prompt v2)
- **One change to `prompts/reasoner_v2.md`** (not yet run on any model, so it is edited in place).
  - "What not to write" banned any claim that the patient "is eligible" or "qualifies". But eligibility for thrombectomy is itself the reason extended-window thrombolysis does not apply (fx02 and fx05 expect it), so the ban could push that reasoning step out of the rationale.
  - The bullet now bans only restating the output's own decision ("is recommended", "the rulebook supports"). It allows a step another decision depends on, stated with the passage that sets the criteria and the inputs that meet them.
  - The bad example's explanation and `test_v2_states_each_targeted_fix` are updated to match.
- **On the evidence behind v2:**
  - Its targets come from Benny's labels of the 28 EXP-12 flags (22 real reasoner errors, 6 verifier mistakes). These are not yet adjudicated, and Parsa's earlier reading was about 9 real and 19 verifier errors.
  - Some targets (rules named in uncertainties, restated decisions) are now handled by the verifier itself: DEC-23 and metric points 13–14, 12 Nov 2026.
  - So the paid v1-vs-v2 comparison is scored with the current verifier (`judge_v3`, the code fixes), so that v2 is not credited for verifier artefacts.
- **Rulebook gap Hana found:** "time counts from last known well when onset is not witnessed" is in the prompt but in no passage. Adding a passage is Parsa's call as rulebook owner.
- Plan §7 note spacing tidied.
- pytest: `tests/test_reasoner_v2.py` and `tests/test_reasoner.py` 15 passed (Mac).
- **Notion sync:** Tasks → "Reasoner prompt v2": In review, with this change.

## 2026-11-12 — Hana (NLP A: reasoner prompt v2, conditions fix)
- **`prompts/reasoner_v2.md`:** the "check a rule's conditions" bullet contradicted the `rule_ids` instruction. It said a rule whose condition is false must not be cited, while `rule_ids` must include the limits a patient fails (and the "9 h is beyond a 4.5 h window" example needs that passage). A model could drop the citation for a blocking factor, bringing back the "right fact, missing citation" error. The bullet now separates two cases: a **blocking rule** (the patient fails a limit for the treatment being decided) is cited in `rule_ids` and `blocking_factors`; a **rule that does not apply** (a different situation, as with the fx03 BP rule) is not cited. Not yet run on a model.
- **`tests/test_reasoner_v2.py`:** one more phrase in `test_v2_states_each_targeted_fix` ("A blocking rule sets a limit").
- pytest: on Windows (Python 3.14), `test_reasoner_v2.py` + `test_reasoner.py` 15 passed (1 unrelated deprecation warning from the anyio pytest plugin). Full suite not re-run.
- **Notion sync:**
  - pending: Tasks → "Reasoner prompt v2 from the EXP-12 findings": note the conditions fix before Parsa's paid v1/v2 run.

## 2026-11-12 — Benny's NLP B work (PR #5), with Parsa's review fixes
- **Benny's deliverables:**
  - `docs/ERROR_TAXONOMY.md`: the 8 types with real examples, detection and criticality.
  - His independent labels of the 28 EXP-12 flags (Drive `EXP-12_benny_labels.csv`): 22 real reasoner errors, 6 verifier mistakes. Parsa's earlier reading was about 9 and 19; the gap is mostly uncertainty claims and true-but-miscited claims.
  - `pipeline/compare.py` and `notebooks/41_nlpB_model_comparison.ipynb` (Task 2).
  - `nlp/verifier/adjudicate.py` and `docs/LABELLING_GUIDE.md` (Task 3).
- **Review fixes (Parsa):**
  - **Cohen's kappa** divided by (1 − po) instead of (1 − pe): it crashed on perfect agreement and returned 1.20 where the true value is 0.545. Fixed, with known-value tests; one of the original tests expected a negative kappa for two constant labellers, where the true value is 0.
  - **`compare.py`** read `metrics.json`, which keeps a run's first scoring. For EXP-12 that showed 0.235 with no critical or omission rate; the current scoring is 0.054 (EXP-23). It now reads the newest scoring by default or a named file, records which file each row came from, takes the source from `source` (current runs) or `mode`, shows an unmeasured rate as n/a rather than 0, and imports matplotlib only to plot.
  - **`adjudicate.py`** read an old run layout (`per_fixture`), so new runs gave no claims. It now:
    - reads the flagged-claims files and refuses held-out cases through the split's own check;
    - writes a blind sheet (claim text, field, declared sources and an opaque case alias; no model, run, case ID or verifier verdict, since showing the verifier's sub-type would anchor the labellers);
    - writes a separate key file for unblinding, and the case inputs per alias;
    - reports agreement with the verifier overall, per sub-type and per model.
  - **Docs:** `ERROR_TAXONOMY.md` and `LABELLING_GUIDE.md` now match the metric:
    - omissions are their own rate, not "always critical";
    - uncertainties are never charged 6 or 7;
    - faulty-derivation criticality follows §5;
    - correct medicine with no source in the inputs is still an error;
    - the null, rounding and restatement rules are stated;
    - `judge_v3` (DEC-23) is named.
  - **Notebook 41:** installs with `GITHUB_TOKEN` (the repo is private), reads EXP-12 at its current scoring, and writes the figure to `runs/comparison/`, not into a finished run.
- **Back to Benny (spec in the PR #5 review comment):**
  - **The sampling design.** It samples only flagged claims, so it measures the verifier's precision but never its misses. Metric §8 asks for a sample across claims, with a consensus label and agreement per sub-type.
  - **The table's shape:** models × (source × arm).
- pytest: 454 passed.
- **Notion sync:** Tasks → Benny's taxonomy, comparison notebook and labelling kit: In review, with this note.

## 2026-11-09 — Hana (NLP A: reasoner prompt v2 from the EXP-12 findings)
- **`prompts/reasoner_v2.md`** (new; `reasoner_v1` unchanged and still the default). Built from Benny's labels of the 28 EXP-12 flags (22 real reasoner errors, 6 verifier mistakes):
  - Right fact, wrong or missing citation (10): cite the passage that states the rule or threshold used; no same-topic padding.
  - Blocking factors checked only against `rule_ids` (fx04, fx05): for a decision against a treatment, `rule_ids` include the passages setting the limits the patient fails.
  - Uncertainties stating rules with no source (8): name what is unknown and the affected rule's ID in square brackets; no rule content there.
  - Conclusions written as claims (6): new "What not to write" section ("is eligible", "the rulebook supports…").
  - Inferences, predictions and added note details (5, two critical): no predictions, no general knowledge, no "only/never" summaries; nothing beyond the quoted note words.
  - Rules applied when their conditions don't hold (fx03 BP, fx05 ASPECTS): check a rule's conditions before using it.
  - The prompt's own instructions restated as rules (fx02 last-known-well): the instructions are not a source.
  - Numbers copied exactly (0 cases in EXP-12; kept as prevention).
  - One worked good/bad example with invented IDs and numbers (`PASSAGE-A/B`, 42.0 mL vs 60 mL) that appear in no passage, fixture or note.
  - About 490 words longer than v1; the system prompt stays cached.
- **`tests/test_reasoner_v2.py`** (7 tests, fake model, no network): v2 holds all 47 passage IDs; v1 kept and v2 contains its input section; one phrase per fix; the example shares no ID or number with any fixture; a fake reply assembles and validates as `reasoner_v2`; a bracketed-ID uncertainty is valid; a fixture run records `prompt_version` `reasoner_v2`.
- **For Parsa:**
  - Paid v1 against v2 run on the fixtures is yours (no paid calls made).
  - Rulebook gap: "time counts from last known well when onset is not witnessed" is in the prompt but in no passage. v2 stops the model restating it; adding a passage is your call.
  - Uncertainties have no sources field, so the verifier gives them no declared sources; whether it should read the bracketed IDs is for the verifier side.
- **Windows fix in two tests:** `test_reasoner.py` (line 107) and the new v2 test read `system_prompt.md` back with `encoding="utf-8"`. It is written as UTF-8, and Windows' default encoding garbled characters such as `–`, so both failed on Windows only. No change on Mac, Linux or Colab.
- pytest: on Windows (Python 3.14), `test_reasoner_v2.py` + `test_reasoner.py` 15 passed. Full suite on Linux: 409 passed, 4 skipped; 3 failed in `test_haemorrhage.py` only because PyTorch was not installed (unrelated).
- **Notion sync:**
  - done: Tasks → "Reasoner prompt v2 from the EXP-12 findings": note for Parsa on the last-known-well gap.
  - pending: same task → In review once the pull request is open; Experiments row for the paid v1/v2 run (Parsa).

## 2026-11-08 — Claude Code session (Vision B Task 1: cross-fitted operating point, code and notebook)
- **Scope:** Phase 3, Vision B, Task 1 from `docs/team/mohamed_vision_b.md`. Code and notebook only; the Kaggle run is Mohamed's, and the threshold decision is Parsa's.
- **`vision/haemorrhage/operating_point.py`** (tested): EXP-6 set 0.278 for 95% sensitivity and measured it on the same validation scans. This module cross-fits instead:
  - validation patients split into two halves with a fixed seed (20261108), whole patients per half, stratified by whether a patient has a bleeding scan;
  - threshold chosen for ≥ 95% sensitivity on one half (`aggregate.threshold_for_sensitivity`), measured on the other, then swapped; the two held-out results are pooled;
  - thresholds are set on the raw top-2 score. Platt calibration is monotonic (checked: temperature > 0), so the calls are identical and the calibrator, itself fit on the full validation split, does not leak into the held-out halves. The proposal is converted to the calibrated scale;
  - 95% Wilson intervals and a patient-level bootstrap (scans of one patient are not independent); 20 seeds for the spread; a sweep of targets 0.90–0.99 for the specificity cost of each extra point of sensitivity;
  - proposal rule, fixed before the run: the lowest full-split target (0.95–0.99) whose held-out sensitivity, median over seeds, reaches 0.95. A stricter alternative (lower Wilson bound ≥ 0.95) is reported beside it;
  - `reasoning_text` drafts the written reasoning from the run's numbers, including EXP-16's ISLES'24 false-positive rate if it is entered.
- **`notebooks/22_visionB_operating_point.ipynb`** (Kaggle): rebuilds the RSNA split and refuses to go on unless its fingerprint matches EXP-6's; keeps validation rows only; refuses to run if `ALLOW_RSNA_TEST` is set. It recomputes the validation slice predictions with EXP-6's `best.pt` from a private weights dataset, and stops if the scan AUC is more than 0.005 from 0.977. Writes `config.json` first; then `metrics.json`, `reasoning_draft.md` and `tradeoff.png`, after an ID check. `REPO_REF` is `mohamed/operating-point`.
  - Sections 3–8 were dry-run locally on four small fake shards with a fake model. Not run on Kaggle or real data: every result and the timing are unmeasured.
- **Plan:** layout lists `operating_point.py` and notebook 22; the §7 Vision B tuning item notes the code is ready and the run and decision pending (not ticked). README lists notebook 22.
- pytest: 468 passed (13 new).
- **Notion sync** (not done; for Mohamed or Parsa):
  - Tasks → Vision B "Operating point": In progress; code and notebook on `mohamed/operating-point`.
  - Experiments → a new row (Status *Running*) before the Kaggle run; afterwards the held-out sensitivity and specificity with intervals, the proposed threshold and the commit.
  - Decisions → Parsa records the threshold once he has chosen between the proposed and strict values.

## 2026-11-07 — Lorna (Vision A error analysis, EXP-15, fold 0; 14 working sessions on 6–7 Nov, folded into one entry at review)
- Task 1 of `docs/team/lorna_vision_a.md`. No held-out case was used; the notebook prints counts only, and case IDs stay in the Drive tables.
- **`vision/perfusion/error_analysis.py`:**
  - `write_config` writes once and records checksums, no case IDs.
  - `join_tables` joins EXP-14 to EXP-1's `within_tmax6` rows. It refuses missing columns, repeated cases, and reference volumes that disagree.
  - `case_features`: size band, occlusion site with its nearest label and distance, territory, onset to imaging, the infarct's Tmax −30 and unreliable shares, and its Tmax > 6 s share.
  - `group_summary` / `summarise_features`, `miss_profile`, `visibility_comparison`, `crossing_counts`.
  - `figure_cases` / `case_figure` / `figure_summary`, with anonymous labels.
  - `build_metrics` / `write_metrics` refuse case IDs and record the installed commit next to `config.json`'s.
- **`tests/test_error_analysis.py`:** 47 synthetic tests.
- **`notebooks/12_visionA_error_analysis.ipynb`** (CPU Colab):
  - installs the exact commit and checks it, with a forced reinstall, because the package version never changes;
  - checks the split and tarball checksums;
  - writes `per_case.csv`, `metrics.json` and `figures/` to Drive `runs/EXP-15_visionA_error_analysis/`.

  `00_template.ipynb` still installs the old way.
- **EXP-15 run at `c442a54`:**
  - 9 of 26 fold-0 cases are Dice-0 misses;
  - volume error and cut-off crossings by size band;
  - figures of every miss and the 3 best cases.
- **`docs/ERROR_ANALYSIS_VISION_A.md`:** a plain-English draft with Lorna's visual readings (drafted with AI help, checked by Lorna).
- **Open points recorded:**
  - territory uses the unconfirmed TopCoW numbering and no distance limit;
  - EXP-15's `config.json` predates the results commit;
  - the first Colab run stopped on EXP-1's two rows per case, now handled.
- `pytest` on Lorna's Windows laptop: 435 passed, 3 skipped, and the same 9 Windows-only failures as before the work (archive listing, bash parse, corpus encoding, nnU-Net splits).
- **Notion sync:**
  - Experiments: EXP-15 (commit, findings, link to the draft).
  - Tasks: the error-analysis row through In progress to In review.

## 2026-11-06 — Claude Code session (Phase 3 team briefs and the template notebook)
- **Roles agreed by Parsa:**
  - Parsa keeps the long GPU runs, the paid model runs, `contract/build.py`, the verifier fixes and the decisions.
  - Lorna: Vision A error analysis and confidence.
  - Mohamed: Vision B operating point and ISLES'24 check.
  - Hana: model shortlist, prompt v2, providers and real inputs.
  - Benny: error taxonomy, comparison notebook and labelling kit.
  - Lorna and Mohamed are the blind labellers.
- **`docs/team/`:** a shared guide (access, rules, branches and pull requests, run records) and one brief per person, with inputs, deliverables and "done when". Teammates make no paid API calls.
- **`notebooks/00_template.ipynb` and `data/template.py`** (tested): install, mount, checksum checks, unpack, and one development case's maps with infarct and occlusion outlines plus the note-safe clinical row.
  - The figure was checked locally on one development case.
  - It has not yet run on a Colab T4, so the 5-minute target is unmeasured.
- **Collect** now also brings back nnU-Net's validation masks (`validation/*.nii.gz`) for folds 1–4. EXP-14's fold-0 masks stayed on the destroyed pod, so they need regenerating for the error analysis.
- **Plan §7:** onboarding started; Phase 3 owners recorded; the old "assign Phase 2 tasks" item marked superseded.
- pytest: 436 passed.
- **Notion sync** (approved by Parsa, done 6 Nov 2026):
  - Documentation → "Team guide (Phase 3)", with one brief subpage per person.
  - Tasks → 17 Phase 3 rows, each linked to its brief: 13 for the team and 4 for Parsa. Lorna, Mohammed Elzeiny and Hana Ayad are set as Owners. Benny's 3 rows have no Owner yet, because his Notion account wasn't found.
  - The two existing Phase 3 rows are assigned to Parsa.
  - The onboarding row is In progress, and the template row is In review.
  - The charter's Person column is filled in.

## 2026-11-06 — Claude Code session (Vision B Task 2: ISLES'24 domain-shift check, code and notebook)
- **Scope:** Phase 3, Vision B, Task 2 from `docs/team/mohamed_vision_b.md`. Code and notebook only; the real run is Mohamed's, on Colab.
- **`vision/haemorrhage/isles_check.py`** (tested): scores each development case's `ncct_native` the way the RSNA shards were built:
  - turned to RAS, then laid out as DICOM stores axial slices (front of the head at the top, patient's right on the left). ISLES'24 sources are LAS or LPS, so this matters;
  - averaged into 5 mm slabs in HU, each native slice weighted by its overlap. 5 mm is RSNA's documented thickness (plan §5.2), not measured: the shards' `z_mm` could confirm it on Kaggle;
  - padded to a square with air, then `rsna.window_hu` (brain, subdural, bone) at 320 px, then JPEG at quality 95 and back, as the shards were stored;
  - scored with EXP-6's top-2 mean and Platt calibration through `infer`, against the calibrated threshold in `calibration.json`.
  - The native thin slices are scored too, as a second reading, to show what the reformatting changes.
  - Reports hold counts, rates with 95% Wilson intervals, percentiles and a histogram. The per-case table holds case IDs and is saved only to Drive. Figures are labelled by rank and score. A held-out request stops the run; any other failure is kept in the table and counted.
- **`notebooks/21_visionB_isles_check.ipynb`:** thin Colab notebook: install, checksums, `config.json` before the run, scoring, results, the top-scored scans with a list of common false-alarm causes, then `metrics.json`, `per_case.csv` (with an empty `review` column), two figures and a read-back. `REPO_REF` is `mohamed/isles-check` until the pull request is merged.
  - Sections 4–8 were dry-run locally on synthetic volumes with a random-weight model and a stubbed Drive. It has not run on Colab or on real data, so its timing and every result are unmeasured.
- **Known limits, to state with the result:** no bleed labels, so positives are *presumed* false positives; the threshold (0.278) is provisional until Task 1; in-plane field of view is not matched to RSNA's (ISLES'24 spacing is 0.305–0.562 mm, `docs/DATA.md`), and is recorded per case so it can be checked. `set_field_of_view` exists for that check but is off.
- **Plan:** layout lists `isles_check.py` and notebook 21; the §7 domain-shift item notes the code is ready and the run pending (not ticked). README lists notebook 21.
- pytest: 455 passed (19 new).
- **Notion sync** (not done; for Mohamed or Parsa):
  - Tasks → Vision B "Domain shift check": In progress; code and notebook ready on `mohamed/isles-check`.
  - Experiments → a new row (Status *Running*) before the Colab run; its ID goes in `EXPERIMENT`. Afterwards: the presumed false-positive rate at 0.278 with its interval, the top-scan review by count, and the commit.

## 2026-11-05 — Claude Code session (DEC-21 rented GPU, EXP-13 and EXP-14, notes, trial passages checked; Phase 2 closed)
- **DEC-21 pod tooling** (commit `5767cd4`): `src/strokecopilot/compute/` and `scripts/pod` (`offers`, `up`, `status`, `collect`, `down`, `monitor`).
  - `up` refuses without Parsa's `--go`, with a secret in the repo, with a balance under 1.5 × the estimated cost, with uncommitted code under `src/`, or with a tarball that fails its checksum. It destroys the pod if setup fails.
  - A watchdog on the pod destroys it at the hour limit. `collect` brings outputs to the Mac and to Drive (via Drive for desktop), then destroys the pod and confirms it is gone.
  - The Mac is the data relay: tarball from Drive for desktop, repo as `git archive` at the pinned commit. No token ever goes on the pod.
  - Host filter relaxed by Parsa (DEC-21 amended): no datacenter requirement.
- **New this session, uncommitted:**
  - `compute/monitor.py`: a local progress page (`scripts/pod monitor EXP-<id>`, served from `runs/EXP-<id>/monitor/`), with epochs, ETA, pseudo Dice, cost and the watchdog deadline.
  - `status` and the monitor read every nnU-Net training log, so a resumed run counts all its epochs.
  - `collect` also brings back `job_part*.log`.
  - **Host contract check.** `up` refuses an offer whose contract ends less than 1 h after the run's hour limit, and `offers` shows the hours left. This came from EXP-14 (below).
- **EXP-13** (dummy, 3 epochs, Vast RTX 4090, California, 0.459 USD/h):
  - Epochs took 37–44 s, against about 121 s on the Colab L4, so about 3× faster.
  - GPU utilisation median 99%, with no worker bottleneck. Resume works.
  - Two setup failures were fixed in the scripts:
    - the PyTorch runtime image has no C compiler for `torch.compile` (gcc is now in the bootstrap);
    - the local network blocks outbound SSH (fixed by using the VPN).
- **EXP-14, nnU-Net fold 0, 300 epochs:**
  - **Dice 0.218 ± 0.249** (median 0.111), Dice on reliable Tmax 0.221, abs volume error 19.6 ± 23.3 mL (signed −7.0 mL), over 26 development cases.
  - **Threshold baseline on the same cases** (EXP-1, within Tmax > 6 s): Dice 0.123, 20.5 mL. Dice nearly doubles, but volume error, which matters more for the rules (§8.1), barely moves.
  - Wrong threshold crossings at 21/31/51/70 mL: 5/5/3/3 of 26.
  - By lesion size:
    - three of the four infarcts above 95 mL score 0.60–0.70; the fourth (101 mL) is predicted empty;
    - lesions under 5 mL average 0.12;
    - 9 of 26 cases score 0.
  - The 3-epoch dummy had a similar mean (0.236) by over-predicting everywhere. A Phase 3 error analysis is needed.
  - **Interrupted and resumed.** Vast stopped the pod at 18:25:53 UTC because the host's contract ended: the offer had about 3.2 h of contract left against a 5.5 h limit, and I had not checked. The epoch-269 checkpoint was on disk. The instance was restarted about an hour later, the watchdog re-armed, and training resumed at epoch 270. Hence the new contract check.
- **GPU spend** (measured from Vast credit; DEC-21 budget 8–15 USD):

  | Run | USD |
  |---|---|
  | First pod, no SSH | 0.11 |
  | EXP-13 | 0.31 |
  | EXP-14 (includes the stopped hour's storage) | 2.14 |
  | **Total** | **2.56** |

  Credit left: 2.44.
- **Estimate for folds 1–4** (not measured): about 3.6 h each, as 3.1 h of training at 37.6 s per epoch plus about 25 min of upload, preprocessing and scoring. That is about 1.8–2.0 USD each and 7–8 USD for the four, about 10 USD for the project so far. Running them needs a top-up of about 8 USD.
- **Notes** (commit `6b1ceb4`): `nlp/notes/generate.py` wrote 129 development notes to Drive `artifacts/isles24_notes_v1/`.
  - Durations only, no invented clock times, and missing rule inputs written as not recorded.
  - New fact keys: `atrial_fibrillation`, `onset_to_door`, `door_to_imaging`, `found_to_imaging`, `found_to_door`. They change the extractor prompt for new runs.
- **Trial passages checked by Parsa** against the DAWN and DEFUSE-3 supplementary protocols, with pages. All 47 passages are now checked.
  - 15 of 18 facts matched, and 1 was found on a second look (DEFUSE-3 pre-stroke mRS 0–2).
  - 2 were corrected: DEFUSE-3 needs a mismatch ratio **above** 1.8 and a mismatch volume **above** 15 mL (strict). The fix is in the passage, `rules.py`, §8.3 and `docs/CLINICIAN_REVIEW.md`. No scenario decision changed, and the scenario hash is unchanged.
  - `corpus/passages.jsonl` sha256 is now `acb29e213b10…`. The reasoner gets the whole rulebook, so its prompt changes too: EXP-12's numbers belong to the earlier corpus.
- **Run records:** every run EXP-1 to EXP-14 now has an Experiments row and a `runs/` folder with metrics. EXP-3 (a stop record, no numbers) and EXP-5 (Parsa's pasted metrics) got their `metrics.json` on Drive.
- **Phase 2 is closed:** every §7 task is ticked, and the onboarding block stays deferred under DEC-6.
- **Still open, not blocking:**
  - a free EXP-12 rescore for the corrected critical rate;
  - Parsa's OK on `HALLUCINATION_METRIC.md` §9 points 8–12 and on the RSNA split.
- pytest: 432 passed.
- **Notion sync:**
  - done: Experiments → EXP-13 and EXP-14 rows;
  - done: Tasks → the 8 Phase 2 rows (nnU-Net baseline, corpus, reasoner v0, verifier v0, haemorrhage classifier, metric definition, scenarios, seeded errors) → Done, with results in Notes; the Colab template and onboarding rows stay in Backlog, noted as deferred under DEC-6.

## 2026-11-05 — Claude Code session (EXP-12 rescore with the rulebook in code; critical-rate fix)
- **EXP-12 rescore** (commit `f4d754f`, no API calls): rate 0.080 (content 0.060) over 351 claims, as before.
  - No omissions: the reasoner ignored no hard rule.
  - No uncertainty omissions: it named every unknown that decided an outcome.
  - Critical rate reported as 0.185.
- **The 0.185 used the wrong denominator.** `docs/HALLUCINATION_METRIC.md` §6 divides by all claims whose criticality is determined, clean ones included. The code left clean claims undetermined, so 0.185 was the share of flagged claims that are critical.
  - **Fix:** a clean, uncited or mis-attributed claim is now "not critical" whenever the rulebook decides at least one decision for the case, and undetermined only when both decisions are.
  - A test covers it. The corrected EXP-12 rate needs one more free rescore.
- pytest: 390 passed.
- **Notion sync:** none until the rescore.

## 2026-11-05 — Claude Code session (scenarios.py and the rulebook in code)
- **`nlp/verifier/rules.py`, the rulebook in code:** `decide()` returns each decision as yes, no or undetermined, with the passage IDs, the inputs that decided it, and any unknown or unobservable inputs that left it undetermined (`docs/HALLUCINATION_METRIC.md` §5).
  - **Thrombectomy** covers ICA/M1 by window, ASPECTS, pre-stroke mRS and age; M2 (dominance unobservable, so undetermined); basilar (PC-ASPECTS unobservable, so undetermined); and DAWN and DEFUSE-3 when ASPECTS is unknown at 6–24 h.
  - **Thrombolysis** covers haemorrhage, absolute and relative Table 8 contraindications, 4.5 h with a disabling deficit, and the extended window (EXTEND, sleep midpoint, TRACE-III), which applies only when thrombectomy is not an option.
  - **Conventions** are listed in the module and added to `docs/CLINICIAN_REVIEW.md` as question 8.
  - It reproduces all ten expected fixture decisions.
- **Omission checks:** haemorrhage with either treatment recommended; INR above 1.7 or a recent head injury with thrombolysis; beyond 4.5 h with no extended route; core 70 mL or more with thrombectomy on DEFUSE-3 alone; no occlusion with thrombectomy. Unknown inputs that leave a decision undetermined but are not named in `uncertainties` are counted separately.
- **Criticality:** for a wrong number or invented finding, the claimed value is substituted and the rules are run again; it is critical if a determined decision changes. For wrong rule, misapplied rule and faulty derivation, it is critical if the claim concerns a deciding input or passage. Uncited and mis-attributed claims are never critical.
- **`nlp/notes/render.py`:** facts to an ED-style note with a few phrasings per field. Every fact's span is verbatim, and silent facts are left out. `notes/generate.py` will reuse it.
- **`scenarios.py`:** 150 synthetic scenarios from seed 20261105.
  - Each has a contract-valid vision JSON, a rendered note with its fact sheet, and the rule-derived expected decisions.
  - Paths are capped per decision pair, with fewer when undetermined, and 16 key routes each have at least 3 scenarios.
  - Decisions: 85 fully determined; thrombolysis yes 40, thrombectomy yes 32.
  - `manifest()` gives counts and a hash.
- **Corpus cross-check (the plan's "review the rule code against the corpus"):** a test checks that every threshold the rules use is stated in the passage they cite, and that every cited passage is about the right treatment.
- **`pipeline/end_to_end.py`:**
  - One case runner for fixtures and scenarios.
  - Criticality and omissions now fill the summary, so the critical and omission rates are no longer None.
  - The rescore works for both sources and for EXP-12's existing folder.
  - Notebook 50 prints the critical and omission rates and each case's omissions.
- pytest: 389 passed.
- **Notion sync:**
  - pending: Tasks → `scenarios.py` done.

## 2026-11-05 — Claude Code session (rescores after the five fixes)
- **EXP-12 rescore** (commit `83085b4`, no API calls):
  - 351 claims (fewer, because ranges like "3 to 5" are no longer read as numbers);
  - hallucination rate 0.080 (content 0.060), against 0.235 at first and 0.132 after the first rescore;
  - attribution accuracy 0.979; 28 flags.
- **EXP-11 rescore** (same commit): seeded recall is unchanged. Lowest strict recall is 0.80 (uncited rule) and false flags are 0.010, so the bar is still met at τ ≤ 0.7. The fixes made the verifier more lenient without losing a planted error.
- **Claude's reading of the 28 flags** (not human adjudication):
  - all 9 real reasoner errors found earlier (6 clear, 3 borderline) are still flagged;
  - about 19 are verifier errors, nearly all in the three groups left for Phase 3:
    - rules in uncertainties with no passage retrieved;
    - judge errors on conditional or negative applications;
    - meta-statements extracted as claims.
  - So, by this reading, the verifier falsely flags about 5% of real claims, and the reasoner's rate is about 2–2.5%.
- **Notion sync:**
  - done: Experiments → EXP-12 row (value 0.080 and this reading).

## 2026-11-05 — Claude Code session (verifier: five code-side fixes from EXP-12)
- **Fixes, from the six groups of verifier errors in EXP-12** (recorded in `docs/HALLUCINATION_METRIC.md` §9 point 12, with tests):
  - Values written as bounds or sets ("6 or more", "within 6 h", "0 or 1", "3 to 5") are checked as such. A false one is a faulty derivation.
  - "null", "unknown" or "undefined" for a null field is supported.
  - A note fact claimed without a value, or with only its unit, is supported when the note states it.
  - "Not given" for a subject outside the fact list needs every word of the subject to be absent from the note, not just one.
  - "3 to 5" ranges are rule text and are not read as numbers.
  - An unused cited passage is charged (7) only to claims that rest on a rule.
- **Left for Phase 3**, because they need prompt changes measured against human adjudication:
  - trivial statements extracted as claims (extractor prompt v2);
  - rules in uncertainties with no passage found above τ (candidate selection);
  - judge errors on conditional and negative applications (judge prompt v3).
- pytest: 350 passed.
- **Notion sync:** none until the rescores.

## 2026-11-05 — Claude Code session (EXP-12 rescore and a reading of every flag)
- **EXP-12 rescore** (commit `56a7534`, no API calls), same 371 claims:
  - hallucination rate 0.132 (content 0.102), down from 0.235;
  - attribution accuracy 0.967;
  - 49 flags.
- **Claude read all 49 flagged claims** (`flagged_rescore.json`). This is not the human adjudication the plan requires.
  - **6 clear real errors:**
    - one invented finding: fx03 "headache and vomiting before the weakness", where the note puts the vomiting in the ambulance, after the weakness;
    - five citation errors: "eligible for thrombectomy" citing only thrombolysis passages (fx02, fx05); the haemorrhage contraindication without `AHA2026-T8-ich` (fx03); two DEFUSE-3 blocking factors with DEFUSE-3 in no `rule_ids` (fx05).
  - **3 borderline:**
    - "the window counts from last known well", our prompt's instruction stated as a rule;
    - a true 4.5 h rule cited to the haemorrhage passage, which the verifier labelled wrong rule rather than mis-cited;
    - "labs are not needed for thrombectomy per the rulebook", which the rulebook does not say.
  - **About 40 verifier errors, in six generic groups:**

    | Group | Count |
    |---|---|
    | Null, unit or bound given as the value ("6 or more", "0 or 1", "mmol/L") | 9 |
    | "Not given" matched on one shared word (`other:` subjects) | 7 |
    | Judge errors on conditional or negative applications | 5 |
    | Rules in uncertainties with no passage retrieved above τ | 6 |
    | Unused passages charged to plain note or vision facts | 6 |
    | Ranges written "3 to 5" read as single numbers, and trivial non-claims | 7 |

- **What this means:**
  - On real output, the verifier's false flags are about 11% of claims, above the 0.10 bar. The 0.010 on the gold outputs was optimistic, because they were written in the verifier's style.
  - By this reading, the reasoner's rate is about 2–2.4%.
  - The Phase 2 acceptance criterion is still met. The real measure of the verifier is Phase 3's human adjudication.
- No code changed.
- **Notion sync:**
  - done: Experiments → EXP-12 row updated with the rescore and this reading.

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
