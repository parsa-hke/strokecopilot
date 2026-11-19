# CHANGELOG

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
