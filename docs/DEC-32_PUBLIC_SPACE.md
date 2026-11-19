# DEC-32: the demo is a public Hugging Face Space

Drafted 19 Nov 2026 in the Phase 4 session, before any held-out case is opened. **Approved as drafted by Parsa on
19 Nov 2026**, with choices a–f as recommended (§10) and one change: the demo's spend limit is small, because he expects
fewer than 20 live runs in all (§7). Parsa chose a public Space on 19 Nov 2026. This document records that choice and fixes how the Space is
built, and it amends plan §8.5, §4.4, Phase 4's demo task and acceptance line, and DEC-31's demo step (§9 below).
Nothing here changes the evaluation: no live answer from the Space is ever counted in a result.

## 1. Decision (the plan §2 row)

> **DEC-32** | The demo is a public Hugging Face Space, replacing §8.5's Gradio app run from Colab. It is a Docker Space on
> free CPU-basic hardware (Parsa's HF Pro account): a small FastAPI server serving a static page (HTML/JS). It has
> three parts:
> - a scrolling story of the pipeline;
> - a case explorer, where every claim is clickable and its source lights up;
> - the results as interactive charts.
>
> It also has one live element, a scenario builder. The visitor sets structured values only (no free text); the project's
> note generator writes the note, and Sonnet 5.5 with prompt v3 and the rulebook, then the verifier (`judge_v3`, τ 0.6),
> run live, with the same settings as EXP-37. Every failure, slow call, or spent budget switches to the nearest
> precomputed scenario, with a message saying so. Spending is capped three ways:
> - a result cache;
> - per-visitor and daily limits in the server;
> - a dedicated Anthropic key in its own console workspace, with a monthly spend limit Parsa sets.
>
> The key lives only in the Space's secrets, which Parsa sets; that is added to §4.4's allowed places for this one key.
> Data on the page:
> - RSNA appears only as aggregate numbers;
> - ISLES'24 images appear with attribution under CC BY-NC-SA 4.0 (checked on Zenodo, 19 Nov 2026), with cases shown
>   by alias, never by ID;
> - guideline text appears only as our rulebook passages.
>
> It is built in `demo/space/` on fixtures, scenarios and development results before unlocking, and filled with held-out
> results after DEC-31's analyses. Every push to the Space is a publish and happens only on Parsa's word. Estimate:
> 15–20 h of session work. Live use is bounded by the workspace limit, kept small (about $5 a month suggested; Parsa expects fewer than 20 live runs) | Decided — Parsa, 19 Nov 2026

## 2. Architecture

- **Docker Space, CPU basic** (2 vCPU, 16 GB RAM, 50 GB disk that does not persist; free; Docker Spaces need a paid
  plan, which HF Pro is). The image holds:
  - a FastAPI server, run by uvicorn on port 7860, as user 1000 (HF's rule);
  - the static frontend;
  - the `strokecopilot` package, installed from a copy assembled at build time;
  - the precomputed bank (§4);
  - `BAAI/bge-base-en-v1.5` on CPU, downloaded at build time, with the 47 passage embeddings precomputed.
- **Why not static-only:** a static Space cannot hide a key. **Why not Gradio:** it would limit the design. A Docker
  Space gives full design freedom, and a server that holds the key.
- **Frontend:** plain HTML, CSS and JS with no build step:
  - one chart library, vendored at a pinned version;
  - no runtime CDN, analytics, cookies or third-party calls, so the page works whenever the container is up.
- **Endpoints:**
  - `GET /` serves the page;
  - `GET /api/bank/...` serves precomputed cases, scenarios and chart data (static JSON);
  - `POST /api/run` takes a builder input and returns a run id;
  - `GET /api/run/{id}/events` streams progress as server-sent events;
  - `GET /healthz` reports the health check (§5).
- **Layout in the repo:**
  - `demo/space/` holds the Dockerfile, the README metadata, the server, the frontend and the build script;
  - the server's logic (input checks, matching, limits, the live runner) goes in `src/strokecopilot/demo/`, with
    tests in `tests/`, as CLAUDE.md asks;
  - `scripts/space_build` assembles `demo/space/build/`, the exact tree that is pushed, and runs the leak scan (§6)
    on it. The existing one-line `src/strokecopilot/demo/app.py` is replaced.

## 3. The page

Every screen carries a fixed banner, *Research prototype — not for clinical use*, that cannot be scrolled away or
dismissed (§8.5). Every API reply carries it too, as a field.

1. **The story.** A scrolling walk through one case, shown by alias:
   - perfusion maps;
   - nnU-Net's core and the Tmax > 6 s penumbra;
   - the bleed check;
   - the rulebook;
   - the reasoner's answer;
   - the verifier's tracing.

   Each step says what was measured and how well it worked (Dice, volume error, cut-off crossings, Vision B's
   sensitivity, the 68 of 129 false alarms and Item 2's negative result), so the weak points are shown, not hidden.
2. **Case explorer.**
   - The cases: the 5 fixtures, the 150 synthetic scenarios, and real cases by alias (development now, held-out
     after Phase 4).
   - Each case shows its note, its contract, both decisions, and the rationale.
   - Clicking a claim lights up its source: the contract field, the note line, or the passage with its ID and pointer.
   - A flagged claim shows its sub-type and the verifier's reason.
   - Real cases also get the predicted and ground-truth vision arms side by side.
   - **Proposed addition:** a model switch showing the six models' saved answers for the same case (EXP-37, EXP-47,
     and the held-out runs). It is free and shows C9 at the level of one case.
3. **Results (interactive charts):**
   - C9's rates per model, with intervals (rescored, points 19–20);
   - the rulebook effect (DEC-29);
   - the human check (EXP-41, κ 0.27, verifier–human agreement 0.61, the verifier's rates as upper bounds);
   - the tuned 4B: its rate against its treatment bias (thrombectomy yes caught 10/35);
   - Item 2;
   - after Phase 4: held-out correctness, error propagation and the ±20% core flips.

   Each chart says what it shows and what it does not, in one sentence.
4. **Scenario builder** (§4).
5. **About:** the team, data attributions and licences, the limits (§9 of the plan), and the code link.

## 4. The live element: the scenario builder

**Inputs.** The fields of a synthetic scenario (`scenarios._draw`), as controls, with nothing typed:
- **Times:** onset type (witnessed, unwitnessed, wake-up), minutes to imaging, and sleep midpoint for wake-up.
- **Patient:** age, sex, NIHSS, pre-stroke mRS, the deficit (from the generator's list).
- **Contraindications and drugs:** contraindication (none, or one of the generator's kinds), anticoagulant and INR,
  antiplatelet.
- **Imaging:** core and penumbra volumes, occlusion site, ASPECTS (unknown or a value), the bleed flag.

Each control is bounded by the range the evaluated sets cover (the 150 scenarios, the fixtures and the development
contracts), with steps fixed at build time and listed on the page. Fields the builder does not expose (imaging clock,
referral, blood pressure, glucose, routine drugs) are drawn by a generator seeded from the inputs' hash. So the same
inputs always give the same note.

**One run:**
1. Validate the inputs (rejected outside the bounds).
2. Build the contract and validate it against `schemas/vision_output.json`.
3. Render the note with `notes/render.py`.
4. Compute `rules.decide`, shown as "what our rulebook says".
5. **Live:** Sonnet 5.5, `reasoner_v3` with the rulebook, effort high, max_tokens 20,000, streamed; the reply is
   validated against `schemas/recommendation.json`.
6. **Live:** the Sonnet extractor (effort medium) and the Haiku judge (`judge_v3`, τ 0.6, top_k 5). Judge calls run
   in parallel.
7. Show the claims and flags as in the case explorer.

These are EXP-37's settings, except that the calls are live rather than batched. A batch request is the same model
and request, at half price and with no latency limit. Progress streams step by step. If incremental parsing holds up
in testing, the rationale entries also appear as the reasoner writes them.

**Precomputed bank and fallback.**
- **Presets** are the 150 scenarios and 5 fixtures, with Sonnet's saved, verified answers from EXP-37
  (`scenarios_rulebook`, `fixtures_rulebook`, rescored traces). They cost nothing new. One scenario has no output
  (149 of 150), so that preset is left out.
- **Nearest scenario:** the bank entry with the same `rules.decide` outcome for both treatments. Among those, the one
  closest on the inputs (normalised distance; categories must match first, then numbers). The page names the values
  that differ.
- **Fallback triggers** (each with a plain message, e.g. "the live model did not answer in time; showing the closest
  precomputed scenario"):
  - any error, refusal, or schema failure after the retries;
  - a step over its time limit;
  - the visitor's limit, the daily cap, or a full queue;
  - live runs switched off.
- **Result cache:** a finished live run is stored under its canonical input hash. A repeat is served from the cache,
  marked "computed live earlier". The cache lives in memory and is lost on restart (§7).

## 5. "Flawless": what the server guarantees

**Timeouts and retries** (values to be set from the dummy):
- reasoner 150 s;
- extractor 120 s;
- each judge call 45 s;
- the whole run 300 s.

The SDK retries 429, 5xx and connection errors twice with backoff. A malformed reply is retried once. Then the run
falls back.

**Limits:**
- **Concurrency:** at most 3 live runs at a time; a 4th falls back at once instead of waiting.
- **Per visitor** (by client IP from the forwarded header, hashed in memory, never logged): 3 live runs an hour,
  10 a day.

**Health check:** `/healthz` reports:
- that the bank loaded;
- that the key is present (yes/no only);
- whether live runs are on;
- spend today against the cap;
- the last live success and failure times.

The page shows a live/precomputed badge from it.

**Kill switch:** a Space variable `LIVE_ENABLED`. Setting it to `0` turns every run into a preset, without a rebuild.

**Tests:**
- unit tests for input checks, matching, limits and the fallback paths, with fake models (no spend);
- a load test against the local container with a fake model that is slow or fails on a schedule: 50 simulated
  visitors, every request answered within its limit, no 5xx, every failure falling back;
- a short load test on the real Space before publishing, with live runs off.

## 6. Data on a public page

**ISLES'24.** Zenodo record 17652035 (v7) is **CC BY-NC-SA 4.0**, rechecked 19 Nov 2026.
- Images, maps and masks may appear with attribution.
- The Space is non-commercial.
- ShareAlike: the Space's README declares `license: cc-by-nc-sa-4.0`.
- The About section cites the dataset paper (Riedel et al., arXiv:2408.11142) and the challenge paper (de la Rosa et
  al., arXiv:2408.10966), as `docs/DATA_USE.md` does, plus TopCoW if the Circle of Willis masks are shown.
- (The Zenodo page lists the dataset paper as arXiv:2408.09259, but that number is an unrelated paper. Ours is
  right.)
- Case IDs never appear: cases have aliases (`Case D-01`…, `Case H-01`… after Phase 4). The alias map stays on Drive.

**RSNA:** aggregate numbers and charts only. No images, IDs or per-scan values (DATA_USE).

**Guidelines:** our rulebook passages only, with their pointers. No PDF text.

**Leak scan.** `scripts/space_build` refuses to finish if the build tree contains any of these:
- an ISLES'24 case ID (checked in code against the split, never printed);
- an RSNA ID pattern;
- a key pattern;
- a file outside the allowed list.

## 7. Spending

**Measured cost per live run.** From EXP-37's logged Sonnet usage (`scenarios_rulebook`, 150 scenarios), priced at live
rates with `providers.PRICES`:

| Part | Per run, warm cache | Notes |
|---|---|---|
| Reasoner (Sonnet 5.5) | about $0.044 | about 4,200 output tokens; 11,900 rulebook tokens read from cache |
| Extractor (Sonnet 5.5) | about $0.031 | about 2,800 output tokens |
| Judge (Haiku 5.5) | about $0.003 | about 4 calls |
| **Total** | **about $0.08** | cold (cache older than 5 min): about $0.11–0.12 |

These come from batch runs, so latency is not measured. From the output sizes it is likely 1–3 minutes; the dummy
measures it.

**Measured (EXP-64, 19 Nov 2026, 5 live runs, $0.45 in all):**

| Cache | Cost per run | Time per run |
|---|---|---|
| Warm | $0.05–0.08 | 45–68 s |
| Cold | $0.10 and $0.15 | 77 and 153 s |

- The judge took up to 97 s, because each Haiku call takes about 6–8 s and the 3–12 calls ran one after another. Live runs now trace claims concurrently (6 workers), with the same verdicts.
- The server reserves $0.16 per run against the daily cap.
- A $5 monthly limit is about 35–90 runs.

**Caps:**
- **The workspace's monthly limit is the hard cap**, set by Parsa. Parsa expects fewer than 20 live runs in all (19 Nov
  2026), so the limit is small: about **$5** suggested, which allows about 40–60 runs (20 runs cost about $1.60–2.40).
  The $20 of the first draft was too high.
- **Changed by Parsa, 20 Nov 2026:** the daily cap is **$5** (about 30 runs at the $0.16 reserve), set as the Space
  variable `DAILY_CAP_USD = 5`; the code's default stays $1. The workspace's monthly limit remains the hard cap.
- The server's daily cap ($1 at first, about 10 runs; adjustable by a Space variable) and the per-visitor limits are soft. They
  live in memory, so a restart resets them. Free CPU Spaces also sleep when unused and restart on the next visit.
- Persisting them would need a paid storage bucket or a write token to a dataset repo (a second secret). Not proposed:
  the workspace limit already bounds the worst case at its own value.

**The same credit pool.** The Space's workspace draws on the same Anthropic credit as Phase 4's runs:
- about $27 left (an estimate; check the console);
- DEC-31's runs, about $6;
- the dummy, about $0.50.

With a limit of about $5 this fits without a top-up, with about $15 to spare (estimates).

**Dummy (before the Space is built out):** 5 live runs (2 cold, 3 warm) from the Mac, with the dedicated key in the
shell. Cost about $0.50. It reports cost and latency per step.

## 8. The key

- A new key, made by Parsa, in a new console workspace (for example `strokecopilot-space`) with a monthly spend limit.
- It goes into the Space's **secrets** (private, not copied when someone duplicates the Space, readable only as an
  environment variable at runtime). Never in code, the repo, a build argument, chat, or Notion.
- For the dummy and local previews, the same key comes from Parsa's shell environment.
- HF's secrets scanner is a second check. The leak scan (§6) is the first.

## 9. Amendments (applied once approved)

1. **Plan §8.5** is replaced by:
   > Public Hugging Face Space (DEC-32), Docker, CPU basic, built in `demo/space/`. A scrolling story of the pipeline;
   > a case explorer (fixtures, scenarios, real cases by alias) with each claim clickable to its source (contract
   > field, note line, passage with ID) and flagged claims showing their reason; results as interactive charts; and a
   > scenario builder that runs Sonnet 5.5 (prompt v3, rulebook) and the verifier live on structured inputs, falling
   > back to the nearest precomputed scenario. Persistent banner: *Research prototype — not for clinical use.* No
   > patient-identifying content exists in the datasets, but the banner is non-negotiable.
2. **Plan §4.4, secrets:** add "The Space's live Anthropic key (DEC-32) lives in the Space's secrets, set by Parsa;
   it is the only secret allowed there."
3. **Plan §7 Phase 4, the demo task:** "`demo/app.py`: pick a case → …" becomes:
   > The public HF Space (DEC-32): story, case explorer with clickable claims and flags in red, results charts, live
   > scenario builder with precomputed fallback; banner on every screen. Built on development results before
   > unlocking, filled with held-out results after the analyses, load-tested, and published on Parsa's word.
4. **Plan §7 Phase 4, acceptance:** "held-out results in Experiments; demo runs from a fresh Colab session in under 10
   minutes" becomes:
   > Held-out results in Experiments. The public Space is live: its live element is load-tested and falls back to
   > precomputed results, the not-for-clinical-use banner is on every screen, and its data follows the licence
   > check (DEC-32 §6).
5. **DEC-31** (`docs/PHASE4_PREREGISTRATION.md`), a dated note, since this is before unlocking:
   - §1 item 4, "the demo (plan §8.5)", becomes "the public Space (DEC-32), built on fixtures, scenarios and
     development results, with the live element dummy-tested";
   - D4 step 9, "Demo on held-out cases", becomes "fill the Space with held-out results, load-test it, and publish
     on Parsa's word";
   - §3 gains a line: the Space's live use is separate from DEC-31's estimate and bounded by the workspace limit.
6. **CLAUDE.md, Environment:** "Secrets come from Colab Secrets or env vars" gains "or, for the Space's live key
   only, the Space's secrets (DEC-32)". This file is Parsa's, so it is changed only if he agrees.

## 10. Choices for Parsa

Each has a recommendation; the draft assumes the recommended one.

| # | Choice | Recommended | Alternative |
|---|---|---|---|
| a | Visibility at publish | **Public**: code and app open, which suits an academic project | **Protected** (HF Pro): app public, source private |
| b | Before going public | A **private** Space for a staging check with the real key (still a push, so on your word) | Go straight to public from the local preview |
| c | Sign-in for live runs | **Off at launch.** Anyone can run it; the limits and the workspace cap bound abuse. A switch turns on "Sign in with HF" for live runs without a redeploy | Always on: less abuse, more friction for markers |
| d | Builder input values | **Continuous controls** within the evaluated ranges, fixed steps | Only the 150 scenarios' grid values: every run nearly a preset, less interesting |
| e | Model switch in the explorer | **Yes** (free, shows C9 per case) | Sonnet only |
| f | Daily cap persistence | **In memory**, hard cap by the workspace limit | Paid storage bucket, or a dataset repo with a second token |

**Time:** about 15–20 h for the Space (the earlier 4–6 h was for a Colab demo), so Phase 4 is about 30–45 h in all.
**Cost:** hardware free; live use at most the workspace limit; the dummy about $0.50.
