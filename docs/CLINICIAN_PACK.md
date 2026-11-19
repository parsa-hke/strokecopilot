# Clinician pack (Phase 5, task A)

Written 20 Nov 2026, before anything is sent to a clinician. **Status: approved and fixed by Parsa, 20 Nov 2026** (§7 and the delivery format the same day). The cases, the
questions and the analysis plan below are fixed. Any change from here needs a dated note in §9
saying why. Replaces `docs/CLINICIAN_REVIEW.md` as the thing we send (that file stays as the record of the Phase 3
draft).

Code: `src/strokecopilot/pipeline/clinician_pack.py` (tests: `tests/test_clinician_pack.py`). Build:

```bash
.venv/bin/python -m strokecopilot.pipeline.clinician_pack --bank demo/space/build/bank --out runs/EXP-67_clinician_review/pack
```

The output goes to a gitignored run folder (and Drive), because instrument 2 holds the generated notes of real
development cases (by alias, as on the public Space). The build refuses to finish if the leak scan finds a development
case ID, an RSNA ID or a key pattern. No held-out case is used, and the bank loader refuses one.

## 1. What this is, and what we may claim

Two clinicians each receive two instruments: a review of our rulebook and test cases, and a written interview on the
system's answers. They answer independently.

- **What it gives us:** expert opinion on whether our rulebook reads the guideline correctly, how far two clinicians
  agree on it, and their view of the system's answers, flags and trustworthiness.
- **What we may not claim:** clinical validation, accuracy against clinicians, or safety. Two reviewers are expert
  opinion. Every sentence in the reports and paper says "two clinicians reviewed …" or "in the opinion of two
  clinicians …", never "validated".
- **What it cannot change:** under DEC-31's freeze, nothing is re-run or re-scored because of an answer (§7).

Every page of both instruments carries: *"Research prototype, not for clinical use. Your answers are expert opinion on a
student research project. They are not a clinical validation, and we will not describe them as one."* Clinicians see
only our own-words rulebook (DEC-18), never guideline or trial text.

## 2. Delivery

- **Format: Google Forms, generated (Parsa, 20 Nov 2026).** The Markdown instruments stay the record; the forms are a
  rendering of them, built in code so nothing is retyped:
  - `build_forms.gs`: an Apps Script holding `forms_spec.json`. Parsa runs `buildForms()` once at script.google.com;
    it creates both forms, each with its own private response Sheet in his Drive, and logs their links. It sends
    nothing. A test checks that every item of both response sheets appears in the forms exactly once, under its ID,
    and that only the consent questions are required (anything else may be skipped, as the consent says).
  - `interview_cases.docx` (from `interview_cases.md`): the nine case pages, which the interview form points to.
    Forms cannot show tables, so the cases travel as a document; it holds no questions and no expected answers.
  - Form 1 (rulebook review): consent, about you, Parts A–D, one page each. Form 2 (interview): consent, one page per
    case, the general questions, then the expected answers with R1, after all the cases.
  - Not chosen: a form inside the public Space (it would change DEC-32 and publish; the Space has no lasting storage;
    anyone could submit) and Word for both (answers would need retyping).
  - Before sending, Parsa opens each form and checks: published, "Collect email addresses" off, the respondent link
    works in a private window.
- **Independence:** each clinician gets their own copy; the instructions ask them not to discuss it with the other
  reviewer; neither sees the other's answers before both have returned theirs.
- **Order:** instrument 1 first, then instrument 2. Instrument 2's expected answers sit at the end, to be read after all
  the cases.
- **Time:** about 45–60 minutes for instrument 1 (76 items, about 5,500 words) and 60–90 minutes for instrument 2 (9
  cases, about 8,800 words). That is long; Parsa may ask them to do it in two sittings.
- **Who sends:** Parsa. This session never sends anything.

## 3. Instrument 1: rulebook and test-case review

It opens with the consent (§5) and four questions about the reviewer, so roles are self-reported. Then 69 scored items,
each answered **agree / disagree / unsure (outside my practice)**, with a comment box, and 3 open questions. "Agree"
means the item is a correct and not misleading reading for adults with acute ischaemic stroke.

| Part | Items | What |
|---|---|---|
| About you | P1–P4 | name (private unless allowed), role and specialty, years in practice (band), how often they take part in acute stroke decisions |
| A | A01–A47 | every passage of `corpus/passages.jsonl` (sha256 `acb29e21…`), with its pointer and page |
| B | B1a–B5b | the five fixtures (inputs and note) and the answer our rules give, one item per treatment |
| C | C01–C12 | the conventions of `nlp/verifier/rules.py` and the stand-ins the evaluation rests on |
| D | D1–D3 | open questions: misleading in practice; a common trap our cases miss; anything else |

Part C's twelve conventions (exact wording in the instrument): time from onset or last known well (C01); times at
imaging, "within 6 h" inclusive (C02); COR 1/2a "yes", 2b "undetermined", 3 "no" (C03); NIHSS ≥ 6 as disabling (C04);
Table 8 absolute "no", relative "undetermined" (C05); unobservable inputs give "undetermined" (C06); ASPECTS unknown at
6–24 h: DAWN or DEFUSE-3 for "yes", otherwise "undetermined" (C07); EXTEND for extended-window thrombolysis, only when
thrombectomy is not an option (C08); TRACE-III with "penumbra volume > 15 mL" read as total Tmax > 6 s (C09); core under
1 mL: ratio not computed, threshold taken as met (C10); haemorrhage gives "no" for both treatments (C11); final infarct
on follow-up MRI as the stand-in for core at presentation (C12).

**Found while building it:** `docs/CLINICIAN_REVIEW.md`'s "our own choices" line gives TRACE-III as "mismatch > 15 mL",
but the code (`rules._trace3`) and the rulebook passage use total Tmax > 6 s volume > 15 mL. The pack states what the
code does (C09) and asks.

## 4. Instrument 2: written interview

### 4.1 The cases: a rule, fixed before sending

Nine cases, chosen in code by a seeded draw (seed **20261120**) within fixed strata, from the saved answers in the
Space's bank (bank `index.json` sha256 `04423393…`). Nothing is run. The strata make sure the clinicians see answers
that are right, answers with flags, answers that are wrong and caught, and answers that are wrong and missed, so the set
cannot flatter the system.

| Stratum | Rule | Pool | Chosen | Model |
|---|---|---|---|---|
| F | 2 of the 5 fixtures | 5 | fx04 (large core), fx05 (wake-up, borderline) | Sonnet 5.5 |
| S1 | a scenario where Sonnet agrees with the rules on every determined decision and nothing is flagged | 44 | scenario-057 | Sonnet 5.5 |
| S2 | a scenario where Sonnet agrees with the rules and at least one claim is flagged | 88 | scenario-056 | Sonnet 5.5 |
| S3 | a scenario where Sonnet's decision differs from the rules | 17 | scenario-025 | Sonnet 5.5 |
| D1 | a development case, predicted vision, no bleed flag (V case excluded) | 10 | D-12 | Sonnet 5.5 |
| D2 | a development case, predicted vision, with a bleed flag (V case excluded) | 9 | D-03 | Sonnet 5.5 |
| V | a development case with a vision error the verifier missed (DEC-31 D3's definitions) | **1** | D-05 | Sonnet 5.5 |
| M | a scenario where Gemma 3 27B's decision differs from the rules and the verifier caught it | 72 | scenario-032 | Gemma 3 27B |

Notes on the draw:
- **V has one case**, so it was not drawn: D-05 is the only development case among EXP-37's 20 where the rules on the
  predicted contract disagree with the truth on a determined decision. A false bleed flag (probability 0.46) turns a
  "yes" for thrombolysis into "no"; Sonnet follows the contract, and the verifier flags nothing, because the claims
  restate the contract faithfully.
- **S3's case** is a bleed with an M1 occlusion, where Sonnet recommends thrombectomy and our rules say no (convention
  C11). Nothing is flagged. It also tests C11 itself.
- **M** uses Gemma 3 27B, the open mid-size model, so a weaker model and a caught error appear. Its case has 7 flags,
  1 of them critical.
- The nine are shown in a seeded order (cases 1–9: D-12, D-03, fx04, D-05, scenario-032, scenario-056, fx05,
  scenario-025, scenario-057), so the strata are not visible.
- Each case shows the model's name, the imaging findings, the triage note, the full answer (decisions, cited passages,
  reasons, rationale with sources, uncertainties) and the verifier's flags in plain words. Each can be opened on the
  Space (https://huggingface.co/spaces/parsa-hke/stroke-copilot) by typing its code into the case explorer.
- The expected answer for each case (the rules on the ground-truth contract for real cases) is printed at the end, with
  one question per case (R1).

### 4.2 The questions

Per case (Q1–Q5 on a five-point scale, strongly disagree to strongly agree):
- **Q1** The thrombolysis recommendation is appropriate, given the information shown.
- **Q2** The thrombectomy recommendation is appropriate, given the information shown.
- **Q3** The rationale explains the recommendation in a way a clinician could check.
- **Q4** The uncertainties listed are the ones that matter for this patient.
- **Q5** The verifier's flags point to real problems in the answer (N/A if nothing was flagged).
- **Q6** Did you notice a claim that is wrong or misleading but was not flagged? If yes, which?
- **Q7** If this answer reached you in practice, you would: ignore it / use it only after checking every claim / use it
  after checking the flagged claims / rely on it.
- **Q8** (open) What, if anything, is wrong, missing or misleading here? What would you have recommended?
- **R1** (after all cases) Do you agree with our expected answer for this case? agree / disagree / unsure, with a
  comment.

General, after the cases:
- **G1** A tool like this would be useful in acute stroke care (five-point scale). Where in the pathway, if anywhere?
- **G2** Under what conditions would you trust its recommendation, and when would you never?
- **G3** Does the claim-by-claim check with flagged claims help you, or does it add noise?
- **G4** On real acute-stroke scans our imaging model flags about half as possible bleeds, almost all false alarms. How
  should a tool like this present an uncertain bleed finding?
- **G5** Which failure, seen here or imagined, worries you most?
- **G6** Anything else?

## 5. Consent

At the start of both instruments. The clinicians are known to the team, so no withdrawal date or address is needed:

> Taking part is voluntary, and you may skip any question. We will use your answers in three university assessments of
> the MAAI 901/902 group project at UOWD (two course reports and one research paper). We keep your answers on the team's
> private drive and share them with no one outside the team and the course staff. You may withdraw your answers at any
> time before the reports are submitted, by telling Parsa or any member of the team.
>
> I have read this and agree to take part (tick).
>
> How may we quote you? by name and role / by role only / do not quote me; summarise my answers only.
> May we thank you by name in the acknowledgements? yes / no

A clinician who ticks "do not quote me" is summarised only, and counted. Ethics clearance for the expert interviews was
confirmed by Parsa with the professors (20 Nov 2026).

## 6. Analysis plan (fixed before any answer returns)

**Instrument 1, agreement between the two reviewers.**
- Unit: the scored item (69: Parts A–C; Part D is open text and is summarised, not scored; P1–P4 describe the
  reviewers).
- Each reviewer's counts of agree / disagree / unsure, overall and per part (A, B, C).
- **Agreement:** raw percent agreement on the three categories over items both answered; Cohen's κ with a 95%
  bootstrap interval over items (10,000 resamples, seed 20261104, as elsewhere); and PABAK beside κ, because "agree"
  will dominate and κ falls when one category does (the κ paradox). "Unsure" is a category, not missing; a blank is
  missing and counted.
- **Items with any "disagree":** listed with both reviewers' comments (as consent allows). Two team members classify each
  independently as (a) wording or clarity, (b) our reading of the guideline is wrong, (c) practice differs from the
  guideline, not our reading, or (d) other; they settle differences by discussion, and both the comments and the
  classes are published in the appendix. Class (b) in Parts A–C goes to §7.
- One sentence for the reports, for example: "Two clinicians reviewed the 47 rulebook passages, five test cases and 12
  conventions independently; they agreed with X and Y of 69 items, agreeing with each other on Z% (κ …, PABAK …)." One
  figure: a stacked bar per part and reviewer.

**Instrument 2, summary of the interview.**
- Ratings (Q1–Q5, G1) are tabulated per case and question, the two clinicians shown separately. With n = 2, no means,
  tests or κ; counts only (for example "Q1 agree or strongly agree in 15 of 18 case ratings").
- Their agreement: exact and within-one-point agreement over case–question pairs.
- Pre-specified contrasts, descriptive only: Q1–Q2 on cases where the model agrees with the rules (F, S1, S2, D1, D2)
  against cases where it does not (S3, V, M); Q5 and Q6 on cases with flags against cases without.
- R1: per case, agreement with our expected answer, added to instrument 1's evidence on the rules.
- Q7: counts per option, per case and overall.
- **Open answers (Q6, Q8, G1–G6):** coded against a frame fixed now: correctness of the recommendation; usefulness of
  the rationale; uncertainty handling; verifier flags (helpful / noise / missed an error); conditions for trust; fit to
  clinical workflow; safety concerns; uncertain bleed findings; other. Two team members code independently, then agree.
  Each code is reported with how many clinicians (of 2) and cases raised it. Every quote used in a report or the paper
  carries its code and that count and points to an excerpt ID in the appendix. A critical code gets at least the same
  space as a positive one; nothing is quoted without consent.

**Reporting limits, stated wherever results appear:** two reviewers; expert opinion, not validation; nine cases chosen
by rule, not a sample of practice; the reviewers saw text, not images; real cases' notes are generated from records.

## 7. If a clinician finds a rule error

**Approved by Parsa, 20 Nov 2026** (his default, with the counting in code added):
1. Nothing is changed or re-run: the rulebook, `rules.py`, scenarios, contracts and every result stay as they are
   (DEC-31's freeze).
2. For a class (b) item, count in code the decisions it touches: scenarios (of 150) and development real-case truths
   (of 129) whose rule decision cites the passage in its `rule_ids` or turns on the convention, and among those, how
   many model answers were scored right or wrong against it. Held-out truths are counted only with Parsa's OK, as a
   post hoc analysis labelled so.
3. Report it as a limitation with those counts, quoting the clinician's reason. If the two reviewers disagree, both views
   are reported, with no tie-break.

## 8. Before sending (Parsa)

- [x] Approve this document (cases, questions, analysis plan): Parsa, 20 Nov 2026.
- [x] Format: Google Forms, generated (Parsa, 20 Nov 2026; §2).
- [ ] Run `build_forms.gs`, check both forms, then send each clinician the two form links and the case document.
- [x] Ethics clearance for the expert interviews (confirmed by Parsa, 20 Nov 2026).
- [x] Consent without a withdrawal date or address; roles self-reported in P1–P4 (Parsa, 20 Nov 2026).
- [x] §7, the rule-error policy (Parsa, 20 Nov 2026).
- [x] Decisions row DEC-34 and EXP-67 (Parsa, 20 Nov 2026).

## 9. Fixed files

Built at this commit's code with seed 20261120 (`runs/EXP-67_clinician_review/pack/manifest.json`):

| File | sha256 |
|---|---|
| `build_forms.gs` | `4986a0f391ec3c9d684faa6020696a9cbadfea6c231fad02a31302f63ce27822` |
| `forms_spec.json` | `dd1d017052770de46fe87a124c56ee77f9fe1cabad32a7c15d5a5627f72cd400` |
| `instrument1_rulebook_review.md` | `2df20369925fc2ca10c90f682ce1048e969d8d87bc19a87800352e839f23b55e` |
| `instrument2_interview.md` | `dda3f826c47cfe35cc0e91346fe19b411946a72de743ed08e3c06bcfcf9339a8` |
| `interview_cases.md` | `82b94589c99d2fcdcb8f40b94e3c9f16ca8d24376e824044edeccf9e640ff33d` |
| `selection.json` | `6d8ce709004ab6a3b6652a996cfcb8020e97b851addf1c1e180993376f88b8a5` |
| `instrument1_responses.csv` | `488b0f296f1441244a88acfc8867f01275b8d2b6ef463aa749de955664cc1c22` |
| `instrument2_responses.csv` | `6a054c05da40f84687278d9dd2f3eadccbb4b01ddb64ab7b04bcba57792a9bd4` |
| `interview_cases.docx` | `6fa2f1c400a2a009abcb0ab00eb1c1dfea1e88100c445e3d7a9a304f00265d42` |

Fixed at approval (20 Nov 2026). Any change from here, to a case, a question, the wording or the analysis plan, needs a
dated note below saying why, and a rebuild with its new checksums.

### Changes after approval

None yet.
