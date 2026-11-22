# Clinician pack (Phase 5, task A)

Written 20 Nov 2026, before anything is sent to a clinician. **Status: approved and fixed by Parsa, 20 Nov 2026** (§7 and the delivery format the same day). The
items and the analysis plan below are fixed. Any change from here needs a dated note in §9
saying why. Replaces `docs/CLINICIAN_REVIEW.md` as the thing we send (that file stays as the record of the Phase 3
draft).

Code: `src/strokecopilot/pipeline/clinician_pack.py` (tests: `tests/test_clinician_pack.py`). Build:

```bash
.venv/bin/python -m strokecopilot.pipeline.clinician_pack --bank demo/space/build/bank --out runs/EXP-67_clinician_review/pack
```

The output goes to a gitignored run folder (and Drive). The build refuses to finish if the leak scan finds a
development case ID, an RSNA ID or a key pattern. No held-out case is used, and the bank loader refuses one.

## 1. What this is, and what we may claim

Two clinicians each receive a review of our rulebook, our test cases and the conventions our rules apply. They answer
independently.

- **What it gives us:** expert opinion on whether our rulebook reads the guideline correctly, and how far two
  clinicians agree on it.
- **What we may not claim:** clinical validation, accuracy against clinicians, or safety. Two reviewers are expert
  opinion. Every sentence in the reports and paper says "two clinicians reviewed …" or "in the opinion of two
  clinicians …", never "validated".
- **What it cannot change:** under DEC-31's freeze, nothing is re-run or re-scored because of an answer (§7).

Every page of the review carries: *"Research prototype, not for clinical use. Your answers are expert opinion on a
student research project. They are not a clinical validation, and we will not describe them as one."* Clinicians see
only our own-words rulebook (DEC-18), never guideline or trial text.

## 2. Delivery

- **Format: Google Forms, generated (Parsa, 20 Nov 2026).** The Markdown review stays the record; the form is a
  rendering of it, built in code so nothing is retyped:
  - `build_forms.gs`: an Apps Script holding `forms_spec.json`. Parsa runs `buildForms()` once at script.google.com;
    it creates the form, with its own private response Sheet in his Drive, and logs its links. It sends
    nothing. A test checks that every item of the response sheet appears in the form exactly once, under its ID,
    and that only the consent questions are required (anything else may be skipped, as the consent says).
  - The form: consent, about you, Parts A–D, one page each.
  - Not chosen: a form inside the public Space (it would change DEC-32 and publish; the Space has no lasting storage;
    anyone could submit) and Word (answers would need retyping).
  - Before sending, Parsa opens the form and checks: published, "Collect email addresses" off, the respondent link
    works in a private window.
- **Independence:** each clinician gets their own copy; the instructions ask them not to discuss it with the other
  reviewer; neither sees the other's answers before both have returned theirs.
- **Time:** about 45–60 minutes (76 items, about 5,500 words).
- **Who sends:** Parsa. This session never sends anything.

## 3. The rulebook and test-case review

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

## 4. The twelve conventions (Part C)

In short (exact wording in the review): time from onset or last known well (C01); times at
imaging, "within 6 h" inclusive (C02); COR 1/2a "yes", 2b "undetermined", 3 "no" (C03); NIHSS ≥ 6 as disabling (C04);
Table 8 absolute "no", relative "undetermined" (C05); unobservable inputs give "undetermined" (C06); ASPECTS unknown at
6–24 h: DAWN or DEFUSE-3 for "yes", otherwise "undetermined" (C07); EXTEND for extended-window thrombolysis, only when
thrombectomy is not an option (C08); TRACE-III with "penumbra volume > 15 mL" read as total Tmax > 6 s (C09); core under
1 mL: ratio not computed, threshold taken as met (C10); haemorrhage gives "no" for both treatments (C11); final infarct
on follow-up MRI as the stand-in for core at presentation (C12).

**Found while building it:** `docs/CLINICIAN_REVIEW.md`'s "our own choices" line gives TRACE-III as "mismatch > 15 mL",
but the code (`rules._trace3`) and the rulebook passage use total Tmax > 6 s volume > 15 mL. The pack states what the
code does (C09) and asks.

## 5. Consent

At the start of the review. The clinicians are known to the team, so no withdrawal date or address is needed:

> Taking part is voluntary, and you may skip any question. We will use your answers in three university assessments of
> the MAAI 901/902 group project at UOWD (two course reports and one research paper). We keep your answers on the team's
> private drive and share them with no one outside the team and the course staff. You may withdraw your answers at any
> time before the reports are submitted, by telling Parsa or any member of the team.
>
> I have read this and agree to take part (tick).
>
> How may we quote you? by name and role / by role only / do not quote me; summarise my answers only.
> May we thank you by name in the acknowledgements? yes / no

A clinician who ticks "do not quote me" is summarised only, and counted. Ethics clearance for the expert review was
confirmed by Parsa with the professors (20 Nov 2026).

## 6. Analysis plan (fixed before any answer returns)

**Agreement between the two reviewers.**
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

**Reporting limits, stated wherever results appear:** two reviewers; expert opinion, not validation; the reviewers saw
text, not images; the five test cases are invented.

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

- [x] Approve this document (items, analysis plan): Parsa, 20 Nov 2026.
- [x] Format: Google Forms, generated (Parsa, 20 Nov 2026; §2).
- [x] Run `build_forms.gs`, check the form, then send each clinician its link (answers received, 23 Nov 2026).
- [x] Ethics clearance for the expert review (confirmed by Parsa, 20 Nov 2026).
- [x] Consent without a withdrawal date or address; roles self-reported in P1–P4 (Parsa, 20 Nov 2026).
- [x] §7, the rule-error policy (Parsa, 20 Nov 2026).
- [x] Decisions row DEC-34 and EXP-67 (Parsa, 20 Nov 2026).

## 9. Fixed files

Built by this commit's code (`manifest.json` in the build folder):

| File | sha256 |
|---|---|
| `instrument1_rulebook_review.md` | `2df20369925fc2ca10c90f682ce1048e969d8d87bc19a87800352e839f23b55e` |
| `instrument1_responses.csv` | `488b0f296f1441244a88acfc8867f01275b8d2b6ef463aa749de955664cc1c22` |
| `forms_spec.json` | `7474b38e1cdc1f9c2a6245eca6599d8300ee4f0e9d3e41535f5ab28396c0290a` |
| `build_forms.gs` | `346d7e972f9638c63a57c6a79aa03e6b4fd2e78c8c725399171b468cc910fb4f` |

Fixed at approval (20 Nov 2026). Any change from here, to an item, the wording or the analysis plan, needs a
dated note below saying why, and a rebuild with its new checksums.

### Changes after approval

- **23 Nov 2026 (Parsa):** this document and the build now cover the rulebook and test-case review only, which is what
  the two clinicians answered and what the reports use. The review itself is unchanged: `instrument1_rulebook_review.md`
  and `instrument1_responses.csv` rebuild to the checksums recorded at approval, and the form's 171 items are the same,
  item for item. `forms_spec.json` and `build_forms.gs` now hold that one form, so their checksums above are new. The
  analysis plan for the review (§6) and the rule-error policy (§7) are as approved.
