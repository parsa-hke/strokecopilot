"""The clinician pack (Phase 5, task A): two instruments for two clinicians, fixed before anything is sent.

Instrument 1, rulebook and fixture review: every rulebook passage, every fixture decision and every rule convention the
code uses, each answered agree / disagree / unsure with a comment. The two clinicians answer independently, so their
agreement can be reported (docs/CLINICIAN_PACK.md §6).

Instrument 2, written interview: a fixed set of cases, chosen here by a seeded rule over strata (below), each shown with
the reasoner's answer and the verifier's flags, then fixed questions. The selection rule, the seed and the questions are
written down before anything is sent, so nobody can say the cases were picked to flatter the system.

Case strata for instrument 2 (all from saved answers in the Space's bank; nothing is run):
- F: 2 of the 5 fixtures, Sonnet 5.5;
- S1: a scenario where Sonnet agrees with the rules on every determined decision and the verifier flags nothing;
- S2: a scenario where Sonnet agrees with the rules and the verifier flags at least one claim;
- S3: a scenario where Sonnet's decision differs from the rules;
- D1, D2: development real cases on predicted vision, Sonnet; one without and one with a bleed flag in the contract;
- V: a development case with a vision error the verifier missed (DEC-31 D3's definitions: the rules on the predicted
  contract already disagree with the truth, Sonnet's answer is wrong against the truth, and no claim of that decision is
  flagged);
- M: a scenario where Gemma 3 27B's decision differs from the rules and the verifier caught it (a flagged claim of that
  decision), so the clinicians also see a weaker open model and a caught error.
Held-out cases are never used. Real development cases appear only by their Space alias (D-xx).

Output (`write`): the two instruments as Markdown (the record), a response sheet per instrument as CSV, the selection
with its pool sizes, and the Google Forms route chosen by Parsa (20 Nov 2026): `forms_spec.json`, `build_forms.gs` (an
Apps Script that builds both forms from it) and `interview_cases.md` / `.docx` (the case pages the interview form points
to, without questions or expected answers). A manifest records the sha256 of every file. Instrument 2 holds generated notes of real cases, so
the output goes to a gitignored run folder, never to docs/.
"""

import csv
import hashlib
import json
import random
from pathlib import Path

from strokecopilot.config import CORPUS_DIR
from strokecopilot.pipeline import heldout_analysis as ha

SEED = 20261120
SPACE_URL = "https://huggingface.co/spaces/parsa-hke/stroke-copilot"
SONNET = "claude-sonnet-5-5"
OTHER_MODEL = "gemma-3-27b"
MODEL_NAMES = {SONNET: "Claude Sonnet 5.5", OTHER_MODEL: "Gemma 3 27B"}
DECISIONS = ha.DECISIONS
DISCLAIMER = ("Research prototype, not for clinical use. Your answers are expert opinion on a student research project. "
              "They are not a clinical validation, and we will not describe them as one.")
RESPONSES = ("agree", "disagree", "unsure")
SUBTYPE_WORDS = {
    "wrong_number": "wrong number",
    "invented_finding": "finding or fact with no source in the inputs",
    "wrong_rule": "rule not stated in the rulebook",
    "misapplied_rule": "rule applied outside its scope",
    "faulty_derivation": "wrong calculation or comparison",
    "uncited_rule": "true rule, but no rulebook passage cited",
    "misattributed_source": "true, but the cited source does not support it",
}

# The conventions of nlp/verifier/rules.py (its docstring) and the stand-ins the evaluation rests on, in plain words.
CONVENTIONS = (
    "Time windows count from symptom onset, or from last known well when onset was not witnessed. The guideline counts "
    "the thrombectomy windows from symptom onset.",
    "Times are taken at imaging, because the inputs give no treatment time. \"Within 6 h\" includes exactly 6 h 0 min.",
    "COR 1 and 2a count as \"yes\", COR 2b as \"undetermined\" (left to judgement), and COR 3 or no applicable "
    "recommendation as \"no\".",
    "NIHSS 6 or more counts as a disabling deficit. NIHSS under 6 within 4.5 h is left undetermined, because whether "
    "it is disabling cannot be read from the inputs.",
    "Table 8 absolute contraindications give \"no\". Relative ones, including a DOAC within 48 h, give "
    "\"undetermined\".",
    "A recommendation that needs an input the system cannot observe (significant mass effect, M2 dominance, "
    "PC-ASPECTS, extensive hypodensity) is \"undetermined\", and the missing input is named.",
    "With ASPECTS unknown at 6-24 h, a \"yes\" for thrombectomy needs the DAWN or DEFUSE-3 criteria. If neither is met "
    "the decision is \"undetermined\", not \"no\", because ASPECTS could still qualify.",
    "Salvageable penumbra for extended-window thrombolysis uses the trial criteria in Table 3: EXTEND at 4.5-9 h (or "
    "within 9 h of the midpoint of sleep): mismatch ratio > 1.2, Tmax > 6 s minus core > 10 mL, core < 70 mL. "
    "Extended-window thrombolysis is considered only when thrombectomy is not an option.",
    "TRACE-III at 4.5-24 h (COR 2b, so \"undetermined\"): ratio > 1.8, core < 70 mL, and Table 3's \"penumbra volume "
    "> 15 mL\" read as the total Tmax > 6 s volume (core included) above 15 mL, not Tmax > 6 s minus core.",
    "When the core is under 1 mL the mismatch ratio is not computed, and its threshold is taken as met.",
    "A haemorrhage on CT gives \"no\" for both treatments: the ischaemic-stroke thrombectomy recommendations do not "
    "apply.",
    "For real cases, the correct answer is computed by these rules on the reference final infarct (follow-up MRI, 2-9 "
    "days later, in patients who all reperfused) used as the core at presentation, with no bleed presumed.",
)

FIXTURE_TITLES = {
    "fx01_thrombolysis_clear": "Witnessed onset 2.5 h, M2 occlusion",
    "fx02_thrombectomy_large_mismatch": "Last known well 9 h, M1 occlusion, large mismatch",
    "fx03_haemorrhage_present": "Witnessed onset 2 h, haemorrhage on CT",
    "fx04_large_core_no_mismatch": "Witnessed onset 6 h, M1 occlusion, large core",
    "fx05_borderline_onset": "Wake-up stroke, last known well 10 h, M1 occlusion",
}

OPEN_QUESTIONS_1 = (
    "Is anything in the rulebook misleading in practice, even where it matches the guideline?",
    "Is there a common clinical situation or trap that our five test cases and these rules miss?",
    "Anything else we should know?",
)

# Instrument 2's questions. Ratings use a five-point scale; Q6 and the general questions are as written.
LIKERT = ("strongly disagree", "disagree", "neither", "agree", "strongly agree")
CASE_QUESTIONS = (
    ("Q1", "likert", "The thrombolysis recommendation is appropriate, given the information shown."),
    ("Q2", "likert", "The thrombectomy recommendation is appropriate, given the information shown."),
    ("Q3", "likert", "The rationale explains the recommendation in a way a clinician could check."),
    ("Q4", "likert", "The uncertainties listed are the ones that matter for this patient."),
    ("Q5", "likert_na", "The verifier's flags point to real problems in the answer (N/A if nothing was flagged)."),
    ("Q6", "choice", "Did you notice a claim that is wrong or misleading but was not flagged? If yes, which?"),
    ("Q7", "choice", "If this answer reached you in practice, you would: ignore it / use it only after checking every "
                     "claim / use it after checking the flagged claims / rely on it."),
    ("Q8", "open", "What, if anything, is wrong, missing or misleading here? What would you have recommended?"),
)
REVEAL_QUESTION = ("R1", "choice", "Having seen it, do you agree with our expected answer for this case? "
                                   "(agree / disagree / unsure, with a comment)")
GENERAL_QUESTIONS = (
    ("G1", "likert_open", "A tool like this would be useful in acute stroke care. Where in the pathway, if anywhere?"),
    ("G2", "open", "Under what conditions would you trust its recommendation, and when would you never?"),
    ("G3", "open", "Does the claim-by-claim check with flagged claims help you, or does it add noise?"),
    ("G4", "open", "On real acute-stroke scans, our imaging model flags about half as possible bleeds, almost all false "
                   "alarms. How should a tool like this present an uncertain bleed finding?"),
    ("G5", "open", "Which failure, seen here or imagined, worries you most?"),
    ("G6", "open", "Anything else?"),
)

CONSENT = (
    "Taking part is voluntary, and you may skip any question. We will use your answers in three university "
    "assessments of the MAAI 901/902 group project at UOWD (two course reports and one research paper). We keep your "
    "answers on the team's private drive and share them with no one outside the team and the course staff. You may "
    "withdraw your answers at any time before the reports are submitted, by telling Parsa or any member of the team.\n\n"
    "- [ ] I have read this and agree to take part.\n\n"
    "How may we quote you? Tick one:\n"
    "- [ ] by name and role (for example \"Dr A. B., consultant neurologist\")\n"
    "- [ ] by role only (for example \"a consultant neurologist\")\n"
    "- [ ] do not quote me; summarise my answers only\n\n"
    "May we thank you by name in the acknowledgements? [ ] yes [ ] no"
)

# Asked once, in instrument 1, so the report can describe the reviewers in their own words.
ABOUT_YOU = (
    ("P1", "Your name (kept private unless you allow it above)."),
    ("P2", "Your current role and specialty (for example consultant neurologist, emergency physician, "
           "neuroradiologist, stroke registrar)."),
    ("P3", "Years in clinical practice since qualifying: under 5 / 5-10 / 11-20 / over 20."),
    ("P4", "How often do you take part in acute stroke treatment decisions? never / occasionally / regularly."),
)


# --------------------------------------------------------------------------------------------- Inputs

def load_passages(path=None):
    path = Path(path or CORPUS_DIR / "passages.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_bank(bank_dir):
    """{alias: case} for the fixtures, scenarios and development cases of the Space's bank (never held-out ones)."""
    bank_dir = Path(bank_dir)
    index = json.loads((bank_dir / "index.json").read_text())
    cases = {}
    for row in index["cases"]:
        if row["kind"] not in ("fixture", "scenario", "development"):
            raise ValueError(f"unexpected case kind in the bank: {row['kind']}")
        case = json.loads((bank_dir / "cases" / f"{row['id']}.json").read_text())
        case["index"] = row
        cases[row["id"]] = case
    return cases


# --------------------------------------------------------------------------------------- Case strata

def _arm(case):
    return case["arms"]["predicted"] if case["kind"] == "development" else case["arms"]["rulebook"]


def _answer(case, model):
    a = _arm(case)["answers"].get(model)
    return a if a and a.get("available") else None


def _disagrees(answer):
    """Decisions where the answer differs from a determined rule decision on the contract it was given."""
    return [d for d, v in answer["decisions"].items() if v.get("rulebook") is not None and not v["agrees"]]


def _caught(answer, decision):
    return ha._flagged(answer["claims"], decision)[0]


def _vision_errors_missed(case, model=SONNET):
    """D3: decisions where rules(predicted) != truth, the answer != truth, and no claim of the decision is flagged."""
    a = _answer(case, model)
    if a is None:
        return []
    pred, truth = case["arms"]["predicted"]["rules"], case["arms"]["groundtruth"]["rules"]
    out = []
    for d in DECISIONS:
        t = ha.WORD[truth[d]["value"]]
        if t is None or ha.WORD[pred[d]["value"]] == t:
            continue
        if a["recommendation"][d]["recommend"] != t and not _caught(a, d):
            out.append(d)
    return out


def pools(cases):
    """{stratum: sorted aliases} (see the module docstring)."""
    p = {k: [] for k in ("F", "S1", "S2", "S3", "D1", "D2", "V", "M")}
    for alias, case in sorted(cases.items()):
        kind, s = case["kind"], _answer(case, SONNET)
        if kind == "fixture" and s:
            p["F"].append(alias)
        elif kind == "scenario":
            if s:
                if _disagrees(s):
                    p["S3"].append(alias)
                elif s["counts"]["flagged"] == 0:
                    p["S1"].append(alias)
                else:
                    p["S2"].append(alias)
            g = _answer(case, OTHER_MODEL)
            if g and any(_caught(g, d) for d in _disagrees(g)):
                p["M"].append(alias)
        elif kind == "development" and s:
            if _vision_errors_missed(case):
                p["V"].append(alias)
            else:
                bleed = case["arms"]["predicted"]["vision"]["haemorrhage"]["present"]
                p["D2" if bleed else "D1"].append(alias)
    return p


PLAN = (("F", 2, SONNET), ("S1", 1, SONNET), ("S2", 1, SONNET), ("S3", 1, SONNET), ("D1", 1, SONNET),
        ("D2", 1, SONNET), ("V", 1, SONNET), ("M", 1, OTHER_MODEL))


def select(cases, seed=SEED):
    """The fixed selection: [{"n", "alias", "stratum", "model"}] in a seeded order, plus the pool sizes."""
    rng = random.Random(seed)
    p = pools(cases)
    chosen, used = [], set()
    for stratum, k, model in PLAN:
        pool = [a for a in p[stratum] if a not in used]
        if len(pool) < k:
            raise ValueError(f"stratum {stratum} has {len(pool)} cases, needs {k}")
        for alias in rng.sample(pool, k):
            chosen.append({"alias": alias, "stratum": stratum, "model": model})
            used.add(alias)
    rng.shuffle(chosen)
    for i, c in enumerate(chosen, 1):
        c["n"] = i
    return chosen, {k: len(v) for k, v in p.items()}


# ---------------------------------------------------------------------------------------- Instrument 1

def _tick(item_id):
    return f"**{item_id}** [ ] Agree [ ] Disagree [ ] Unsure / outside my practice. Comment:"


def _fixture_lines(case):
    v, facts_note = case["arms"]["rulebook"]["vision"], case["arms"]["rulebook"]["note"]
    return [*_vision_table(v), "", "Triage note:", "", *_quote(facts_note)]


def instrument1(passages, cases):
    """(markdown, rows) for the rulebook and fixture review. rows: one dict per item, for the response sheet."""
    rows = []
    L = ["# Stroke Decision Copilot: rulebook and test-case review", "", f"*{DISCLAIMER}*", "",
         "Thank you for reviewing our work. Our system suggests whether thrombolysis and thrombectomy are indicated, "
         "and every result we report is scored against the rules below. If a rule is wrong, our results are wrong, "
         "so we need your honest reading. Please answer on your own, without discussing it with the other reviewer.",
         "",
         "For each item, tick one box and add a comment if you disagree or are unsure. **Agree** means the item is a "
         "correct and not misleading reading for adults with acute ischaemic stroke. The rules are our own words; they "
         "are based on the 2026 AHA/ASA guideline (*Stroke* 2026;57:e316-e436) and the DAWN and DEFUSE-3 trials, and "
         "each gives the section and page it rests on. About 45-60 minutes.", "",
         "## Consent", "", CONSENT, "", "## About you", ""]
    for item, q in ABOUT_YOU:
        L += [f"**{item}.** {q}", "", "Answer:", ""]
        rows.append({"item_id": item, "section": "about", "ref": "", "item": q})

    L += ["## Part A. The rulebook", "",
          f"{len(passages)} passages, in the order of the guideline. COR = class of recommendation.", ""]
    topic = None
    for i, p in enumerate(passages, 1):
        if p.get("topic") != topic:
            topic = p.get("topic")
            L += [f"### {topic}", ""]
        item = f"A{i:02d}"
        L += [f"**{item}. {p['id']}** ({p['pointer']}, p. {p['page']})", "", p["text"], "", _tick(item), ""]
        rows.append({"item_id": item, "section": "rulebook", "ref": p["id"], "item": p["text"]})

    L += ["## Part B. Five invented test cases and the answers our rules give", "",
          "Volumes are in mL. \"Tmax > 6 s\" is the whole hypoperfused volume, core included; mismatch volume is "
          "Tmax > 6 s minus core. Each case has two items, one per treatment.", ""]
    fixtures = sorted(a for a, c in cases.items() if c["kind"] == "fixture")
    for j, alias in enumerate(fixtures, 1):
        case = cases[alias]
        L += [f"### Case B{j}: {FIXTURE_TITLES.get(alias, alias)}", "", *_fixture_lines(case), ""]
        for d, letter in zip(DECISIONS, "ab"):
            r = case["index"]["rules"][d]
            item = f"B{j}{letter}"
            text = f"{d.capitalize()}: {r['value']}. {r['reason']}"
            L += [f"**{item}. Expected answer.** {text}", "", _tick(item), ""]
            rows.append({"item_id": item, "section": "fixtures", "ref": alias, "item": text})

    L += ["## Part C. How the rules are applied", "",
          "Choices we made where the guideline leaves room, or where our inputs are limited. \"Undetermined\" means "
          "the case is left out of our accuracy figures rather than guessed.", ""]
    for k, text in enumerate(CONVENTIONS, 1):
        item = f"C{k:02d}"
        L += [f"**{item}.** {text}", "", _tick(item), ""]
        rows.append({"item_id": item, "section": "conventions", "ref": "", "item": text})

    L += ["## Part D. Open questions", ""]
    for k, q in enumerate(OPEN_QUESTIONS_1, 1):
        item = f"D{k}"
        L += [f"**{item}.** {q}", "", "Answer:", ""]
        rows.append({"item_id": item, "section": "open", "ref": "", "item": q})
    L += ["---", "", f"*{DISCLAIMER}*", ""]
    return "\n".join(L), rows


# ---------------------------------------------------------------------------------------- Instrument 2

def _quote(text):
    return ["> " + line if line else ">" for line in text.strip().splitlines()]


def _fmt(x, unit=""):
    if x is None:
        return "not available"
    return f"{x:g}{unit}" if isinstance(x, (int, float)) else str(x)


def _vision_table(v):
    h = v["haemorrhage"]
    rows = [
        ("Haemorrhage on plain CT", f"{'yes' if h['present'] else 'no'} (model probability {h['confidence']:.2f})"),
        ("Core (mL)", _fmt(v.get("core_ml"))),
        ("Tmax > 6 s, core included (mL)", _fmt(v.get("penumbra_ml"))),
        ("Mismatch ratio", _fmt(v.get("mismatch_ratio"))),
        ("Mismatch volume (mL)", _fmt(v.get("mismatch_volume_ml"))),
        ("Occlusion site", _fmt(v.get("occlusion_site"))),
        ("ASPECTS", _fmt(v.get("aspects"))),
        ("Confidence in the core volume (0-1)", _fmt((v.get("confidence") or {}).get("core"))),
        ("Perfusion data quality (0-1)", _fmt((v.get("confidence") or {}).get("penumbra"))),
        ("Warnings", "; ".join(v.get("warnings") or []) or "none"),
    ]
    return ["| Imaging finding | Value |", "|---|---|", *[f"| {a} | {b} |" for a, b in rows]]


def _source(s):
    if s["type"] == "vision":
        return f"imaging: {s['field']}"
    if s["type"] == "note":
        return f"note: \"{s.get('quote', '')}\""
    if s["type"] == "passage":
        return f"rulebook: {s.get('id')}"
    return s["type"]


def _answer_lines(answer):
    rec = answer["recommendation"]
    L = ["**Recommendation**", ""]
    for d in DECISIONS:
        r = rec[d]
        L.append(f"- **{d.capitalize()}: {'recommended' if r['recommend'] else 'not recommended'}.** "
                 f"Rules cited: {', '.join(r.get('rule_ids') or []) or 'none'}.")
        for b in r.get("blocking_factors") or []:
            L.append(f"  - Reason against: {b}")
    L += ["", "**Rationale**", ""]
    for k, e in enumerate(rec.get("rationale") or [], 1):
        src = "; ".join(_source(s) for s in e.get("sources") or []) or "no source given"
        L.append(f"{k}. {e['claim']} *[{src}]*")
    L += ["", "**Uncertainties named by the model**", ""]
    L += [f"- {u}" for u in rec.get("uncertainties") or []] or ["- none"]
    claims = answer["claims"]
    flagged = [c for c in claims if c.get("flagged")]
    L += ["", f"**Verifier:** {len(claims)} claims checked, {len(flagged)} flagged.", ""]
    for c in flagged:
        crit = {True: "yes", False: "no"}.get(c.get("critical"), "cannot tell")
        L.append(f"- Flagged: \"{c['text']}\". Problem: {SUBTYPE_WORDS.get(c['subtype'], c['subtype'])}. "
                 f"Would change a decision if believed: {crit}. Verifier's reason: {c.get('reason', '')}")
    return L, sorted({i for d in DECISIONS for i in rec[d].get("rule_ids") or []}
                     | {s.get("id") for e in rec.get("rationale") or [] for s in e.get("sources") or []
                        if s["type"] == "passage" and s.get("id")})


KIND_WORDS = {"fixture": "an invented test case with a hand-written note",
              "scenario": "an invented test case with a generated note",
              "development": "a real ISLES'24 case (de-identified, CC BY-NC-SA 4.0) with imaging read by our models and "
                             "a note generated from its clinical record"}


def _question_lines(qs, case_n):
    L = []
    for qid, kind, text in qs:
        if kind.startswith("likert"):
            scale = " ".join(f"[ ] {x}" for x in LIKERT) + (" [ ] N/A" if kind == "likert_na" else "")
            L += [f"**{case_n}{qid}.** {text}", "", scale, ""]
            if kind == "likert_open":
                L += ["Comment:", ""]
        else:
            L += [f"**{case_n}{qid}.** {text}", "", "Answer:", ""]
    return L


def _reveal_lines(case, n):
    return _reveal_body(case, n) + [""] + _question_lines((REVEAL_QUESTION,), f"{n}") + [""]


def _reveal_body(case, n):
    if case["kind"] == "development":
        pred, gt = case["arms"]["predicted"], case["arms"]["groundtruth"]
        diffs = ha.contract_differences(pred["vision"], gt["vision"])
        L = [f"**Case {n}.** Our expected answer uses the reference follow-up imaging, not the models' reading: "
             f"core {_fmt(gt['vision']['core_ml'])} mL against {_fmt(pred['vision']['core_ml'])} mL read by our model, "
             f"and no bleed presumed, because every ISLES'24 patient went on to reperfusion treatment (the model said "
             f"{'bleed' if pred['vision']['haemorrhage']['present'] else 'no bleed'}). Differences that matter for "
             f"the rules: {', '.join(diffs) or 'none'}.", ""]
        for d in DECISIONS:
            L.append(f"- {d.capitalize()}: expected **{gt['rules'][d]['value']}** "
                     f"({gt['rules'][d]['reason'].rstrip('.')}). "
                     f"On the models' reading the rules give {pred['rules'][d]['value']}.")
    else:
        L = [f"**Case {n}.**", ""]
        for d in DECISIONS:
            r = case["index"]["rules"][d]
            L.append(f"- {d.capitalize()}: expected **{r['value']}** ({r['reason'].rstrip('.')}).")
    return L


def instrument2(cases, chosen, passages):
    """(markdown, rows) for the written interview."""
    by_id = {p["id"]: p for p in passages}
    rows, cited = [], set()
    L = ["# Stroke Decision Copilot: written interview", "", f"*{DISCLAIMER}*", "",
         "## Before you start", "",
         f"You will see {len(chosen)} cases. For each one: the imaging findings our models produced, the triage note, "
         "the language model's recommendation with its rationale, and what our verifier flagged. Please answer the "
         "questions after each case before reading on. Our expected answers are at the end; please read them only "
         "after all the cases. Answer on your own, without discussing it with the other reviewer. About 60-90 minutes.",
         "",
         f"Each case can also be opened on our demo page, {SPACE_URL}: go to the case explorer, type the case's code "
         "into the search box, and choose the model named for the case. The demo shows the same saved answer.",
         "",
         "Volumes are in mL; \"Tmax > 6 s\" is the whole hypoperfused volume, core included. Rulebook passages are "
         "cited by ID, and the passages cited in these cases are printed at the end.", "",
         "## Consent", "", CONSENT, ""]
    for c in chosen:
        n = c["n"]
        lines, ids = _case_lines(cases[c["alias"]], c)
        cited |= set(ids)
        L += lines + ["", "### Your view", ""] + _question_lines(CASE_QUESTIONS, f"{n}") + [""]
        for qid, _, text in CASE_QUESTIONS:
            rows.append({"case": n, "alias": c["alias"], "question_id": f"{n}{qid}", "question": text})
    L += ["## General questions", ""] + _question_lines(GENERAL_QUESTIONS, "")
    rows += [{"case": "", "alias": "", "question_id": qid, "question": text} for qid, _, text in GENERAL_QUESTIONS]
    L += ["## Our expected answers (read after all the cases)", "",
          "Our expected answers come from the rules you can review in the other document, applied in code. For real "
          "cases they use the reference follow-up imaging. \"Undetermined\" means the rules cannot decide from the "
          "inputs, and the case is left out of our accuracy figures.", ""]
    for c in chosen:
        L += _reveal_lines(cases[c["alias"]], c["n"])
        rows.append({"case": c["n"], "alias": c["alias"], "question_id": f"{c['n']}R1",
                     "question": REVEAL_QUESTION[2]})
    L += _cited_lines(cited, by_id) + ["---", "", f"*{DISCLAIMER}*", ""]
    return "\n".join(L), rows


def _case_lines(case, c):
    """(markdown lines, cited passage IDs) for one interview case: inputs, the answer and the verifier's flags."""
    arm = _arm(case)
    L = [f"## Case {c['n']} (code {c['alias']})", "",
         f"This is {KIND_WORDS[case['kind']]}. Answer by {MODEL_NAMES[c['model']]}.", "",
         *_vision_table(arm["vision"]), "", "**Triage note**", "", *_quote(arm["note"]), ""]
    lines, ids = _answer_lines(_answer(case, c["model"]))
    return L + lines, ids


def _cited_lines(cited, by_id):
    L = ["## Rulebook passages cited in these cases", ""]
    for pid in sorted(cited):
        p = by_id.get(pid)
        L += [f"**{pid}** ({p['pointer']}, p. {p['page']}): {p['text']}" if p else
              f"**{pid}**: not a passage in our rulebook (a citation error by the model).", ""]
    return L


def case_document(cases, chosen, passages):
    """The interview's case pages for the Google Form route: inputs, answers and flags, with no questions and no
    expected answers (the form shows those only after the cases)."""
    by_id = {p["id"]: p for p in passages}
    cited = set()
    L = ["# Stroke Decision Copilot: interview cases", "", f"*{DISCLAIMER}*", "",
         f"These are the {len(chosen)} cases for the interview form. Read each case here, then answer its section of the "
         "form before going on to the next. For each case: the imaging findings our models produced, the triage note, "
         "the language model's recommendation with its rationale, and what our verifier flagged.", "",
         f"Each case can also be opened on our demo page, {SPACE_URL}: go to the case explorer, type the case's code "
         "into the search box, and choose the model named for the case.", "",
         "Volumes are in mL; \"Tmax > 6 s\" is the whole hypoperfused volume, core included. The rulebook passages cited "
         "in these cases are printed at the end.", ""]
    for c in chosen:
        lines, ids = _case_lines(cases[c["alias"]], c)
        cited |= set(ids)
        L += lines + [""]
    L += _cited_lines(cited, by_id) + ["---", "", f"*{DISCLAIMER}*", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------- Google Forms

CONSENT_FORM = CONSENT.split("\n\n")[0]
QUOTE_CHOICES = ("By name and role", "By role only", "Do not quote me; summarise my answers only")
CHOICES = {
    "P3": ("Under 5", "5-10", "11-20", "Over 20"),
    "P4": ("Never", "Occasionally", "Regularly"),
    "Q7": ("Ignore it", "Use it only after checking every claim", "Use it after checking the flagged claims",
           "Rely on it"),
}
RESPONSE_WORDS = ("Agree", "Disagree", "Unsure / outside my practice")


def _plain(lines):
    return "\n".join(x.replace("**", "").replace("> ", "").rstrip(">") for x in lines).strip()


def _vision_plain(v):
    return [line.strip("|").replace(" | ", ": ").strip() for line in _vision_table(v)[2:]]


def _consent_items():
    return [{"type": "page", "title": "Consent", "help": f"{DISCLAIMER}\n\n{CONSENT_FORM}"},
            {"type": "checkbox", "title": "Consent", "choices": ["I have read this and agree to take part."],
             "required": True},
            {"type": "mc", "title": "How may we quote you?", "choices": list(QUOTE_CHOICES), "required": True},
            {"type": "mc", "title": "May we thank you by name in the acknowledgements?", "choices": ["Yes", "No"],
             "required": True}]


def _rated(item_id, title, help_text=""):
    return [{"type": "mc", "title": title, "help": help_text, "choices": list(RESPONSE_WORDS)},
            {"type": "paragraph", "title": f"{item_id} comment (optional)"}]


def form_spec(passages, cases, chosen):
    """Both forms as data: {"forms": [{title, description, confirmation, items}]}. Every item of the two response
    sheets appears once, its title starting with its ID; nothing is required beyond consent (anything may be skipped).
    Item types: page, header, mc, checkbox, text, paragraph."""
    confirmation = f"Thank you. Your answers are saved. {DISCLAIMER}"
    f1 = _consent_items() + [{"type": "page", "title": "About you"}]
    for item, q in ABOUT_YOU:
        f1.append({"type": "mc", "title": f"{item}. {q.split(':')[0]}", "choices": list(CHOICES[item])}
                  if item in CHOICES else {"type": "text", "title": f"{item}. {q}"})
    f1.append({"type": "page", "title": "Part A. The rulebook",
               "help": "For each passage, choose one answer and add a comment if you disagree or are unsure. Agree means "
                       "the passage is a correct and not misleading reading for adults with acute ischaemic stroke. "
                       "COR = class of recommendation."})
    topic = None
    for i, p in enumerate(passages, 1):
        if p.get("topic") != topic:
            topic = p.get("topic")
            f1.append({"type": "header", "title": topic})
        f1 += _rated(f"A{i:02d}", f"A{i:02d}. {p['id']}", f"{p['pointer']}, p. {p['page']}\n\n{p['text']}")
    f1.append({"type": "page", "title": "Part B. Five invented test cases and the answers our rules give",
               "help": "Volumes are in mL. Tmax > 6 s is the whole hypoperfused volume, core included; mismatch volume "
                       "is Tmax > 6 s minus core. Each case has one item per treatment."})
    fixtures = sorted(a for a, c in cases.items() if c["kind"] == "fixture")
    for j, alias in enumerate(fixtures, 1):
        arm = cases[alias]["arms"]["rulebook"]
        f1.append({"type": "header", "title": f"Case B{j}: {FIXTURE_TITLES.get(alias, alias)}",
                   "help": "\n".join(_vision_plain(arm["vision"])) + "\n\nTriage note:\n" + arm["note"].strip()})
        for d, letter in zip(DECISIONS, "ab"):
            r = cases[alias]["index"]["rules"][d]
            f1 += _rated(f"B{j}{letter}", f"B{j}{letter}. Expected answer. {d.capitalize()}: {r['value']}.", r["reason"])
    f1.append({"type": "page", "title": "Part C. How the rules are applied",
               "help": "Choices we made where the guideline leaves room, or where our inputs are limited. "
                       "\"Undetermined\" means the case is left out of our accuracy figures rather than guessed."})
    for k, text in enumerate(CONVENTIONS, 1):
        f1 += _rated(f"C{k:02d}", f"C{k:02d}. {text}")
    f1.append({"type": "page", "title": "Part D. Open questions"})
    f1 += [{"type": "paragraph", "title": f"D{k}. {q}"} for k, q in enumerate(OPEN_QUESTIONS_1, 1)]

    f2 = _consent_items()
    f2.append({"type": "page", "title": "Before you start",
               "help": f"You will see {len(chosen)} cases, in the case document we sent with this form. Read each case "
                       "there, then answer its section here before going on. Our expected answers come at the end of "
                       "this form, after all the cases. Answer on your own, without discussing it with the other "
                       f"reviewer. About 60-90 minutes. The cases can also be opened on {SPACE_URL}."})
    for c in chosen:
        n = c["n"]
        f2.append({"type": "page", "title": f"Case {n} (code {c['alias']})",
                   "help": f"Read Case {n} in the case document (answer by {MODEL_NAMES[c['model']]}), then answer."})
        for qid, kind, text in CASE_QUESTIONS:
            title = f"{n}{qid}. {text}"
            if kind.startswith("likert"):
                choices = list(LIKERT) + (["N/A, nothing was flagged"] if kind == "likert_na" else [])
                f2.append({"type": "mc", "title": title, "choices": [x.capitalize() for x in choices]})
            elif qid in CHOICES:
                f2.append({"type": "mc", "title": f"{n}{qid}. If this answer reached you in practice, you would:",
                           "choices": list(CHOICES[qid])})
            else:
                f2.append({"type": "paragraph", "title": title})
    f2.append({"type": "page", "title": "General questions"})
    for qid, kind, text in GENERAL_QUESTIONS:
        if kind == "likert_open":
            f2.append({"type": "mc", "title": f"{qid}. {text.split('.')[0]}.",
                       "choices": [x.capitalize() for x in LIKERT]})
            f2.append({"type": "paragraph", "title": f"{qid} comment: {text.split('. ', 1)[1]}"})
        else:
            f2.append({"type": "paragraph", "title": f"{qid}. {text}"})
    f2.append({"type": "page", "title": "Our expected answers",
               "help": "Our expected answers come from the rules you reviewed in the other form, applied in code. For "
                       "real cases they use the reference follow-up imaging. \"Undetermined\" means the rules cannot "
                       "decide from the inputs, and the case is left out of our accuracy figures."})
    for c in chosen:
        n = c["n"]
        f2.append({"type": "header", "title": f"Case {n} (code {c['alias']})",
                   "help": _plain(_reveal_body(cases[c["alias"]], n))})
        f2 += _rated(f"{n}R1", f"{n}R1. Do you agree with our expected answer for case {n}?")
    return {"forms": [
        {"title": "Stroke Decision Copilot: rulebook and test-case review", "confirmation": confirmation,
         "description": f"{DISCLAIMER}\n\nThank you for reviewing our work. Every result we report is scored against "
                        "these rules, so we need your honest reading. Please answer on your own. About 45-60 minutes. "
                        "You can come back and edit your answers.", "items": f1},
        {"title": "Stroke Decision Copilot: written interview", "confirmation": confirmation,
         "description": f"{DISCLAIMER}\n\nPlease have the case document open beside this form. About 60-90 minutes. "
                        "You can come back and edit your answers.", "items": f2}]}


APPS_SCRIPT = """// Generated by strokecopilot.pipeline.clinician_pack from the fixed pack (docs/CLINICIAN_PACK.md). Do not edit by
// hand: change the code and rebuild, so the forms always match the recorded items.
//
// Run buildForms() once at script.google.com (new project, paste this file, Run). It creates two Google Forms, each
// with its own response Sheet, in your Drive, and logs their links (View > Logs). It sends nothing to anyone.
// Afterwards, open each form, check it is published and that "Collect email addresses" is off, then send the
// respondent link yourself.

const SPEC = %SPEC%;

function buildForms() {
  SPEC.forms.forEach(function (f) {
    const form = FormApp.create(f.title);
    form.setDescription(f.description)
        .setProgressBar(true)
        .setAllowResponseEdits(true)
        .setCollectEmail(false)
        .setShowLinkToRespondAgain(false)
        .setConfirmationMessage(f.confirmation);
    f.items.forEach(function (it) { addItem(form, it); });
    const sheet = SpreadsheetApp.create(f.title + ' (responses)');
    form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getId());
    try { form.setPublished(true); } catch (e) { Logger.log('Publish it by hand: ' + e); }
    Logger.log(f.title + '\\n  respondent link: ' + form.getPublishedUrl() + '\\n  edit: ' + form.getEditUrl() +
               '\\n  responses: ' + sheet.getUrl() + '\\n  items: ' + f.items.length);
  });
}

function addItem(form, it) {
  let item;
  if (it.type === 'page') item = form.addPageBreakItem();
  else if (it.type === 'header') item = form.addSectionHeaderItem();
  else if (it.type === 'mc') item = form.addMultipleChoiceItem().setChoiceValues(it.choices);
  else if (it.type === 'checkbox') item = form.addCheckboxItem().setChoiceValues(it.choices);
  else if (it.type === 'text') item = form.addTextItem();
  else if (it.type === 'paragraph') item = form.addParagraphTextItem();
  else throw new Error('Unknown item type: ' + it.type);
  item.setTitle(it.title);
  if (it.help) item.setHelpText(it.help);
  if (it.required) item.setRequired(true);
}
"""


def apps_script(spec):
    return APPS_SCRIPT.replace("%SPEC%", json.dumps(spec, ensure_ascii=False, indent=1))


def to_docx(markdown_path, docx_path):
    """Word copy of a Markdown file through pandoc, reproducible (SOURCE_DATE_EPOCH). False if pandoc is missing."""
    import os
    import shutil
    import subprocess
    if not shutil.which("pandoc"):
        return False
    subprocess.run(["pandoc", str(markdown_path), "-o", str(docx_path)], check=True,
                   env={**os.environ, "SOURCE_DATE_EPOCH": "1795651200"})
    return True


# ------------------------------------------------------------------------------------------- Writing

def _write_csv(path, rows, extra):
    fields = list(rows[0]) + list(extra)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{k: "" for k in extra}})


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(bank_dir, out_dir, seed=SEED, passages_path=None):
    """Build both instruments and write them with a manifest. Returns the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    passages = load_passages(passages_path)
    cases = load_bank(bank_dir)
    chosen, sizes = select(cases, seed)
    md1, rows1 = instrument1(passages, cases)
    md2, rows2 = instrument2(cases, chosen, passages)
    spec = form_spec(passages, cases, chosen)
    files = {
        "instrument1_rulebook_review.md": md1,
        "instrument2_interview.md": md2,
        "interview_cases.md": case_document(cases, chosen, passages),
        "forms_spec.json": json.dumps(spec, ensure_ascii=False, indent=1),
        "build_forms.gs": apps_script(spec),
        "selection.json": json.dumps({"seed": seed, "plan": [list(p) for p in PLAN], "pool_sizes": sizes,
                                      "cases": chosen}, indent=2),
    }
    for name, text in files.items():
        (out / name).write_text(text)
    _write_csv(out / "instrument1_responses.csv", rows1, ("response", "comment"))
    _write_csv(out / "instrument2_responses.csv", rows2, ("response", "comment"))
    names = sorted(files) + ["instrument1_responses.csv", "instrument2_responses.csv"]
    if to_docx(out / "interview_cases.md", out / "interview_cases.docx"):
        names.append("interview_cases.docx")
    manifest = {"seed": seed, "bank_index_sha256": _sha(Path(bank_dir) / "index.json"),
                "passages_sha256": _sha(passages_path or CORPUS_DIR / "passages.jsonl"),
                "items_instrument1": len(rows1), "cases": [(c["n"], c["alias"], c["stratum"], c["model"]) for c in chosen],
                "pool_sizes": sizes, "files": {n: _sha(out / n) for n in names}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    problems = leak_scan(out)
    if problems:
        raise ValueError(f"The pack failed the leak scan ({len(problems)} files). Delete {out} and check.")
    return manifest


def leak_scan(folder):
    """The bank's scan (development case IDs, compared in code and never printed; RSNA IDs; key patterns)."""
    from strokecopilot.data import splits
    from strokecopilot.demo import bank
    return bank.scan(folder, list(splits.development_ids()))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", default="demo/space/build/bank")
    ap.add_argument("--out", default="runs/EXP-67_clinician_review/pack")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    m = write(args.bank, args.out, args.seed)
    print(json.dumps({k: m[k] for k in ("seed", "items_instrument1", "cases", "pool_sizes")}, indent=1))


if __name__ == "__main__":
    main()
