"""The clinician pack (Phase 5, task A): the rulebook and test-case review for two clinicians, fixed before it is sent.

Every rulebook passage, every fixture decision and every rule convention the code uses, each answered agree / disagree /
unsure with a comment. The two clinicians answer independently, so their agreement can be reported
(docs/CLINICIAN_PACK.md §6). Only the fixtures of the Space's bank are shown; held-out cases are never used, and the
bank loader refuses one.

Output (`write`): the review as Markdown (the record), its response sheet as CSV, and the Google Forms route chosen by
Parsa (20 Nov 2026): `forms_spec.json` and `build_forms.gs` (an Apps Script that builds the form from it). A manifest
records the sha256 of every file. The output goes to a gitignored run folder, never to docs/.
"""

import csv
import hashlib
import json
from pathlib import Path

from strokecopilot.config import CORPUS_DIR
from strokecopilot.pipeline import heldout_analysis as ha

DECISIONS = ha.DECISIONS
DISCLAIMER = ("Research prototype, not for clinical use. Your answers are expert opinion on a student research project. "
              "They are not a clinical validation, and we will not describe them as one.")
RESPONSES = ("agree", "disagree", "unsure")
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

# Asked at the start, so the report can describe the reviewers in their own words.
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


# ------------------------------------------------------------------------------------------ The review

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


# ------------------------------------------------------------------------------------- Google Forms

CONSENT_FORM = CONSENT.split("\n\n")[0]
QUOTE_CHOICES = ("By name and role", "By role only", "Do not quote me; summarise my answers only")
CHOICES = {
    "P3": ("Under 5", "5-10", "11-20", "Over 20"),
    "P4": ("Never", "Occasionally", "Regularly"),
}
RESPONSE_WORDS = ("Agree", "Disagree", "Unsure / outside my practice")


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


def form_spec(passages, cases):
    """The form as data: {"forms": [{title, description, confirmation, items}]}. Every item of the response sheet
    appears once, its title starting with its ID; nothing is required beyond consent (anything may be skipped).
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

    return {"forms": [
        {"title": "Stroke Decision Copilot: rulebook and test-case review", "confirmation": confirmation,
         "description": f"{DISCLAIMER}\n\nThank you for reviewing our work. Every result we report is scored against "
                        "these rules, so we need your honest reading. Please answer on your own. About 45-60 minutes. "
                        "You can come back and edit your answers.", "items": f1}]}


APPS_SCRIPT = """// Generated by strokecopilot.pipeline.clinician_pack from the fixed pack (docs/CLINICIAN_PACK.md). Do not edit by
// hand: change the code and rebuild, so the forms always match the recorded items.
//
// Run buildForms() once at script.google.com (new project, paste this file, Run). It creates the Google Form, with
// its own response Sheet, in your Drive, and logs its links (View > Logs). It sends nothing to anyone.
// Afterwards, open the form, check it is published and that "Collect email addresses" is off, then send the
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


def write(bank_dir, out_dir, passages_path=None):
    """Build the review and write it with a manifest. Returns the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    passages = load_passages(passages_path)
    cases = load_bank(bank_dir)
    md1, rows1 = instrument1(passages, cases)
    spec = form_spec(passages, cases)
    files = {
        "instrument1_rulebook_review.md": md1,
        "forms_spec.json": json.dumps(spec, ensure_ascii=False, indent=1),
        "build_forms.gs": apps_script(spec),
    }
    for name, text in files.items():
        (out / name).write_text(text)
    _write_csv(out / "instrument1_responses.csv", rows1, ("response", "comment"))
    names = sorted(files) + ["instrument1_responses.csv"]
    manifest = {"bank_index_sha256": _sha(Path(bank_dir) / "index.json"),
                "passages_sha256": _sha(passages_path or CORPUS_DIR / "passages.jsonl"),
                "items_instrument1": len(rows1), "files": {n: _sha(out / n) for n in names}}
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
    args = ap.parse_args(argv)
    m = write(args.bank, args.out)
    print(json.dumps({k: m[k] for k in ("items_instrument1", "files")}, indent=1))


if __name__ == "__main__":
    main()
