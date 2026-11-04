"""Reasoner output -> atomic claims (docs/HALLUCINATION_METRIC.md §1-2, plan §8.4).

Three sources of claims, all with a pointer back to the field and entry they came from:
- the LLM extractor (`prompts/claim_extractor_v1.md`), which splits rationale entries, blocking
  factors and fact-stating uncertainties into atomic claims and labels what each one is about,
  using fixed vocabularies of vision fields and note facts. It never sees the inputs' values, so
  it cannot correct a claim while extracting it;
- a regex for numbers with units: a number the LLM's claims do not cover becomes a claim of its own;
- each decision's rule_ids entry, as a citation claim.

The extractor gives numbers as written ("18.0"), because a number's stated precision decides how
it is compared (§3b).
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from strokecopilot.config import PROMPTS_DIR
from strokecopilot.nlp.notes.facts import FACT_KEYS

EXTRACTOR_PROMPT = "claim_extractor_v1"
EXTRACTOR_MODEL = "claude-sonnet-5-5"  # with effort medium; held fixed in the EXP-11 bake-off
EXTRACTOR_EFFORT = "medium"

VISION_SUBJECTS = {
    "haemorrhage.present": "intracranial haemorrhage on plain CT (true or false)",
    "haemorrhage.confidence": "probability that a haemorrhage is present",
    "core_ml": "infarct core volume, mL",
    "penumbra_ml": "total hypoperfused (Tmax > 6 s) volume, core included, mL",
    "mismatch_ratio": "penumbra volume divided by core volume",
    "mismatch_volume_ml": "penumbra minus core, mL (salvageable tissue)",
    "occlusion_site": "site of the occlusion: ICA, M1, M2, basilar or unknown",
    "large_vessel_occlusion": "whether there is a large-vessel occlusion (true or false)",
    "aspects": "ASPECTS score",
    "confidence.core": "the vision model's confidence in the core volume",
    "confidence.penumbra": "the vision model's confidence in the penumbra volume",
    "warnings": "vision pipeline warnings, such as a missing scan or artefact",
}
# Inputs the pipeline cannot observe (plan §8.3): a claim that one is unknown is supported.
UNOBSERVED_SUBJECTS = {
    "mass_effect": "significant mass effect",
    "m2_dominance": "whether an M2 occlusion is dominant, codominant or non-dominant",
    "extensive_hypodensity": "extensive clear hypodensity on CT",
    "pc_aspects": "posterior-circulation ASPECTS",
    "dwi_flair": "DWI or FLAIR findings",
}
KINDS = ("vision", "patient", "absence", "rule", "comparison")
OPS = ("<", "<=", ">", ">=", "==", "!=")
FIELDS = ("rationale", "thrombolysis.blocking_factors", "thrombectomy.blocking_factors", "uncertainties")

UNITS = r"mL|ml|cc|hours?|hrs?|h|minutes?|mins?|min|seconds?|s|%|mmHg|mmol/L|mg/dL|years?|y"
_UNIT_RE = re.compile(rf"^({UNITS})\b")
_KEYWORD_BEFORE = re.compile(r"(NIHSS|ASPECTS|mRS|INR|ratio|score)\D{0,12}$", re.I)
_RANGE_BEFORE = re.compile(r"\d\s*(?:[–-]|\bto\b)\s*$")
_PASSAGE_ID = re.compile(r"\b[A-Z][A-Za-z0-9]*-[\w.\-]+")
_NUMBER = re.compile(r"(?<![\w.§/–-])(\d+(?:\.\d+)?)(?![\w.]*\d)(?!\.\d)(?!\s*[–-]\s*\d)(?!\s+to\s+\d)")  # ranges such as 6-10 or 6 to 10 are rule text


@dataclass
class Claim:
    id: str
    field: str
    entry: int
    text: str
    kind: str
    subject: str | None = None
    value: str | None = None
    unit: str | None = None
    applies_to_patient: bool = False
    comparison: dict | None = None
    origin: str = "llm"  # llm, regex or citation
    declared_sources: list = field(default_factory=list)
    derived: bool = False

    def to_dict(self):
        return asdict(self)


def entries(rec):
    """(field, entry index, text, declared sources, derived) for every field that can hold claims."""
    out = [("rationale", i, item["claim"], item["sources"], item["derived"]) for i, item in enumerate(rec["rationale"])]
    for decision in ("thrombolysis", "thrombectomy"):
        # The schema gives blocking factors no sources. They rest on the rule_ids of both decisions, own decision
        # first: a reason against one treatment often rests on the other's rule ("eligible for thrombectomy").
        other = "thrombectomy" if decision == "thrombolysis" else "thrombolysis"
        ids = list(dict.fromkeys(rec[decision]["rule_ids"] + rec[other]["rule_ids"]))
        cited = [{"type": "passage", "id": pid} for pid in ids]
        out += [(f"{decision}.blocking_factors", i, text, cited, False) for i, text in enumerate(rec[decision]["blocking_factors"])]
    out += [("uncertainties", i, text, [], False) for i, text in enumerate(rec["uncertainties"])]
    return out


def numbers(text):
    """[(number as written, unit or None, start)] for numbers worth checking: with a unit, or after a score name.

    Passage IDs, section numbers (4.6.1), § references and ranges (6-24 h) are skipped.
    """
    masked = _PASSAGE_ID.sub(lambda m: " " * len(m.group()), text)
    out = []
    # "2 h 30 min" is one time: report it once, in minutes, and mask it so its parts are not read again.
    for m in re.finditer(r"(?<![\d.])(\d+)\s*h\s+(\d+)\s*min\b", masked):
        out.append((str(int(m.group(1)) * 60 + int(m.group(2))), "min", m.start()))
        masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
    for match in _NUMBER.finditer(masked):
        if _RANGE_BEFORE.search(masked[: match.start()]):
            continue  # the upper end of a range
        rest = masked[match.end():].lstrip()
        unit = _UNIT_RE.match(rest)
        if unit or _KEYWORD_BEFORE.search(masked[: match.start()]):
            out.append((match.group(1), unit.group(1) if unit else None, match.start()))
    return out


def _covered(value, claims):
    target = float(value)
    for claim in claims:
        values = [claim.value]
        if claim.comparison:
            values += [claim.comparison["left"].get("value"), claim.comparison["right"].get("value")]
        for v in values:
            try:
                if v is not None and float(v) == target:
                    return True
            except ValueError:
                continue
    return False


def regex_claims(rec, claims):
    """A claim for every checkable number in a field entry that no extracted claim covers."""
    out = []
    for field_name, entry, text, sources, derived in entries(rec):
        mine = [c for c in claims if c.field == field_name and c.entry == entry]
        for k, (value, unit, _) in enumerate(numbers(text)):
            if not _covered(value, mine):
                out.append(Claim(id=f"{field_name}[{entry}].n{k}", field=field_name, entry=entry, text=text, kind="number",
                                 value=value, unit=unit, origin="regex", declared_sources=sources, derived=derived))
    return out


def citation_claims(rec):
    """One claim per decision rule_ids entry: "this decision rests on passage X"."""
    out = []
    for decision in ("thrombolysis", "thrombectomy"):
        for i, passage_id in enumerate(rec[decision]["rule_ids"]):
            out.append(Claim(id=f"{decision}.rule_ids[{i}]", field=f"{decision}.rule_ids", entry=i,
                             text=f"Passage {passage_id} is a rule about {decision}.", kind="citation", subject=decision,
                             value=passage_id, origin="citation", declared_sources=[{"type": "passage", "id": passage_id}]))
    return out


# --- LLM extraction -------------------------------------------------------------------------

_OPERAND = {
    "type": "object", "additionalProperties": False,
    "required": ["subject", "value", "unit", "from_rule"],
    "properties": {
        "subject": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "from_rule": {"type": "boolean"},
    },
}
EXTRACTOR_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["claims"],
    "properties": {"claims": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["field", "entry", "text", "kind", "subject", "value", "unit", "applies_to_patient", "comparison"],
        "properties": {
            "field": {"type": "string", "enum": list(FIELDS)},
            "entry": {"type": "integer"},
            "text": {"type": "string"},
            "kind": {"type": "string", "enum": list(KINDS)},
            "subject": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "applies_to_patient": {"type": "boolean"},
            "comparison": {"anyOf": [{"type": "null"}, {
                "type": "object", "additionalProperties": False, "required": ["left", "op", "right"],
                "properties": {"left": _OPERAND, "op": {"type": "string", "enum": list(OPS)}, "right": _OPERAND},
            }]},
        },
    }}},
}


def _vocabulary(names):
    return "\n".join(f"- `{key}`: {text}" for key, text in names.items())


def system_prompt(version=EXTRACTOR_PROMPT):
    """The extractor prompt with its vocabularies filled in. The same for every case, so it caches."""
    template = (Path(PROMPTS_DIR) / f"{version}.md").read_text()
    return (template.replace("{{VISION_SUBJECTS}}", _vocabulary(VISION_SUBJECTS))
            .replace("{{FACT_KEYS}}", _vocabulary(FACT_KEYS))
            .replace("{{UNOBSERVED_SUBJECTS}}", _vocabulary(UNOBSERVED_SUBJECTS)))


def user_message(rec):
    """The fields to split, numbered, and nothing about the case's inputs."""
    items = [{"field": f, "entry": i, "text": text} for f, i, text, _, _ in entries(rec)]
    return "Split these entries into atomic claims.\n\n" + json.dumps(items, indent=1)


def _from_llm(item, rec_entries, counters):
    key = (item["field"], item["entry"])
    if key not in rec_entries:
        return None  # an entry that does not exist: ignored, never scored
    sources, derived = rec_entries[key]
    counters[key] = counters.get(key, -1) + 1
    subject = item["subject"]
    if subject is not None and subject not in VISION_SUBJECTS and subject not in FACT_KEYS and subject not in UNOBSERVED_SUBJECTS:
        subject = f"other:{subject}"  # outside every vocabulary: nothing in the inputs can support it
    return Claim(id=f"{item['field']}[{item['entry']}].{counters[key]}", field=item["field"], entry=item["entry"],
                 text=item["text"], kind=item["kind"], subject=subject, value=item["value"], unit=item["unit"],
                 applies_to_patient=item["applies_to_patient"], comparison=item["comparison"],
                 declared_sources=sources, derived=derived)


def extract(rec, llm, version=EXTRACTOR_PROMPT):
    """All claims of one recommendation: LLM claims, regex number claims and citation claims."""
    reply = llm(system_prompt(version), user_message(rec), EXTRACTOR_SCHEMA)
    rec_entries = {(f, i): (sources, derived) for f, i, _, sources, derived in entries(rec)}
    counters = {}
    claims = [c for c in (_from_llm(item, rec_entries, counters) for item in reply["claims"]) if c is not None]
    return claims + regex_claims(rec, claims) + citation_claims(rec)
