"""Claim -> its sources, and the hallucination sub-type if it has none (docs/HALLUCINATION_METRIC.md §3-4).

Every claim is traced on its own, ignoring the sources the model declared, then the declared
sources are compared with what was found:
- (a) vision fields and (b) note facts are compared in code, a number after rounding the true value
  to the precision the claim states, in the claim's unit;
- (c) rules go to an LLM judge with the passages the claim cites plus the retriever's top 5 above a
  similarity threshold tau. The judge returns supported / contradicted / not_addressed per passage
  and, for a claim that applies a rule to this patient, whether the passage's scope fits;
- comparisons are recomputed in code from the true values of their inputs.

Each hallucination gets one sub-type, the first that applies: 1 wrong number, 2 invented finding,
3 wrong rule, 4 misapplied rule, 5 faulty derivation, 6 uncited rule, 7 mis-attributed source.
Criticality needs the rulebook in code (scenarios.py), so it is left None here.
"""

import json
import math
import operator
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from strokecopilot.config import PROMPTS_DIR
from strokecopilot.nlp.notes.facts import FACT_KEYS
from strokecopilot.nlp.verifier.claims import UNOBSERVED_SUBJECTS, VISION_SUBJECTS

SUBTYPES = {1: "wrong_number", 2: "invented_finding", 3: "wrong_rule", 4: "misapplied_rule",
            5: "faulty_derivation", 6: "uncited_rule", 7: "misattributed_source"}
CONTENT_FALSE = {1, 2, 3, 4, 5}
JUDGE_PROMPT = "judge_v2"
# DEC-20 (EXP-11 bake-off, 4 Nov 2026): the cheaper judge that meets the seeded-error bar, and the similarity a retrieved
# passage needs before the judge sees it. The claim extractor is claims.EXTRACTOR_MODEL.
JUDGE_MODEL = "claude-haiku-5-5"
JUDGE_EFFORT = "medium"
TAU = 0.6
TOP_K = 5
LARGE_VESSEL = {"ICA", "M1", "basilar"}

# unit -> (dimension, factor to the dimension's base unit)
_UNITS = {
    "ml": ("volume", 1.0), "cc": ("volume", 1.0),
    "s": ("time", 1 / 60), "sec": ("time", 1 / 60), "second": ("time", 1 / 60), "seconds": ("time", 1 / 60),
    "min": ("time", 1.0), "mins": ("time", 1.0), "minute": ("time", 1.0), "minutes": ("time", 1.0),
    "h": ("time", 60.0), "hr": ("time", 60.0), "hrs": ("time", 60.0), "hour": ("time", 60.0), "hours": ("time", 60.0),
    "%": ("fraction", 0.01), "fraction": ("fraction", 1.0),
    "y": ("years", 1.0), "year": ("years", 1.0), "years": ("years", 1.0),
    "mmhg": ("pressure", 1.0), "mmol/l": ("glucose", 1.0), "mg/dl": ("glucose", 1 / 18.016),
}
VISION_UNITS = {"core_ml": "mL", "penumbra_ml": "mL", "mismatch_volume_ml": "mL", "mismatch_ratio": None,
                "haemorrhage.confidence": "fraction", "confidence.core": "fraction", "confidence.penumbra": "fraction", "aspects": None}
_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne}
_TRUE, _FALSE = {"true", "yes", "present", "positive"}, {"false", "no", "absent", "negative", "none"}
_STOP = {"and", "the", "with", "from", "for", "was", "has", "his", "her", "of", "on", "in", "at", "to", "a", "an"}


@dataclass
class Inputs:
    """What the reasoner was given, plus the verifier's own tools."""

    vision: dict
    note: str
    facts: object  # FactSheet
    corpus: dict  # passage id -> {"text", "source", "section", ...}
    judge: object = None  # callable(claim_text, passages, case_inputs, applied) -> {id: (verdict, scope_fits)}
    retriever: object = None  # callable(text, k) -> [(passage id, similarity)]
    tau: float = TAU


@dataclass
class Trace:
    claim_id: str
    field: str
    kind: str
    traceable: bool
    subtype: int | None
    subtype_name: str | None
    content_false: bool
    reason: str
    used: list = field(default_factory=list)  # supporting sources found
    supported_passages: list = field(default_factory=list)
    declared_ok: bool = True
    incomplete: bool = False
    critical: bool | None = None
    substitution: dict | None = None  # for criticality: the claimed value of a vision or note subject

    def to_dict(self):
        return asdict(self)


# --- values and units ---------------------------------------------------------------------

def _unit(unit):
    if unit is None:
        return (None, 1.0)
    return _UNITS.get(unit.strip().lower(), (unit.strip().lower(), 1.0))


def _decimals(text):
    return len(text.split(".", 1)[1]) if "." in text else 0


_COMPOUND = re.compile(r"^\s*(\d+)\s*(?:h|hr|hrs|hours?)\s*(\d+)\s*(?:min|mins|minutes?)\s*$", re.I)


def _compound(claimed, claimed_unit):
    """'2 h 30 min' (as the value, or split across value and unit) -> ('150', 'min'); anything else unchanged."""
    joined = f"{claimed} {claimed_unit or ''}".strip()
    match = _COMPOUND.match(str(claimed)) or _COMPOUND.match(joined)
    if match:
        return str(int(match.group(1)) * 60 + int(match.group(2))), "min"
    return claimed, claimed_unit


def number_matches(claimed, claimed_unit, true_value, true_unit):
    """True if the true value, in the claim's unit and rounded to its stated precision, equals the claim."""
    claimed, claimed_unit = _compound(claimed, claimed_unit)
    try:
        claimed_value = float(claimed)
    except (TypeError, ValueError):
        return False
    c_dim, c_factor = _unit(claimed_unit if claimed_unit is not None else true_unit)
    t_dim, t_factor = _unit(true_unit)
    if c_dim != t_dim:
        return False
    in_claim_unit = true_value * t_factor / c_factor
    return math.isclose(round(in_claim_unit + 0.0, _decimals(str(claimed))), claimed_value, abs_tol=1e-9)


def _base(value, unit):
    dim, factor = _unit(unit)
    return dim, value * factor


def _bool(text):
    text = str(text).strip().lower()
    return True if text in _TRUE else False if text in _FALSE else None


def _tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in _STOP and len(t) > 1}


def _clock(text):
    match = re.search(r"(\d{1,2})[:.h](\d{2})", str(text))
    return f"{int(match.group(1)):02d}:{match.group(2)}" if match else None


def _vision_value(vision, subject):
    if subject == "large_vessel_occlusion":
        site = vision.get("occlusion_site")
        return None if site in (None, "unknown") else site in LARGE_VESSEL
    value = vision
    for part in subject.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _vision_field(subject):
    return "occlusion_site" if subject == "large_vessel_occlusion" else subject


# --- checks: each returns (sub-type or None, used sources, reason) -----------------------------

_BOUNDS = (("less than", "<"), ("under", "<"), ("below", "<"), ("at most", "<="), ("more than", ">"), ("over", ">"),
           ("above", ">"), ("at least", ">="), ("exceeds", ">"), ("exceeding", ">"))


_UNKNOWN_WORDS = {"null", "none", "unknown", "undefined", "not available", "not known", "not computed", "n/a", "na"}
_VALUE_BOUND = re.compile(r"^\s*(within|under|below|less than|at most|over|above|more than|at least)\s+(\d+(?:\.\d+)?)\s*$", re.I)
_VALUE_TAIL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+or\s+(more|higher|above|greater|less|lower|below|fewer)\s*$", re.I)
_VALUE_SET = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:or|to|-|–)\s*(\d+(?:\.\d+)?)\s*$", re.I)
_BOUND_OPS = {"within": "<=", "under": "<", "below": "<", "less than": "<", "at most": "<=", "over": ">", "above": ">",
              "more than": ">", "at least": ">=", "more": ">=", "higher": ">=", "greater": ">=", "less": "<=", "lower": "<=", "fewer": "<="}


def _is_unknown(value):
    return value is None or str(value).strip().lower() in _UNKNOWN_WORDS


def value_constraint(value):
    """A value written as a bound or a set ("6 or more", "within 6", "0 or 1", "3 to 5") -> (test(number), text),
    else None. The bound is tested in the claim's unit; a set or range includes its ends."""
    text = str(value or "")
    m = _VALUE_BOUND.match(text)
    if m:
        op, bound = _BOUND_OPS[m.group(1).lower()], float(m.group(2))
        return (lambda x: _OPS[op](x, bound)), f"{op} {bound:g}"
    m = _VALUE_TAIL.match(text)
    if m:
        op, bound = _BOUND_OPS[m.group(2).lower()], float(m.group(1))
        return (lambda x: _OPS[op](x, bound)), f"{op} {bound:g}"
    m = _VALUE_SET.match(text)
    if m:
        low, high = sorted((float(m.group(1)), float(m.group(2))))
        return (lambda x: low <= x <= high), f"{low:g} to {high:g}"
    return None


def _constrained(value, unit, true_value, true_unit):
    """(holds, description) when the claimed value is a bound or set, else None. Compared in the claim's unit."""
    found = value_constraint(value)
    if found is None:
        return None
    test, text = found
    c_dim, c_factor = _unit(unit if unit is not None else true_unit)
    t_dim, t_factor = _unit(true_unit)
    if c_dim != t_dim:
        return False, text
    return test(true_value * t_factor / c_factor), text


def _bound(text, value):
    """The operator when the text gives the value as a bound ("the core is under 1 mL"), else None."""
    for word, op in _BOUNDS:
        if re.search(rf"\b{word}\s+{re.escape(str(value))}(?![\d.])", str(text), re.I):
            return op
    return None


def check_vision(subject, value, unit, vision):
    true = _vision_value(vision, subject)
    used = [{"type": "vision", "field": _vision_field(subject)}]
    if true is None and subject != "warnings" and _is_unknown(value):
        return None, used, f"{subject} is null, and the claim says it is unknown."
    if true is None and subject != "warnings":
        if value is not None and _number_or_none(value) == 0:
            return 1, used, f"{subject} is null (unknown), but the claim gives 0."
        return 2, used, f"{subject} is null (unknown), but the claim gives it a value."
    if subject == "warnings":
        if any(_tokens(value) <= _tokens(w) for w in vision.get("warnings", [])):
            return None, used, "Matches a warning."
        return 2, used, "No vision warning says this."
    if isinstance(true, bool):
        return (None, used, "Matches.") if _bool(value) == true else (2, used, f"{subject} is {true}.")
    if isinstance(true, (int, float)):
        if number_matches(value, unit, float(true), VISION_UNITS.get(subject)):
            return None, used, "Matches after rounding."
        bound = _constrained(value, unit, float(true), VISION_UNITS.get(subject))
        if bound is not None:
            return (None, used, f"{subject} {true} is {bound[1]}.") if bound[0] else (5, used, f"{subject} {true} is not {bound[1]}.")
        return 1, used, f"{subject} is {true}."
    return (None, used, "Matches.") if str(value).strip().casefold() == str(true).casefold() else (2, used, f"{subject} is {true}.")


def _number_or_none(text):
    try:
        return float(_compound(text, None)[0])
    except (TypeError, ValueError):
        return None


def check_fact(subject, value, unit, facts):
    fact = facts.get(subject)
    if fact is None:
        return 2, [], f"The note does not state {subject}."
    used = [{"type": "note", "key": subject, "span": fact.span}]
    if fact.type == "not_recorded":
        return 2, used, f"The note says {subject} is not recorded."
    if value is None or (fact.unit and str(value).strip().lower() == fact.unit.lower()):
        # No value claimed ("glucose was measured"), or only the unit ("glucose is given in mmol/L"): the note states it.
        return None, used, f"The note states {subject}."
    if fact.type == "number":
        if number_matches(value, unit, float(fact.value), fact.unit):
            return None, used, "Matches after rounding."
        bound = _constrained(value, unit, float(fact.value), fact.unit)
        if bound is not None:
            return (None, used, f"{subject} {fact.value} is {bound[1]}.") if bound[0] else (5, used, f"{subject} {fact.value} is not {bound[1]}.")
        return 1, used, f"The note gives {fact.value} {fact.unit or ''}".strip() + "."
    if fact.type == "bool":
        return (None, used, "Matches.") if _bool(value) == fact.value else (2, used, f"The note says {subject} is {fact.value}.")
    if fact.type == "time":
        return (None, used, "Matches.") if _clock(value) == fact.value else (2, used, f"The note gives {fact.value}.")
    claimed = _tokens(value)
    if claimed and claimed <= _tokens(fact.value) | _tokens(fact.span):
        return None, used, "Matches the note's wording."
    return 2, used, "The note does not say this."


def _warning_for(text, vision):
    """The pipeline warning a claim's text restates, if any ("CTA was not acquired" restates "CTA not acquired")."""
    claimed = _tokens(text)
    for w in vision.get("warnings", []):
        if _tokens(w) and _tokens(w) <= claimed:
            return w
    return None


def check_absence(subject, inputs, text=None):
    if text and _warning_for(text, inputs.vision):
        return None, [{"type": "vision", "field": "warnings"}], "A pipeline warning says it is missing."
    if subject == "occlusion_site" and inputs.vision.get("occlusion_site") == "unknown":
        return None, [{"type": "vision", "field": "occlusion_site"}], "The occlusion site is reported as unknown."
    if subject in VISION_SUBJECTS:
        if _vision_value(inputs.vision, subject) is None:
            return None, [{"type": "vision", "field": _vision_field(subject)}], "The field is null."
        return 2, [{"type": "vision", "field": _vision_field(subject)}], f"{subject} is known."
    fact = inputs.facts.get(subject)
    if fact is not None:
        used = [{"type": "note", "key": subject, "span": fact.span}]
        return (None, used, "The note says it is not recorded.") if fact.type == "not_recorded" else (2, used, "The note states it.")
    if subject in UNOBSERVED_SUBJECTS:
        return None, [], "The pipeline cannot observe it."
    # The claim is that the note does not give it. A silent note makes that true (§3b is about claims *for or
    # against* a value, which silence supports neither way).
    if subject in FACT_KEYS:
        return None, [], f"The note does not state {subject}."
    name = _tokens((subject or "").removeprefix("other:").replace("_", " "))
    if name and not name <= _tokens(inputs.note):
        return None, [], "The note does not mention it."
    return 2, [], "The note mentions it, so it is not missing."


def _subject_check(subject, value, unit, inputs, text=None):
    if subject is None or subject.startswith("other:"):
        if text and _warning_for(text, inputs.vision):
            return None, [{"type": "vision", "field": "warnings"}], "Restates a pipeline warning."
        return 2, [], "The claim is about something no input covers."
    if subject in VISION_SUBJECTS:
        return check_vision(subject, value, unit, inputs.vision)
    return check_fact(subject, value, unit, inputs.facts)


def _true_number(subject, inputs):
    """(value in base unit, dimension) of a vision or note number, or None."""
    if subject in VISION_SUBJECTS:
        value = _vision_value(inputs.vision, subject)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            dim, base = _base(float(value), VISION_UNITS.get(subject))
            return base, dim
        return None
    fact = inputs.facts.get(subject) if subject else None
    if fact is not None and fact.type == "number":
        dim, base = _base(float(fact.value), fact.unit)
        return base, dim
    return None


def check_comparison(comparison, inputs):
    """(sub-type, used, reason, rule_needed). Inputs first, then the comparison itself."""
    used, sides, rule_needed = [], [], False
    for side in ("left", "right"):
        operand = comparison[side]
        if operand["from_rule"]:
            rule_needed = True
            number = _number_or_none(operand["value"])
            if number is None:
                return 5, used, f"The {side} threshold is not a number.", rule_needed
            dim, base = _base(number, operand["unit"])
            sides.append((base, dim))
            continue
        subtype, u, reason = _subject_check(operand["subject"], operand["value"], operand["unit"], inputs)
        used += u
        if subtype is not None:
            return subtype, used, f"{side}: {reason}", rule_needed
        true = _true_number(operand["subject"], inputs)
        if true is None:
            return 5, used, f"{side} is not a number that can be compared.", rule_needed
        sides.append(true)
    (left, l_dim), (right, r_dim) = sides
    if l_dim != r_dim:
        return 5, used, f"Compares {l_dim} with {r_dim}.", rule_needed
    if not _OPS[comparison["op"]](left, right):
        return 5, used, f"With the true values, {left:g} {comparison['op']} {right:g} is false.", rule_needed
    return None, used, "The comparison holds with the true values.", rule_needed


def _states_number(text, value):
    return re.search(rf"(?<![\d.]){re.escape(value)}(?![\d]|\.\d)", text) is not None


def check_threshold(claim, inputs):
    """(sub-type, supported passage ids, reason) for the rule part of a comparison, in code.

    The comparison's application to this patient is the arithmetic (check_comparison). What is left is whether each
    threshold it takes from a rule is stated in a passage: in a cited one (supported), only in a retrieved one
    (supported, so attribution marks the rule uncited), or in none (wrong rule)."""
    values = [op["value"] for op in (claim.comparison["left"], claim.comparison["right"]) if op.get("from_rule") and op.get("value")]
    cited = list(dict.fromkeys(s["id"] for s in claim.declared_sources if s["type"] == "passage"))
    missing = [pid for pid in cited if pid not in inputs.corpus]
    if missing:
        return 3, [], f"Cites passages that do not exist: {', '.join(missing)}."
    found = [pid for pid in cited if all(_states_number(inputs.corpus[pid]["text"], v) for v in values)]
    if found:
        return None, found, "The threshold is stated in a cited passage."
    if inputs.retriever is not None:
        found = [pid for pid, score in inputs.retriever(claim.text, TOP_K) if score >= inputs.tau
                 and all(_states_number(inputs.corpus[pid]["text"], v) for v in values)]
        if found:
            return None, found, "The threshold is stated in a passage the claim does not cite."
    return 3, [], f"No passage states the threshold {', '.join(values)}."


# --- rules ---------------------------------------------------------------------------------

JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "verdict", "scope_fits", "reason"],
        "properties": {
            "id": {"type": "string"},
            "verdict": {"type": "string", "enum": ["supported", "contradicted", "not_addressed"]},
            "scope_fits": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            "reason": {"type": "string"},
        },
    }}},
}


class Judge:
    """LLM judge for rule claims (§3c). One call per claim, with all its candidate passages."""

    def __init__(self, llm, version=JUDGE_PROMPT):
        self.llm = llm
        self.version = version
        self.system = (Path(PROMPTS_DIR) / f"{version}.md").read_text()

    def __call__(self, claim_text, passages, case_inputs, applied):
        user = json.dumps({"claim": claim_text, "applies_to_patient": applied, "case_inputs": case_inputs,
                           "passages": [{"id": p["id"], "text": p["text"]} for p in passages]}, indent=1)
        reply = self.llm(self.system, user, JUDGE_SCHEMA)
        wanted = {p["id"] for p in passages}
        return {v["id"]: (v["verdict"], v["scope_fits"]) for v in reply["verdicts"] if v["id"] in wanted}


def _case_inputs(inputs):
    vision = {k: v for k, v in inputs.vision.items() if k not in ("schema_version", "model_version", "case_id")}
    return {"vision_output": vision, "triage_note": inputs.note}


def check_rule(text, cited, applied, inputs):
    """(sub-type, supported passage ids, reason) for the rule part of a claim."""
    missing = [pid for pid in cited if pid not in inputs.corpus]
    if missing:
        return 3, [], f"Cites passages that do not exist: {', '.join(missing)}."
    candidates = list(dict.fromkeys(cited))
    if inputs.retriever is not None:
        candidates += [pid for pid, score in inputs.retriever(text, TOP_K) if score >= inputs.tau and pid not in candidates]
    if not candidates:
        return 3, [], "No passage is a candidate."
    if inputs.judge is None:
        raise ValueError("Rule claims need a judge.")
    verdicts = inputs.judge(text, [{"id": pid, **inputs.corpus[pid]} for pid in candidates], _case_inputs(inputs), applied)
    stated = [pid for pid, (verdict, _) in verdicts.items() if verdict == "supported"]
    if not stated:
        return 3, [], "No candidate passage states this rule."
    fits = [pid for pid in stated if not applied or verdicts[pid][1]]
    if not fits:
        return 4, stated, "A passage states the rule, but its scope does not fit this patient."
    return None, fits, "Supported by the corpus."


# --- attribution -----------------------------------------------------------------------------

def _norm(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def _overlaps(quote, span, note):
    note_n, quote_n, span_n = _norm(note), _norm(quote), _norm(span)
    q = note_n.find(quote_n)
    s = note_n.find(span_n)
    return q >= 0 and s >= 0 and q < s + len(span_n) and s < q + len(quote_n)


def _backs(source, trace, note):
    """True if a declared source supports part of this claim."""
    if source["type"] == "vision":
        return any(u["type"] == "vision" and u["field"] == source["field"] for u in trace.used)
    if source["type"] == "note":
        return any(u["type"] == "note" and _overlaps(source["quote"], u["span"], note) for u in trace.used)
    return source["id"] in trace.supported_passages


def attribute(traces, declared, note, own_sources=True, charge=True):
    """Sub-types 6 and 7 and the incomplete flag, for the claims of one entry, which share its sources.

    A claim needs a declared source for each part it rests on. A part left uncovered while the entry
    declares a source that supports nothing in it is mis-attributed (7): the wrong source was given.
    Otherwise an uncovered rule is uncited (6), and an uncovered vision or note part is incomplete,
    which still counts as clean. A declared source that supports nothing is charged to a claim even
    when every claim is covered.

    Blocking factors have no sources of their own and inherit their decision's rule_ids
    (own_sources False), so a passage they do not use is not held against them.
    """
    orphans = [s for s in declared if not any(_backs(s, t, note) for t in traces)] if own_sources and charge else []
    # A passage the entry cites but no claim uses is charged only to a claim that rests on a rule: a passage is never
    # the source of a patient or imaging fact, and an unextracted rule sentence often explains it (EXP-12).
    if not any(t._rule_part for t in traces):
        orphans = [s for s in orphans if s["type"] != "passage"]
    charged = set()
    for t in traces if charge else []:
        if t.subtype is not None:
            continue
        needs = {u["type"] for u in t.used} | ({"passage"} if t._rule_part else set())
        unmet = {kind for kind in needs if not any(s["type"] == kind and _backs(s, t, note) for s in declared)}
        if t.kind == "citation":
            unmet.discard("passage")
        mine = orphans if t._rule_part else [s for s in orphans if s["type"] != "passage"]
        if unmet and mine:
            t.subtype, t.reason = 7, "A declared source supports nothing in this entry, and this claim lacks a supporting one."
            charged.update(id(s) for s in orphans)
        elif "passage" in unmet:
            t.subtype, t.reason = 6, "The rule is in the corpus, but no passage the entry cites supports it."
        elif unmet:
            t.incomplete = True
    content_errors = any(t.subtype in CONTENT_FALSE for t in traces)
    for s in orphans:
        if id(s) in charged or content_errors:  # an unsupported source most likely belongs to the false claim
            continue
        clean = [t for t in traces if t.subtype is None and (s["type"] != "passage" or t._rule_part)]
        if clean:
            clean[0].subtype, clean[0].reason = 7, f"Declared source {s.get('field') or s.get('id') or 'note quote'} supports nothing in this entry."
            clean[0].incomplete = False
    for t in traces:
        t.traceable = t.subtype is None
        t.subtype_name = SUBTYPES.get(t.subtype)
        t.content_false = t.subtype in CONTENT_FALSE
        t.declared_ok = t.subtype != 7


# --- one claim -----------------------------------------------------------------------------

def _number_claim(claim, inputs):
    """A regex number no extracted claim covered: any input with that number and a compatible unit.

    A number with no unit only matches inputs without one, such as scores, so "6" cannot match 6.1 mmol/L.
    """
    def fits(true, true_unit):
        return (claim.unit is not None or true_unit is None) and number_matches(claim.value, claim.unit, float(true), true_unit)

    for subject in VISION_UNITS:
        true = _vision_value(inputs.vision, subject)
        if isinstance(true, (int, float)) and not isinstance(true, bool) and fits(true, VISION_UNITS[subject]):
            return None, [{"type": "vision", "field": subject}], [], f"Matches {subject}."
    for fact in inputs.facts:
        if fact.type == "number" and fits(fact.value, fact.unit):
            return None, [{"type": "note", "key": fact.key, "span": fact.span}], [], f"Matches the note's {fact.key}."
    cited = [s["id"] for s in claim.declared_sources if s["type"] == "passage" and s["id"] in inputs.corpus]
    pattern = re.compile(rf"(?<![\d.]){re.escape(claim.value)}(?![\d]|\.\d)")
    for pid in cited:
        if pattern.search(inputs.corpus[pid]["text"]):
            return None, [], [pid], f"A number stated in {pid}."
    if inputs.retriever is not None:
        for pid, score in inputs.retriever(claim.text, TOP_K):
            if score >= inputs.tau and pattern.search(inputs.corpus[pid]["text"]):
                return None, [], [pid], f"A number stated in {pid}, which the entry does not cite."
    return 1, [], [], "This number is in no input and no passage."


def trace_claim(claim, inputs):
    """Content check of one claim (sub-types 1-5). Attribution needs the whole entry: see trace_all."""
    subtype, used, supported, reason, rule_part = None, [], [], "", False
    cited = [s["id"] for s in claim.declared_sources if s["type"] == "passage"]
    if claim.kind == "number":
        subtype, used, supported, reason = _number_claim(claim, inputs)
    elif claim.kind in ("vision", "patient"):
        subtype, used, reason = _subject_check(claim.subject, claim.value, claim.unit, inputs, claim.text)
        op = _bound(claim.text, claim.value) if subtype in (None, 1) else None  # a bound is never an equality
        true = _true_number(claim.subject, inputs) if op else None
        number = _number_or_none(claim.value) if op else None
        if true is not None and number is not None:
            dim, base = _base(number, claim.unit or (VISION_UNITS.get(claim.subject) if claim.subject in VISION_SUBJECTS else None))
            if dim == true[1]:
                subtype, reason = (None, "The bound holds with the true value.") if _OPS[op](true[0], base) else (5, f"With the true value, {true[0]:g} {op} {base:g} is false.")
    elif claim.kind == "absence":
        subtype, used, reason = check_absence(claim.subject, inputs, claim.text)
    elif claim.kind == "comparison":
        subtype, used, reason, rule_part = check_comparison(claim.comparison, inputs)
    elif claim.kind == "citation" and claim.value in inputs.corpus and "treatment" in inputs.corpus[claim.value]:
        # The metric's definition (§1): the passage exists and is a rule about that treatment. The rulebook records
        # each passage's treatment, so no judge is needed.
        treatment = inputs.corpus[claim.value]["treatment"]
        if treatment in (claim.subject, "both"):
            supported, reason = [claim.value], f"{claim.value} is a rule about {claim.subject}."
        else:
            subtype, reason = 3, f"{claim.value} is about {treatment}, not {claim.subject}."
    elif claim.kind in ("rule", "citation"):
        rule_part = True
    else:
        raise ValueError(f"Unknown claim kind {claim.kind!r}.")

    if rule_part and (subtype is None or subtype == 5):
        # Rule content comes before the derivation in the sub-type order. A comparison's application to this patient
        # is the arithmetic, checked in code above; its rule part is only whether the threshold comes from a passage.
        if claim.kind == "comparison":
            rule_subtype, supported, rule_reason = check_threshold(claim, inputs)
        else:
            rule_subtype, supported, rule_reason = check_rule(claim.text, cited, claim.applies_to_patient, inputs)
        if rule_subtype is not None:
            subtype, reason = rule_subtype, rule_reason
        elif subtype is None:
            reason = rule_reason
    substitution = None
    if subtype in (1, 2) and claim.kind in ("vision", "patient") and claim.subject and not claim.subject.startswith("other:"):
        substitution = {"subject": claim.subject, "value": claim.value, "unit": claim.unit}
    trace = Trace(claim_id=claim.id, field=claim.field, kind=claim.kind, traceable=subtype is None, subtype=subtype,
                  subtype_name=SUBTYPES.get(subtype), content_false=subtype in CONTENT_FALSE, reason=reason, used=used,
                  supported_passages=supported, substitution=substitution)
    trace._rule_part = rule_part
    return trace


def trace_all(claims, inputs):
    """Trace every claim, then settle attribution entry by entry. Returns traces in claim order."""
    traces = [trace_claim(claim, inputs) for claim in claims]
    groups = {}
    for claim, trace in zip(claims, traces):
        groups.setdefault((claim.field, claim.entry), (claim.declared_sources, []))[1].append(trace)
    for (field_name, _), (declared, group) in groups.items():
        # Uncertainties have no sources in the schema, so only their content is scored there (§9).
        attribute(group, declared, inputs.note, own_sources=not field_name.endswith(".blocking_factors"),
                  charge=field_name != "uncertainties")
    return traces
