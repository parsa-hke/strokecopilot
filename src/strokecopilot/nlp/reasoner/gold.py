"""Gold answers for fine-tuning a small reasoner (Item 4; DEC-30, pending). A scenario's inputs and the rulebook's
verdicts (rules.decide) -> the reply prompt v3 asks for: both decisions, their rule_ids, blocking factors, a sourced
rationale and the uncertainties.

How each part is built:
- recommend is True only for a "yes" verdict. "Undetermined" (COR 2b, a relative contraindication, an unobservable
  input) is False, as prompt v3 says ("recommend a treatment only when a passage supports it"), and the reason goes
  in `uncertainties`.
- rule_ids are the verdict's passages about that decision's own treatment (prompt v3, EXP-39).
- The rationale states each condition of the deciding passages as one comparison of an input with the passage's
  threshold ("NIHSS 16 is 6 or more."), citing the input (an imaging field or the note's own span) and the passage,
  derived true. A condition with an unknown input is not stated; it goes to `uncertainties`. Facts with no threshold
  (a haemorrhage, a contraindication, no occlusion) are stated plainly. A sentence used twice is written once, with
  the sources of both uses.
- blocking_factors are the failed conditions and contraindications behind a decision against a treatment.
- uncertainties name each unknown input of a verdict with the passage ID in square brackets, as prompt v3 asks, plus
  a COR 2b or relative-contraindication route that the rulebook leaves to judgement.

Numbers are copied as the inputs write them: imaging values as in vision_output.json (12.0 mL), times as the note
writes them (6 h 0 min). check() runs the code checks every gold answer must pass before training.
"""

import json
from dataclasses import dataclass

from strokecopilot.contract.validate import validate_recommendation
from strokecopilot.nlp.corpus import ingest
from strokecopilot.nlp.verifier import rules

LVO = rules.LVO
LVO_OR_BASILAR = LVO + ("basilar",)


def hm(minutes):
    """240 -> '4 h 0 min', the note's own format."""
    minutes = int(round(minutes))
    return f"{minutes // 60} h {minutes % 60} min"


def _num(value):
    """A number as vision_output.json writes it (12.0 stays 12.0)."""
    return json.dumps(value)


@dataclass(frozen=True)
class Condition:
    """One condition of a passage: the input it reads, a test, and the two ways to state the comparison."""
    input: str
    test: object  # value -> bool
    met: str
    failed: str


def _c(key, test, met, failed):
    return Condition(key, test, met, failed)


W45 = _c("time", lambda t: t <= rules.WINDOW_IVT, "is within 4.5 hours", "is beyond 4.5 hours")
W45_9 = _c("time", lambda t: rules.WINDOW_IVT < t <= rules.EXTENDED_IVT, "is between 4.5 and 9 hours", "is beyond 9 hours")
W45_24 = _c("time", lambda t: rules.WINDOW_IVT < t <= rules.EVT_LATE, "is between 4.5 and 24 hours", "is beyond 24 hours")
W6 = _c("time", lambda t: t <= rules.EVT_EARLY, "is within 6 hours", "is beyond 6 hours")
W6_24 = _c("time", lambda t: rules.EVT_EARLY < t <= rules.EVT_LATE, "is between 6 and 24 hours", "is beyond 24 hours")
W6_16 = _c("time", lambda t: rules.EVT_EARLY < t <= rules.DEFUSE3_LATE, "is between 6 and 16 hours", "is beyond 16 hours")
W24 = _c("time", lambda t: t <= rules.EVT_LATE, "is within 24 hours", "is beyond 24 hours")
SITE_LVO = _c("site", lambda s: s in LVO, "is an ICA or M1 occlusion", "is not an ICA or M1 occlusion")
SITE_LVO_B = _c("site", lambda s: s in LVO_OR_BASILAR, "is a large vessel occlusion", "is not a large vessel occlusion")
SITE_M2 = _c("site", lambda s: s == "M2", "is an M2 occlusion", "is not an M2 occlusion")
SITE_BASILAR = _c("site", lambda s: s == "basilar", "is a basilar occlusion", "is not a basilar occlusion")
NIHSS6 = _c("nihss", lambda n: n >= 6, "is 6 or more", "is below 6")
NIHSS10 = _c("nihss", lambda n: n >= 10, "is 10 or more", "is below 10")
NIHSS6_9 = _c("nihss", lambda n: 6 <= n <= 9, "is between 6 and 9", "is not between 6 and 9")
NIHSS20 = _c("nihss", lambda n: n >= 20, "is 20 or more", "is below 20")
MRS01 = _c("mrs", lambda m: m <= 1, "is 0 or 1", "is above 1")
MRS2 = _c("mrs", lambda m: m == 2, "is 2", "is not 2")
MRS34 = _c("mrs", lambda m: m in (3, 4), "is 3 or 4", "is not 3 or 4")
MRS02 = _c("mrs", lambda m: m <= 2, "is 2 or less", "is above 2")
ASPECTS3_10 = _c("aspects", lambda a: 3 <= a <= 10, "is between 3 and 10", "is below 3")
ASPECTS6 = _c("aspects", lambda a: a >= 6, "is 6 or more", "is below 6")
ASPECTS3_5 = _c("aspects", lambda a: 3 <= a <= 5, "is between 3 and 5", "is outside 3 to 5")
ASPECTS0_2 = _c("aspects", lambda a: a <= 2, "is between 0 and 2", "is above 2")
AGE80 = _c("age", lambda a: a < 80, "is under 80", "is 80 or over")
AGE80_OVER = _c("age", lambda a: a >= 80, "is 80 or over", "is under 80")
AGE18_90 = _c("age", lambda a: 18 <= a <= 90, "is between 18 and 90", "is outside 18 to 90")
CORE70 = _c("core", lambda v: v < 70, "is below 70 mL", "is 70 mL or more")
CORE21 = _c("core", lambda v: v < 21, "is under 21 mL", "is 21 mL or more")
CORE31 = _c("core", lambda v: v < 31, "is under 31 mL", "is 31 mL or more")
CORE31_51 = _c("core", lambda v: 31 <= v < 51, "is from 31 mL up to under 51 mL", "is outside 31 to under 51 mL")
RATIO12 = _c("ratio", lambda r: r > 1.2, "is above 1.2", "is not above 1.2")
RATIO18 = _c("ratio", lambda r: r > 1.8, "is above 1.8", "is not above 1.8")
MISMATCH10 = _c("mismatch", lambda v: v > 10, "is above 10 mL", "is not above 10 mL")
MISMATCH15 = _c("mismatch", lambda v: v > 15, "is above 15 mL", "is not above 15 mL")
PENUMBRA15 = _c("penumbra", lambda v: v > 15, "is above 15 mL", "is not above 15 mL")

# The conditions each passage sets, in the passage's order. Only passages a verdict can rest on are listed.
CONDITIONS = {
    "AHA2026-4.6.1-within-4.5h": (W45,),
    "AHA2026-4.6.3-wake-up-perfusion": (W45_9,),
    "AHA2026-4.6.3-lvo-4.5-24h": (SITE_LVO_B, W45_24),
    "AHA2026-T3-extend": (RATIO12, MISMATCH10, CORE70),
    "AHA2026-T3-trace3": (RATIO18, PENUMBRA15, CORE70),
    "AHA2026-4.7.2-0-6h-aspects-3-10": (SITE_LVO, W6, NIHSS6, MRS01, ASPECTS3_10),
    "AHA2026-4.7.2-0-6h-mrs-2": (SITE_LVO, W6, NIHSS6, ASPECTS6, MRS2),
    "AHA2026-4.7.2-0-6h-mrs-3-4": (SITE_LVO, W6, NIHSS6, ASPECTS6, MRS34),
    "AHA2026-4.7.2-0-6h-aspects-0-2": (SITE_LVO, W6, ASPECTS0_2, AGE80, NIHSS6, MRS01),
    "AHA2026-4.7.2-6-24h-aspects-6-10": (SITE_LVO, W6_24, NIHSS6, MRS01, ASPECTS6),
    "AHA2026-4.7.2-6-24h-aspects-3-5": (SITE_LVO, W6_24, ASPECTS3_5, AGE80, NIHSS6, MRS01),
    "AHA2026-4.7.2-m2-dominant": (SITE_M2, W6, MRS01, NIHSS6, ASPECTS6),
    "AHA2026-4.7.3-basilar": (SITE_BASILAR, W24, MRS01, NIHSS10),
    "AHA2026-4.7.3-basilar-nihss-6-9": (SITE_BASILAR, W24, NIHSS6_9, MRS01),
    "DAWN-inclusion-clinical": (SITE_LVO, MRS01, W6_24),
    "DEFUSE3-inclusion-clinical": (AGE18_90, SITE_LVO, NIHSS6, MRS02, W6_16),
    "DEFUSE3-inclusion-imaging": (CORE70, RATIO18, MISMATCH15),
}
DAWN_GROUPS = {  # rules._dawn's text -> the group's conditions in DAWN-inclusion-mismatch-groups
    "age 80 or over, NIHSS 10 or more, core under 21 mL": (AGE80_OVER, NIHSS10, CORE21),
    "under 80, NIHSS 10 or more, core under 31 mL": (AGE80, NIHSS10, CORE31),
    "under 80, NIHSS 20 or more, core 31 to under 51 mL": (AGE80, NIHSS20, CORE31_51),
}
DAWN = "DAWN-inclusion-mismatch-groups"


class _Answer:
    """Collects rationale entries (one per sentence, sources merged), blocking factors and uncertainties."""

    def __init__(self, vision, facts, c):
        self.vision, self.facts, self.c = vision, facts, c
        self.entries, self.order = {}, []
        self.blocking = {"thrombolysis": [], "thrombectomy": []}
        self.uncertainties = []

    # --- inputs as sources ---
    def span(self, key):
        fact = self.facts.get(key)
        return None if fact is None or fact.type == "not_recorded" else fact.span

    def note(self, key):
        span = self.span(key)
        return [{"type": "note", "quote": span}] if span else []

    @staticmethod
    def vis(field):
        return [{"type": "vision", "field": field}]

    def value(self, key):
        """(text for the claim, sources) of an input, or None when it is unknown."""
        c, v = self.c, self.vision
        if key == "time":
            if c.minutes is None:
                return None
            label = "Time from symptom onset to imaging" if c.time_basis == "onset_to_imaging" else "Time from last known well to imaging"
            return f"{label} {hm(c.minutes)}", self.note(c.time_basis)
        if key == "sleep_mid":
            if c.sleep_mid_minutes is None:
                return None
            return (f"Time from the midpoint of sleep to imaging, {hm(c.sleep_mid_minutes)},",
                    self.note("lkw_clock") + self.note("found_clock") + self.note("lkw_to_imaging"))
        simple = {"site": ("The occlusion site", "occlusion_site", None), "aspects": ("ASPECTS", "aspects", None),
                  "core": ("Core volume", "core_ml", "mL"), "ratio": ("Mismatch ratio", "mismatch_ratio", None),
                  "mismatch": ("Mismatch volume", "mismatch_volume_ml", "mL"),
                  "penumbra": ("Hypoperfused volume", "penumbra_ml", "mL")}
        if key in simple:
            label, field, unit = simple[key]
            raw = v.get(field)
            if raw is None:
                return None
            text = raw if isinstance(raw, str) else _num(raw)
            return f"{label} {text}{' ' + unit if unit else ''}", self.vis(field)
        notes = {"nihss": ("NIHSS", "nihss"), "mrs": ("Pre-stroke mRS", "premorbid_mrs"), "age": ("Age", "age")}
        if key in notes:
            label, fact_key = notes[key]
            fact = self.facts.get(fact_key)
            if fact is None or fact.type == "not_recorded":
                return None
            return f"{label} {_num(fact.value)}", self.note(fact_key)
        raise KeyError(key)

    def case_value(self, key):
        c = self.c
        return {"time": c.minutes, "site": c.site, "aspects": c.aspects, "core": c.core, "ratio": c.ratio,
                "mismatch": c.mismatch, "penumbra": c.penumbra, "nihss": c.nihss, "mrs": c.mrs, "age": c.age,
                "sleep_mid": c.sleep_mid_minutes}[key]

    # --- writing ---
    def add(self, text, sources, derived):
        if text in self.entries:
            for s in sources:
                if s not in self.entries[text]["sources"]:
                    self.entries[text]["sources"].append(s)
            return text
        self.entries[text] = {"claim": text, "sources": list(sources), "derived": derived}
        self.order.append(text)
        return text

    def condition(self, cond, passage):
        """The comparison of one condition with its input, added to the rationale: (sentence, met) or None if the
        input is unknown."""
        got = self.value(cond.input)
        value = self.case_value(cond.input)
        if got is None or value is None:
            return None
        label, sources = got
        met = bool(cond.test(value))
        if cond.input == "site":  # "The occlusion site is M1." reads better than "The occlusion site M1 is ..."
            text = f"The occlusion site is {value}." if met else f"The occlusion site is {value}, {cond.failed[3:]}."
        else:
            text = f"{label} {cond.met if met else cond.failed}."
        return self.add(text, sources + [{"type": "passage", "id": passage}], True), met

    def conditions(self, passage, only=None, conds=None):
        """Every known condition of a passage (or `conds`), stated; returns [(sentence, met)]. only="met" or
        "failed" keeps those alone."""
        out = []
        for cond in conds or CONDITIONS[passage]:
            if cond.input == "ratio" and self.c.ratio is None:
                continue  # null under a 1 mL core; rules.py takes its threshold as met, and it is not stated
            got = self.condition_check(cond)
            if got is None or (only == "met" and not got) or (only == "failed" and got):
                continue
            out.append(self.condition(cond, passage))
        return out

    def condition_check(self, cond):
        value = self.case_value(cond.input)
        return None if value is None else bool(cond.test(value))

    def block(self, decision, text):
        if text not in self.blocking[decision]:
            self.blocking[decision].append(text)

    def uncertain(self, text):
        if text not in self.uncertainties:
            self.uncertainties.append(text)


# --- uncertainties for unknown inputs ---------------------------------------------------------------------------

def _first(ids, wanted):
    for i in ids:
        if i in wanted:
            return i
    return ids[0] if ids else None


def _unknowns(a, verdict):
    ids = verdict.rule_ids
    for unknown in verdict.unknown:
        if unknown == "aspects":
            pid = _first(ids, [p for p, conds in CONDITIONS.items() if any(c.input == "aspects" for c in conds)])
            a.uncertain(f"ASPECTS is not reported; it is needed for [{pid}].")
        elif unknown == "mass_effect":
            pid = _first(ids, ["AHA2026-4.7.2-6-24h-aspects-3-5", "AHA2026-4.7.2-0-6h-aspects-0-2"])
            a.uncertain(f"Significant mass effect cannot be observed; it is a condition of [{pid}].")
        elif unknown == "m2_dominance":
            a.uncertain("M2 dominance cannot be observed; it decides between [AHA2026-4.7.2-m2-dominant] and "
                        "[AHA2026-4.7.2-distal-no-benefit].")
        elif unknown == "pc_aspects":
            a.uncertain(f"PC-ASPECTS cannot be observed; it is needed for [{_first(ids, ['AHA2026-4.7.3-basilar'])}].")
        elif unknown == "disabling":
            a.uncertain("Whether the deficit is disabling is not stated; it decides between [AHA2026-4.6.1-within-4.5h] "
                        "and [AHA2026-4.6.1-mild-nondisabling].")
        elif unknown == "onset_time":
            a.uncertain("The time of symptom onset or last known well is not reported; it is needed for "
                        "[AHA2026-4.6.1-within-4.5h].")
        elif unknown == "nihss":
            a.uncertain(f"NIHSS is not reported; it is needed for [{ids[0] if ids else 'AHA2026-4.6.1-within-4.5h'}].")
        elif unknown == "premorbid_mrs":
            a.uncertain(f"Pre-stroke mRS is not reported; it is needed for [{ids[0] if ids else 'AHA2026-4.7.2-0-6h-aspects-3-10'}].")


# --- thrombectomy ------------------------------------------------------------------------------------------------

def _thrombectomy(a, v):
    c, d = a.c, "thrombectomy"
    reason = v.reason
    if reason.startswith("Haemorrhage on CT"):
        a.block(d, a.add("Haemorrhage is present on CT.", a.vis("haemorrhage.present"), False))
        return
    if reason.startswith("No occlusion is identified"):
        if c.site == "unknown":
            a.block(d, a.add("The occlusion site is unknown.", a.vis("occlusion_site"), False))
            if "CTA not acquired" in (a.vision.get("warnings") or []):
                a.add("CTA was not acquired.", a.vis("warnings"), False)
        else:
            a.block(d, a.add("No occlusion is identified.", a.vis("occlusion_site"), False))
        return
    if v.value == rules.UNDETERMINED and not v.rule_ids:  # time, NIHSS or mRS unknown
        _unknowns(a, v)
        return
    if c.site == "basilar":
        pid = v.rule_ids[0]
        if v.value == rules.UNDETERMINED:
            a.conditions(pid)
            _unknowns(a, v)
            if pid.endswith("nihss-6-9"):
                a.uncertain(f"The rulebook gives only a COR 2b recommendation for NIHSS 6 to 9 [{pid}]; it leaves the "
                            "decision to judgement.")
        else:
            for text, met in a.conditions(pid, only="failed"):
                a.block(d, text)
        return
    if c.site == "M2":
        if v.value == rules.UNDETERMINED:
            a.conditions("AHA2026-4.7.2-m2-dominant", only="met")
            _unknowns(a, v)
        else:
            for text, _ in a.conditions("AHA2026-4.7.2-m2-dominant", only="failed"):
                a.block(d, text)
        return
    # ICA or M1
    if reason.startswith("Outside the ICA/M1 criteria"):
        early = c.minutes is not None and c.minutes <= rules.EVT_EARLY
        pid = "AHA2026-4.7.2-0-6h-aspects-3-10" if early else "AHA2026-4.7.2-6-24h-aspects-6-10"
        conds = [x for x in CONDITIONS[pid] if x.input in ("nihss", "mrs", "time")]
        if not early and c.minutes is not None and c.minutes > rules.EVT_LATE:
            conds = [W6_24] + [x for x in conds if x.input != "time"]
        failed = [t for t, met in a.conditions(pid, conds=conds) if not met]
        if c.mrs is not None and c.mrs > 4:
            failed.append(a.add(f"Pre-stroke mRS {_num(a.facts.get('premorbid_mrs').value)} is above 4.",
                                a.note("premorbid_mrs") + [{"type": "passage", "id": "AHA2026-4.7.2-0-6h-mrs-3-4"}], True))
        for text in failed:
            a.block(d, text)
        return
    if reason.startswith("Within 6 h, ASPECTS is unknown"):
        a.conditions("AHA2026-4.7.2-0-6h-aspects-3-10", only="met")
        _unknowns(a, v)
        return
    if v.value == rules.YES and v.rule_ids[0].startswith("AHA2026"):
        a.conditions(v.rule_ids[0])
        return
    if reason.startswith("0-6 h, mRS 3-4"):
        pid = "AHA2026-4.7.2-0-6h-mrs-3-4"
        a.conditions(pid)
        a.uncertain(f"The rulebook gives only a COR 2b recommendation for pre-stroke mRS 3 or 4 [{pid}]; it leaves "
                    "the decision to judgement.")
        return
    if reason.startswith(("0-6 h, ASPECTS 0-2", "6-24 h, ASPECTS 3-5")):
        a.conditions(v.rule_ids[0])
        _unknowns(a, v)
        return
    if reason.startswith("0-6 h, outside"):
        for text, _ in a.conditions("AHA2026-4.7.2-0-6h-aspects-3-10", only="failed"):
            a.block(d, text)
        return
    if reason.startswith("6-24 h, outside"):
        for pid in ("AHA2026-4.7.2-6-24h-aspects-6-10", "AHA2026-4.7.2-6-24h-aspects-3-5"):
            for text, _ in a.conditions(pid, only="failed"):
                a.block(d, text)
        return
    if reason.startswith("ASPECTS unknown; DAWN group met"):
        group = reason.split("(", 1)[1].rstrip(").")
        a.conditions(DAWN, conds=DAWN_GROUPS[group])
        a.conditions("DAWN-inclusion-clinical")
        return
    if reason.startswith("ASPECTS unknown; DEFUSE-3"):
        a.conditions("DEFUSE3-inclusion-clinical")
        a.conditions("DEFUSE3-inclusion-imaging")
        return
    if reason.startswith("ASPECTS unknown and neither"):
        _dawn_missed(a)
        a.conditions("DEFUSE3-inclusion-clinical", only="failed")
        a.conditions("DEFUSE3-inclusion-imaging", only="failed")
        _unknowns(a, v)
        return
    raise ValueError(f"No gold template for the thrombectomy route {reason!r}.")


def _dawn_missed(a):
    """For a case outside every DAWN group: the condition that misses in the group its age falls in."""
    c = a.c
    if c.age is None or c.nihss is None or c.core is None:
        return
    groups = [g for g in DAWN_GROUPS.values() if a.condition_check(g[0])]
    for conds in groups:
        failed = [x for x in conds[1:] if a.condition_check(x) is False]
        if failed:
            a.condition(conds[0], DAWN)  # the age group the case falls in, so the miss reads in context
        for cond in failed:
            a.condition(cond, DAWN)


# --- thrombolysis ------------------------------------------------------------------------------------------------

def _not_eligible_for_thrombectomy(a, evt):
    """The step prompt v3 allows: the patient's thrombectomy status, when it decides extended-window thrombolysis.
    It cites only the passages whose conditions decide (EXP-40: a general passage such as late-window-trials or
    distal-no-benefit was charged as a source that supports nothing). With no such passage (no occlusion, a
    haemorrhage) nothing is added: the fact stated for thrombectomy already says it. Returns the sentence or None."""
    ids = [{"type": "passage", "id": i} for i in evt.rule_ids if i in CONDITIONS or i == DAWN]
    if not ids:
        return None
    if evt.value == rules.YES:
        return a.add("The patient meets the criteria for thrombectomy.", ids, True)
    return a.add("The patient does not meet the criteria for thrombectomy.", ids, True)


def _thrombolysis(a, v, evt):
    c, d = a.c, "thrombolysis"
    reason = v.reason
    if reason.startswith("Acute intracranial haemorrhage"):
        a.block(d, a.add("Haemorrhage is present on CT, an absolute contraindication.",
                         a.vis("haemorrhage.present") + [{"type": "passage", "id": "AHA2026-T8-ich"}], True))
        return
    if reason.startswith("INR above 1.7"):
        a.block(d, a.add(f"INR {_num(a.facts.get('inr').value)} is above 1.7, an absolute contraindication.",
                         a.note("inr") + [{"type": "passage", "id": "AHA2026-T8-coagulopathy"}], True))
        return
    if reason.startswith("Recent head injury"):
        a.block(d, a.add("A recent head injury is an absolute contraindication.",
                         a.note("head_trauma") + [{"type": "passage", "id": "AHA2026-T8-trauma-surgery"}], True))
        return
    if reason.startswith("Relative contraindication"):
        words = {"previous_ich": "A previous intracranial haemorrhage", "recent_surgery": "Recent surgery",
                 "recent_bleeding": "Recent bleeding", "anticoagulant": "Direct oral anticoagulant use"}
        pid = v.rule_ids[0]
        for key in v.deciding:
            a.block(d, a.add(f"{words[key]} is a relative contraindication.",
                             a.note(key) + [{"type": "passage", "id": pid}], True))
        a.uncertain(f"Whether benefit outweighs risk is not settled by the inputs; the relative contraindication calls "
                    f"for an individual judgement [{pid}].")
        return
    if c.minutes is None:
        _unknowns(a, v)
        return
    if c.minutes <= rules.WINDOW_IVT:
        a.conditions("AHA2026-4.6.1-within-4.5h")
        if c.nihss is not None:
            a.add(f"NIHSS on arrival is {_num(a.facts.get('nihss').value)}.", a.note("nihss"), False)
        if v.value == rules.YES:
            a.add("No haemorrhage is detected on CT.", a.vis("haemorrhage.present"), False)
        _unknowns(a, v)
        return
    # beyond 4.5 h
    beyond = a.condition(W45, "AHA2026-4.6.1-within-4.5h")[0]
    if evt.value == rules.YES:
        a.block(d, beyond)
        a.block(d, _not_eligible_for_thrombectomy(a, evt))  # a "yes" always rests on a passage with conditions
        a.add("The extended-window thrombolysis recommendations apply only to patients who cannot have thrombectomy.",
              [{"type": "passage", "id": "AHA2026-4.6.3-wake-up-perfusion"},
               {"type": "passage", "id": "AHA2026-4.6.3-lvo-4.5-24h"}], False)
        return
    if evt.value == rules.UNDETERMINED:
        a.block(d, beyond)
        a.uncertain("Thrombectomy eligibility is undetermined; extended-window thrombolysis "
                    "[AHA2026-4.6.3-wake-up-perfusion] [AHA2026-4.6.3-lvo-4.5-24h] applies only to patients who cannot "
                    "have thrombectomy.")
        return
    _not_eligible_for_thrombectomy(a, evt)
    wake = "AHA2026-4.6.3-wake-up-perfusion"
    if reason.startswith("Extended window"):
        if c.minutes <= rules.EXTENDED_IVT:
            a.condition(W45_9, wake)
        else:
            a.condition(_c("sleep_mid", lambda t: t <= rules.EXTENDED_IVT, "is within 9 hours", "is beyond 9 hours"), wake)
        a.conditions("AHA2026-T3-extend")
        return
    if reason.startswith("LVO 4.5-24 h"):
        pid = "AHA2026-4.6.3-lvo-4.5-24h"
        a.conditions(pid)
        a.conditions("AHA2026-T3-trace3")
        a.uncertain(f"The rulebook gives only a COR 2b recommendation for thrombolysis in a large vessel occlusion "
                    f"seen 4.5 to 24 hours after onset [{pid}]; it leaves the decision to judgement.")
        return
    if reason.startswith("Beyond 4.5 h with no extended-window route"):
        a.block(d, beyond)
        in_window = c.minutes <= rules.EXTENDED_IVT or (c.wake_up and c.sleep_mid_minutes is not None
                                                         and c.sleep_mid_minutes <= rules.EXTENDED_IVT)
        if not in_window:
            # The time since last known well is enough here; a computed sleep-midpoint time is not stated, because
            # the verifier cannot trace a number derived from two clock times to the note (EXP-40).
            a.block(d, a.condition(W45_9, wake)[0])
        else:
            for text, _ in a.conditions("AHA2026-T3-extend", only="failed"):
                a.block(d, text)
        lvo = "AHA2026-4.6.3-lvo-4.5-24h"
        failed = a.conditions(lvo, only="failed")
        for text, _ in failed:
            a.block(d, text)
        if not failed:
            for text, _ in a.conditions("AHA2026-T3-trace3", only="failed"):
                a.block(d, text)
        return
    raise ValueError(f"No gold template for the thrombolysis route {reason!r}.")


# --- the answer --------------------------------------------------------------------------------------------------

def _own(ids, treatment, corpus):
    return [i for i in ids if corpus[i].get("treatment") in (treatment, "both")]


def gold_answer(vision, facts, corpus=None):
    """The reply prompt v3 asks for (no run fields): {thrombolysis, thrombectomy, rationale, uncertainties}."""
    corpus = corpus or ingest.load()
    c = rules.case_from(vision, facts)
    verdicts = rules.decide(c)
    a = _Answer(vision, facts, c)
    _thrombectomy(a, verdicts["thrombectomy"])
    _thrombolysis(a, verdicts["thrombolysis"], verdicts["thrombectomy"])
    if verdicts["thrombolysis"].value == rules.UNDETERMINED and verdicts["thrombectomy"].value != rules.UNDETERMINED:
        _unknowns(a, verdicts["thrombolysis"])
    out = {}
    for decision in ("thrombolysis", "thrombectomy"):
        v = verdicts[decision]
        recommend = v.value == rules.YES
        blocking = [] if recommend else a.blocking[decision]
        # The verifier checks blocking factors against the decisions' rule_ids only, so the passages they rest on are
        # listed too, when they are about this treatment (EXP-40: EXTEND, TRACE-III and the 4.5-24 h passage).
        ids = list(v.rule_ids)
        for text in blocking:
            ids += [s["id"] for s in a.entries.get(text, {}).get("sources", []) if s["type"] == "passage"]
        out[decision] = {"recommend": recommend, "rule_ids": _own(list(dict.fromkeys(ids)), decision, corpus),
                         "blocking_factors": blocking}
    out["rationale"] = [a.entries[t] for t in a.order]
    out["uncertainties"] = a.uncertainties
    return out


def gold_record(case_id, reply, model="gold", prompt_version="reasoner_v3"):
    return {"case_id": case_id, "model": model, "prompt_version": prompt_version, **json.loads(json.dumps(reply))}


def check(vision, facts, note, reply, corpus=None):
    """The code checks a gold answer must pass (Item 4, step 2): [] when it passes, else the problems.

    - it validates against schemas/recommendation.json;
    - each decision matches the verdict (yes -> True, otherwise False);
    - every rule_id names a passage about its own decision's treatment, and every passage source exists;
    - every note quote is verbatim in the note, and every vision source names a known field;
    - rules.omissions finds no omission and no unnamed unknown;
    - a decision against a treatment has blocking factors unless its verdict is undetermined."""
    corpus = corpus or ingest.load()
    problems = []
    rec = gold_record(vision["case_id"], reply)
    try:
        validate_recommendation(rec)
    except Exception as e:  # ContractError
        problems.append(f"schema: {str(e)[:200]}")
    c = rules.case_from(vision, facts)
    verdicts = rules.decide(c)
    for decision, v in verdicts.items():
        if reply[decision]["recommend"] != (v.value == rules.YES):
            problems.append(f"{decision}: recommend does not match the verdict {v.value}")
        for i in reply[decision]["rule_ids"]:
            if i not in corpus:
                problems.append(f"{decision}: rule_id {i} is not in the rulebook")
            elif corpus[i].get("treatment") not in (decision, "both"):
                problems.append(f"{decision}: rule_id {i} is about {corpus[i].get('treatment')}")
        if v.value == rules.NO and not reply[decision]["blocking_factors"]:
            problems.append(f"{decision}: a decision against with no blocking factor")
    for entry in reply["rationale"]:
        for s in entry["sources"]:
            if s["type"] == "note" and s["quote"] not in note:
                problems.append(f"note quote not verbatim: {s['quote'][:60]}")
            if s["type"] == "passage" and s["id"] not in corpus:
                problems.append(f"passage source {s['id']} is not in the rulebook")
    found, missing = rules.omissions(reply, c, verdicts)
    problems += [f"omission: {x}" for x in found] + [f"uncertainty omission: {x}" for x in missing]
    return problems
