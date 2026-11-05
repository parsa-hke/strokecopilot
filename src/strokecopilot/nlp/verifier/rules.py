"""The rulebook in code: inputs -> each decision as yes, no or undetermined, with the passages and inputs that decided
it. docs/HALLUCINATION_METRIC.md §5; plan §8.3 (the rule summary) and corpus/RULEBOOK.md (the passages).

Used three ways:
- expected recommendations for the synthetic scenarios (scenarios.py);
- omission checks: an input that must change a decision and was ignored;
- criticality: substitute a hallucinated value into the inputs and decide again.

Conventions, all ours and listed for the clinician review (docs/CLINICIAN_REVIEW.md):
- Time windows count from symptom onset, or from last known well when onset was not witnessed. "Within 6 h" includes
  6 h 0 min; times are at imaging, because the inputs give no treatment time.
- COR 1 and 2a are "yes", COR 2b is "undetermined" (the guideline leaves it to judgement), COR 3 or no applicable
  recommendation is "no".
- A deficit with NIHSS 6 or more is taken as disabling; NIHSS under 6 within 4.5 h is undetermined, because whether it
  is disabling is a clinical judgement the inputs cannot settle.
- Table 8 absolute contraindications give "no"; relative ones give "undetermined".
- A recommendation that needs an input the system cannot observe (significant mass effect, M2 dominance, PC-ASPECTS)
  is undetermined, and that input is reported.
- With ASPECTS unknown, 6 to 24 h, the trial criteria (DAWN, DEFUSE-3) are the route to "yes"; if neither is met the
  decision is undetermined, because ASPECTS could still meet the guideline recommendation.
- Salvageable penumbra for extended-window thrombolysis uses Table 3: EXTEND for 4.5 to 9 h, TRACE-III for 4.5 to
  24 h. With a core under 1 mL the mismatch ratio is null but its threshold is taken as met.
"""

from dataclasses import dataclass, field

YES, NO, UNDETERMINED = "yes", "no", "undetermined"
LVO = ("ICA", "M1")
WINDOW_IVT = 270  # 4.5 h, minutes
EXTENDED_IVT = 540  # 9 h
EVT_EARLY = 360  # 6 h
EVT_LATE = 1440  # 24 h
DEFUSE3_LATE = 960  # 16 h


@dataclass
class Verdict:
    value: str
    rule_ids: list = field(default_factory=list)
    deciding: list = field(default_factory=list)  # input subjects the outcome rests on
    unknown: list = field(default_factory=list)  # unknown or unobservable inputs that leave it undetermined
    reason: str = ""

    def to_dict(self):
        return {"value": self.value, "rule_ids": self.rule_ids, "deciding": self.deciding, "unknown": self.unknown,
                "reason": self.reason}


@dataclass
class Case:
    """Everything the rules read, in base units: minutes, mL, mmol/L. None means unknown."""
    haemorrhage: bool | None = None
    core: float | None = None
    penumbra: float | None = None
    ratio: float | None = None
    mismatch: float | None = None
    site: str | None = None
    aspects: int | None = None
    age: float | None = None
    nihss: float | None = None
    mrs: float | None = None
    minutes: float | None = None  # from onset, or from last known well when onset was not witnessed
    time_basis: str | None = None  # "onset_to_imaging" or "lkw_to_imaging"
    wake_up: bool = False
    sleep_mid_minutes: float | None = None  # from the midpoint of sleep to imaging, for a wake-up stroke
    inr: float | None = None
    anticoagulant: bool | None = None
    head_trauma: bool | None = None
    recent_surgery: bool | None = None
    previous_ich: bool | None = None
    recent_bleeding: bool | None = None


def _clock_minutes(text):
    h, m = str(text).split(":")
    return int(h) * 60 + int(m)


def case_from(vision, facts):
    """A Case from a vision_output.json dict and a FactSheet."""
    def fact(key):
        f = facts.get(key)
        return None if f is None or f.type == "not_recorded" else f.value

    witnessed = fact("onset_witnessed")
    onset, lkw = fact("onset_to_imaging"), fact("lkw_to_imaging")
    if onset is not None and witnessed is not False:
        minutes, basis = float(onset), "onset_to_imaging"
    elif lkw is not None:
        minutes, basis = float(lkw), "lkw_to_imaging"
    else:
        minutes, basis = None, None
    wake = bool(fact("wake_up_stroke"))
    sleep_mid = None
    if wake and lkw is not None and fact("lkw_clock") and fact("found_clock"):
        asleep = (_clock_minutes(fact("found_clock")) - _clock_minutes(fact("lkw_clock"))) % 1440
        sleep_mid = float(lkw) - asleep / 2
    h = vision.get("haemorrhage") or {}
    num = lambda k: None if fact(k) is None else float(fact(k))
    return Case(haemorrhage=h.get("present"), core=vision.get("core_ml"), penumbra=vision.get("penumbra_ml"),
                ratio=vision.get("mismatch_ratio"), mismatch=vision.get("mismatch_volume_ml"),
                site=vision.get("occlusion_site"), aspects=vision.get("aspects"), age=num("age"), nihss=num("nihss"),
                mrs=num("premorbid_mrs"), minutes=minutes, time_basis=basis, wake_up=wake, sleep_mid_minutes=sleep_mid,
                inr=num("inr"), anticoagulant=fact("anticoagulant"), head_trauma=fact("head_trauma"),
                recent_surgery=fact("recent_surgery"), previous_ich=fact("previous_ich"),
                recent_bleeding=fact("recent_bleeding"))


# --- thrombectomy ---------------------------------------------------------------------------------

def _dawn(c):
    """The DAWN clinical-core mismatch group met, or None."""
    if None in (c.age, c.nihss, c.core):
        return None
    if c.age >= 80 and c.nihss >= 10 and c.core < 21:
        return "age 80 or over, NIHSS 10 or more, core under 21 mL"
    if c.age < 80 and c.nihss >= 10 and c.core < 31:
        return "under 80, NIHSS 10 or more, core under 31 mL"
    if c.age < 80 and c.nihss >= 20 and 31 <= c.core < 51:
        return "under 80, NIHSS 20 or more, core 31 to under 51 mL"
    return None


def _defuse3(c):
    if None in (c.age, c.nihss, c.mrs, c.core, c.mismatch) or c.minutes > DEFUSE3_LATE:
        return False
    # Ratio above 1.8 and mismatch above 15 mL, strictly: checked against the DEFUSE-3 paper by Parsa, 5 Nov 2026
    ratio_ok = c.ratio > 1.8 if c.ratio is not None else c.core < 1
    return 18 <= c.age <= 90 and c.nihss >= 6 and c.mrs <= 2 and c.core < 70 and ratio_ok and c.mismatch > 15


def thrombectomy(c):
    base = ["occlusion_site", c.time_basis or "time", "nihss", "premorbid_mrs"]
    if c.haemorrhage:
        return Verdict(NO, [], ["haemorrhage.present"], reason="Haemorrhage on CT: the ischaemic thrombectomy recommendations do not apply.")
    if c.site in (None, "none", "unknown"):
        return Verdict(NO, [], ["occlusion_site"], reason="No occlusion is identified, so no thrombectomy recommendation applies.")
    if c.minutes is None:
        return Verdict(UNDETERMINED, [], [], ["onset_time"], "The time from onset or last known well is unknown.")
    if None in (c.nihss, c.mrs):
        return Verdict(UNDETERMINED, [], [], [k for k, v in (("nihss", c.nihss), ("premorbid_mrs", c.mrs)) if v is None],
                       "NIHSS or pre-stroke mRS is unknown.")
    t = c.minutes

    if c.site == "basilar":
        if t <= EVT_LATE and c.mrs <= 1 and c.nihss >= 6:
            ids = ["AHA2026-4.7.3-basilar"] if c.nihss >= 10 else ["AHA2026-4.7.3-basilar-nihss-6-9"]
            return Verdict(UNDETERMINED, ids, base, ["pc_aspects"], "Basilar occlusion: PC-ASPECTS cannot be observed.")
        return Verdict(NO, ["AHA2026-4.7.3-basilar"], base, reason="Outside the basilar criteria.")

    if c.site == "M2":
        if t <= EVT_EARLY and c.mrs <= 1 and c.nihss >= 6 and (c.aspects is None or c.aspects >= 6):
            unknown = ["m2_dominance"] + (["aspects"] if c.aspects is None else [])
            return Verdict(UNDETERMINED, ["AHA2026-4.7.2-m2-dominant", "AHA2026-4.7.2-distal-no-benefit"], base + ["aspects"],
                           unknown, "M2 occlusion: dominance cannot be observed (dominant: COR 2a; otherwise COR 3).")
        return Verdict(NO, ["AHA2026-4.7.2-distal-no-benefit", "AHA2026-4.7.2-m2-dominant"], base + ["aspects"],
                       reason="M2 occlusion outside the dominant-M2 criteria.")

    # ICA or M1
    if c.nihss < 6 or c.mrs > 4 or t > EVT_LATE:
        return Verdict(NO, ["AHA2026-4.7.2-0-6h-aspects-3-10", "AHA2026-4.7.2-6-24h-aspects-6-10"], base,
                       reason="Outside the ICA/M1 criteria (NIHSS under 6, pre-stroke mRS over 4, or beyond 24 h).")
    a = c.aspects
    if t <= EVT_EARLY:
        if a is None:
            return Verdict(UNDETERMINED, ["AHA2026-4.7.2-0-6h-aspects-3-10"], base, ["aspects"], "Within 6 h, ASPECTS is unknown.")
        if 3 <= a <= 10 and c.mrs <= 1:
            return Verdict(YES, ["AHA2026-4.7.2-0-6h-aspects-3-10"], base + ["aspects"], reason="0-6 h, ASPECTS 3-10, mRS 0-1: COR 1.")
        if a >= 6 and c.mrs == 2:
            return Verdict(YES, ["AHA2026-4.7.2-0-6h-mrs-2"], base + ["aspects"], reason="0-6 h, ASPECTS 6 or more, mRS 2: COR 2a.")
        if a >= 6 and c.mrs in (3, 4):
            return Verdict(UNDETERMINED, ["AHA2026-4.7.2-0-6h-mrs-3-4"], base + ["aspects"], reason="0-6 h, mRS 3-4: COR 2b.")
        if a <= 2 and c.mrs <= 1 and c.age is not None and c.age < 80:
            return Verdict(UNDETERMINED, ["AHA2026-4.7.2-0-6h-aspects-0-2"], base + ["aspects", "age"], ["mass_effect"],
                           "0-6 h, ASPECTS 0-2: COR 2a only without significant mass effect, which cannot be observed.")
        return Verdict(NO, ["AHA2026-4.7.2-0-6h-aspects-3-10"], base + ["aspects"], reason="0-6 h, outside the ASPECTS and mRS criteria.")
    # 6 to 24 h
    if a is not None:
        if a >= 6 and c.mrs <= 1:
            return Verdict(YES, ["AHA2026-4.7.2-6-24h-aspects-6-10"], base + ["aspects"], reason="6-24 h, ASPECTS 6 or more, mRS 0-1: COR 1.")
        if 3 <= a <= 5 and c.mrs <= 1 and c.age is not None and c.age < 80:
            return Verdict(UNDETERMINED, ["AHA2026-4.7.2-6-24h-aspects-3-5"], base + ["aspects", "age"], ["mass_effect"],
                           "6-24 h, ASPECTS 3-5: COR 1 only without significant mass effect, which cannot be observed.")
        return Verdict(NO, ["AHA2026-4.7.2-6-24h-aspects-6-10", "AHA2026-4.7.2-6-24h-aspects-3-5"], base + ["aspects", "age"],
                       reason="6-24 h, outside the ASPECTS, mRS and age criteria.")
    dawn = _dawn(c) if c.mrs <= 1 else None
    if dawn:
        return Verdict(YES, ["DAWN-inclusion-mismatch-groups", "DAWN-inclusion-clinical", "AHA2026-4.7.2-late-window-trials"],
                       base + ["age", "core_ml"], reason=f"ASPECTS unknown; DAWN group met ({dawn}).")
    if _defuse3(c):
        return Verdict(YES, ["DEFUSE3-inclusion-imaging", "DEFUSE3-inclusion-clinical", "AHA2026-4.7.2-late-window-trials"],
                       base + ["age", "core_ml", "mismatch_ratio", "mismatch_volume_ml"], reason="ASPECTS unknown; DEFUSE-3 criteria met.")
    return Verdict(UNDETERMINED, ["AHA2026-4.7.2-6-24h-aspects-6-10", "DAWN-inclusion-mismatch-groups", "DEFUSE3-inclusion-imaging"],
                   base + ["age", "core_ml", "mismatch_ratio", "mismatch_volume_ml"], ["aspects"],
                   "ASPECTS unknown and neither DAWN nor DEFUSE-3 met: ASPECTS could still meet the guideline recommendation.")


# --- thrombolysis ---------------------------------------------------------------------------------

def _extend(c):
    if c.core is None or c.mismatch is None:
        return False
    ratio_ok = c.ratio > 1.2 if c.ratio is not None else c.core < 1
    return ratio_ok and c.mismatch > 10 and c.core < 70


def _trace3(c):
    if c.core is None or c.penumbra is None:
        return False
    ratio_ok = c.ratio > 1.8 if c.ratio is not None else c.core < 1
    return ratio_ok and c.penumbra > 15 and c.core < 70


def thrombolysis(c, evt=None):
    if c.haemorrhage:
        return Verdict(NO, ["AHA2026-T8-ich"], ["haemorrhage.present"], reason="Acute intracranial haemorrhage: absolute contraindication.")
    if c.inr is not None and c.inr > 1.7:
        return Verdict(NO, ["AHA2026-T8-coagulopathy"], ["inr"], reason="INR above 1.7: absolute contraindication.")
    if c.head_trauma:
        return Verdict(NO, ["AHA2026-T8-trauma-surgery"], ["head_trauma"], reason="Recent head injury: absolute contraindication.")
    relative = [k for k in ("previous_ich", "recent_surgery", "recent_bleeding", "anticoagulant") if getattr(c, k)]
    if relative:
        ids = ["AHA2026-T8-doac"] if relative == ["anticoagulant"] else ["AHA2026-T8-relative"]
        return Verdict(UNDETERMINED, ids, relative, reason=f"Relative contraindication ({', '.join(relative)}): individual judgement.")
    if c.minutes is None:
        return Verdict(UNDETERMINED, [], [], ["onset_time"], "The time from onset or last known well is unknown.")
    t = c.minutes
    if t <= WINDOW_IVT:
        if c.nihss is None:
            return Verdict(UNDETERMINED, ["AHA2026-4.6.1-within-4.5h"], [c.time_basis], ["nihss"], "Within 4.5 h; NIHSS unknown.")
        if c.nihss >= 6:
            return Verdict(YES, ["AHA2026-4.6.1-within-4.5h"], [c.time_basis, "nihss", "haemorrhage.present"],
                           reason="Within 4.5 h with a disabling deficit: COR 1.")
        return Verdict(UNDETERMINED, ["AHA2026-4.6.1-within-4.5h", "AHA2026-4.6.1-mild-nondisabling"], [c.time_basis, "nihss"],
                       ["disabling"], "Within 4.5 h, NIHSS under 6: depends on whether the deficit is disabling.")
    evt = evt or thrombectomy(c)
    if evt.value == YES:
        return Verdict(NO, ["AHA2026-4.6.3-wake-up-perfusion", "AHA2026-4.6.3-lvo-4.5-24h", "AHA2026-4.6.1-within-4.5h"],
                       [c.time_basis] + evt.deciding,
                       reason="Beyond 4.5 h and eligible for thrombectomy: the extended-window recommendations do not apply.")
    if evt.value == UNDETERMINED:
        return Verdict(UNDETERMINED, ["AHA2026-4.6.3-wake-up-perfusion", "AHA2026-4.6.3-lvo-4.5-24h"], [c.time_basis] + evt.deciding,
                       evt.unknown, "Beyond 4.5 h: depends on thrombectomy eligibility, which is undetermined.")
    in_extended = t <= EXTENDED_IVT or (c.wake_up and c.sleep_mid_minutes is not None and c.sleep_mid_minutes <= EXTENDED_IVT)
    perfusion = [c.time_basis, "core_ml", "mismatch_ratio", "mismatch_volume_ml"]
    if in_extended and _extend(c):
        return Verdict(YES, ["AHA2026-4.6.3-wake-up-perfusion", "AHA2026-T3-extend"], perfusion,
                       reason="Extended window, not eligible for thrombectomy, salvageable penumbra (EXTEND): COR 2a.")
    if c.site in LVO + ("basilar",) and t <= EVT_LATE and _trace3(c):
        return Verdict(UNDETERMINED, ["AHA2026-4.6.3-lvo-4.5-24h", "AHA2026-T3-trace3"], perfusion + ["penumbra_ml", "occlusion_site"],
                       reason="LVO 4.5-24 h without thrombectomy, TRACE-III penumbra: COR 2b.")
    return Verdict(NO, ["AHA2026-4.6.1-within-4.5h", "AHA2026-4.6.3-wake-up-perfusion"], perfusion,
                   reason="Beyond 4.5 h with no extended-window route met.")


def decide(c):
    """{"thrombolysis": Verdict, "thrombectomy": Verdict} for a Case."""
    evt = thrombectomy(c)
    return {"thrombolysis": thrombolysis(c, evt), "thrombectomy": evt}


# --- omissions (docs/HALLUCINATION_METRIC.md §5) ---------------------------------------------------

# How a reasoner names an unknown input in `uncertainties`.
UNKNOWN_WORDS = {
    "aspects": ("aspects",), "mass_effect": ("mass effect",), "m2_dominance": ("dominan",),
    "pc_aspects": ("pc-aspects", "pc aspects", "posterior circulation aspects"), "onset_time": ("onset", "last known well"),
    "nihss": ("nihss",), "premorbid_mrs": ("mrs", "rankin"), "disabling": ("disabling",),
}


def omissions(rec, c, verdicts):
    """(omissions, uncertainty omissions): each a list of short descriptions. An omission is an input that a hard
    rule says must change a decision, ignored by the recommendation; an uncertainty omission is an unknown input that
    leaves a decision undetermined and is not named in `uncertainties`."""
    ivt, evt = rec["thrombolysis"]["recommend"], rec["thrombectomy"]["recommend"]
    out = []
    if c.haemorrhage and ivt:
        out.append("haemorrhage present, thrombolysis recommended")
    if c.haemorrhage and evt:
        out.append("haemorrhage present, thrombectomy recommended")
    if ivt and c.inr is not None and c.inr > 1.7:
        out.append("INR above 1.7, thrombolysis recommended")
    if ivt and c.head_trauma:
        out.append("recent head injury, thrombolysis recommended")
    v = verdicts["thrombolysis"]
    if ivt and c.minutes is not None and c.minutes > WINDOW_IVT and v.value == NO and not c.haemorrhage:
        out.append("beyond 4.5 h with no extended-window route met, thrombolysis recommended")
    ids = rec["thrombectomy"]["rule_ids"]
    if evt and c.core is not None and c.core >= 70 and ids and all(i.startswith("DEFUSE3") for i in ids):
        out.append("core 70 mL or more, thrombectomy recommended on DEFUSE-3 alone")
    if evt and c.site in ("none", "unknown") and not c.haemorrhage:
        out.append("no occlusion identified, thrombectomy recommended")
    named = " ".join(rec.get("uncertainties", [])).lower()
    missing = []
    for decision, verdict in verdicts.items():
        if verdict.value != UNDETERMINED:
            continue
        for unknown in verdict.unknown:
            words = UNKNOWN_WORDS.get(unknown, (unknown.replace("_", " "),))
            if not any(w in named for w in words):
                missing.append(f"{decision}: {unknown} not named in uncertainties")
    return out, sorted(set(missing))


# --- criticality (docs/HALLUCINATION_METRIC.md §5) -------------------------------------------------

def substitute(vision, facts, subject, value, unit):
    """Copies of the inputs with `subject` set to the claimed value, in the input's own unit."""
    import copy

    from strokecopilot.nlp.notes.facts import Fact, FactSheet
    from strokecopilot.nlp.verifier import trace as tr

    vision = copy.deepcopy(vision)
    if subject in tr.VISION_UNITS or subject in ("occlusion_site", "haemorrhage.present", "aspects"):
        if subject == "haemorrhage.present":
            vision["haemorrhage"]["present"] = bool(tr._bool(value))
        elif subject == "occlusion_site":
            vision["occlusion_site"] = str(value)
        elif "." in subject:
            outer, inner = subject.split(".")
            vision[outer][inner] = tr._number_or_none(value)
        else:
            number = tr._number_or_none(value)
            if number is not None and unit is not None and tr.VISION_UNITS.get(subject):
                number = number * tr._unit(unit)[1] / tr._unit(tr.VISION_UNITS[subject])[1]
            vision[subject] = int(number) if subject == "aspects" and number is not None else number
        return vision, facts
    old = facts.get(subject)
    kept = [f for f in facts if f.key != subject]
    number = tr._number_or_none(value)
    if number is not None:
        if old is not None and old.unit and unit:
            number = number * tr._unit(unit)[1] / tr._unit(old.unit)[1]
        new = Fact(key=subject, type="number", value=number, span="(substituted)", unit=old.unit if old else unit)
    elif tr._bool(value) is not None:
        new = Fact(key=subject, type="bool", value=tr._bool(value), span="(substituted)")
    else:
        new = Fact(key=subject, type="text", value=str(value), span="(substituted)")
    return vision, FactSheet(kept + [new])


def criticality(trace, claim, vision, facts, verdicts=None):
    """True, False or None (undetermined) for one claim's trace, per §5 and §6.

    The critical rate's denominator is every claim whose criticality is determined (§6), so a clean claim is False
    (not critical) whenever the rulebook decides at least one decision for the case, and None otherwise."""
    subtype = trace["subtype"]
    verdicts = verdicts or decide(case_from(vision, facts))
    determined = {d: v for d, v in verdicts.items() if v.value != UNDETERMINED}
    if not determined:
        return None
    if subtype is None or subtype in (6, 7):
        return False
    if subtype in (1, 2):
        sub = trace.get("substitution")
        if not sub:
            return None
        try:
            new_vision, new_facts = substitute(vision, facts, sub["subject"], sub["value"], sub.get("unit"))
            after = decide(case_from(new_vision, new_facts))
        except (TypeError, ValueError, KeyError, AttributeError):
            return None
        return any(after[d].value != v.value for d, v in determined.items())
    # 3, 4, 5: critical if the claim concerns a criterion the rulebook lists as deciding for this case
    deciding = {s for v in determined.values() for s in v.deciding}
    passages = {i for v in determined.values() for i in v.rule_ids}
    subjects = {claim.get("subject")}
    for side in ("left", "right"):
        operand = (claim.get("comparison") or {}).get(side) or {}
        subjects.add(operand.get("subject"))
    cited = {s.get("id") for s in claim.get("declared_sources", []) if s.get("type") == "passage"}
    return bool(subjects & deciding) or bool(cited & passages)


def annotate(rec, claims, traces, vision, facts):
    """Fill each trace's `critical` and return the case's (omission list, uncertainty omission list, verdicts)."""
    c = case_from(vision, facts)
    verdicts = decide(c)
    by_id = {cl["id"] if isinstance(cl, dict) else cl.id: (cl if isinstance(cl, dict) else cl.to_dict()) for cl in claims}
    for t in traces:
        t["critical"] = criticality(t, by_id.get(t["claim_id"], {}), vision, facts, verdicts)
    found, missing = omissions(rec, c, verdicts)
    return found, missing, verdicts
