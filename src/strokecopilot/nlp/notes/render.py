"""Triage note text from structured facts, with a fact sheet whose spans are verbatim in the note. Plan §8.3.

Used by scenarios.py now and by notes/generate.py for the ISLES'24 development cases. A few phrasings per field give
controlled variation, so the reasoner does not see one fixed format. A fact left as None is not written: the note is
silent on it, which is not the same as "not recorded" (use `not_recorded` for that).
"""

from strokecopilot.nlp.notes.facts import Fact, FactSheet

MRS_WORDS = {0: "no symptoms", 1: "no significant disability", 2: "slight disability", 3: "moderate disability",
             4: "moderately severe disability", 5: "severe disability"}


def _hm(minutes):
    minutes = int(round(minutes))
    return f"{minutes // 60} h {minutes % 60} min"


def _clock(minutes):
    minutes = int(round(minutes)) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class _Writer:
    def __init__(self, rng):
        self.rng, self.lines, self.facts = rng, [], []

    def pick(self, options):
        return options[self.rng.randrange(len(options))]

    def fact(self, key, type_, value, span, unit=None):
        self.facts.append(Fact(key=key, type=type_, value=value, span=span, unit=unit))


def render(spec, rng):
    """(note text, FactSheet) for a spec dict. Keys (all optional except age, sex, nihss):
    age, sex ("male"/"female"), onset ("witnessed"/"unwitnessed"/"wake_up"), minutes (to imaging), imaging_clock
    (minutes after midnight), sleep_minutes (wake-up only: time asleep before waking), deficits, referral, nihss, mrs,
    medications (list), anticoagulant (None/False/"apixaban"/"warfarin"...), inr, antiplatelet (bool), history (list),
    head_trauma, recent_surgery, previous_ich, recent_bleeding (True/False/None), bp (sys, dia), glucose (mmol/L),
    not_recorded (list of keys the note says are not recorded)."""
    w = _Writer(rng)
    noun = {"male": "man", "female": "woman"}[spec["sex"]]
    age_span = f"{spec['age']}-year-old"
    w.fact("age", "number", spec["age"], age_span, "years")
    w.fact("sex", "text", spec["sex"], noun)
    deficits = spec.get("deficits", "right-sided weakness and slurred speech")
    onset, minutes = spec.get("onset", "witnessed"), spec.get("minutes")
    imaging = spec.get("imaging_clock", 14 * 60)
    first = f"{age_span} {noun}."
    if minutes is None:
        first += f" Found with {deficits}; time of onset not known."
        w.fact("deficits", "text", deficits, deficits)
    elif onset == "witnessed":
        clock = _clock(imaging - minutes)
        span = w.pick([f"Witnessed sudden onset of {deficits} at {clock}", f"Sudden {deficits} at {clock}, witnessed by family"])
        first += f" {span}."
        w.fact("onset_witnessed", "bool", True, span)
        w.fact("onset_clock", "time", clock, clock)
        w.fact("deficits", "text", deficits, deficits)
    elif onset == "unwitnessed":
        lkw = _clock(imaging - minutes)
        found = _clock(imaging - min(minutes, 60) + 15)
        first += f" Last seen well at {lkw}; found at {found} with {deficits}. Onset not witnessed."
        w.fact("lkw_clock", "time", lkw, f"Last seen well at {lkw}")
        w.fact("found_clock", "time", found, f"found at {found}")
        w.fact("onset_witnessed", "bool", False, "Onset not witnessed")
        w.fact("deficits", "text", deficits, deficits)
    else:  # wake-up
        lkw_minutes = imaging - minutes
        woke = lkw_minutes + spec.get("sleep_minutes", 480)
        lkw, found = _clock(lkw_minutes), _clock(woke)
        first += f" Went to bed well at {lkw}; woke at {found} with {deficits}. Wake-up stroke; onset not witnessed."
        w.fact("lkw_clock", "time", lkw, f"Went to bed well at {lkw}")
        w.fact("found_clock", "time", found, f"woke at {found}")
        w.fact("wake_up_stroke", "bool", True, "Wake-up stroke")
        w.fact("onset_witnessed", "bool", False, "onset not witnessed")
        w.fact("deficits", "text", deficits, deficits)
    if spec.get("referral"):
        referral = spec["referral"][0].upper() + spec["referral"][1:]
        first += f" {referral}."
        w.fact("referral", "text", spec["referral"], referral)
    w.lines.append(first)

    if minutes is not None:
        basis, key = ("symptom onset", "onset_to_imaging") if onset == "witnessed" else ("last known well", "lkw_to_imaging")
        span = f"Time from {basis} to imaging: {_hm(minutes)}."
        w.lines.append(span)
        w.fact(key, "number", int(round(minutes)), span, "min")

    span = w.pick([f"NIHSS on arrival: {spec['nihss']}.", f"Initial NIHSS {spec['nihss']}.", f"NIHSS {spec['nihss']} on arrival."])
    w.lines.append(span)
    w.fact("nihss", "number", spec["nihss"], span)
    if spec.get("mrs") is not None:
        span = f"Pre-stroke mRS: {spec['mrs']} ({MRS_WORDS[spec['mrs']]})."
        w.lines.append(span)
        w.fact("premorbid_mrs", "number", spec["mrs"], span)

    meds = list(spec.get("medications", []))
    anticoag = spec.get("anticoagulant")
    if anticoag:
        meds.append(anticoag if anticoag != "apixaban" else "apixaban 5 mg twice daily, last dose this morning")
    if spec.get("antiplatelet"):
        meds.append("aspirin 75 mg daily")
    med_line = f"Medications: {', '.join(meds)}." if meds else "Medications: none."
    if meds:
        w.fact("medications", "text", ", ".join(meds), med_line)
    extra = []
    if anticoag is False and spec.get("antiplatelet") is False:
        extra.append("No anticoagulant or antiplatelet use.")
        w.fact("anticoagulant", "bool", False, extra[-1])
        w.fact("antiplatelet", "bool", False, extra[-1])
    elif anticoag is False:
        extra.append("No anticoagulant use.")
        w.fact("anticoagulant", "bool", False, extra[-1])
    elif anticoag:
        w.fact("anticoagulant", "bool", True, anticoag if anticoag != "apixaban" else "apixaban")
    if spec.get("antiplatelet"):
        w.fact("antiplatelet", "bool", True, "aspirin")
    w.lines.append(" ".join([med_line] + extra))

    history = list(spec.get("history", []))
    hist = f"History: {', '.join(history)}." if history else "History: nil of note."
    for item in history:
        key = {"hypertension": "hypertension", "type 2 diabetes": "diabetes", "atrial fibrillation": None,
               "hyperlipidaemia": "hyperlipidaemia", "ischaemic heart disease": "ischaemic_heart_disease",
               "chronic kidney disease stage 3": "chronic_kidney_disease"}.get(item)
        if key:
            w.fact(key, "bool", True, item)
    positives = {"head_trauma": "Fell and struck his head with brief loss of consciousness 5 days ago",
                 "recent_surgery": "Hip replacement 6 days ago", "previous_ich": "Previous intracerebral haemorrhage in 2019",
                 "recent_bleeding": "Upper gastrointestinal bleed 10 days ago"}
    parts = [hist]
    negatives = [k for k in ("recent_surgery", "head_trauma", "previous_ich") if spec.get(k) is False]
    for key, text in positives.items():
        if spec.get(key):
            text = text if spec["sex"] == "male" or key != "head_trauma" else text.replace("his", "her")
            parts.append(text + ".")
            w.fact(key, "bool", True, text)
    if negatives:
        words = {"recent_surgery": "recent surgery", "head_trauma": "head trauma", "previous_ich": "previous intracranial haemorrhage"}
        listed = [words[k] for k in negatives]
        sentence = "No " + (listed[0] if len(listed) == 1 else ", ".join(listed[:-1]) + " or " + listed[-1]) + "."
        parts.append(sentence)
        for key in negatives:
            w.fact(key, "bool", False, sentence)
    w.lines.append(" ".join(parts))

    obs = []
    if spec.get("bp"):
        s, d = spec["bp"]
        obs.append(f"BP {s}/{d} mmHg")
        w.fact("bp_systolic", "number", s, f"BP {s}/{d} mmHg", "mmHg")
        w.fact("bp_diastolic", "number", d, f"BP {s}/{d} mmHg", "mmHg")
    if spec.get("glucose") is not None:
        obs.append(f"capillary glucose {spec['glucose']} mmol/L")
        w.fact("glucose", "number", spec["glucose"], obs[-1], "mmol/L")
    if spec.get("inr") is not None:
        obs.append(f"INR {spec['inr']}")
        w.fact("inr", "number", spec["inr"], obs[-1])
    if obs:
        w.lines.append("Observations: " + ", ".join(obs) + ".")
    for key in spec.get("not_recorded", []):
        span = f"{key.replace('_', ' ').capitalize()}: not recorded."
        w.lines.append(span)
        w.fact(key, "not_recorded", None, span)

    note = "ED STROKE TRIAGE NOTE\n" + "\n".join(w.lines)
    sheet = FactSheet(w.facts)
    sheet.check_against(note)
    return note, sheet
