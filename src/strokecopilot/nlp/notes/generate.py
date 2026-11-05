"""ISLES'24 clinical row -> ED triage note and fact sheet (plan §8.3, Phase 2).

Only columns tagged "allowed" in data/clinical_columns.py are read; an untagged column raises, and the forbidden ones
(door to groin, TICI, post-admission NIHSS and mRS...) are dropped before anything else sees the row. Every fact in the
note comes from one allowed column, so the verifier can trace patient-fact claims:

- times are durations, as the data has them; no clock time is invented. Onset to imaging = onset to door + door to
  imaging. A wake-up stroke has no last-known-well time in the data, so the note says so and gives the time from
  symptoms being found (alert to door + door to imaging);
- a missing rule input (onset time, NIHSS, pre-stroke mRS, anticoagulation, INR) is written as "not recorded", so the
  reasoner treats it as unknown (docs/DATA.md: INR is missing in 61 rows, pre-stroke mRS in 60, onset in about 61);
- glucose is converted from mg/dL to mmol/L (one decimal), the unit the fixtures and the verifier use;
- phrasing varies a little per case, from a seed and the case ID, so a note is the same on every run.

Development cases only before Phase 4 (splits.check_ids refuses held-out IDs). Logs print counts, never case IDs.
"""

import json
import os
import random
import re

from strokecopilot.data import clinical_columns, splits
from strokecopilot.nlp.notes.facts import Fact, FactSheet

SEED = 20261105
EXPECTED_DEVELOPMENT = 129
SLIM_SHA256 = "b0afd16e4cbfb16a69fb5190083fc3869f5dcc6c92f86f10135b0de1f59f4728"
GLUCOSE_MG_PER_MMOL = 18.016
MRS_WORDS = {0: "no symptoms", 1: "no significant disability", 2: "slight disability", 3: "moderate disability",
             4: "moderately severe disability", 5: "severe disability"}
HISTORY = (("Hypertension", "hypertension", "hypertension"), ("Atrial fibrillation", "atrial_fibrillation", "atrial fibrillation"),
           ("Diabetes", "diabetes", "diabetes"), ("Hyperlipidemia", "hyperlipidaemia", "hyperlipidaemia"))
_TIME = re.compile(r"^\s*(?:(\d+)\s+days?,?\s*)?(\d+):(\d{1,2})(?::(\d{1,2})(?:\.\d+)?)?\s*$")


def allowed_fields(row):
    """The row with forbidden columns removed. Raises on a column with no tag, so nothing new leaks by default."""
    return {k: v for k, v in row.items() if k == "case" or clinical_columns.tag_of(k) == "allowed"}


def _missing(value):
    return value is None or (isinstance(value, float) and value != value) or str(value).strip() in ("", "nan", "NaN", "None")


def number(value):
    return None if _missing(value) else float(value)


def flag(value, unknown=(99,)):
    """0/1 column -> False/True; missing or an unknown code -> None."""
    v = number(value)
    if v is None or v in unknown:
        return None
    if v not in (0, 1):
        raise ValueError(f"Expected 0 or 1, got {v}.")
    return v == 1


def minutes(value):
    """Duration text ("hh:mm:ss", with or without microseconds, or "N days, hh:mm:ss") -> minutes, or None."""
    if _missing(value):
        return None
    m = _TIME.match(str(value))
    if not m:
        raise ValueError(f"Unreadable duration {value!r}.")
    days, h, mi, s = (int(g) if g else 0 for g in m.groups())
    return days * 1440 + h * 60 + mi + s / 60


def _hm(total):
    total = int(round(total))
    return f"{total // 60} h {total % 60:02d} min"


def spec_from_row(row):
    """The note's content from a clinical row: plain values only, every one from an allowed column."""
    r = allowed_fields(row)
    sex = {"M": "male", "F": "female"}.get(str(r.get("Sex")).strip()) if not _missing(r.get("Sex")) else None
    onset_door, alert_door, door_img = minutes(r.get("Onset to door")), minutes(r.get("Alert to door")), minutes(r.get("Door to imaging"))
    glucose = number(r.get("Glucose"))
    return {
        "age": None if number(r.get("Age")) is None else int(number(r.get("Age"))),
        "sex": sex,
        "wake_up": flag(r.get("Wake-up")), "in_house": flag(r.get("In-House")), "referral": flag(r.get("Referral")),
        "onset_to_door": onset_door, "alert_to_door": alert_door, "door_to_imaging": door_img,
        "nihss": None if number(r.get("NIHSS at admission")) is None else int(number(r.get("NIHSS at admission"))),
        "mrs": None if number(r.get("mRS premorbid")) is None else int(number(r.get("mRS premorbid"))),
        "anticoagulant": flag(r.get("Anticoagulation")), "antiplatelet": flag(r.get("PAIs")),
        "statin": flag(r.get("Lipid lowering drugs")),
        "history": {key: flag(r.get(col)) for col, key, _ in HISTORY},
        "glucose_mmol": None if glucose is None else round(glucose / GLUCOSE_MG_PER_MMOL, 1),
        "inr": None if number(r.get("INR")) is None else round(number(r.get("INR")), 2),
    }


class _Note:
    def __init__(self, rng):
        self.rng, self.lines, self.facts = rng, [], []

    def pick(self, options):
        return options[self.rng.randrange(len(options))]

    def fact(self, key, type_, value, span, unit=None):
        self.facts.append(Fact(key=key, type=type_, value=value, span=span, unit=unit))

    def not_recorded(self, key, label):
        span = f"{label}: not recorded."
        self.lines.append(span)
        self.fact(key, "not_recorded", None, span)


def render_case(spec, rng):
    """(note text, FactSheet) for a spec from spec_from_row."""
    n = _Note(rng)
    if spec["age"] is None or spec["sex"] is None:
        who = "Adult patient" if spec["age"] is None else f"{spec['age']}-year-old patient"
    else:
        who = f"{spec['age']}-year-old {'man' if spec['sex'] == 'male' else 'woman'}"
    first = who + n.pick([" admitted with suspected acute ischaemic stroke.", " presenting with suspected acute stroke."])
    n.lines.append(first)
    if spec["age"] is not None:
        n.fact("age", "number", spec["age"], f"{spec['age']}-year-old", "years")
    if spec["sex"] is not None and spec["age"] is not None:
        n.fact("sex", "text", spec["sex"], "man" if spec["sex"] == "male" else "woman")

    if spec["in_house"]:
        span = "In-hospital stroke: symptoms began while in hospital"
        n.lines.append(span + ".")
        n.fact("referral", "text", "in-hospital stroke", span)
    elif spec["referral"]:
        span = n.pick(["Transferred from an external hospital", "Referred from another clinic"])
        n.lines.append(span + ".")
        n.fact("referral", "text", "transfer from an external clinic", span)
    elif spec["referral"] is False:
        span = "Came directly to this hospital"
        n.lines.append(span + ".")
        n.fact("referral", "text", "direct admission", span)

    door = "arrival" if not spec["in_house"] else "neurologist assessment"
    if spec["wake_up"]:
        span = "Woke with symptoms (wake-up stroke); onset not witnessed"
        n.lines.append(span + ".")
        n.fact("wake_up_stroke", "bool", True, "wake-up stroke")
        n.fact("onset_witnessed", "bool", False, "onset not witnessed")
        n.not_recorded("lkw_to_imaging", "Time last known well")
        if spec["alert_to_door"] is not None and spec["door_to_imaging"] is not None:
            span = f"Time from symptoms being found to imaging: {_hm(spec['alert_to_door'] + spec['door_to_imaging'])}."
            n.lines.append(span)
            n.fact("found_to_imaging", "number", int(round(spec["alert_to_door"] + spec["door_to_imaging"])), span, "min")
        elif spec["alert_to_door"] is not None:
            span = f"Time from symptoms being found to {door}: {_hm(spec['alert_to_door'])}."
            n.lines.append(span)
            n.fact("found_to_door", "number", int(round(spec["alert_to_door"])), span, "min")
    elif spec["onset_to_door"] is not None and spec["door_to_imaging"] is not None:
        total = spec["onset_to_door"] + spec["door_to_imaging"]
        span = n.pick([f"Time from symptom onset to imaging: {_hm(total)}.", f"Symptom onset {_hm(total)} before imaging."])
        n.lines.append(span)
        n.fact("onset_to_imaging", "number", int(round(total)), span, "min")
    elif spec["onset_to_door"] is not None:
        span = f"Time from symptom onset to {door}: {_hm(spec['onset_to_door'])}."
        n.lines.append(span)
        n.fact("onset_to_door", "number", int(round(spec["onset_to_door"])), span, "min")
        n.not_recorded("door_to_imaging", f"Time from {door} to imaging")
    else:
        n.not_recorded("onset_to_imaging", "Time of symptom onset")
    if spec["wake_up"] is None:
        n.not_recorded("wake_up_stroke", "Wake-up stroke")

    if spec["nihss"] is None:
        n.not_recorded("nihss", "NIHSS on arrival")
    else:
        span = n.pick([f"NIHSS on arrival: {spec['nihss']}.", f"Initial NIHSS {spec['nihss']}.", f"NIHSS {spec['nihss']} on arrival."])
        n.lines.append(span)
        n.fact("nihss", "number", spec["nihss"], span)
    if spec["mrs"] is None:
        n.not_recorded("premorbid_mrs", "Pre-stroke mRS")
    else:
        span = f"Pre-stroke mRS: {spec['mrs']} ({MRS_WORDS.get(spec['mrs'], 'see score')})."
        n.lines.append(span)
        n.fact("premorbid_mrs", "number", spec["mrs"], span)

    meds = []
    if spec["anticoagulant"]:
        meds.append("an anticoagulant (agent not recorded)")
        n.fact("anticoagulant", "bool", True, "an anticoagulant")
    if spec["antiplatelet"]:
        meds.append("an antiplatelet agent")
        n.fact("antiplatelet", "bool", True, "an antiplatelet agent")
    if spec["statin"]:
        meds.append("a statin")
    line = f"Medications: {', '.join(meds)}." if meds else "Medications: none of note."
    if meds:
        n.fact("medications", "text", ", ".join(meds), line)
    negatives = [w for w, v in (("anticoagulant", spec["anticoagulant"]), ("antiplatelet", spec["antiplatelet"])) if v is False]
    if negatives:
        neg = "No " + " or ".join(negatives) + " use."
        line += " " + neg
        for key in ("anticoagulant", "antiplatelet"):
            if spec[key] is False:
                n.fact(key, "bool", False, neg)
    n.lines.append(line)
    if spec["anticoagulant"] is None:
        n.not_recorded("anticoagulant", "Anticoagulant use")

    yes = [label for col, key, label in HISTORY if spec["history"][key]]
    no = [label for col, key, label in HISTORY if spec["history"][key] is False]
    parts = [f"History: {', '.join(yes)}." if yes else "History: none of the listed conditions."]
    for col, key, label in HISTORY:
        if spec["history"][key]:
            n.fact(key, "bool", True, label)
    if no and yes:
        sentence = "No " + (no[0] if len(no) == 1 else ", ".join(no[:-1]) + " or " + no[-1]) + "."
        parts.append(sentence)
        for col, key, label in HISTORY:
            if spec["history"][key] is False:
                n.fact(key, "bool", False, sentence)
    elif no:
        parts = ["History: no " + (no[0] if len(no) == 1 else ", ".join(no[:-1]) + " or " + no[-1]) + "."]
        for col, key, label in HISTORY:
            if spec["history"][key] is False:
                n.fact(key, "bool", False, parts[0])
    n.lines.append(" ".join(parts))

    labs = []
    if spec["glucose_mmol"] is not None:
        labs.append(f"glucose {spec['glucose_mmol']} mmol/L")
        n.fact("glucose", "number", spec["glucose_mmol"], labs[-1], "mmol/L")
    if spec["inr"] is not None:
        labs.append(f"INR {spec['inr']}")
        n.fact("inr", "number", spec["inr"], labs[-1])
    if labs:
        n.lines.append("Admission labs: " + ", ".join(labs) + ".")
    if spec["inr"] is None:
        n.not_recorded("inr", "INR")

    note = "ED STROKE TRIAGE NOTE\n" + "\n".join(n.lines)
    sheet = FactSheet(n.facts)
    sheet.check_against(note)
    return note, sheet


def note_for(case_id, row, seed=SEED):
    """(note, FactSheet) for one case: the same text on every run for a given seed."""
    return render_case(spec_from_row(row), random.Random(f"{seed}:{case_id}"))


def generate(slim_root, cases, out_dir, folds=None, log=print):
    """Write <case>.txt and <case>.facts.json for each case, plus manifest.json (counts only). Development cases only."""
    import pandas as pd

    splits.check_ids(cases, folds)
    clinical = pd.read_csv(os.path.join(slim_root, "clinical.csv"), dtype=object).set_index("case")
    os.makedirs(out_dir, exist_ok=True)
    counts = {"notes": 0, "not_recorded": {}, "wake_up": 0, "onset_known": 0}
    for case in cases:
        row = {k: (None if pd.isna(v) else v) for k, v in clinical.loc[case].items()}
        note, sheet = note_for(case, row)
        with open(os.path.join(out_dir, f"{case}.txt"), "w") as f:
            f.write(note + "\n")
        with open(os.path.join(out_dir, f"{case}.facts.json"), "w") as f:
            json.dump({"facts": [vars(fact) for fact in sheet]}, f, indent=1)
        counts["notes"] += 1
        counts["wake_up"] += bool(sheet.get("wake_up_stroke") and sheet.get("wake_up_stroke").value)
        counts["onset_known"] += sheet.get("onset_to_imaging") is not None and sheet.get("onset_to_imaging").type == "number"
        for fact in sheet:
            if fact.type == "not_recorded":
                counts["not_recorded"][fact.key] = counts["not_recorded"].get(fact.key, 0) + 1
    counts["seed"] = SEED
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(counts, f, indent=1)
    log(f"Wrote {counts['notes']} notes: onset time known in {counts['onset_known']}, wake-up {counts['wake_up']}; "
        f"not recorded: {counts['not_recorded']}.")
    return counts


def main(argv=None):
    """python -m strokecopilot.nlp.notes.generate --tar <isles24_slim_v1.tar> --out <folder>

    Reads only clinical.csv and manifest.json from the tarball (checked by sha256 first), writes notes for the 129
    development cases, and prints counts only."""
    import argparse
    import hashlib
    import tarfile
    import tempfile

    from strokecopilot.data import isles24

    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    h = hashlib.sha256()
    with open(a.tar, "rb") as f:
        for block in iter(lambda: f.read(16 * 2**20), b""):
            h.update(block)
    if h.hexdigest() != SLIM_SHA256:
        raise SystemExit("The tarball is not the recorded isles24_slim_v1.tar.")
    with tempfile.TemporaryDirectory() as tmp:
        wanted = {f"{isles24.SLIM_NAME}/clinical.csv", f"{isles24.SLIM_NAME}/manifest.json"}
        with tarfile.open(a.tar) as t:
            members = [m for m in t.getmembers() if m.name.lstrip("./") in wanted]
            if len(members) != 2:
                raise SystemExit("clinical.csv or manifest.json is missing from the tarball.")
            t.extractall(tmp, members=members, filter="data")
        root = os.path.join(tmp, isles24.SLIM_NAME)
        folds = splits.load_folds()
        cases = isles24.list_cases(root, folds)
        if len(cases) != EXPECTED_DEVELOPMENT or set(cases) != set(splits.development_ids(folds)):
            raise SystemExit(f"Expected the 129 development cases, found {len(cases)}.")
        generate(root, cases, a.out, folds)
    print(f"Notes written to {a.out}")


if __name__ == "__main__":
    main()
