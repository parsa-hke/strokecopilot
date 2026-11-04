"""Fact sheets: what a triage note states, as checkable entries (docs/HALLUCINATION_METRIC.md §3b).

One entry per fact: key, type, value, unit and the verbatim span of the note that states it.
Types are number, bool (true, or false for a stated negative such as "no anticoagulant use"),
text, time (a clock time, HH:MM) and not_recorded (the note says the fact is missing). A fact the
note is silent on has no entry, so nothing supports a claim about it either way.

Generated notes get their sheet from notes/generate.py; the five fixture notes have hand-written
sheets next to them in fixtures/notes/<stem>.facts.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from strokecopilot.config import FIXTURE_NOTES_DIR

TYPES = ("number", "bool", "text", "time", "not_recorded")

# Keys a note may state, with the description the claim extractor sees. Values are never shown to it.
FACT_KEYS = {
    "age": "patient age, years",
    "sex": "patient sex",
    "onset_witnessed": "whether symptom onset was witnessed",
    "wake_up_stroke": "whether symptoms were found on waking",
    "onset_clock": "clock time of symptom onset",
    "lkw_clock": "clock time the patient was last known well",
    "found_clock": "clock time symptoms were found or the patient woke with them",
    "onset_to_imaging": "time from symptom onset to imaging",
    "lkw_to_imaging": "time from last known well to imaging",
    "deficits": "presenting symptoms and deficits",
    "nihss": "NIHSS score on arrival",
    "premorbid_mrs": "pre-stroke modified Rankin Scale score",
    "medications": "current medications",
    "anticoagulant": "anticoagulant use",
    "antiplatelet": "antiplatelet use",
    "inr": "INR",
    "hypertension": "history of hypertension",
    "diabetes": "history of diabetes",
    "hyperlipidaemia": "history of hyperlipidaemia",
    "smoker": "current smoker",
    "ischaemic_heart_disease": "history of ischaemic heart disease",
    "chronic_kidney_disease": "history of chronic kidney disease",
    "recent_surgery": "recent surgery",
    "head_trauma": "recent head trauma",
    "previous_ich": "previous intracranial haemorrhage",
    "recent_bleeding": "recent bleeding",
    "bp_systolic": "systolic blood pressure on arrival, mmHg",
    "bp_diastolic": "diastolic blood pressure on arrival, mmHg",
    "glucose": "capillary glucose, mmol/L",
    "referral": "how the patient arrived (ambulance, transfer)",
}


@dataclass(frozen=True)
class Fact:
    key: str
    type: str
    value: object
    span: str
    unit: str | None = None


class FactSheet:
    def __init__(self, facts):
        self.facts = {}
        for fact in facts:
            if fact.key not in FACT_KEYS:
                raise ValueError(f"Unknown fact key {fact.key!r}.")
            if fact.type not in TYPES:
                raise ValueError(f"Unknown fact type {fact.type!r} for {fact.key}.")
            if fact.key in self.facts:
                raise ValueError(f"Fact {fact.key} is listed twice.")
            self.facts[fact.key] = fact

    def get(self, key):
        return self.facts.get(key)

    def __iter__(self):
        return iter(self.facts.values())

    def check_against(self, note):
        """Raise if a span is not verbatim in the note."""
        for fact in self:
            if fact.span not in note:
                raise ValueError(f"The span for {fact.key} is not in the note.")

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls([Fact(**entry) for entry in json.load(f)["facts"]])


def fixture_note(stem):
    """(note text, FactSheet) for a fixture, by file stem such as fx02_thrombectomy_large_mismatch."""
    folder = Path(FIXTURE_NOTES_DIR)
    note = (folder / f"{stem}.txt").read_text()
    sheet = FactSheet.from_json(folder / f"{stem}.facts.json")
    sheet.check_against(note)
    return note, sheet
