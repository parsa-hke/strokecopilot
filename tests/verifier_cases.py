"""Shared test material for the verifier: a clean fx02 output, its extracted claims, a tiny corpus and a fake judge.

The corpus is synthetic text written for tests (TEST- IDs), not guideline wording.
"""

import copy

CORPUS = {
    "TEST-IVT-4.5H": {"text": "IV thrombolysis is recommended within 4.5 hours of symptom onset or last known well for disabling deficits."},
    "TEST-DEFUSE3": {"text": "DEFUSE-3 enrolled patients 6 to 16 hours from last known well with an ischaemic core under 70 mL, a mismatch ratio of 1.8 or more and a mismatch volume of 15 mL or more."},
    "TEST-EVT-6-24": {"text": "For ICA or M1 occlusion 6 to 24 hours from last known well with NIHSS 6 or more, ASPECTS 6 to 10 and pre-stroke mRS 0 to 1, thrombectomy is recommended."},
    "TEST-EXT-IVT": {"text": "Extended-window thrombolysis is reasonable only for patients who are not eligible for thrombectomy."},
}

R_EVT = "Thrombectomy is recommended for this patient in the 6-24 h window with ASPECTS 6-10 and pre-stroke mRS 0-1."
R_EXT = "Extended-window thrombolysis does not apply because thrombectomy is possible."
R_IVT = "Time from last known well to imaging, 9 h, is beyond the 4.5 h thrombolysis window."
R_CORE = "Core 18.0 mL is below the DEFUSE-3 limit of 70 mL."

# claim text -> {passage id: (verdict, scope_fits)}; anything missing is not_addressed.
JUDGE_TABLE = {
    R_EVT: {"TEST-EVT-6-24": ("supported", True)},
    R_EXT: {"TEST-EXT-IVT": ("supported", True)},
    R_IVT: {"TEST-IVT-4.5H": ("supported", None)},
    R_CORE: {"TEST-DEFUSE3": ("supported", None)},
    "Passage TEST-EVT-6-24 is a rule about thrombectomy.": {"TEST-EVT-6-24": ("supported", None)},
    "Passage TEST-EXT-IVT is a rule about thrombolysis.": {"TEST-EXT-IVT": ("supported", None)},
}


class FakeJudge:
    def __init__(self, table=None):
        self.table = JUDGE_TABLE if table is None else table
        self.calls = []

    def __call__(self, claim_text, passages, case_inputs, applied):
        self.calls.append((claim_text, [p["id"] for p in passages], applied))
        row = self.table.get(claim_text, {})
        return {p["id"]: row.get(p["id"], ("not_addressed", None)) for p in passages}


def note_source(quote):
    return {"type": "note", "quote": quote}


def vision_source(field):
    return {"type": "vision", "field": field}


def passage_source(pid):
    return {"type": "passage", "id": pid}


CLEAN_REC = {
    "case_id": "fixture-fx02",
    "model": "test-model",
    "prompt_version": "reasoner_v1",
    "thrombolysis": {"recommend": False, "rule_ids": ["TEST-EXT-IVT"], "blocking_factors": [R_EXT]},
    "thrombectomy": {"recommend": True, "rule_ids": ["TEST-EVT-6-24"], "blocking_factors": []},
    "rationale": [
        {"claim": "Core volume is 18.0 mL and the total hypoperfused volume is 95.0 mL.", "sources": [vision_source("core_ml"), vision_source("penumbra_ml")], "derived": False},
        {"claim": R_IVT, "sources": [note_source("Time from last known well to imaging: 9 h 0 min."), passage_source("TEST-IVT-4.5H")], "derived": True},
        {"claim": "The occlusion is in M1 and the NIHSS is 16.", "sources": [vision_source("occlusion_site"), note_source("NIHSS on arrival: 16.")], "derived": False},
        {"claim": R_CORE, "sources": [vision_source("core_ml"), passage_source("TEST-DEFUSE3")], "derived": True},
        {"claim": R_EVT, "sources": [passage_source("TEST-EVT-6-24")], "derived": False},
    ],
    "uncertainties": ["Significant mass effect is not reported.", "Whether reperfusion will succeed is uncertain."],
}


def _item(field, entry, text, kind, subject=None, value=None, unit=None, applies=False, comparison=None):
    return {"field": field, "entry": entry, "text": text, "kind": kind, "subject": subject, "value": value, "unit": unit,
            "applies_to_patient": applies, "comparison": comparison}


def operand(subject=None, value=None, unit=None, from_rule=False):
    return {"subject": subject, "value": value, "unit": unit, "from_rule": from_rule}


# What a correct extractor returns for CLEAN_REC.
CLEAN_EXTRACTION = {"claims": [
    _item("rationale", 0, "Core volume is 18.0 mL.", "vision", "core_ml", "18.0", "mL"),
    _item("rationale", 0, "Total hypoperfused volume is 95.0 mL.", "vision", "penumbra_ml", "95.0", "mL"),
    _item("rationale", 1, "Time from last known well to imaging is 9 h.", "patient", "lkw_to_imaging", "9", "h"),
    _item("rationale", 1, R_IVT, "comparison", comparison={"left": operand("lkw_to_imaging", "9", "h"), "op": ">", "right": operand(None, "4.5", "h", True)}),
    _item("rationale", 2, "The occlusion is in M1.", "vision", "occlusion_site", "M1"),
    _item("rationale", 2, "NIHSS is 16.", "patient", "nihss", "16"),
    _item("rationale", 3, R_CORE, "comparison", comparison={"left": operand("core_ml", "18.0", "mL"), "op": "<", "right": operand(None, "70", "mL", True)}),
    _item("rationale", 4, R_EVT, "rule", applies=True),
    _item("thrombolysis.blocking_factors", 0, R_EXT, "rule", applies=True),
    _item("uncertainties", 0, "Significant mass effect is not reported.", "absence", "mass_effect"),
]}


def fake_extractor(reply):
    calls = []

    def llm(system, user, schema):
        calls.append((system, user, schema))
        return copy.deepcopy(reply)

    llm.calls = calls
    return llm
