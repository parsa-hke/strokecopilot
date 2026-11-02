"""Which ISLES'24 clinical columns may reach a triage note (plan §8.3).

Names are the real workbook headers, which differ from the data dictionary in places. Every column
is tagged, so a new or renamed column fails `tag_of` instead of leaking into a note by default.
"""

ALLOWED_FOR_NOTES = {
    "Sex": "demographic",
    "Age": "demographic; decimal in the data, integer in the dictionary",
    "Atrial fibrillation": "comorbidity",
    "Hypertension": "comorbidity",
    "Diabetes": "comorbidity",
    "Hyperlipidemia": "comorbidity; 99 means unknown",
    "Anticoagulation": "pre-stroke medication",
    "Lipid lowering drugs": "pre-stroke medication; the dictionary calls it Statins",
    "PAIs": "pre-stroke medication: platelet aggregation inhibitors",
    "Glucose": "admission lab, altered +/-5%",
    "Leucocytes": "admission lab, altered +/-5%",
    "CRP": "admission lab, altered +/-5%",
    "INR": "admission lab, altered +/-5%",
    "Wake-up": "presentation",
    "In-House": "presentation",
    "Referral": "presentation",
    "Onset to door": "time; needed for onset to imaging",
    "Alert to door": "time; recorded for wake-up strokes only",
    "NIHSS at admission": "admission exam",
    "mRS premorbid": "pre-stroke function",
    "Door to imaging": "time; needed for onset to imaging",
}

FORBIDDEN_FOR_NOTES = {
    "Center": "study site, not a clinical fact; not in the dictionary",
    "mRS at admission": "not in the §8.3 note fields; easily confused with premorbid mRS, which the rules use",
    "Door to groin": "reveals thrombectomy",
    "Door to first series": "reveals thrombectomy",
    "Time of intervention": "reveals thrombectomy",
    "Door to recanalization": "reveals thrombectomy",
    "TICI postinterventional": "outcome of treatment",
    "NIHSS 24h": "after treatment",
    "mRS 24h": "after treatment",
    "NIHSS discharge": "after treatment",
    "mRS discharge": "after treatment",
    "mRS 3 months": "after treatment",
}

BASELINE_COLUMNS = (
    "Center", "Sex", "Age", "Atrial fibrillation", "Hypertension", "Diabetes", "Hyperlipidemia", "Anticoagulation",
    "Lipid lowering drugs", "PAIs", "Glucose", "Leucocytes", "CRP", "INR", "Wake-up", "In-House", "Referral",
    "Onset to door", "Alert to door", "NIHSS at admission", "mRS at admission", "mRS premorbid", "Door to imaging",
    "Door to groin", "Door to first series", "Time of intervention", "Door to recanalization",
)  # fmt: skip

OUTCOME_COLUMNS = ("TICI postinterventional", "NIHSS 24h", "mRS 24h", "NIHSS discharge", "mRS discharge", "mRS 3 months")


def tag_of(column):
    if column in ALLOWED_FOR_NOTES:
        return "allowed"
    if column in FORBIDDEN_FOR_NOTES:
        return "forbidden"
    raise KeyError(f"Clinical column {column!r} has no note tag. Tag it in clinical_columns.py (plan §8.3).")
