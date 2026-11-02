"""Note tags for the ISLES'24 clinical columns (plan §8.3)."""

import pytest

from strokecopilot.data import clinical_columns as cc


def test_every_real_column_has_exactly_one_tag():
    columns = set(cc.BASELINE_COLUMNS) | set(cc.OUTCOME_COLUMNS)
    assert columns == set(cc.ALLOWED_FOR_NOTES) | set(cc.FORBIDDEN_FOR_NOTES)
    assert not set(cc.ALLOWED_FOR_NOTES) & set(cc.FORBIDDEN_FOR_NOTES)


def test_outcome_columns_are_all_forbidden():
    assert {cc.tag_of(c) for c in cc.OUTCOME_COLUMNS} == {"forbidden"}


@pytest.mark.parametrize(
    "column",
    ["Door to groin", "Time of intervention", "Door to first series", "Door to recanalization", "TICI postinterventional"],
)
def test_section_8_3_treatment_columns_are_forbidden(column):
    assert cc.tag_of(column) == "forbidden"


@pytest.mark.parametrize("column", ["mRS premorbid", "NIHSS at admission", "Onset to door", "Door to imaging", "INR", "Anticoagulation"])
def test_section_8_3_note_fields_are_allowed(column):
    assert cc.tag_of(column) == "allowed"


def test_unknown_column_is_refused():
    with pytest.raises(KeyError, match="no note tag"):
        cc.tag_of("mTICI")
