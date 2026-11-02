"""Clinical survey on synthetic workbooks (plan §5.1, §8.3). Needs the [isles] extra."""

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

from strokecopilot.data import clinical_survey as cs  # noqa: E402

CASES = [f"sub-stroke{i:04d}" for i in (1, 2, 3, 7, 9, 12, 20, 33, 41, 55, 60, 71)]


def _write(root, case, kind, row):
    path = root / case / cs.KINDS[kind].format(case=case)
    path.parent.mkdir(parents=True)
    pd.DataFrame([row]).to_excel(path, index=False)


@pytest.fixture
def phenotype(tmp_path):
    for i, case in enumerate(CASES):
        age = "unknown" if i == 0 else 50 + i
        _write(tmp_path, case, "baseline", {"id": case, "Age": age, "Sex": "F" if i % 2 else "M", "INR": None if i == 1 else 1.0 + i / 10})
        if i != 2:
            _write(tmp_path, case, "outcome", {"id": case, "mTICI": "2c" if i % 2 else "3"})
    return tmp_path


@pytest.fixture
def surveyed(phenotype):
    cases = cs.find_cases(str(phenotype))
    tables, missing = cs.read_tables(str(phenotype), cases)
    return cs.survey(tables, cases), missing


def test_finds_cases_and_missing_workbooks(surveyed):
    report, missing = surveyed
    assert report["n_cases"] == len(CASES)
    assert missing == [("outcome", "sub-stroke0003")]
    assert report["kinds"]["outcome"]["workbooks"] == len(CASES) - 1


def test_column_counts_and_types(surveyed):
    baseline = surveyed[0]["kinds"]["baseline"]
    assert baseline["rows_per_sheet"] == {"1": len(CASES)}
    assert baseline["workbooks_holding_their_case_id"] == len(CASES)
    assert type(baseline["workbooks_holding_their_case_id"]) is int  # numpy ints reach the JSON as strings
    assert baseline["columns"]["INR"]["null"] == 1
    age = baseline["columns"]["Age"]
    assert age["types"] == {"text": 1, "number": len(CASES) - 1}
    assert age["text_values"] == {"unknown": 1}
    assert age["p5_p50_p95"][1] == 56.0


def test_low_cardinality_values_are_counted(surveyed):
    assert surveyed[0]["kinds"]["outcome"]["columns"]["mTICI"]["value_counts"] == {"2c": 6, "3": 5}


def test_report_never_prints_case_ids(surveyed):
    text = cs.format_survey(surveyed[0], [])
    assert "'id': non-null" in text
    assert not any(case in text for case in CASES)


def test_numeric_text_is_flagged():
    column = cs._survey_column([f"1,{i}" for i in range(20)], 20, 20)
    assert column["text_parses_as_number"] == 20
    assert "value_counts" not in column and "text_values" not in column


def test_column_absent_from_some_workbooks(tmp_path):
    _write(tmp_path, "sub-stroke0001", "baseline", {"Age": 60})
    _write(tmp_path, "sub-stroke0002", "baseline", {"Age": 61, "Extra": 1})
    cases = cs.find_cases(str(tmp_path))
    tables, _ = cs.read_tables(str(tmp_path), cases)
    info = cs.survey(tables, cases)["kinds"]["baseline"]
    assert info["column_layouts"] == 2
    assert info["columns"]["Extra"]["cases_without_column"] == 1
    assert "ABSENT FROM 1 workbooks" in cs.format_survey({"n_cases": 2, "kinds": {"baseline": info}}, [])


def test_dictionary_reads_every_sheet_and_skips_empty_rows(tmp_path):
    path = tmp_path / "dict.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([["Age", "years"], [None, None], ["INR", None]]).to_excel(writer, sheet_name="A", index=False, header=False)
        pd.DataFrame([["mTICI", "score"]]).to_excel(writer, sheet_name="B", index=False, header=False)
    dictionary = cs.read_dictionary(path)
    assert dictionary == {"A": [["Age", "years"], ["INR"]], "B": [["mTICI", "score"]]}
    assert "Age | years" in cs.format_dictionary(dictionary)
