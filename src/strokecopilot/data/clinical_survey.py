"""Survey the ISLES'24 clinical Excel files without showing any patient's row (plan §5.1, §8.3).

Each case has one baseline and one outcome workbook. Reports are aggregate only: values appear only
as counts for columns with few distinct values, and numbers only as percentiles. Columns with many
distinct text values, which is where case IDs would be, are never printed.
"""

import datetime as dt
import glob
import json
import numbers
import os
from collections import Counter, defaultdict

import pandas as pd

KINDS = {"baseline": "ses-01/{case}_ses-01_demographic_baseline.xlsx", "outcome": "ses-02/{case}_ses-02_outcome.xlsx"}
MAX_VALUES_SHOWN = 10


def read_dictionary(path):
    """Every non-empty row of every sheet in the data dictionary, as lists of strings."""
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    return {name: [[str(v) for v in row if pd.notna(v)] for row in df.values if pd.notna(row).any()] for name, df in sheets.items()}


def find_cases(phenotype_dir):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(phenotype_dir, "*")) if os.path.isdir(p))


def read_tables(phenotype_dir, cases, progress_every=50):
    """{kind: {case: {sheet: DataFrame}}}, plus (kind, case) pairs whose workbook is missing."""
    tables, missing = defaultdict(dict), []
    for i, case in enumerate(cases, 1):
        for kind, template in KINDS.items():
            path = os.path.join(phenotype_dir, case, template.format(case=case))
            if os.path.exists(path):
                tables[kind][case] = pd.read_excel(path, sheet_name=None, dtype=object)
            else:
                missing.append((kind, case))
        if i % progress_every == 0:
            print(f"Read {i} of {len(cases)} cases", flush=True)
    return dict(tables), missing


def _value_type(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, numbers.Number):
        return "number"
    if isinstance(value, (dt.datetime, dt.date, dt.time, dt.timedelta, pd.Timestamp)):
        return "datetime"
    return "text"


def _parses_as_number(text):
    try:
        float(str(text).strip().replace(",", "."))
        return True
    except ValueError:
        return False


def _survey_column(values, cases_present, n_cases):
    present = [v for v in values if pd.notna(v)]
    types = Counter(_value_type(v) for v in present)
    distinct = Counter(str(v).strip() for v in present)
    column = {
        "cases_with_column": cases_present,
        "cases_without_column": n_cases - cases_present,
        "non_null": len(present),
        "null": len(values) - len(present),
        "types": dict(types),
        "distinct": len(distinct),
    }
    if len(distinct) <= MAX_VALUES_SHOWN:
        column["value_counts"] = dict(distinct.most_common())

    numbers_only = [float(v) for v in present if _value_type(v) == "number"]
    if numbers_only:
        p5, p50, p95 = pd.Series(numbers_only).quantile([0.05, 0.5, 0.95]).tolist()
        column["p5_p50_p95"] = [round(p5, 3), round(p50, 3), round(p95, 3)]

    texts = [v for v in present if _value_type(v) == "text"]
    if texts:
        column["text_parses_as_number"] = sum(_parses_as_number(v) for v in texts)
        text_distinct = Counter(str(v).strip() for v in texts)
        if len(text_distinct) <= MAX_VALUES_SHOWN:
            column["text_values"] = dict(text_distinct.most_common())
    return column


def survey(tables, cases):
    report = {"n_cases": len(cases), "kinds": {}}
    for kind, by_case in tables.items():
        sheets = Counter(tuple(book) for book in by_case.values())
        rows, column_sets = Counter(), Counter()
        values, present_in = defaultdict(list), defaultdict(set)
        id_in_file = 0
        for case, book in by_case.items():
            found_id = False
            for sheet, df in book.items():
                rows[len(df)] += 1
                column_sets[tuple(str(c) for c in df.columns)] += 1
                for col in df.columns:
                    name = str(col)
                    values[name].extend(df[col].tolist())
                    present_in[name].add(case)
                    found_id = found_id or df[col].astype(str).str.strip().eq(case).any()
            id_in_file += found_id
        report["kinds"][kind] = {
            "workbooks": len(by_case),
            "sheet_layouts": [{"sheets": list(k), "workbooks": v} for k, v in sheets.most_common()],
            "rows_per_sheet": {str(k): v for k, v in sorted(rows.items())},
            "column_layouts": len(column_sets),
            "workbooks_holding_their_case_id": id_in_file,
            "columns": {name: _survey_column(vals, len(present_in[name]), len(by_case)) for name, vals in values.items()},
        }
    return report


def format_dictionary(dictionary):
    out = []
    for sheet, rows in dictionary.items():
        out += ["", f"DICTIONARY SHEET: {sheet} ({len(rows)} rows)"]
        out += ["  " + " | ".join(row) for row in rows]
    return "\n".join(out)


def format_survey(report, missing):
    out = [f"CASES {report['n_cases']}", f"MISSING WORKBOOKS {len(missing)}: {missing[:10]}"]
    for kind, info in report["kinds"].items():
        out += [
            "",
            f"{kind.upper()}: {info['workbooks']} workbooks, sheets {info['sheet_layouts']}, rows per sheet {info['rows_per_sheet']}, "
            f"{info['column_layouts']} column layout(s), {info['workbooks_holding_their_case_id']} hold their own case ID",
        ]
        for name, c in info["columns"].items():
            line = f"  {name!r}: non-null {c['non_null']}, null {c['null']}, distinct {c['distinct']}, types {c['types']}"
            if c["cases_without_column"]:
                line += f", ABSENT FROM {c['cases_without_column']} workbooks"
            if "p5_p50_p95" in c:
                line += f", p5/p50/p95 {c['p5_p50_p95']}"
            if "value_counts" in c:
                line += f", values {c['value_counts']}"
            elif "text_values" in c:
                line += f", text values {c['text_values']}"
            elif c.get("text_parses_as_number"):
                line += f", {c['text_parses_as_number']} text values parse as numbers"
            out.append(line)
    return "\n".join(out)


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
