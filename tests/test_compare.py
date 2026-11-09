"""Tests for pipeline/compare.py."""

import json
from pathlib import Path

import pytest

from strokecopilot.pipeline.compare import build_table, load_run, plot_table


@pytest.fixture
def exp12_folder(tmp_path):
    """Minimal EXP-12-style run folder."""
    folder = tmp_path / "EXP-12_test"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "experiment": "EXP-12",
        "mode": "fixtures",
        "reasoner": {"model": "claude-sonnet-5-5"},
    }))
    (folder / "metrics.json").write_text(json.dumps({
        "summary": {
            "hallucination_rate": {"value": 0.235, "numerator": 87, "denominator": 371, "ci95": None},
            "critical_hallucination_rate": {"value": None, "numerator": 0, "denominator": 0, "ci95": None},
            "omission_rate": {"value": None, "numerator": 0, "denominator": 0, "ci95": None},
        },
        "per_fixture": [],
    }))
    return folder


def test_load_run(exp12_folder):
    run = load_run(exp12_folder)
    assert run["experiment"] == "EXP-12"
    assert run["model"] == "claude-sonnet-5-5"
    assert run["source"] == "fixtures"
    assert run["arm"] == "with_rulebook"


def test_build_table_one_row(exp12_folder):
    table = build_table([exp12_folder])
    assert len(table) == 1
    assert table[0]["experiment"] == "EXP-12"
    assert table[0]["hallucination_rate"] == 0.235
    assert table[0]["critical_hallucination_rate"] is None
    assert table[0]["omission_rate"] is None


def test_build_table_two_rows(exp12_folder, tmp_path):
    folder2 = tmp_path / "EXP-13_test"
    folder2.mkdir()
    (folder2 / "config.json").write_text(json.dumps({
        "experiment": "EXP-13", "mode": "scenarios",
        "reasoner": {"model": "claude-sonnet-5-5"},
    }))
    (folder2 / "metrics.json").write_text(json.dumps({
        "summary": {
            "hallucination_rate": {"value": 0.10, "numerator": 10, "denominator": 100, "ci95": [0.05, 0.15]},
            "critical_hallucination_rate": {"value": 0.02, "numerator": 2, "denominator": 100, "ci95": None},
            "omission_rate": {"value": 0.01, "numerator": 1, "denominator": 100, "ci95": None},
        },
        "per_fixture": [],
    }))
    table = build_table([exp12_folder, folder2])
    assert len(table) == 2
    assert table[1]["experiment"] == "EXP-13"
    assert table[1]["hallucination_rate"] == 0.10


def test_plot_table_writes_file(exp12_folder, tmp_path):
    table = build_table([exp12_folder])
    out = tmp_path / "figure.png"
    plot_table(table, out)
    assert out.exists()
    assert out.stat().st_size > 0