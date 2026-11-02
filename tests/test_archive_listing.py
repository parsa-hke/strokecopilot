"""Archive listing summary on a hand-written `7z l -slt` output (plan §7, Phase 1)."""

import stat

import pytest

from strokecopilot.data import archive_listing as al

SLT = """
7-Zip (a) 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21

Listing archive: train.7z

--
Path = train.7z
Type = 7z
Physical Size = 1000
Headers Size = 50
Method = LZMA2:26
Solid = +
Blocks = 2

----------
Path = train
Size = 0
Packed Size = 0
Attributes = D drwxr-xr-x
Encrypted = -
Block =

----------
"""


def _file(path, size, block):
    return f"Path = {path}\nSize = {size}\nPacked Size = \nAttributes = A -rw-r--r--\nEncrypted = -\nMethod = LZMA2:26\nBlock = {block}\n\n"


LISTING = (
    SLT.rsplit("----------\n", 1)[0]
    + _file("train/clinical_data-description.xlsx", 10, 0)
    + _file("train/sub-stroke0001/sub-stroke0001_cbf.nii.gz", 100, 0)
    + _file("train/sub-stroke0001/sub-stroke0001_tmax.nii.gz", 200, 0)
    + _file("train/sub-stroke0002/sub-stroke0002_cbf.nii.gz", 300, 1)
    + _file("train/sub-stroke0002/sub-stroke0002_tmax.nii.gz", 400, 1)
    + "Path = train/empty.txt\nSize = 0\nAttributes = A -rw-r--r--\nEncrypted = -\nBlock = \n"
)


@pytest.fixture
def summary():
    return al.summarise(*al.parse_slt(LISTING))


def _pattern(summary, name):
    return next(p for p in summary["patterns"] if p["pattern"].endswith(name))


def test_parse_reads_archive_properties_and_entries():
    archive, entries = al.parse_slt(LISTING)
    assert archive["Solid"] == "+" and archive["Physical Size"] == "1000"
    assert entries[0] == {"path": "train", "size": 0, "block": None, "encrypted": False, "is_dir": True}
    assert entries[2]["size"] == 100 and entries[2]["block"] == "0"
    assert len(entries) == 7


def test_parse_rejects_other_text():
    with pytest.raises(ValueError):
        al.parse_slt("7-Zip error: cannot open file")


def test_totals_and_cases(summary):
    assert summary["n_files"] == 6 and summary["n_dirs"] == 1
    assert summary["unpacked_bytes"] == 1010
    assert summary["n_cases"] == 2
    assert (summary["first_case"], summary["last_case"]) == ("sub-stroke0001", "sub-stroke0002")
    assert summary["archive"]["solid"] is True and summary["archive"]["blocks"] == 2
    assert summary["extensions"][".nii.gz"] == {"files": 4, "bytes": 1000}


def test_patterns_hide_the_case_id(summary):
    assert len(summary["patterns"]) == 2
    cbf = _pattern(summary, "{case}_cbf.nii.gz")
    assert cbf["pattern"] == "train/{case}/{case}_cbf.nii.gz"
    assert (cbf["cases"], cbf["files"], cbf["bytes"], cbf["missing_cases"]) == (2, 2, 400, [])


def test_decode_cost_counts_from_block_start_to_last_wanted_file(summary):
    # block 0: xlsx 10 + cbf 100 + tmax 200; block 1: cbf 300 + tmax 400
    assert summary["shared_files"][0] == {"path": "train/clinical_data-description.xlsx", "bytes": 10, "decode_bytes": 10}
    assert _pattern(summary, "{case}_cbf.nii.gz")["decode_bytes"] == 110 + 300
    assert _pattern(summary, "{case}_tmax.nii.gz")["decode_bytes"] == 310 + 700


def test_decode_cost_in_a_non_solid_archive_is_just_the_file():
    entries = [
        {"path": "sub-stroke1/a", "size": 5, "block": "0", "encrypted": False, "is_dir": False},
        {"path": "sub-stroke1/b", "size": 7, "block": "1", "encrypted": False, "is_dir": False},
        {"path": "sub-stroke2/a", "size": 9, "block": "2", "encrypted": False, "is_dir": False},
    ]
    summary = al.summarise({}, entries)
    assert _pattern(summary, "{case}/a")["decode_bytes"] == 14


def test_missing_case_is_named():
    listing = LISTING.replace("sub-stroke0002/sub-stroke0002_cbf", "sub-stroke0002/sub-stroke0002_other")
    summary = al.summarise(*al.parse_slt(listing))
    assert _pattern(summary, "{case}_cbf.nii.gz")["missing_cases"] == ["sub-stroke0002"]
    assert summary["perfusion_maps"]["cbf"]["max_cases"] == 1
    assert any("cbf map" in p for p in al.find_problems(summary, 1000, 2))


def test_perfusion_maps_match_whole_words_only(summary):
    assert summary["perfusion_maps"]["tmax"]["max_cases"] == 2
    assert summary["perfusion_maps"]["mtt"] == {"max_cases": 0, "patterns": []}
    entries = [{"path": "sub-stroke1/sub-stroke1_rcbfx.nii.gz", "size": 1, "block": "0", "encrypted": False, "is_dir": False}]
    assert al.summarise({}, entries)["perfusion_maps"]["cbf"]["max_cases"] == 0


def test_find_problems(summary):
    assert any("mtt" in p for p in al.find_problems(summary, 1000, 2))
    problems = al.find_problems(summary, 999, 149)
    assert any("not the verified 999" in p for p in problems)
    assert any("Found 2 case IDs, expected 149" in p for p in problems)


def test_report_shows_findings(summary):
    report = al.format_report(summary, ["a problem"], local_free_bytes=1)
    assert "solid=True" in report and "a problem" in report and "DOES NOT FIT" in report
    assert "train/{case}/{case}_cbf.nii.gz" in report
    assert "clinical_data-description.xlsx" in report


def _script(tmp_path, body):
    path = tmp_path / "fake7z"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_list_archive_writes_output_only_on_success(tmp_path):
    out = tmp_path / "listing.txt"
    al.list_archive("x.7z", str(out), seven_zip=_script(tmp_path, "echo listed $1 $2 $3"))
    assert out.read_text().strip() == "listed l -slt x.7z"
    assert not (tmp_path / "listing.txt.part").exists()


def test_list_archive_raises_on_failure(tmp_path):
    out = tmp_path / "listing.txt"
    with pytest.raises(RuntimeError, match="broken"):
        al.list_archive("x.7z", str(out), seven_zip=_script(tmp_path, "echo broken >&2; exit 2"))
    assert not out.exists()


def test_list_archive_prints_heartbeat_and_times_out(tmp_path, capsys):
    out = tmp_path / "listing.txt"
    al.list_archive("x.7z", str(out), seven_zip=_script(tmp_path, "sleep 2"), heartbeat_s=1)
    assert "Still reading" in capsys.readouterr().out
    out.unlink()
    with pytest.raises(TimeoutError):
        al.list_archive("x.7z", str(out), seven_zip=_script(tmp_path, "sleep 30"), timeout_s=1)
    assert not out.exists()
