"""RSNA patient split on a synthetic scan table: patients stay whole, rates hold, test stays closed."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from strokecopilot.data import rsna_split as rs  # noqa: E402


@pytest.fixture(scope="module")
def scans():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(3000):
        bleed = rng.random() < 0.4
        for s in range(1 + int(rng.random() < 0.15)):
            rows.append({"patient_id": f"ID_p{p:05d}", "series_id": f"ID_s{p:05d}_{s}", "any": int(bleed), "slices": 30})
    return pd.DataFrame(rows)


def test_split_keeps_patients_whole_and_shares(scans):
    table = rs.add_split(scans, rs.assign(scans))
    report = rs.check(table)
    assert set(table["split"]) == set(rs.SPLITS)
    assert abs(report["patient_share"]["train"] - 0.8) < 0.002 and abs(report["patient_share"]["val"] - 0.1) < 0.002
    assert abs(report["val"]["patient_bleed_rate"] - report["all"]["patient_bleed_rate"]) < 0.002
    assert "ID_" not in str(report)


def test_split_ignores_row_order_and_depends_on_seed(scans):
    a = rs.assign(scans)
    assert rs.assign(scans.sample(frac=1, random_state=3)) == a
    assert rs.fingerprint(a) == rs.fingerprint(dict(reversed(list(a.items()))))
    assert rs.fingerprint(rs.assign(scans, seed=1)) != rs.fingerprint(a)


def test_check_catches_drift_and_split_patients(scans):
    table = rs.add_split(scans, rs.assign(scans))
    drifted = table.copy()
    drifted.loc[(drifted["split"] == "val") & (drifted["any"] == 0), "split"] = "train"
    with pytest.raises(ValueError, match="val scan bleed rate"):
        rs.check(drifted)
    broken = table.copy()
    row = broken.index[broken["patient_id"].duplicated()][0]
    broken.loc[row, "split"] = "test" if broken.loc[row, "split"] != "test" else "val"
    with pytest.raises(ValueError, match="more than one split"):
        rs.check(broken)


def test_missing_patient_raises(scans):
    with pytest.raises(ValueError, match="without a split"):
        rs.add_split(scans, {})


def test_test_split_is_closed(scans, monkeypatch):
    table = rs.add_split(scans, rs.assign(scans))
    monkeypatch.delenv(rs.TEST_FLAG, raising=False)
    assert len(rs.rows(table, "val")) > 0
    with pytest.raises(rs.TestSplitError):
        rs.rows(table, "test")
    monkeypatch.setenv(rs.TEST_FLAG, "1")
    assert len(rs.rows(table, "test")) > 0
