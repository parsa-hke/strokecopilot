"""data/template.py on a synthetic case (no patient data). Needs the [isles] extra."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from strokecopilot.data import isles24, template  # noqa: E402

SHAPE = (20, 24, 10)


def case(infarct_slice=6, empty=False):
    arrays = {name: np.random.default_rng(0).uniform(0, 50, SHAPE).astype(np.float32) for name in ("ncct", "cbf", "cbv", "mtt", "tmax")}
    infarct = np.zeros(SHAPE, bool)
    if not empty:
        infarct[5:10, 5:10, infarct_slice] = True
        infarct[5:7, 5:7, 2] = True
    occlusion = np.zeros(SHAPE, bool)
    occlusion[3, 3, infarct_slice] = True
    arrays.update(infarct=infarct, occlusion=occlusion)
    clinical = {"case": "dev-1", "Age": "71.2", "INR": None, "Door to groin": "01:00:00"}
    return isles24.Case(id="dev-1", arrays=arrays, spacing_mm=1.5, affine=np.eye(4), clinical=clinical,
                        native_volumes={"infarct_ml": 12.34, "tmax6_ml": 80.0})


def test_largest_slice_and_empty_mask():
    assert template.largest_infarct_slice(case(infarct_slice=6)) == 6
    assert template.largest_infarct_slice(case(empty=True)) == SHAPE[2] // 2


def test_figure_has_one_panel_per_map():
    fig = template.case_figure(case())
    assert len(fig.axes) == len(template.PANELS)
    assert "slice 6" in fig._suptitle.get_text()


def test_note_row_drops_forbidden_columns_and_marks_missing():
    row = template.note_row(case())
    assert "Door to groin" not in row and "case" not in row
    assert row["INR"] == "not recorded" and row["Age"] == "71.2"


def test_summary_lists_reference_volumes():
    text = template.summary(case())
    assert "infarct_ml 12.3 mL" in text and "1.5 mm" in text
