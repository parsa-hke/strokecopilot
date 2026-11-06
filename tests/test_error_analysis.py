"""Vision A error analysis on synthetic cases. Needs numpy and scipy; no real data.

Step 1 sets up the synthetic case builder. Later steps add tests for each function in
strokecopilot.vision.perfusion.error_analysis as it is filled in.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from strokecopilot.data.isles24 import Case  # noqa: E402
from strokecopilot.vision.perfusion import error_analysis  # noqa: E402
from strokecopilot.vision.perfusion.thresholds import reliable  # noqa: E402
from strokecopilot.vision.perfusion.volumes import mask_ml  # noqa: E402

SHAPE = (40, 30, 20)
SPACING = 2.0  # 8 mm3 per voxel, so 125 voxels = 1 mL


def synthetic_case(infarct_box=(slice(24, 34), slice(8, 20), slice(6, 14)), no_data_box=None, occlusion_label=6):
    """A small fake case in the shape isles24.load_case() returns.

    Brain at x 4-35 with Tmax 1 s; a delay (Tmax 10 s) over the infarct box on the right side.
    no_data_box sets Tmax to -30 there (no perfusion data). The occlusion is a few voxels next to
    one Circle of Willis label (6 = right ICA by default).
    """
    cbf = np.zeros(SHAPE, dtype=np.float32)
    tmax = np.full(SHAPE, -30.0, dtype=np.float32)
    brain = (slice(4, 36), slice(4, 26), slice(4, 16))
    cbf[brain], tmax[brain] = 50.0, 1.0
    infarct = np.zeros(SHAPE, dtype=bool)
    if infarct_box is not None:
        infarct[infarct_box] = True
        tmax[infarct_box], cbf[infarct_box] = 10.0, 10.0
    if no_data_box is not None:
        tmax[no_data_box] = -30.0
    cow = np.zeros(SHAPE, dtype=np.int16)
    cow[20:24, 14:16, 8:10] = occlusion_label
    occlusion = np.zeros(SHAPE, dtype=bool)
    occlusion[22:24, 14:16, 10:11] = True
    return Case(
        id="synthetic",
        arrays={"cbf": cbf, "cbv": cbf / 10, "mtt": np.full(SHAPE, 4.0, np.float32), "tmax": tmax,
                "infarct": infarct, "occlusion": occlusion, "cow": cow},
        spacing_mm=SPACING,
        affine=np.diag([SPACING, SPACING, SPACING, 1.0]),
        clinical={"case": "synthetic", "Onset to door": 90.0, "Door to imaging": 30.0},
        native_volumes={"infarct_ml": mask_ml(infarct, SPACING), "tmax6_ml": mask_ml(tmax > 6, SPACING)},
    )


def test_synthetic_case_volumes():
    case = synthetic_case()
    assert case.native_volumes["infarct_ml"] == pytest.approx(10 * 12 * 8 / 125)
    assert synthetic_case(infarct_box=None).native_volumes["infarct_ml"] == 0.0


def test_synthetic_case_no_data_region():
    box = (slice(24, 29), slice(8, 20), slice(6, 14))  # half the infarct
    case = synthetic_case(no_data_box=box)
    valid = reliable(case.arrays["tmax"], case.arrays["cbf"])
    infarct = case.arrays["infarct"]
    assert np.count_nonzero(infarct & valid) / np.count_nonzero(infarct) == pytest.approx(0.5)


def test_later_steps_not_built_yet():
    with pytest.raises(NotImplementedError):
        error_analysis.case_features(synthetic_case())
