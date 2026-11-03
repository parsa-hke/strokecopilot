"""Volumes and the occlusion-site rule on synthetic masks. Needs numpy and scipy."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from strokecopilot.contract.schema import VisionOutput  # noqa: E402
from strokecopilot.vision.perfusion import volumes as vol  # noqa: E402

SHAPE = (40, 40, 20)


def _cow(label, where):
    cow = np.zeros(SHAPE, dtype=np.uint8)
    cow[where] = label
    return cow


def _occlusion(where):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[where] = True
    return mask


def test_mask_ml_isotropic_and_anisotropic():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:10, :10, :10] = True  # 1,000 voxels
    assert vol.mask_ml(mask, 1.5) == pytest.approx(3.375)
    assert vol.mask_ml(mask, (1.0, 1.0, 2.0)) == pytest.approx(2.0)
    assert vol.mask_ml(np.zeros(SHAPE, dtype=bool), 1.5) == 0.0


def test_contract_volumes_round_and_validate():
    fields = vol.contract_volumes(18.04, 95.049)
    assert fields == {"core_ml": 18.0, "penumbra_ml": 95.0, "mismatch_ratio": 5.28, "mismatch_volume_ml": 77.0}
    output = {
        "schema_version": "0.1", "case_id": "x", "model_version": "test",
        "haemorrhage": {"present": False, "confidence": 0.1}, "occlusion_site": "unknown", "aspects": None,
        "confidence": {"core": 0.5, "penumbra": 0.5}, "warnings": [], **fields,
    }
    VisionOutput.model_validate(output)


def test_contract_volumes_small_core_and_null():
    assert vol.contract_volumes(0.64, 22.4)["mismatch_ratio"] is None
    assert vol.contract_volumes(None, 22.4) == {"core_ml": None, "penumbra_ml": 22.4, "mismatch_ratio": None, "mismatch_volume_ml": None}


def test_ica_within_limit():
    cow = _cow(4, (slice(10, 20), slice(10, 14), slice(8, 12)))
    occ = _occlusion((slice(21, 23), slice(10, 14), slice(8, 12)))  # 2-3 voxels from the label
    evidence = vol.occlusion_evidence(occ, cow, 1.5)
    assert evidence["nearest_label"] == 4 and evidence["site_share"] == 1.0
    assert evidence["distance_mm"] == pytest.approx(3.75)
    assert vol.occlusion_site(evidence) == ("ICA", "ok")


def test_basilar_too_far_is_unknown():
    cow = _cow(1, (slice(10, 12), slice(10, 12), slice(5, 15)))
    occ = _occlusion((slice(30, 32), slice(10, 12), slice(5, 15)))  # 18+ voxels away
    evidence = vol.occlusion_evidence(occ, cow, 1.5)
    assert evidence["site"] == "basilar"
    assert vol.occlusion_site(evidence) == ("unknown", "too_far")
    assert vol.occlusion_site(evidence, limit_mm=100.0) == ("basilar", "ok")


def test_mca_is_unknown_even_when_overlapping():
    cow = _cow(7, (slice(10, 20), slice(10, 14), slice(8, 12)))
    occ = _occlusion((slice(12, 14), slice(10, 14), slice(8, 12)))
    evidence = vol.occlusion_evidence(occ, cow, 1.5)
    assert evidence["distance_mm"] == 0.0 and evidence["outside_share"] == 0.0
    assert vol.occlusion_site(evidence) == ("unknown", "nearest_MCA")


def test_other_label_is_unknown():
    cow = _cow(11, (slice(10, 20), slice(10, 14), slice(8, 12)))
    occ = _occlusion((slice(12, 14), slice(10, 14), slice(8, 12)))
    assert vol.occlusion_site(vol.occlusion_evidence(occ, cow, 1.5)) == ("unknown", "nearest_other")


def test_split_between_sites_is_ambiguous():
    cow = np.zeros(SHAPE, dtype=np.uint8)
    cow[5:8, 10:14, 8:12] = 4   # ICA on one side
    cow[32:35, 10:14, 8:12] = 7  # MCA on the other
    occ = _occlusion((slice(12, 28), slice(10, 14), slice(8, 12)))  # halfway between
    evidence = vol.occlusion_evidence(occ, cow, 1.5)
    assert 0.25 < evidence["site_share"] < 0.75
    assert vol.occlusion_site(evidence, limit_mm=100.0)[0] == "unknown"


def test_empty_and_missing_inputs():
    cow = _cow(4, (slice(10, 20), slice(10, 14), slice(8, 12)))
    assert vol.occlusion_site(vol.occlusion_evidence(np.zeros(SHAPE, bool), cow, 1.5)) == ("unknown", "empty_mask")
    occ = _occlusion((slice(12, 14), slice(10, 14), slice(8, 12)))
    assert vol.occlusion_site(vol.occlusion_evidence(occ, np.zeros(SHAPE, np.uint8), 1.5)) == ("unknown", "no_labels")
    assert vol.occlusion_site(vol.occlusion_evidence(occ, np.zeros((5, 5, 5), np.uint8), 1.5)) == ("unknown", "shape_mismatch")
