"""What 00_template.ipynb shows for one case: the maps at the largest infarct slice, and the note-safe clinical row.

Kept here, not in the notebook, so the notebook stays thin and the figure has a test.
"""

import numpy as np

from strokecopilot.data import isles24

# Display windows: NCCT in HU, CBF and CBV in their source units, MTT and Tmax in seconds
PANELS = (("ncct", 0, 80), ("cbf", None, None), ("cbv", None, None), ("mtt", 0, 15), ("tmax", 0, 12))


def largest_infarct_slice(case):
    """The axial slice with the most final-infarct voxels, or the middle slice when the mask is empty."""
    lesion = case.arrays["infarct"]
    counts = lesion.sum(axis=(0, 1))
    return int(counts.argmax()) if counts.any() else lesion.shape[2] // 2


def _window(array, low, high):
    if low is None:
        finite = array[np.isfinite(array) & (array > 0)]
        low, high = (0.0, float(np.percentile(finite, 99))) if finite.size else (0.0, 1.0)
    return low, high


def case_figure(case, z=None):
    """A matplotlib figure: one panel per map at slice z, with the final infarct outlined in red and the occlusion in yellow."""
    import matplotlib.pyplot as plt

    z = largest_infarct_slice(case) if z is None else z
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.2 * len(PANELS), 3.6))
    for ax, (name, low, high) in zip(axes, PANELS):
        image = case.arrays[name][:, :, z]
        low, high = _window(case.arrays[name], low, high)
        ax.imshow(image.T, cmap="gray", vmin=low, vmax=high, origin="lower")
        for mask, colour in (("infarct", "red"), ("occlusion", "yellow")):
            if case.arrays[mask][:, :, z].any():
                ax.contour(case.arrays[mask][:, :, z].T, levels=[0.5], colors=colour, linewidths=0.8)
        ax.set_title(name.upper(), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{case.id}, slice {z}: final infarct in red, occlusion in yellow", fontsize=10)
    fig.tight_layout()
    return fig


def note_row(case):
    """The clinical values a triage note may use (plan §8.3), with missing values shown as 'not recorded'."""
    return {column: ("not recorded" if value is None else value) for column, value in case.note_fields.items()}


def summary(case):
    """Short text about the case: spacing, shape and the reference volumes measured before resampling."""
    shape = case.arrays["tmax"].shape
    volumes = ", ".join(f"{k} {v:.1f} mL" for k, v in sorted(case.native_volumes.items()))
    return f"{case.id}: grid {shape[0]}x{shape[1]}x{shape[2]} at {case.spacing_mm} mm; reference volumes: {volumes}."


def load_first_development_case(root, folds=None):
    """The first development case in the tarball (never a held-out one)."""
    cases = isles24.list_cases(root, folds)
    return isles24.load_case(cases[0], root, folds=folds)
