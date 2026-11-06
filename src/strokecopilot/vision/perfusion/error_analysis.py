"""Vision A error analysis: where and why the infarct model fails (plan §7 Phase 3, §8.1; brief docs/team/lorna_vision_a.md).

Built one step at a time. This file is the skeleton: each function below says which step fills it in,
and raises NotImplementedError until then.

1. Skeleton and synthetic tests (this file, tests/test_error_analysis.py).
2. join_tables(): line up the nnU-Net (EXP-14) and threshold (EXP-1) per-case tables by case.
3. case_features(): per-case possible causes of failure, from a loaded Case.
4. (Colab) run case_features() over the development cases and join it to the step-2 table.
5. group_summary(): scores per group of a feature, as counts and summaries.
6. (Colab, then docs) images of failed cases and the write-up.

Outputs that leave Drive carry counts and summaries only, never case IDs (docs/team/README.md §3).
Volume error matters more than Dice, because the treatment rules are volume cut-offs
(evaluate.DECISION_CUTOFFS_ML).
"""

from strokecopilot.vision.perfusion.evaluate import DECISION_CUTOFFS_ML  # noqa: F401  (used from step 2)


def join_tables(nnunet, threshold):
    """Step 2. One row per case with both models' scores side by side, plus where they disagree.

    nnunet, threshold: the per_case.csv tables of EXP-14 and EXP-1 as pandas DataFrames.
    """
    raise NotImplementedError("Step 2")


def case_features(case):
    """Step 3. Possible causes of failure for one loaded isles24.Case, as a flat dict.

    Planned: reference infarct volume, territory (anterior or posterior), occlusion site
    (volumes.occlusion_site), onset to imaging in minutes, share of the brain and of the infarct
    with no reliable Tmax (thresholds.reliable), and simple artefact signals.
    """
    raise NotImplementedError("Step 3")


def group_summary(table, by):
    """Step 5. Cases, Dice and absolute volume error per group of column `by`. No case IDs."""
    raise NotImplementedError("Step 5")
