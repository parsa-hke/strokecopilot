"""Report-only read-outs of saved results: the training-log parser, the threshold trade-off and the centre test."""

import numpy as np
import pytest

from strokecopilot.analysis import report_figures as rf

LOG = """\
2026-11-05 15:32:07.802197: Epoch 0
2026-11-05 15:32:07.802300: Current learning rate: 0.01
2026-11-05 15:33:15.571777: train_loss -0.0898
2026-11-05 15:33:15.571954: val_loss -0.1664
2026-11-05 15:33:15.572000: Pseudo dice [np.float32(0.3265)]
2026-11-05 15:33:15.572128: Epoch time: 67.77 s
2026-11-05 15:33:16.367650: Epoch 1
2026-11-05 15:33:53.659729: train_loss -0.1343
2026-11-05 15:33:53.659904: val_loss -0.1542
2026-11-05 15:33:53.659953: Pseudo dice [0.2883]
2026-11-05 15:33:54.408135: Epoch 2
2026-11-05 15:34:30.000000: train_loss -0.2
"""


def test_parser_reads_finished_epochs_and_drops_the_unfinished_one():
    assert rf.parse_training_log(LOG) == [(0, -0.0898, -0.1664, 0.3265), (1, -0.1343, -0.1542, 0.2883)]


def test_resumed_logs_join_and_the_moving_average_follows_nnunet():
    resumed = LOG.replace("Epoch 0", "Epoch 5").replace("Epoch 1", "Epoch 6").replace("Epoch 2", "Epoch 7")
    curve = rf.training_curve([LOG, resumed])
    assert curve["epoch"].tolist() == [0, 1, 5, 6]
    assert curve["pseudo_dice_ema"][0] == 0.3265
    assert abs(curve["pseudo_dice_ema"][1] - (0.9 * 0.3265 + 0.1 * 0.2883)) < 1e-12


def test_curve_summary_finds_the_lowest_validation_window():
    epochs = np.arange(40)
    val = np.where(epochs < 20, -0.1 - 0.01 * epochs, -0.3 + 0.005 * (epochs - 20))  # falls, then creeps back up
    curve = {"epoch": epochs, "train_loss": -0.02 * epochs, "val_loss": val, "pseudo_dice": 0.3 + 0.0 * epochs,
             "pseudo_dice_ema": 0.3 + 0.001 * epochs}
    s = rf.curve_summary(curve, window=4)
    assert s["val_loss_lowest_window"][0] <= 20 <= s["val_loss_lowest_window"][1]
    assert s["val_loss_last_window_mean"] > s["val_loss_lowest_window_mean"]
    assert s["train_loss_last_window_mean"] < s["train_loss_first_window_mean"]
    assert s["pseudo_dice_ema_best_epoch"] == 39


def test_tradeoff_counts_scans_at_or_above_each_threshold():
    sweep = [{"target": 0.95, "threshold_calibrated": 0.3, "cross_fitted": {"sensitivity": 0.94, "specificity": 0.91}},
             {"target": 0.96, "threshold_calibrated": 0.175, "cross_fitted": {"sensitivity": 0.96, "specificity": 0.88}}]
    rows = rf.tradeoff(sweep, [0.05, 0.175, 0.2, 0.3, 0.9], chosen=0.175)
    assert [r["threshold"] for r in rows] == [0.3, 0.175]  # strictest first
    assert [r["isles_flagged"] for r in rows] == [2, 4]
    assert [r["chosen"] for r in rows] == [False, True]
    assert rows[1]["isles_rate"] == 0.8 and rows[1]["isles_rate_ci95"][0] < 0.8 < rows[1]["isles_rate_ci95"][1]


def test_centre_test_is_labelled_post_hoc_and_keeps_no_values():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    row = rf.centre_test({"1": rng.normal(0.18, 0.2, 60), "2": rng.normal(0.29, 0.2, 40)})
    assert row["post_hoc"] is True and "_values" not in row
    assert row["n"] == [60, 40] and row["form"] == "Welch"
    assert row["ci"][0] < row["mean_difference"] < row["ci"][1]
    with pytest.raises(ValueError):
        rf.centre_test({"1": [0.1, 0.2]})
