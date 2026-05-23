"""Unit tests for the pure helpers in compare_diff_algos.py.

These avoid spinning up subprocesses or real training; they exercise the
parsing / math / IO bits that are most likely to silently break.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import csv

import numpy as np
import pytest
import yaml

from photoinjector_rl.emittance_target import compare_diff_algos as cda


# ---- _max_epochs_for_budget -------------------------------------------------


def test_max_epochs_for_budget(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump({"params": {"config":
                              {"num_actors": 4096, "steps_num": 16}}}, f)
    # 500k / (4096 * 16) = 7.6 → ceil = 8
    assert cda._max_epochs_for_budget(cfg_path, 500_000) == 8
    # 1 / (4096 * 16) = tiny → floor at 1
    assert cda._max_epochs_for_budget(cfg_path, 1) == 1
    # Exact match
    assert cda._max_epochs_for_budget(cfg_path, 4096 * 16) == 1
    assert cda._max_epochs_for_budget(cfg_path, 4096 * 16 + 1) == 2


# ---- _read_diffrl_csv -------------------------------------------------------


def test_read_diffrl_csv_roundtrips(tmp_path):
    p = tmp_path / "lc.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean_episode_loss", "wall_time"])
        w.writerow([100, -1.5, 0.5])
        w.writerow([200, -2.1, 1.2])
    steps, losses, walls = cda._read_diffrl_csv(p)
    assert steps.tolist() == [100, 200]
    np.testing.assert_allclose(losses, [-1.5, -2.1])
    np.testing.assert_allclose(walls, [0.5, 1.2])


# ---- _read_ppo_progress ----------------------------------------------------


def test_read_ppo_progress_skips_rows_without_ep_rew(tmp_path):
    """SB3 progress.csv contains union-of-keys per row; before the first
    rollout completes, `rollout/ep_rew_mean` is empty. We must skip those
    rows rather than crash with ValueError.
    """
    run_dir = tmp_path / "run"
    pp = run_dir / "tb" / "PPO_1"
    pp.mkdir(parents=True)
    csv_path = pp / "progress.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time/total_timesteps", "rollout/ep_rew_mean",
                    "time/time_elapsed"])
        w.writerow([1024, "", "5"])
        w.writerow([2048, "12.5", "10"])
        w.writerow([3072, "", "15"])
        w.writerow([4096, "45.0", "20"])
    result = cda._read_ppo_progress(run_dir)
    assert result is not None
    steps, rewards, walls = result
    assert steps.tolist() == [2048, 4096]
    np.testing.assert_allclose(rewards, [12.5, 45.0])
    np.testing.assert_allclose(walls, [10.0, 20.0])


def test_read_ppo_progress_missing_directory(tmp_path):
    assert cda._read_ppo_progress(tmp_path / "nope") is None


# ---- _aggregate -------------------------------------------------------------


def test_aggregate_uses_unbiased_stderr():
    grid = np.linspace(0, 10, 11)
    curves = [
        (np.array([0.0, 10.0]), np.array([0.0, 10.0])),  # y = x
        (np.array([0.0, 10.0]), np.array([0.0, 20.0])),  # y = 2x
    ]
    mean, stderr = cda._aggregate(curves, grid)
    # At grid = 5: mean = (5 + 10)/2 = 7.5; std_ddof1 over [5, 10] = 3.5355; stderr = 3.5355/sqrt(2) = 2.5
    assert mean[5] == pytest.approx(7.5)
    assert stderr[5] == pytest.approx(2.5)


def test_aggregate_single_seed_stderr_zero():
    grid = np.array([0.0, 1.0, 2.0])
    curves = [(np.array([0.0, 2.0]), np.array([1.0, 3.0]))]
    mean, stderr = cda._aggregate(curves, grid)
    assert np.all(stderr == 0.0)


# ---- _write_summary --------------------------------------------------------


def test_write_summary_handles_empty(tmp_path):
    """An all-failed sweep yields zero rows; the CSV should still be written
    with the header, not raise IndexError.
    """
    cda._write_summary({"shac": []}, tmp_path)
    out = tmp_path / "summary.csv"
    assert out.exists()
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert rows == []  # header only
    with open(out) as f:
        header = f.readline().strip().split(",")
    assert header == cda.SUMMARY_FIELDS


def test_write_summary_handles_none_terminal(tmp_path):
    per_algo = {
        "shac": [
            {"seed": 0, "run_dir": "x", "terminal_emit": None},
            {"seed": 1, "run_dir": "y", "terminal_emit":
             np.array([1e-11, 2e-11, 3e-11])},
        ],
    }
    cda._write_summary(per_algo, tmp_path)
    with open(tmp_path / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["n_rollouts"] == "0"
    assert rows[0]["terminal_emit_median_m2"] == "nan"
    assert rows[1]["n_rollouts"] == "3"
    assert float(rows[1]["terminal_emit_median_m2"]) == pytest.approx(2e-11)


# ---- _plot_results crash safety --------------------------------------------


def test_plot_results_tolerates_failed_eval(tmp_path):
    """If an algo has all-None terminal_emit, plotting must not crash."""
    import matplotlib
    matplotlib.use("Agg")
    per_algo = {
        "shac": [
            {"seed": 0, "run_dir": "x", "terminal_emit": None,
             "learning_curve": (np.array([1, 2]), np.array([-1.0, -2.0]))},
        ],
        "bptt": [
            {"seed": 0, "run_dir": "y",
             "terminal_emit": np.array([1e-11, 2e-11]),
             "learning_curve": (np.array([1, 2]), np.array([-1.5, -2.5]))},
        ],
    }
    cda._plot_results(per_algo, tmp_path)
    assert (tmp_path / "compare.png").exists()
