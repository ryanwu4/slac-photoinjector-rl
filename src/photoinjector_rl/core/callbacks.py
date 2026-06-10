"""
Custom SB3 callbacks for the photoinjector PPO training loops (shared).
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend (this box is SSH'd, no display)
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from photoinjector_rl.core.settings import SETTING_BOUNDS, SETTING_KEYS

# 5 controllable knobs vs. 6 hidden distgen knobs (canonical SETTING_KEYS order).
_KNOB_KEYS = SETTING_KEYS[:5]
_DISTGEN_KEYS = SETTING_KEYS[5:]
_KNOB_LO = np.array([SETTING_BOUNDS[k][0] for k in _KNOB_KEYS], dtype=np.float32)
_KNOB_HI = np.array([SETTING_BOUNDS[k][1] for k in _KNOB_KEYS], dtype=np.float32)
_KNOB_RANGE = _KNOB_HI - _KNOB_LO


def _short(key: str) -> str:
    """Compact axis label for a XOPT key."""
    return key.replace("distgen:", "").replace(":value", "")


def _is_saturated(knobs_phys: np.ndarray, eps_frac: float = 0.01) -> bool:
    """True if any knob is within eps_frac of its physical bound."""
    eps = eps_frac * _KNOB_RANGE
    return bool(
        np.any(knobs_phys <= _KNOB_LO + eps)
        or np.any(knobs_phys >= _KNOB_HI - eps)
    )


# ---------------------------------------------------------------------------
# Multi-seed rollout diagnostic plot
# ---------------------------------------------------------------------------


class RolloutDiagnosticCallback(BaseCallback):
    """At each eval cadence, run N deterministic rollouts -- one per seed in
    `diag_seeds` -- and render a 6-panel figure with all seeds overlaid:

        - top:    norm_emit_4d trajectory per seed (log-y)
        - 1..5:   physical knob trajectories per seed (one line per seed)
        - legend identifies seeds; suptitle shows the timestep.

    Pinning the seed list means every diagnostic plot starts from the same
    cathode realizations, so you can read policy improvement directly off a
    stack of these plots. Plotting multiple seeds together flags
    seed-overfitting: if one seed's trajectory descends but others don't,
    that's a generalization failure.

    If a `wandb_run` is provided, the rendered PNG is also uploaded to W&B
    under `diag/rollout` (step-keyed).
    """

    def __init__(
        self,
        diag_env,
        eval_freq: int,
        out_dir: str | Path,
        diag_seeds: Sequence[int] = (9999,),
        wandb_run=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.diag_env = diag_env
        self.eval_freq = int(eval_freq)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.diag_seeds = tuple(int(s) for s in diag_seeds)
        self.wandb_run = wandb_run

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.render_rollout()
        return True

    def _rollout_one(self, seed: int) -> dict:
        env = self.diag_env
        obs, info = env.reset(seed=seed)
        knobs_traj = [info["knobs_phys"].copy()]
        emit_traj = [info["emit_m2"]]
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            knobs_traj.append(info["knobs_phys"].copy())
            emit_traj.append(info["emit_m2"])
            done = bool(term or trunc)
        return {
            "seed": seed,
            "knobs": np.stack(knobs_traj),  # (T+1, 5)
            "emit": np.array(emit_traj),    # (T+1,)
        }

    def render_rollout(self) -> Path:
        rollouts = [self._rollout_one(s) for s in self.diag_seeds]
        n_seeds = len(rollouts)
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0.05, 0.85, n_seeds))

        fig, axes = plt.subplots(6, 1, figsize=(7.5, 9.5), sharex=True)

        # Emittance.
        for r, c in zip(rollouts, colors):
            axes[0].plot(np.arange(len(r["emit"])), r["emit"],
                         color=c, linewidth=1.3, label=f"seed={r['seed']}")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("norm_emit_4d\n(m²)", fontsize=9)
        axes[0].grid(True, which="both", linestyle=":", alpha=0.4)
        axes[0].legend(loc="upper right", fontsize=7, ncol=min(n_seeds, 4))

        # Knob trajectories with physical bounds.
        for i, key in enumerate(_KNOB_KEYS):
            ax = axes[i + 1]
            for r, c in zip(rollouts, colors):
                ax.plot(np.arange(len(r["emit"])), r["knobs"][:, i],
                        color=c, linewidth=1.0)
            lo, hi = SETTING_BOUNDS[key]
            ax.axhline(lo, color="crimson", linestyle="--",
                       linewidth=0.6, alpha=0.5)
            ax.axhline(hi, color="crimson", linestyle="--",
                       linewidth=0.6, alpha=0.5)
            ax.set_ylabel(_short(key), fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.4)
        axes[-1].set_xlabel("env step")

        # Summary line: terminal emit per seed.
        term_emits = ", ".join(
            f"{r['seed']}: {r['emit'][-1]:.2e}" for r in rollouts
        )
        fig.suptitle(
            f"diag rollouts @ {self.num_timesteps} timesteps  "
            f"({n_seeds} seeds, deterministic policy)\n"
            f"terminal emit m²:  {term_emits}",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        out_path = self.out_dir / f"rollout_step_{self.num_timesteps:08d}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        if self.verbose:
            mins = [float(r["emit"].min()) for r in rollouts]
            print(f"[diag] step={self.num_timesteps}  "
                  f"min_emit per seed: {[f'{m:.3e}' for m in mins]}  -> {out_path}")

        if self.wandb_run is not None:
            import wandb
            self.wandb_run.log(
                {"diag/rollout": wandb.Image(str(out_path))},
                step=self.num_timesteps,
            )

        return out_path


# ---------------------------------------------------------------------------
# Per-episode metrics callback (logs to SB3 logger -> TB/CSV/W&B)
# ---------------------------------------------------------------------------


class EpisodeMetricsCallback(BaseCallback):
    """Records useful per-episode summaries (not exposed by SB3 by default):

        rollout/terminal_emit_m2   norm_emit_4d at the final step
        rollout/min_emit_m2        best norm_emit_4d hit during the episode
        rollout/action_mean_abs    mean |action_t| over the episode
        rollout/saturation_rate    fraction of steps where ANY knob was
                                   within 1% of its physical bound

    Values land in TB, progress.csv, and W&B via the standard SB3 logger
    path. Useful for diagnosing the failure modes the agent flagged in the
    first run (knob saturation, action zigzag, terminal vs trajectory
    reward disagreement).
    """

    def __init__(self, sat_eps_frac: float = 0.01, verbose: int = 0):
        super().__init__(verbose)
        self.sat_eps_frac = float(sat_eps_frac)
        # Per-env buffers keyed by env_idx so a VecEnv with n_envs>1 doesn't
        # commingle transitions across parallel rollouts.
        self._emits: dict[int, list[float]] = {}
        self._actions: dict[int, list[float]] = {}
        self._sat_steps: dict[int, list[int]] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        actions = self.locals.get("actions", None)

        for env_idx, info in enumerate(infos):
            # Skip Monitor's reset-only info (no emit_m2 yet).
            if "emit_m2" not in info:
                continue

            emits = self._emits.setdefault(env_idx, [])
            acts = self._actions.setdefault(env_idx, [])
            sats = self._sat_steps.setdefault(env_idx, [])

            emits.append(float(info["emit_m2"]))
            if actions is not None:
                a = np.asarray(actions[env_idx]).reshape(-1)
                acts.append(float(np.mean(np.abs(a))))
            sats.append(
                1 if _is_saturated(info["knobs_phys"], self.sat_eps_frac) else 0
            )

            # Monitor adds an "episode" key on truncation / termination.
            if "episode" in info:
                ep_len = max(len(emits), 1)
                self.logger.record("rollout/terminal_emit_m2", emits[-1])
                self.logger.record("rollout/min_emit_m2", float(min(emits)))
                self.logger.record(
                    "rollout/action_mean_abs",
                    float(np.mean(acts)) if acts else 0.0,
                )
                self.logger.record(
                    "rollout/saturation_rate",
                    float(sum(sats)) / ep_len,
                )
                emits.clear()
                acts.clear()
                sats.clear()

        return True
