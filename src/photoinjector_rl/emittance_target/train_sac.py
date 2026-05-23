"""
Train Stable-Baselines3 SAC against the PhotoinjectorEnv surrogate-driven env.

Usage:
    python -m photoinjector_rl.emittance_target.train_sac \\
        --ckpt trained/emittance_target/checkpoints/best-XXX.ckpt \\
        --norm-json processed/emittance_target_norm.json \\
        --out-dir trained/sac_emittance_target \\
        --total-timesteps 100000 \\
        --wandb-project photoinjector-rl

Logging:
    - `--wandb-project NAME` enables Weights & Biases dashboarding (the same
      metrics SB3 normally pumps to TensorBoard). Recommended when SSH'd into
      a remote box -- view the dashboard from your laptop's browser, no port
      forwarding needed.
    - `--wandb-mode offline` keeps everything local (writes to wandb/ dir);
      sync later with `wandb sync wandb/<run>`. Useful for air-gapped hosts.
    - If `--wandb-project` is omitted, only the local TB / CSV logs under
      <out-dir>/tb/ are written. Tail <out-dir>/tb/SAC_*/progress.csv to see
      metrics without any UI at all.

Outputs (under --out-dir):
    sac_final.zip                — SB3 model archive
    eval/                        — EvalCallback per-eval episode logs
    eval/best_model.zip          — best-eval-mean-reward checkpoint
    tb/                          — TensorBoard + CSV logs
    ckpts/                       — periodic save_freq checkpoints
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_latest_run_id

from .callbacks import EpisodeMetricsCallback, RolloutDiagnosticCallback
from .env import PhotoinjectorEnv


def build_env(
    ckpt: str,
    norm_json: str,
    *,
    seed: int | None = None,
    device: str = "cpu",
    terminal_emit_bonus: float = 0.0,
) -> Monitor:
    """Construct the env with default parameters and wrap with Monitor.

    `device` is forwarded to `PhotoinjectorEnv.from_checkpoint` so the
    surrogate lives on the same hardware as SAC's policy / critic networks
    (avoids per-step host<->device tensor transfers).

    `terminal_emit_bonus` weights the terminal-step reward extra (default 0 =
    library default). Pass through to env so train + eval + diag all share
    the same reward shape.
    """
    env = PhotoinjectorEnv.from_checkpoint(
        ckpt_path=ckpt, norm_json=norm_json, device=device,
        terminal_emit_bonus=terminal_emit_bonus,
    )
    if seed is not None:
        env.reset(seed=seed)
    return Monitor(env)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="path to EmittanceMLP checkpoint")
    p.add_argument("--norm-json", required=True, help="path to normalization JSON")
    p.add_argument("--out-dir", default="trained/sac_emittance_target")

    # SAC hyperparameters (SB3 defaults, exposed as CLI flags).
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--ent-coef", default="auto",
                   help="'auto' learns the entropy coefficient; or pass a fixed float")
    p.add_argument("--policy", default="MlpPolicy")

    # Eval + checkpoints.
    p.add_argument("--eval-freq", type=int, default=5_000,
                   help="run an eval rollout every N timesteps")
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--ckpt-freq", type=int, default=20_000,
                   help="save a model checkpoint every N timesteps")
    p.add_argument("--diag-seeds", type=int, nargs="+",
                   default=[9999, 9998, 9997, 9996],
                   help="list of seeds for diagnostic rollouts; one trajectory "
                        "per seed is overlaid in each diag PNG. More seeds "
                        "exposes seed-overfitting failure modes.")

    # Reward shaping.
    p.add_argument("--terminal-bonus", type=float, default=0.0,
                   help="multiplier applied to the final-step reward as an "
                        "additive bonus: r_final += K * (-y_norm). Set >0 to "
                        "weight terminal emittance more heavily. 0 = unchanged.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   help="device for both SAC nets and surrogate. "
                        "'auto' picks cuda:0 if available else cpu. "
                        "Pass 'cpu' or e.g. 'cuda:1' to target a specific GPU.")
    p.add_argument("--smoke", action="store_true",
                   help="tiny training run for smoke-testing the wiring")

    # Weights & Biases logging.
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name. If omitted, wandb is disabled and "
                        "only local TB / CSV logs are written.")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity (team / username). Defaults to your "
                        "default entity.")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name. Default = auto-generated by W&B.")
    p.add_argument("--wandb-mode", default="online",
                   choices=("online", "offline", "disabled"),
                   help="online (push to wandb.ai), offline (local dir, sync "
                        "later with `wandb sync`), or disabled.")
    return p


def resolve_device(device: str) -> str:
    """Translate 'auto' -> 'cuda:0' or 'cpu' depending on availability; pass
    other strings through after a quick sanity check."""
    import torch
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"--device={device} requested but torch.cuda.is_available() is False. "
            "Pass --device cpu to run on CPU."
        )
    if device.startswith("cuda:"):
        idx = int(device.split(":")[1])
        n = torch.cuda.device_count()
        if idx >= n:
            raise SystemExit(
                f"--device={device} requested but only {n} CUDA device(s) visible."
            )
    return device


def train(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval").mkdir(exist_ok=True)
    (out_dir / "tb").mkdir(exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)

    # Smoke mode: shrink everything.
    if args.smoke:
        args.total_timesteps = 1_024
        args.eval_freq = 256
        args.ckpt_freq = 512
        args.n_eval_episodes = 2
        args.learning_starts = 64

    device = resolve_device(args.device)
    print(f"[train_sac] using device: {device}")

    env = build_env(args.ckpt, args.norm_json, seed=args.seed,
                    device=device, terminal_emit_bonus=args.terminal_bonus)
    eval_env = build_env(args.ckpt, args.norm_json, seed=args.seed + 1000,
                         device=device, terminal_emit_bonus=args.terminal_bonus)
    # Separate env for diagnostic rollouts so EvalCallback's RNG isn't perturbed.
    diag_env = build_env(args.ckpt, args.norm_json, seed=args.diag_seeds[0],
                         device=device, terminal_emit_bonus=args.terminal_bonus)

    # IMPORTANT: wandb.init() MUST run before any SummaryWriter is instantiated
    # -- it monkey-patches torch.utils.tensorboard.SummaryWriter at init time so
    # that subsequent TB writes are mirrored to the W&B run. If you create a
    # logger first (via SB3 or configure_logger), wandb never sees the writes.
    wandb_run = None
    WandbCallback = None
    if args.wandb_project:
        import wandb
        from wandb.integration.sb3 import WandbCallback as _WandbCallback
        WandbCallback = _WandbCallback

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
            sync_tensorboard=True,
            dir=str(out_dir),
            save_code=False,
        )
        print(f"[train_sac] wandb run: {wandb_run.url if wandb_run else '(local)'}")

    # SB3 SAC. Most kwargs are at their library defaults; exposed via CLI so
    # they can be tuned without code edits.
    ent_coef = args.ent_coef
    if ent_coef not in ("auto",) and isinstance(ent_coef, str):
        try:
            ent_coef = float(ent_coef)
        except ValueError:
            pass  # leave as-is (e.g. 'auto_0.1' is also valid)

    tb_dir = str(out_dir / "tb")
    model = SAC(
        policy=args.policy,
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        learning_starts=args.learning_starts,
        ent_coef=ent_coef,
        seed=args.seed,
        device=device,
        tensorboard_log=tb_dir,
        verbose=1,
    )

    # SB3 by default only emits ["stdout", "tensorboard"]. Add CSV explicitly
    # so users without a UI can `tail -f <run_dir>/progress.csv`. The
    # SummaryWriter created here IS the one wandb shimmed above (if W&B is
    # enabled), so its writes will mirror to W&B in real time.
    next_id = get_latest_run_id(tb_dir, "SAC") + 1
    run_dir = os.path.join(tb_dir, f"SAC_{next_id}")
    model.set_logger(configure_logger(
        run_dir, ["stdout", "tensorboard", "csv"],
    ))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "eval"),
        log_path=str(out_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=args.ckpt_freq,
        save_path=str(out_dir / "ckpts"),
        name_prefix="sac",
    )
    diag_cb = RolloutDiagnosticCallback(
        diag_env=diag_env,
        eval_freq=args.eval_freq,
        out_dir=out_dir / "rollouts",
        diag_seeds=args.diag_seeds,
        wandb_run=wandb_run,
        verbose=1,
    )
    metrics_cb = EpisodeMetricsCallback()
    callbacks: list = [eval_cb, ckpt_cb, diag_cb, metrics_cb]

    if WandbCallback is not None:
        callbacks.append(WandbCallback(
            model_save_freq=args.ckpt_freq,
            model_save_path=str(out_dir / "wandb_models"),
            log="all",
            verbose=1,
        ))

    try:
        model.learn(total_timesteps=args.total_timesteps,
                    callback=callbacks,
                    progress_bar=False)
        model.save(out_dir / "sac_final")
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    result = {
        "final_path": str(out_dir / "sac_final.zip"),
        "best_eval_path": str(out_dir / "eval" / "best_model.zip"),
        "tensorboard": str(out_dir / "tb"),
        "total_timesteps": args.total_timesteps,
    }
    if wandb_run is not None:
        result["wandb_url"] = wandb_run.url if hasattr(wandb_run, "url") else None
    return result


def main() -> None:
    args = build_argparser().parse_args()
    result = train(args)
    print("\nDone.")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
