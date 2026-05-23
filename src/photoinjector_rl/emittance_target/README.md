# emittance_target

First-iteration narrow-scope surrogate. MLP that maps:

```
input (11-D)  = the 11 XOPT-sampled knobs (5 Impact + 6 distgen)

output (1-D)  = log10(norm_emit_4d at PR10241) z-scored
              -- norm_emit_4d = sqrt(det(Sigma_4d)) / (mc^2)^2  in m^2
              -- openPMD-beamphysics convention
```

This is the v1 surrogate. Future variants (full 6D phase-space flow, Sigma-matrix
multi-output, sequence rollout) get their own sibling subpackages.

> **Ablation note (2026-05-12):** An earlier draft also concatenated a 27-D
> summary of the realized initial-bunch moments onto the input (38-D total).
> A full-dataset training run found the moments to be **net-harmful**
> (R² 0.99→0.95, MAPE 2.5%→6%): the moments are noisy single-realization
> estimates from 2000 macroparticles, and they add aleatoric noise about a
> value the settings already encode. The moments code path was removed.
> See `memory/project_emittance_target_v1.md` for the full story.

## Files

| File | Role |
|---|---|
| `__init__.py` | Frozen `SETTING_KEYS`, `SETTING_BOUNDS`, `N_INPUT=11`. |
| `preprocess.py` | Walk `archives/*/*.h5` → single `processed/emittance_target.h5` + `*_norm.json`. Uses `beamphysics.ParticleGroup.norm_emit_4d` for the target. |
| `plot_distributions.py` | Pre-flight 14-panel grid (11 knobs + 3 emittances). Red dashed lines mark configured LHS bounds. |
| `dataset.py` | `EmittanceDataset` (in-memory torch Dataset) + `EmittanceDataModule` (Lightning DataModule, 90/10 random split). |
| `model.py` | `EmittanceMLP` (Lightning module). Default arch: `11 → 128 → 128 → 128 → 1`, GELU, MSE in normalized log-space. MAE in m² logged for human eyes. |
| `train.py` | End-to-end entry point. EarlyStopping + ModelCheckpoint, post-training pred-vs-true scatter with R² / MAPE. |
| `env.py` | `PhotoinjectorEnv` — a Gymnasium env that uses the trained MLP as the transition function. 5-D Δknob action space, 6-D obs (5 knobs + z-scored log-emit), 6-D distgen hidden context with optional drift. |
| `train_sac.py` | Stable-Baselines3 SAC against `PhotoinjectorEnv`. CLI flags for SAC hyperparameters, device selection, W&B logging, and reward shaping. |
| `callbacks.py` | `RolloutDiagnosticCallback` (multi-seed deterministic rollout plots per eval cadence), `EpisodeMetricsCallback` (per-episode terminal emit / min emit / action magnitude / knob-saturation rate to TB+CSV+W&B). |
| `policy_scatter.py` | Standalone analysis: runs N rollouts of a trained policy across random distgen seeds, plots terminal emit vs each of the 6 hidden distgen knobs, plus terminal-emit histogram and init-vs-terminal scatter. |

## Workflow

From repo root, inside the `slac-rl` conda env:

```bash
# 1. (one-time-per-sweep) sanity plot — confirm LHS coverage is uniform.
python -m photoinjector_rl.emittance_target.plot_distributions \
    --archives 'archives/train/*.h5' \
    --out plots/emittance_target/distributions.png

# 2. (one-time-per-sweep) preprocess → processed/emittance_target.h5
python -m photoinjector_rl.emittance_target.preprocess \
    --archives 'archives/train/*.h5' \
    --out processed/emittance_target.h5

# 3. train
python -m photoinjector_rl.emittance_target.train \
    --processed processed/emittance_target.h5 \
    --out-dir trained/emittance_target \
    --max-epochs 200 --batch-size 256

# Outputs:
#   trained/emittance_target/checkpoints/best-*.ckpt   (best val_loss)
#   trained/emittance_target/checkpoints/last.ckpt
#   trained/emittance_target/csv_logs/                 (per-epoch metrics)
#   trained/emittance_target/val_pred_vs_true.png      (R² + MAPE in title)
#   trained/emittance_target/final_metrics.json
```

## Inference

```python
from photoinjector_rl.emittance_target.model import EmittanceMLP
from photoinjector_rl.emittance_target.dataset import EmittanceDataset

ds = EmittanceDataset("processed/emittance_target.h5")
model = EmittanceMLP.load_from_checkpoint("trained/emittance_target/checkpoints/best-...ckpt")
emit_m2 = model.predict_physical(ds.x[:10])      # -> (10, 1) tensor in m^2
```

## Gym environment

For RL training, `env.py` provides a Gymnasium-compatible env that uses the
trained MLP as the transition function:

```python
from photoinjector_rl.emittance_target.env import PhotoinjectorEnv

env = PhotoinjectorEnv.from_checkpoint(
    ckpt_path="trained/emittance_target/checkpoints/best-...ckpt",
    norm_json="processed/emittance_target_norm.json",
    max_steps=64,                # episode length
    action_scale=0.05,           # Δknob per step (5% of full range)
    distgen_drift_std=0.0,       # >0 enables per-step Gaussian random walk on the hidden distgen state
)

obs, info = env.reset(seed=0)
# obs: (6,) = [5 knobs in [0,1], z-scored log-emit]
# info: {"emit_m2", "log_emit_norm", "distgen_norm", "knobs_phys", "step_count"}

for _ in range(64):
    action = env.action_space.sample()          # (5,) in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    if truncated:
        break
```

Reward is `-log_emit_norm` (RL maximizer ⇒ minimizes emittance). The 6-D
distgen state is sampled uniformly at `reset()` and is NOT part of the
observation — it's a hidden context the policy must implicitly adapt to.

## RL training (SAC)

```bash
python -m photoinjector_rl.emittance_target.train_sac \
    --ckpt trained/emittance_target/checkpoints/best-XXX.ckpt \
    --norm-json processed/emittance_target_norm.json \
    --out-dir trained/sac_emittance_target \
    --total-timesteps 100000 \
    --wandb-project photoinjector-rl

# Smoke (~1k timesteps, ~2 min):
python -m photoinjector_rl.emittance_target.train_sac \
    --ckpt ... --norm-json ... --smoke

# Pick a specific GPU (surrogate + SAC nets both move there):
python -m photoinjector_rl.emittance_target.train_sac \
    --ckpt ... --norm-json ... --device cuda:1
# Use --device auto (default) to pick cuda:0 if available else cpu.
```

### Logging

Pick one of three modes:

| Mode | Flag | What you get | When to use |
|---|---|---|---|
| **Weights & Biases** | `--wandb-project NAME` | Cloud dashboard at wandb.ai, viewable from any browser. | SSH'd into a remote box and don't want to port-forward. **Recommended.** |
| **W&B offline** | `--wandb-project NAME --wandb-mode offline` | Local dir under `<out-dir>/wandb/`. Sync later with `wandb sync wandb/<run>`. | Air-gapped host or no outbound network. |
| **No UI** | (omit `--wandb-project`) | Plain TB binary + CSV under `<out-dir>/tb/SAC_*/progress.csv`. | `tail -f` the CSV over SSH. |

First-time wandb users: run `wandb login` once, paste the API key from
https://wandb.ai/authorize. The SAC scalars (rollout/ep_rew_mean, train/actor_loss,
train/critic_loss, train/ent_coef, eval/mean_reward, etc.) all sync because we
set `sync_tensorboard=True`.

Outputs land under `--out-dir`:
- `sac_final.zip` — final SB3 model
- `eval/best_model.zip` — best-eval-mean-reward checkpoint
- `tb/` — TensorBoard + CSV logs (mirrored to W&B when enabled).
  Includes the standard SB3 scalars plus four per-episode metrics:
  `rollout/terminal_emit_m2`, `rollout/min_emit_m2`,
  `rollout/action_mean_abs`, `rollout/saturation_rate`.
- `ckpts/` — periodic save_freq checkpoints
- `rollouts/` — multi-seed diagnostic PNGs per eval cadence
  (one trajectory per `--diag-seeds`, default 4 seeds)
- `wandb/` — W&B run artifacts (only with `--wandb-project`)
- `wandb_models/` — W&B model snapshots (only with `--wandb-project`)

### Reward shaping

`--terminal-bonus K` adds `K * (-y_norm)` to the final step's reward
(default 0 = library default per-step reward). Useful when a previous run
showed the policy learning to "transit through good regions" without
settling there — bumping K to 5–20 makes the agent care explicitly about
the endpoint.

### Policy analysis

After training, generate the policy-vs-distgen scatter:

```bash
python -m photoinjector_rl.emittance_target.policy_scatter \
    --algo sac \
    --policy trained/sac_emittance_target/eval/best_model.zip \
    --ckpt trained/emittance_target/checkpoints/best-XXX.ckpt \
    --norm-json processed/emittance_target_norm.json \
    --out plots/policy_scatter.png \
    --n-samples 200 \
    --device cuda:0
```

6 panels of terminal emit vs each hidden distgen knob (color = improvement
ratio), plus terminal-emit histogram and init-vs-terminal scatter. The raw
data is also dumped to `plots/policy_scatter.npz` for downstream analysis.

Pass `--algo ppo` instead to analyze a PPO checkpoint.

## PPO baseline (ablation)

On-policy alternative to SAC, useful as a sanity check that the result
isn't algorithm-specific. PPO is sample-hungrier than SAC but the surrogate
is cheap, so we make up the gap with parallel workers (`--n-envs 16`).
Default config = 500k timesteps, ~5x SAC's sample budget.

```bash
# Smoke (4096 timesteps = one rollout):
python -m photoinjector_rl.emittance_target.train_ppo \
    --ckpt trained/emittance_target/checkpoints/best-XXX.ckpt \
    --norm-json processed/emittance_target_norm.json \
    --smoke

# Full run:
python -m photoinjector_rl.emittance_target.train_ppo \
    --ckpt ... --norm-json ... \
    --out-dir trained/ppo_v1 \
    --total-timesteps 500000 \
    --n-envs 16 \
    --device cuda:0 \
    --wandb-project photoinjector-rl

# Compare against SAC v3 using the same policy_scatter script:
python -m photoinjector_rl.emittance_target.policy_scatter \
    --algo ppo \
    --policy trained/ppo_v1/eval/best_model.zip \
    --ckpt ... --norm-json ... \
    --out trained/ppo_v1/policy_scatter.png \
    --n-samples 200 --device cuda:0
```

Device handling: `--device` controls PPO's policy/value nets. The surrogate
inside each SubprocVecEnv worker is always loaded on CPU (sub-ms inference,
avoids spinning up 16 CUDA contexts for a tiny MLP). **For a small MLP
policy like ours, SB3 recommends running PPO itself on CPU as well** — the
batch is too small to amortize GPU transfer overhead. Use `--device cpu`.

### Fine-tune on Impact-T (warm-start from surrogate policy)

Take the surrogate-trained PPO policy and refine it directly against
Impact-T. Same action/obs spaces, same z-score, same episode shape —
only the transition function swaps from the MLP to a real simulator run.
Budget is small (~2k Impact-T evals total ≈ 45 min wall-clock on 8 CPU
workers).

```bash
# Smoke (1 PPO update, ~5 min):
python -m photoinjector_rl.emittance_target.train_ppo_impact \
    --warm-start trained/ppo_v1/eval/best_model.zip \
    --impact-config configs/impact/ImpactT_config.yaml \
    --distgen-input configs/impact/distgen_template.yaml \
    --norm-json processed/emittance_target_norm.json \
    --smoke

# Full fine-tune:
python -m photoinjector_rl.emittance_target.train_ppo_impact \
    --warm-start trained/ppo_v1/eval/best_model.zip \
    --impact-config configs/impact/ImpactT_config.yaml \
    --distgen-input configs/impact/distgen_template.yaml \
    --norm-json processed/emittance_target_norm.json \
    --out-dir trained/ppo_impact_v1 \
    --total-timesteps 2048 \
    --n-envs 8 \
    --wandb-project photoinjector-rl
```

Key design choices (locked 2026-05-14):

- **Same normalization JSON as surrogate.** Reward distribution stays
  in-distribution, so the loaded value function (trained on the surrogate)
  remains useful for the fine-tune update.
- **Conservative hyperparameters.** `--learning-rate 1e-4` (vs 3e-4 from
  scratch), `--clip-range 0.1` (vs 0.2). We're nudging, not retraining.
- **No Xopt layer.** The env calls `custom_evaluate_impact_with_distgen`
  directly. Xopt's value-add is in optimization generators (LHS, CNSGA);
  we don't need that here — we need episode-structured rollouts, which
  SubprocVecEnv already gives us.
- **Failure handling.** Impact-T can numerically fail at extreme knob
  settings. The env catches any exception in `_forward()`, returns a
  ~5σ-worse z-score (large negative reward), bumps `info["failure_count"]`,
  and lets the episode continue so the agent can back off.

### Paired comparison (compare_algos)

Run two policies on the *same* distgen seeds and report paired stats:

```bash
python -m photoinjector_rl.emittance_target.compare_algos \
    --algo-a sac --policy-a trained/sac_v3/eval/best_model.zip \
    --algo-b ppo --policy-b trained/ppo_v1/eval/best_model.zip \
    --ckpt trained/emittance_target/checkpoints/best-XXX.ckpt \
    --norm-json processed/emittance_target_norm.json \
    --out trained/compare_sac_vs_ppo.png \
    --n-samples 200 --device cuda:0
```

Outputs a 4-panel PNG (terminal-emit histograms, paired scatter, log-ratio
histogram, ratio CDFs) and a `.npz` with the full paired data. Prints a
per-metric table including head-to-head win rate.

## Testing

```bash
pytest tests/ -v
```

- Unit tests run without the trained checkpoint (use mock surrogates).
- Regression tests skip if `trained/emittance_target/checkpoints/best-*.ckpt`
  or `processed/emittance_target_norm.json` are missing. Their golden values
  are checkpoint-specific — update via `/tmp/compute_goldens.py` (or any
  equivalent script) after retraining.

## Design choices (locked 2026-05-12)

- **Why an MLP and not the conditional NF planned originally:** the v1 RL agent
  needs a fast, differentiable reward signal. The conditional NF on 6D clouds is
  a bigger infra investment with uncertain payoff. An MLP on scalar emittance
  hits the actual need.
- **Why `norm_emit_4d` and not `sqrt(eps_x * eps_y)`:** the user picked the
  openPMD-beamphysics definition explicitly. The geometric-mean variant is also
  computable from `pg.norm_emit_x` and `pg.norm_emit_y` if needed later.
- **Why settings-only (no init moments):** see ablation note above.
- **Why log10 on the target:** emittance is bounded-below by zero and
  right-skewed; log-transforming makes the MSE loss landscape behave.
