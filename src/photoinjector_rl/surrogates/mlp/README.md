# emittance_target

First-iteration narrow-scope surrogate + RL. The surrogate is an MLP that maps:

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

This is the **clean** repo: only the code needed to run the full pipeline
(IMPACT sample generation → surrogate → PPO / SHAC / BPTT) for the **hi-fi**
dataset. See the top-level `README.md` for the end-to-end walkthrough.

## Files

| File | Role | Status |
|---|---|---|
| `__init__.py` | Frozen `SETTING_KEYS`, `SETTING_BOUNDS`, `N_INPUT=11`. | working |
| `preprocess.py` | Walk `archives/*/*.h5` → single `processed/*.h5` + `*_norm.json`. Uses `beamphysics.ParticleGroup.norm_emit_4d` for the target. | working |
| `dataset.py` | `EmittanceDataset` (in-memory torch Dataset) + `EmittanceDataModule` (Lightning DataModule, 90/10 random split). | working |
| `model.py` | `EmittanceMLP` (Lightning module). Default arch `11 → 128 → 128 → 128 → 1`, GELU, MSE in normalized log-space. | working |
| `train.py` | Surrogate training entry point. EarlyStopping + ModelCheckpoint, post-training pred-vs-true scatter with R² / MAPE. | working |
| `env.py` | `PhotoinjectorEnv` — Gymnasium env using the trained MLP as the transition fn (for PPO/surrogate). 5-D Δknob action, 6-D obs, 6-D hidden distgen context. | working |
| `impact_env.py` | `ImpactPhotoinjectorEnv` — same interface as `env.py` but each step runs a real Impact-T sim (for PPO/Impact and SHAC/BPTT Impact-eval). | working |
| `diff_env.py` | `DiffPhotoinjectorEnv` — batched, torch-only, **differentiable** wrapper around the surrogate. The harness SHAC/BPTT plug into. | working |
| `callbacks.py` | `RolloutDiagnosticCallback` + `EpisodeMetricsCallback` for the SB3 (PPO) training loop. | working |
| `train_ppo.py` | PPO (Stable-Baselines3) against the surrogate `PhotoinjectorEnv`. | working |
| `train_ppo_impact.py` | PPO fine-tune against the real Impact-T env (`ImpactPhotoinjectorEnv`). | working |
| `train_shac.py` | SHAC driver: wires cfg → `DiffPhotoinjectorEnv` → `diffrl.SHAC`, CSV/TB hooks. | working |
| `train_bptt.py` | BPTT driver: same, for `diffrl.BPTT`. | working |
| `diffrl/utils.py` | `seeding`, `RunningMeanStd`, `CriticDataset`, `AverageMeter`, `TimeReport`, `grad_norm`. | working |
| `diffrl/models.py` | `ActorStochasticMLP`, `ActorDeterministicMLP`, `CriticMLP`. | **OUTLINE — implement by hand** |
| `diffrl/shac.py` | `SHAC` short-horizon actor-critic. | **OUTLINE — implement by hand** |
| `diffrl/bptt.py` | `BPTT` full-episode backprop-through-time. | **OUTLINE — implement by hand** |
| `compare_impact.py`, `compare_n_impact.py`, `compare_impact_all_algos.py` | Evaluate trained policies (PPO `.zip` via `SB3Adapter`, SHAC/BPTT `.pt` via `DiffRLAdapter`) on the real Impact-T env and report paired stats. | working |
| `compare_diff_algos.py` | Surrogate-side benchmark comparing PPO / SHAC / BPTT across seeds with a shared sample budget. Writes `logs/compare_diff_hifi/`. | working |
| `plot_curves_poster.py` | Poster-sized re-plot of the env-step / wall-clock learning curves from a `compare_diff_algos` run (reads its CSVs, no retraining). Writes the `poster/` subdir. | working |

> The three **OUTLINE** files ship as signatures + docstrings + `NotImplementedError`.
> The original NVlabs/DiffRL implementations are stashed (gitignored) under
> `reference/` at the repo root for self-checking. `diffrl/utils.py`, the
> differentiable env, the drivers, and the configs are fully working so there is
> a runnable harness to plug the hand-written algorithm into.

## Workflow

From repo root, inside the `slac-rl` conda env. The hi-fi surrogate dataset and
checkpoint ship preserved, so steps 1–2 are only needed to regenerate.

```bash
# 1. (regenerate-only) preprocess → processed/emittance_target_hifi.h5
python -m photoinjector_rl.surrogates.mlp.preprocess \
    --archives 'archives/train_hifi/*.h5' \
    --out processed/emittance_target_hifi.h5

# 2. (regenerate-only) train surrogate
python -m photoinjector_rl.surrogates.mlp.train \
    --processed processed/emittance_target_hifi.h5 \
    --out-dir trained/emittance_target_hifi \
    --devices 1
# Outputs: trained/emittance_target_hifi/checkpoints/best-*.ckpt, last.ckpt,
#          csv_logs/, val_pred_vs_true.png, final_metrics.json
```

## Inference

```python
from photoinjector_rl.surrogates.mlp.model import EmittanceMLP
from photoinjector_rl.surrogates.mlp.dataset import EmittanceDataset

ds = EmittanceDataset("processed/emittance_target_hifi.h5")
model = EmittanceMLP.load_from_checkpoint(
    "trained/emittance_target_hifi/checkpoints/best-epoch=191-val_loss=0.0060.ckpt")
emit_m2 = model.predict_physical(ds.x[:10])      # -> (10, 1) tensor in m^2
```

## RL training

All three algorithms minimize terminal `norm_emit_4d`. Reward is `-log_emit_norm`
(RL maximizer ⇒ minimizes emittance). The 6-D distgen state is sampled at
`reset()` and is a hidden context the policy must implicitly adapt to.

### PPO (surrogate)

```bash
python -m photoinjector_rl.surrogates.mlp.train_ppo \
    --ckpt trained/emittance_target_hifi/checkpoints/best-epoch=191-val_loss=0.0060.ckpt \
    --norm-json processed/emittance_target_hifi_norm.json \
    --out-dir trained/ppo_emittance_target \
    --total-timesteps 500000 --n-envs 16 --device cpu
# Smoke: add --smoke (one rollout).
```

### PPO (Impact fine-tune)

Warm-start the surrogate-trained policy and refine against Impact-T. Same
action/obs spaces, same z-score, same episode shape — only the transition
function swaps from the MLP to a real simulator run.

```bash
python -m photoinjector_rl.surrogates.mlp.train_ppo_impact \
    --warm-start trained/ppo_emittance_target/eval/best_model.zip \
    --impact-config configs/impact/ImpactT_config.yaml \
    --distgen-input configs/impact/distgen_template.yaml \
    --norm-json processed/emittance_target_hifi_norm.json \
    --out-dir trained/ppo_impact_v1 \
    --total-timesteps 2048 --n-envs 8
# Smoke: add --smoke (one PPO update).
```

The env catches any Impact-T failure in `_forward()`, returns a ~5σ-worse
z-score (large negative reward), bumps `info["failure_count"]`, and continues
so the agent can back off.

### SHAC / BPTT (differentiable, surrogate)

These backprop the reward through the frozen surrogate, so they need the
**differentiable** env (`diff_env.py`). They are surrogate-only by construction
(Impact-T is not differentiable); evaluate the learned policy on Impact below.

```bash
# Smoke both (few epochs):
bash scripts/run_diff_smoke.sh --device cpu --epochs 10

# Or directly:
python -m photoinjector_rl.surrogates.mlp.train_shac \
    --cfg configs/diff_rl/shac_photoinjector.yaml \
    --ckpt trained/emittance_target_hifi/checkpoints/best-epoch=191-val_loss=0.0060.ckpt \
    --norm-json processed/emittance_target_hifi_norm.json \
    --logdir logs/shac --device cuda:0
```

> SHAC/BPTT will raise `NotImplementedError` until you implement `diffrl/shac.py`,
> `diffrl/bptt.py`, and `diffrl/models.py`.

### Evaluate any trained policy on Impact-T

`compare_n_impact.py` loads PPO (`.zip`) and SHAC/BPTT (`.pt`) policies and rolls
them out on the real Impact-T env:

```bash
python -m photoinjector_rl.surrogates.mlp.compare_n_impact \
    --policy ppo=trained/ppo_emittance_target/eval/best_model.zip \
    --policy shac=logs/shac/best_policy.pt \
    --policy bptt=logs/bptt/best_policy.pt \
    --impact-config configs/impact/ImpactT_config.yaml \
    --distgen-input configs/impact/distgen_template.yaml \
    --norm-json processed/emittance_target_hifi_norm.json \
    --seeds 0,1,2,3
```

### Surrogate-side benchmark (PPO vs SHAC vs BPTT)

```bash
python -m photoinjector_rl.surrogates.mlp.compare_diff_algos \
    --ckpt trained/emittance_target_hifi/checkpoints/best-epoch=191-val_loss=0.0060.ckpt \
    --norm-json processed/emittance_target_hifi_norm.json \
    --out-dir logs/compare_diff_hifi \
    --algos ppo,shac,bptt --seeds 0,1,2 --budget 500000 --device cuda:0

# poster-sized learning curves -> logs/compare_diff_hifi/poster/
python -m photoinjector_rl.surrogates.mlp.plot_curves_poster \
    --in-dir logs/compare_diff_hifi --algos ppo,shac,bptt --seeds 0,1,2

# Impact-T head-to-head of the best-per-algo policies -> logs/compare_impact_hifi/
python -m photoinjector_rl.surrogates.mlp.compare_impact_all_algos \
    --compare-dir logs/compare_diff_hifi \
    --impact-config configs/impact/ImpactT_config.yaml \
    --distgen-input configs/impact/distgen_template.yaml \
    --norm-json processed/emittance_target_hifi_norm.json \
    --out logs/compare_impact_hifi/all_algos \
    --n-samples 20 --n-workers 8 --max-steps 64
```

## Testing

```bash
pytest tests/ -v
```

- Env / harness tests run without the trained checkpoint (mock surrogates) and
  should pass immediately.
- `tests/test_diffrl_models.py` is the acceptance spec for your hand-written
  `diffrl/models.py`; it fails with `NotImplementedError` until implemented.
- Regression tests that need a checkpoint skip if it is absent.

## Design choices (locked 2026-05-12)

- **Why an MLP and not the conditional NF planned originally:** the v1 RL agent
  needs a fast, differentiable reward signal. The conditional NF on 6D clouds is
  a bigger infra investment with uncertain payoff. An MLP on scalar emittance
  hits the actual need.
- **Why `norm_emit_4d` and not `sqrt(eps_x * eps_y)`:** the openPMD-beamphysics
  definition was chosen explicitly. The geometric-mean variant is computable from
  `pg.norm_emit_x` and `pg.norm_emit_y` if needed later.
- **Why settings-only (no init moments):** see ablation note above.
- **Why log10 on the target:** emittance is bounded-below by zero and
  right-skewed; log-transforming makes the MSE loss landscape behave.
