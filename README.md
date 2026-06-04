# photoinjector-rl

Reinforcement-learning pipeline for tuning the **FACET-II photoinjector**. The
agent steers the gun/solenoid/quadrupole knobs to shape the electron bunch at
the **PR10241** screen — minimizing 4D normalized emittance, or tracking a
commanded transverse **(aspect, tilt)** shape — while learning against a fast,
differentiable surrogate of the Impact-T beam dynamics simulator and then being
evaluated/fine-tuned on the real simulator.

The core is a **3-stage pipeline**:

```
  (1) Impact-T LHS sweep            (2) Surrogate models             (3) RL on the surrogate          eval / fine-tune
  ---------------------             --------------------             ---------------------            ----------------
  ImpactT_PR10241.in + distgen      EmittanceMLP (scalar)            SHAC / BPTT (first-order MBRL)    real Impact-T
  11-D LHS (xopt) over knobs   -->  11 knobs -> log10 emit_4d   -->  PPO (model-free baseline)    --> compare_n_impact
  archives/train   (lofi 2k)        ConditionalAffineFlow            tune 5 control knobs vs a         impact_eval_tracking
  archives/train_hifi (20k)         11 knobs -> output bunch (P,6)   differentiable bunch property    (sim-to-sim transfer)
```

Two surrogate flavors share one knob convention:

* **`emittance_target/`** — a scalar `EmittanceMLP`: `11 knobs -> log10(norm_emit_4d)`. Fast, deterministic, the v1 reward.
* **`flow_surrogate/`** — a **conditional affine-coupling normalizing flow**: `11 knobs -> a full PR10241 cloud (P,6)`. Any beam statistic is computed from the sampled cloud and, via the reparameterization trick, its gradient flows back to the knobs — the substrate for first-order MBRL and the goal-conditioned shape controller.

RL is done with **SHAC** and **BPTT** (first-order, backprop the reward through
the frozen surrogate) plus a model-free **PPO** baseline (Stable-Baselines3).
Across the emittance and shape-tracking tasks, **BPTT and SHAC tie or beat PPO**;
SHAC needs `target_critic_alpha=0.9–0.95` to stay stable on the stochastic flow
reward (see `flow_surrogate/README.md` and the notes below).

---

## Repo layout

```
photoinjector-rl-clean/
├── pyproject.toml                  package `photoinjector-rl`; deps + extras
├── README.md                       (this file)
├── src/photoinjector_rl/
│   ├── data/                       Impact-T + distgen runner (xopt evaluator
│   │                               `custom_evaluate_impact_with_distgen`,
│   │                               archiving, fingerprinting)
│   ├── emittance_target/           scalar-emittance pipeline (the v1 surrogate)
│   │   ├── __init__.py             AUTHORITATIVE SETTING_KEYS / SETTING_BOUNDS / N_INPUT=11
│   │   ├── preprocess.py train.py  archives -> processed h5 -> EmittanceMLP
│   │   ├── model.py dataset.py     Lightning MLP (11 -> 128³ -> 1) + DataModule
│   │   ├── env.py impact_env.py    Gymnasium envs (surrogate / real Impact-T)
│   │   ├── diff_env.py vec_env.py  differentiable + vectorized env wrappers
│   │   ├── train_{ppo,shac,bptt}.py  RL drivers (surrogate)
│   │   ├── train_ppo_impact.py     PPO fine-tune on real Impact-T
│   │   ├── compare_{impact,n_impact,impact_all_algos}.py  Impact-T eval of policies
│   │   ├── compare_diff_algos.py   surrogate-side PPO/SHAC/BPTT benchmark
│   │   ├── eval_drift_sweep*.py    eval-time distgen-jitter robustness sweeps
│   │   └── diffrl/                 reused SHAC/BPTT engine (shac.py, bptt.py,
│   │                               models.py, utils.py) — auto-sizes actor/critic
│   │                               from num_obs, so it serves all envs unchanged
│   └── flow_surrogate/             conditional-flow surrogate + shape controllers
│       ├── model.py preprocess.py  ConditionalAffineFlow (Lightning) + h5 builder
│       ├── train.py dataset.py     flow training + FlowDataModule
│       ├── properties.py           PROPERTY_REGISTRY (emit, sigma, energy,
│       │                           aspect_ratio, Stokes s1/s2, tilt, ...)
│       ├── diff_env.py vec_env.py  FlowBunchEnv (differentiable) + vec env
│       ├── train_{ppo,shac,bptt}.py  RL drivers on the flow surrogate
│       ├── compare_algos.py        flow-side PPO/SHAC/BPTT benchmark
│       ├── shape_env.py shape_targets.py   fixed-(aspect,tilt) shape control
│       ├── moving_shape_env.py moving_shape_cli.py  goal-conditioned MOVING-target
│       │                                            (aspect,tilt) tracking controller
│       ├── eval_tracking.py animate_tracking.py     surrogate tracking eval / GIFs
│       ├── impact_eval_tracking.py plot_impact_summary.py  REAL Impact-T tracking eval
│       └── README.md               flow surrogate design notes
├── configs/
│   ├── impact/   ImpactT_PR10241.in, ImpactT_config.yaml, distgen_template.yaml, rfdata*
│   ├── sweep/    lhs_train_hifi.yaml, lhs_smoke_hifi.yaml  (xopt LHS designs)
│   └── diff_rl/  shac_flow.yaml, bptt_flow.yaml, shac_photoinjector.yaml,
│                 bptt_photoinjector.yaml, moving_eval.yaml
├── scripts/      run_sweep_local.sh, regen_hifi.sh, run_diff_smoke.sh,
│                 run_impact_tracking_eval.zsh, eval_drift_sweep.sh, compare_hifi_drift.sh
├── tests/        ~122 test functions (env / harness / surrogate / diffrl)
├── archives/     Impact-T sweep outputs: train/ (lofi 2k particles, 8³ mesh),
│                 train_hifi/ (20k, 32³), smoke_hifi/  — each ~10k *.h5 runs
├── processed/    preprocessed tensors: emittance_target_hifi.h5, flow_surrogate.h5 (+ *_norm.json)
├── trained/      surrogate checkpoints: emittance_target_hifi/, flow_surrogate/
├── logs/         RL run dirs (compare_*, move_*, shape_*, impact_eval, ...)
└── figures/      result plots / JSON / animations (impact_tracking_*, move_*, shape_demo, ...)
```

> All of `archives/ processed/ trained/ logs/ figures/ workdir/` are
> gitignored — they are large data/artifact dirs, not source.

---

## Setup / environment

The conda env **`slac-rl`** already has the full stack installed (lume-impact,
distgen, pmd-beamphysics, torch, lightning, stable-baselines3, gymnasium, xopt,
matplotlib, tensorboard, wandb). **Do not rebuild it.**

The run convention is **not** a normal `pip install` — the package is used
in-place via `PYTHONPATH`:

```bash
cd /home/rwu4/photoinjector-rl/photoinjector-rl-clean
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
export PYTHONPATH=$PWD/src
# Impact-T runs also need the binary (auto-located from <conda>/bin, but explicit is safe):
export IMPACTT_BIN=/home/rwu4/miniconda3/envs/slac-rl/bin/ImpactTexe
```

Then invoke entry points as modules, e.g.
`$PY -m photoinjector_rl.flow_surrogate.train_shac ...`.

---

## Key conventions

* **The 11-D knob vector** (the single source of truth is
  `emittance_target/__init__.py`: `SETTING_KEYS`, `SETTING_BOUNDS`,
  `N_INPUT=11`), all **min-max normalized to `[0,1]`** via `SETTING_BOUNDS`:

  | # | key | bounds | role |
  |---|---|---|---|
  | 1 | `SOL10111:solenoid_field_scale` | (-0.32, -0.2) | control (solenoid) |
  | 2 | `CQ10121:b1_gradient` | (-0.2, 0.2) | control (normal quad) |
  | 3 | `SQ10122:b1_gradient` | (-0.2, 0.2) | control (skew quad) |
  | 4 | `GUNF:rf_field_scale` | (4.696e7, 5.309e7) | control (gun RF amplitude) |
  | 5 | `GUNF:theta0_deg` | (-70, -58) | control (gun RF phase, deg) |
  | 6 | `distgen:r_dist:sigma_xy:value` | (1.0, 2.5) | distgen context |
  | 7 | `distgen:r_dist:truncation_radius:value` | (2.0, 3.5) | distgen context |
  | 8 | `distgen:start:MTE:value` | (100, 1000) | distgen context (cathode MTE) |
  | 9 | `distgen:t_dist:sigma_t:value` | (0.8, 1.5) | distgen context |
  | 10 | `distgen:transforms:s1:scale` | (0.6, 1.4) | distgen context |
  | 11 | `distgen:transforms:r1:angle:value` | (-90, 90) | distgen context (deg) |

  The RL agent **acts on only the first 5 (control) knobs**; the 6 distgen knobs
  are a **hidden context** sampled at `reset()` that the policy must adapt to.

* **Output screen `PR10241`** (z ≈ 0.95 m) — the terminal beam where all rewards
  are measured. Surrogates/sims read the deck `configs/impact/ImpactT_PR10241.in`.

* **`norm_emit_4d`** = `sqrt(det Σ_4d) / (mc²)²` (Σ_4d = cov of `(x,px,y,py)`,
  `mc² = 0.511e6 eV`), matching `openPMD-beamphysics`'s `ParticleGroup.norm_emit_4d`.
  This is the default RL objective; reward = `-log10(norm_emit_4d)` z-scored.

* **Checkpoint formats:**
  * SHAC / BPTT (`diffrl`): `best_policy.pt` / `final_policy.pt` — the **actor weights + obs RunningMeanStd**, loaded by the `DiffRLAdapter`. (`best_policy.pt` is the recommended eval target; a late SHAC collapse can degrade the final.)
  * PPO (Stable-Baselines3): a `.zip` (`ppo_final.zip` / `eval/best_model.zip`), loaded by the `SB3Adapter`.

* **Fidelities:** *lofi* = 2000 particles / 8³ mesh (`archives/train`, ~9 s/run);
  *hifi* = 20000 / 32³ (`archives/train_hifi`, ~35 s/run). Both share the same
  11-D LHS design so artifacts are directly comparable.

---

## How to run each stage

All commands assume the env exports above (`$PY`, `PYTHONPATH`, `IMPACTT_BIN`)
and are run from the repo root. The preprocessed datasets and surrogate
checkpoints ship preserved, so stages 1–2 are **regenerate-only**.

### 1. Data sweep (Impact-T + distgen LHS)

xopt runs the LHS over the 11 knobs, each point a full Impact-T+distgen sim,
archiving one `*.h5` per run.

```bash
# local MPI sweep (driven by configs/sweep/*.yaml)
./scripts/run_sweep_local.sh configs/sweep/lhs_smoke_hifi.yaml     # smoke
./scripts/run_sweep_local.sh configs/sweep/lhs_train_hifi.yaml     # full 10k hifi

# end-to-end hifi regen (sweep -> preprocess -> EmittanceMLP), idempotent/resumable:
./scripts/regen_hifi.sh                # see --skip-sweep / --force-* flags
```

### 2. Surrogate training

```bash
# scalar EmittanceMLP (hifi)
$PY -m photoinjector_rl.emittance_target.preprocess \
    --archives 'archives/train_hifi/*.h5' --out processed/emittance_target_hifi.h5
$PY -m photoinjector_rl.emittance_target.train \
    --processed processed/emittance_target_hifi.h5 \
    --out-dir trained/emittance_target_hifi --devices 1
# shipped: trained/emittance_target_hifi/...best-epoch=191-val_loss=0.0060.ckpt (R²=0.995, MAPE 2.7%)

# conditional flow (lofi: 1500 particles/run)
$PY -m photoinjector_rl.flow_surrogate.preprocess \
    --archives 'archives/train/*.h5' --out processed/flow_surrogate.h5 --target-particles 1500
$PY -m photoinjector_rl.flow_surrogate.train \
    --processed processed/flow_surrogate.h5 --out-dir trained/flow_surrogate --devices 1
# shipped: trained/flow_surrogate/...best-epoch=493-val_loss=-0.9555.ckpt (emit_4d err ~5.4%)
```

### 3. RL training on the surrogate

```bash
# --- scalar-MLP emittance task (emittance_target) ---
CKPT=trained/emittance_target_hifi/checkpoints/best-epoch=191-val_loss=0.0060.ckpt
NORM=processed/emittance_target_hifi_norm.json

$PY -m photoinjector_rl.emittance_target.train_ppo \
    --ckpt $CKPT --norm-json $NORM --out-dir trained/ppo_emittance_target \
    --total-timesteps 500000 --n-envs 16 --device cpu
$PY -m photoinjector_rl.emittance_target.train_shac \
    --cfg configs/diff_rl/shac_photoinjector.yaml --ckpt $CKPT --norm-json $NORM \
    --logdir logs/shac --device cuda:0
$PY -m photoinjector_rl.emittance_target.train_bptt \
    --cfg configs/diff_rl/bptt_photoinjector.yaml --ckpt $CKPT --norm-json $NORM \
    --logdir logs/bptt --device cuda:0
# fast smoke of SHAC+BPTT: ./scripts/run_diff_smoke.sh --device cpu --epochs 10

# --- conditional-flow task (flow_surrogate): any registry property ---
FCKPT=trained/flow_surrogate/checkpoints/best-epoch=493-val_loss=-0.9555.ckpt
FNORM=processed/flow_surrogate_norm.json

$PY -m photoinjector_rl.flow_surrogate.train_shac \
    --cfg configs/diff_rl/shac_flow.yaml --flow-ckpt $FCKPT --norm-json $FNORM \
    --property norm_emit_4d --reward-mode minimize --logdir logs/shac_flow --device cuda:0
$PY -m photoinjector_rl.flow_surrogate.train_bptt \
    --cfg configs/diff_rl/bptt_flow.yaml --flow-ckpt $FCKPT --norm-json $FNORM \
    --logdir logs/bptt_flow --device cuda:0
$PY -m photoinjector_rl.flow_surrogate.train_ppo \
    --flow-ckpt $FCKPT --norm-json $FNORM --out-dir logs/ppo_flow --total-timesteps 500000

# head-to-head PPO/SHAC/BPTT on the flow (writes logs/compare_flow/, evals best_policy):
$PY -m photoinjector_rl.flow_surrogate.compare_algos \
    --flow-ckpt $FCKPT --norm-json $FNORM --out-dir logs/compare_flow \
    --algos ppo,shac,bptt --seeds 0,1,2 --budget 500000 --device cuda:0
```

Other reward modes / properties (flow drivers + `compare_algos`):
`--property {norm_emit_x,norm_emit_y,sigma_x,sigma_y,energy_spread,mean_energy,aspect_ratio,...}`
and `--reward-mode {minimize,maximize,target} [--target <v>]`. `norm_emit_x/y`
and `sigma_x/y` are the most knob-controllable; `sigma_z` is context-dominated.

### 4. Evaluate trained policies on real Impact-T

```bash
# roll out PPO (.zip) + SHAC/BPTT (.pt) policies on the real simulator, paired stats:
$PY -m photoinjector_rl.emittance_target.compare_n_impact \
    --policy ppo=trained/ppo_emittance_target/eval/best_model.zip \
    --policy shac=logs/shac/best_policy.pt \
    --policy bptt=logs/bptt/best_policy.pt \
    --impact-config configs/impact/ImpactT_config.yaml \
    --distgen-input configs/impact/distgen_template.yaml \
    --norm-json $NORM --seeds 0,1,2,3

# single-policy version: emittance_target/compare_impact.py
# best-per-algo from a compare_diff_algos run: emittance_target/compare_impact_all_algos.py
```

PPO can also be **fine-tuned on real Impact-T** by warm-starting the
surrogate-trained policy: `emittance_target/train_ppo_impact.py`.

---

## The moving-target shape controller

A goal-conditioned controller that drives the bunch to a **time-varying**
transverse shape setpoint within a single rollout. The target is the
rotation-aware **Stokes shape vector** `(s1, s2) = ((σx²−σy²)/(σx²+σy²),
2·cov_xy/(σx²+σy²))`, which encodes both **aspect ratio** (`σx/σy`) and
projected **tilt angle**, is strongly controllable by the quads (`R²_ctrl ≈ 0.9`),
and is distgen-independent. One policy tracks a domain-randomized
`(s1*(t), s2*(t))` schedule (curriculum: static → step → smooth) using a
9-D goal-conditioned observation `[knobs(5), s1_cur, s2_cur, s1*(t), s2*(t)]`.

It was **validated sim-to-sim on real Impact-T**: policies trained on the lofi
flow track real Impact-T at both lofi and hifi with settled tilt MAE ≈ 0.5–1.1°
on smooth schedules (SHAC ≈ BPTT > PPO), fidelity-robust to ≤ 0.7°.

Full write-up, training commands, and the complete eval matrix:
**[`docs/moving_target_controller.md`](docs/moving_target_controller.md)**.
One-liners to reproduce the headline results:

```bash
# surrogate-side tracking figure (loads best_policy.pt / ppo_final.zip per algo)
$PY -m photoinjector_rl.flow_surrogate.eval_tracking \
    --flow-ckpt $FCKPT --norm-json $FNORM \
    --shac logs/move_shac/seed0 --bptt logs/move_bptt/seed0 --ppo logs/move_ppo \
    --traj-config configs/diff_rl/moving_eval.yaml --out figures/move_tracking.png

# full REAL Impact-T eval matrix (3 models × 3 held-out schedules × {lofi,hifi}):
./scripts/run_impact_tracking_eval.zsh                # lofi then hifi, fans out in parallel
```

(Train one with the `--moving-shape` flag on any flow driver, e.g.
`train_shac --cfg configs/diff_rl/shac_flow.yaml --flow-ckpt $FCKPT --norm-json $FNORM
--moving-shape --logdir logs/move_shac`; the curriculum is configured in the
YAML `diff_env.curriculum` block.)

---

## Tests

```bash
PYTHONPATH=$PWD/src $PY -m pytest tests/ -q     # ~118 pass, a few skip
```

Env / harness / surrogate / `diffrl` tests run without checkpoints (mock
surrogates); regression tests that need a checkpoint skip if it is absent.

---

## Outputs

| dir | contents |
|---|---|
| `archives/` | raw Impact-T sweep runs (`*.h5`, one per LHS point), per fidelity |
| `processed/` | preprocessed training tensors + `*_norm.json` normalization stats |
| `trained/` | surrogate checkpoints (`emittance_target_hifi/`, `flow_surrogate/`) + `final_metrics.json` + val plots |
| `logs/` | RL run dirs — `best_policy.pt`/`final_policy.pt` or `ppo_final.zip`, `learning_curve.csv`, TensorBoard; compare/move/shape/impact_eval runs |
| `figures/` | result plots, tracking JSON, summary CSVs, and GIF animations |
