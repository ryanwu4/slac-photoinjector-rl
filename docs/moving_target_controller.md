# Moving-Target (Goal-Conditioned) Shape Controller for the FACET-II Photoinjector

A goal-conditioned ("moving-target") reinforcement-learning controller that drives
the **transverse beam shape** — the eigen aspect ratio *and* the tilt of the
PR10241 bunch — to a **time-varying setpoint** within a single rollout. One policy
tracks *any* `(aspect, tilt)` schedule because the setpoint is part of the
observation. The controllers are trained entirely on a conditional
normalizing-flow surrogate of Impact-T and then validated on the **real Impact-T
simulation** (sim-to-sim transfer).

---

## 1. Overview / motivation

Earlier shape work (`shape_env.py`, `ShapeTargetEnv`) trained a *separate* policy
per fixed `(aspect, tilt)` target. The moving-target controller generalizes this:
a **single** policy learns to track an arbitrary, time-varying shape setpoint
`(s1*(t), s2*(t))` that is fed to the policy through the observation. At every step
the policy sees the current beam shape and the commanded shape and acts on the 5
control knobs to close the gap — so the same policy handles a step staircase, a
continuous tilt rotation, an aspect ramp, or a random walk, all chosen at runtime.

This makes the controller useful as a *shape-on-demand* actuator: command an
ellipse orientation and elongation trajectory, and the beam follows it.

---

## 2. Method

### 2.1 Stokes-like shape vector

The transverse shape is parameterized by a normalized 2nd-moment ("Stokes") vector
computed from the `(x, y)` cloud (`flow_surrogate/properties.py`):

```
s1 = (σx² − σy²) / (σx² + σy²)            # x/y elongation,  ∈ [−1, 1]
s2 = 2 · cov_xy / (σx² + σy²)             # x–y coupling,    ∈ [−1, 1]
```

Both are differentiable in the particle coordinates (centered vars/cov are smooth),
so the reward gradient backpropagates through the sampled bunch to the knobs.

The magnitude/angle of `(s1, s2)` map to physical shape descriptors:

```
r       = √(s1² + s2²)                    # radius in the shape disk
aspect  = √((1 + r) / (1 − r))            # eigen aspect ratio σ_major/σ_minor ≥ 1
tilt    = ½ · atan2(s2, s1)               # major-axis tilt angle ∈ (−90°, 90°]
```

The inverse map (used to convert commanded `(aspect, tilt)` into a setpoint) is
`aspect_tilt_to_s`: `r* = (a²−1)/(a²+1)`, `s1* = r*·cos(2θ*)`, `s2* = r*·sin(2θ*)`.
Note the factor of **2θ**: the Stokes angle is twice the physical tilt, which is
why the ellipse orientation has a 180° period (handled by the circular tilt-error
metric). Setpoints stay inside the reachable disk `r < r_max` (default `0.85`);
about 80% of `(s1, s2)` is reachable across distgen contexts.

### 2.2 Goal-conditioned observation and reward

`MovingShapeEnv` (`flow_surrogate/moving_shape_env.py`) subclasses the
differentiable `FlowBunchEnv` and overrides only a few seams. The observation is
**9-D**:

```
obs (9-D) = [ knobs(5), s1_cur, s2_cur, s1*(t), s2*(t) ]
```

- `knobs(5)` — the 5 normalized control knobs, kept **differentiable** so SHAC/BPTT's
  dynamics chain is intact.
- `s1_cur, s2_cur` — the current achieved shape, a **detached sensor** (read off the
  flow-sampled bunch each forward).
- `s1*(t), s2*(t)` — the commanded setpoint at the current step.

The dense, per-step tracking reward is

```
reward = − ‖ (s1(t), s2(t)) − (s1*(t), s2*(t)) ‖ / scale      (scale ≈ 0.3)
```

which is fully differentiable w.r.t. the action. The reward is computed inside the
env (`_sample_property_ynorm`) via a duck-typed `ShapeTargetSpec` that carries only
`scale`.

### 2.3 Per-step target indexing (the crux)

The setpoint at step `t` is indexed by the env's within-episode counter
`_step_count` (clamped to `T−1`), **not** reset by `initialize_trajectory()`
(which only cuts the autograd graph). This is essential because SHAC's short
backprop horizon (`steps_num=16`) sits *inside* the 64-step episode: the target
must advance with the episode, not the horizon. Tests
`test_moving_shape_target_advances_with_step_count` and
`test_moving_shape_initialize_trajectory_keeps_step_count` lock this behavior in.

### 2.4 Training-time curriculum

Per-episode target trajectories are sampled by `shape_targets.py` with a
**static → step → smooth** curriculum, all parameters in `CurriculumConfig`
(populated from the YAML `diff_env.curriculum` block via `from_dict`):

- **static** — constant target for the whole episode (dominates early training).
- **piecewise-constant steps** — hold `k` steps, jump; holds shorten as difficulty rises.
- **smooth** — tilt rotation at fixed aspect, aspect ramp at fixed tilt, or an OU
  random walk in the disk; faster/noisier as difficulty rises.

A shared `CurriculumState.progress` (0→1) is advanced each epoch by the training
driver — SHAC/BPTT via a wrapped `step_metrics_hook` (`make_progress_hook`), PPO
via an SB3 `CurriculumCallback` — biasing per-episode difficulty from mostly-static
early to the full regime mix late.

### 2.5 Controlled vs hidden DOFs, and design footprint

The policy controls only the **5 Impact knobs** (`SETTING_KEYS[:5]`:
`SOL10111` solenoid, `CQ10121` normal quad, `SQ10122` skew quad, `GUNF` rf field
scale, `GUNF` phase). The **6 distgen knobs** (cathode/beam initial conditions) are
fixed hidden context. Control of `(s1, s2)` is cooperative/solenoid-dominant
(R²_ctrl ≈ 0.90 for each of `s1`, `s2`; flow R² ≈ 0.97–0.98) rather than
one-knob-per-DOF, but the policy learns it.

The design touches **only `flow_surrogate/`**. `diffrl/` and `emittance_target/`
are byte-for-byte unchanged: the actor/critic/`obs_rms` auto-size from `num_obs`,
so setting `num_obs = 9` is all that is needed for the goal-conditioned obs.

---

## 3. Algorithms

Three controllers are trained for comparison, all on the same flow surrogate and
the same `MovingShapeEnv` reward:

- **SHAC** — first-order MBRL, short backprop horizon (`steps_num=16`) + a learned
  critic. `target_critic_alpha=0.95` (slow target critic) keeps it stable for the
  full run on the moving target.
- **BPTT** — first-order MBRL, full-episode backprop (`steps_num=64`), no critic.
- **PPO** — model-free / zeroth-order baseline (Stable-Baselines3), via the
  GPU-batched `MovingShapeVecEnv`.

---

## 4. Files added / changed (all under `src/photoinjector_rl/flow_surrogate/`)

New modules:

- `shape_targets.py` — `CurriculumConfig` + `CurriculumState`, the
  `sample_shape_trajectory` generator (static/step/smooth), and the deterministic
  held-out eval builders (`eval_staircase`, `eval_tilt_rotation`, `eval_aspect_ramp`,
  `build_eval_trajectories`).
- `moving_shape_env.py` — `MovingShapeEnv` (9-D goal-conditioned obs, per-step target
  indexing, tracking reward) and `MovingShapeVecEnv` (SB3 VecEnv wrapper for PPO).
- `moving_shape_cli.py` — shared CLI/wiring: `add_moving_shape_args`,
  `apply_moving_shape_overrides`, `build_moving_env_fn`, `make_progress_hook`,
  `load_moving_config`, and the PPO `CurriculumCallback`.
- `eval_tracking.py` — surrogate tracking eval on held-out schedules (s-space RMSE,
  circular tilt MAE, log-aspect MAE, per-segment settling) + the achieved-vs-commanded
  plot.
- `impact_eval_tracking.py` — sim-to-sim transfer: rolls a trained policy on **real
  Impact-T** (sequential closed loop), with a flow-vs-Impact `parity_check`.
- `animate_tracking.py` — per-case GIFs (x–y panel + 6-D corner + knob histories).
- `plot_impact_summary.py` — the lofi-vs-hifi 2×3 summary figure + metrics CSV.

Touched (added a `--moving-shape` branch / args only; default paths unchanged):

- `properties.py` — added the shape math: `_s1`, `_s2`, `_eigen_aspect`,
  `_tilt_angle_deg`, `aspect_tilt_to_s`, `s_to_aspect_tilt`, and `ShapeTargetSpec`.
- `train_shac.py`, `train_bptt.py`, `train_ppo.py` — `--moving-shape` selects the
  goal-conditioned env + curriculum; non-moving paths untouched.
- `configs/diff_rl/shac_flow.yaml`, `bptt_flow.yaml` — added the `diff_env.curriculum`
  block.
- `configs/diff_rl/moving_eval.yaml` — held-out eval-schedule spec.
- `scripts/run_impact_tracking_eval.zsh` — batch Impact-T eval driver.
- `tests/test_flow_diff_env.py` — moving-target tests (curriculum, obs layout,
  step-count indexing, reward differentiability, info-cache survival).

---

## 5. Results

### 5.1 Surrogate tracking (`figures/move_tracking.png`, `move_tracking.json`)

Settled-window (skip first 8 settling-from-random-init steps) metrics on the three
held-out schedules, evaluated on the flow surrogate (64 rollouts, 512 particles):

| schedule        | metric                | SHAC  | BPTT  | PPO   |
|-----------------|-----------------------|-------|-------|-------|
| tilt_rotation   | tilt MAE (settled)    | 2.2°  | 3.0°  | 3.9°  |
| tilt_rotation   | aspect log-MAE        | 0.020 | 0.022 | 0.025 |
| aspect_ramp     | tilt MAE (settled)    | 1.9°  | 2.6°  | 3.7°  |
| aspect_ramp     | aspect log-MAE        | 0.024 | 0.026 | 0.035 |
| staircase       | tilt MAE (settled)    | 15.0° | 12.3° | 22.9° |
| staircase       | s-RMSE (settled)      | 0.43  | 0.38  | 0.55  |

On the smooth schedules settled tilt MAE is ~2–4° with aspect within ~2–4% (log
space). The staircase (sharp setpoint jumps vs the small `action_scale=0.05` per
step) is the hard case — the controller lags each jump — but BPTT (full-horizon
credit assignment) handles steps best. Ranking on the surrogate: **SHAC ≈ BPTT > PPO**.

> Note: raw aspect RMSE is intentionally *not* used as the headline metric because
> `aspect = √((1+r)/(1−r))` blows up as `r → 1`; the robust metrics are s-space RMSE
> and **log-aspect MAE** on the settled window.

### 5.2 Animated GIFs (`figures/move_anim/*.gif`)

`move_staircase.gif`, `move_tilt_rotation.gif`, `move_aspect_ramp.gif` (~5–6 MB
each). Each frame shows a large x–y transverse panel, the full 6-D corner plot
(x, y, z, px, py, pz), and a right-hand column of 5 control-knob-history panels.
SHAC/BPTT/PPO bunches are overlaid and color-coded, with the commanded ellipse
drawn dashed. All three controllers roll from an **identical deterministic start**
and share a fixed latent `z` (`stochastic_init=False`, `torch.manual_seed` before
each sample), so the clouds morph smoothly and differ only by policy. The quads
(`CQ10121`/`SQ10122`) visibly oscillate to set aspect/tilt while the GUNF amplitude
stays flat.

### 5.3 Impact-T transfer (sim-to-sim)

The full **3 models × 3 schedules × {lofi, hifi} = 18-rollout matrix** ran with
**0 Impact-T failures**. Impact at fixed knobs+distgen is deterministic (Hammersley
distgen), so the real-sim track has no per-step shot noise and is actually
*smoother* than the MC-sampled surrogate.

**Parity pre-check** (`figures/impact_parity_lofi.png`): flow-predicted vs real
Impact-T `(s1, s2)` at 6 random knob settings (fixed distgen) — **MAE
s1 = 0.023, s2 = 0.029**. The surrogate is faithful across the shape disk. (Parity
ran only on the standalone first run; the batch ran without `--parity`, so there is
no hifi parity plot.)

**Per-schedule Impact-T tilt MAE (settled, degrees)** from
`figures/impact_tracking_summary_metrics.csv`:

| schedule       | SHAC lofi / hifi | BPTT lofi / hifi | PPO lofi / hifi |
|----------------|------------------|------------------|-----------------|
| tilt_rotation  | 0.80 / 0.81      | 0.92 / 1.11      | 3.23 / 3.04     |
| aspect_ramp    | 0.60 / 0.48      | 0.91 / 1.04      | 2.36 / 2.22     |
| staircase      | 13.03 / 12.69    | 9.16 / 8.48      | 21.59 / 21.42   |

Key findings:

1. **Transfer holds at both fidelities.** Policies trained on the *lofi* flow track
   real Impact-T at *hifi* (32³ mesh, 20k particles) as well as at lofi.
2. **Fidelity-robust.** `|Δ tilt MAE|` between lofi and hifi is **≤ 0.7°** (mostly
   ≤ 0.3°) — aspect/tilt is linear-optics-dominated and captured even at lofi.
3. **Smooth schedules are tight:** tilt MAE 0.5–1.1° for SHAC/BPTT, 2.2–3.2° for
   PPO; aspect log-MAE ~1–5%.
4. **Staircase is the hard case:** BPTT (8.5–9.2°) < SHAC (12.7–13.0°) < PPO (~21°)
   — the small per-step action scale lags sharp jumps; BPTT's full-horizon credit
   assignment wins here.
5. **Ranking SHAC ≈ BPTT > PPO** (BPTT best on the staircase).
6. **Real Impact often *beats* the surrogate's own prediction** — e.g. SHAC
   tilt_rotation lofi: Impact 0.80° vs surrogate 1.05° — because Impact is
   deterministic while the surrogate adds MC sampling noise.

Summary figure: `figures/impact_tracking_summary.png` (2×3 grid, rows =
aspect/tilt, cols = schedules; color = controller, linestyle = fidelity, lofi
solid / hifi dotted — same-color curves nearly coincide). Per-case plots/JSON are
`figures/impact_tracking_<algo>_<traj>_<fid>.{png,json}`.

---

## 6. Replication — all commands (verbatim)

Set up the environment first (run from the repo root). Verify against the argparse
in each module before changing flags.

```bash
cd /home/rwu4/photoinjector-rl/photoinjector-rl-clean
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
export PYTHONPATH=$PWD/src
CKPT=trained/flow_surrogate/checkpoints/best-epoch=493-val_loss=-0.9555.ckpt
NORM=processed/flow_surrogate_norm.json
export IMPACTT_BIN=/home/rwu4/miniconda3/envs/slac-rl/bin/ImpactTexe   # Impact-T runs only
```

### Step 1 — Train the moving-target controllers on the flow surrogate

```bash
$PY -m photoinjector_rl.flow_surrogate.train_shac --cfg configs/diff_rl/shac_flow.yaml \
  --flow-ckpt $CKPT --norm-json $NORM --logdir logs/move_shac/seed0 --seed 0 --max-epochs 1000 --moving-shape
$PY -m photoinjector_rl.flow_surrogate.train_bptt --cfg configs/diff_rl/bptt_flow.yaml \
  --flow-ckpt $CKPT --norm-json $NORM --logdir logs/move_bptt/seed0 --seed 0 --max-epochs 245 --moving-shape
$PY -m photoinjector_rl.flow_surrogate.train_ppo --flow-ckpt $CKPT --norm-json $NORM \
  --out-dir logs/move_ppo --seed 0 --total-timesteps 500000 --moving-shape
```

### Step 2 — Surrogate tracking eval → `figures/move_tracking.png`

```bash
$PY -m photoinjector_rl.flow_surrogate.eval_tracking --flow-ckpt $CKPT --norm-json $NORM \
  --traj-config configs/diff_rl/moving_eval.yaml \
  --shac logs/move_shac/seed0 --bptt logs/move_bptt/seed0 --ppo logs/move_ppo \
  --episode-length 64 --n-rollouts 64 --n-particles 512 --out figures/move_tracking.png
```

### Step 3 — Animated GIFs → `figures/move_anim/*.gif`

```bash
$PY -m photoinjector_rl.flow_surrogate.animate_tracking --flow-ckpt $CKPT --norm-json $NORM \
  --traj-config configs/diff_rl/moving_eval.yaml \
  --shac logs/move_shac/seed0 --bptt logs/move_bptt/seed0 --ppo logs/move_ppo \
  --episode-length 64 --n-vis 1200 --stride 2 --fps 12 --device cuda:0 --out-dir figures/move_anim
```

### Step 4 — Impact-T evaluation (full 3×3×2 matrix)

```bash
./scripts/run_impact_tracking_eval.zsh            # lofi (9 parallel) then hifi (9 parallel)
```

Preview the commands without running anything:

```bash
./scripts/run_impact_tracking_eval.zsh --dry-run
```

Single-run form (one model / one schedule / one fidelity, with the parity check):

```bash
$PY -m photoinjector_rl.flow_surrogate.impact_eval_tracking --flow-ckpt $CKPT --norm-json $NORM \
  --shac logs/move_shac/seed0 --which-traj tilt_rotation --fidelity lofi --episode-length 64 --parity-check
```

### Step 5 — Impact-T lofi-vs-hifi summary → `figures/impact_tracking_summary.png` (+ `_metrics.csv`)

```bash
$PY -m photoinjector_rl.flow_surrogate.plot_impact_summary --log-dir logs/impact_eval \
  --out figures/impact_tracking_summary.png
```

### Tests

```bash
$PY -m pytest tests/test_flow_diff_env.py -q
```

---

## 7. Key CLI options and config

### Training drivers (`--moving-shape` mode)

- `--moving-shape` — switch to the goal-conditioned `MovingShapeEnv` (9-D obs);
  ignores `--property` / `--shape-aspect`.
- `--curriculum` / `--no-curriculum` — force the difficulty ramp on/off, overriding
  the YAML (`--no-curriculum` uses the full difficulty mix from the start). Default
  is `None` so the config value is honored.
- `--r-max <float>` — override `curriculum.r_max` (reachable shape-disk radius).
- `--action-rate-penalty <λ>` (SHAC/BPTT) — add `−λ·‖Δaction‖` for smoother control
  (0 = off).
- `--shape-scale <float>` — tracking reward scale (default 0.3 from config).
- PPO only: `--moving-config <yaml>` supplies a `curriculum` block (since PPO does
  not take a `--cfg`).

### `diff_env.curriculum` YAML block (SHAC/BPTT)

Maps directly to `CurriculumConfig` fields. Key knobs: `enabled` (ramp on/off),
`r_max`, the regime-mix probabilities (`p_static_base`, `p_static_slope`,
`p_static_floor`, `p_steps`), the step-hold fractions (`hold_frac_easy/hard`,
`hold_min`), the smooth-regime ranges (`smooth_r_lo`, `tilt_turns_easy/hard`,
`ou_sigma_easy/hard`, `ou_decay`, `ou_init_frac`), and `difficulty_bias_lo`.
See `configs/diff_rl/shac_flow.yaml` for the annotated defaults.

### Eval-schedule spec (`configs/diff_rl/moving_eval.yaml`, `--traj-config`)

Named held-out builders under `eval_trajectories`:

- `staircase` — `segments`: list of `[aspect, tilt_deg]`, equal holds (step-response).
- `tilt_rotation` — `aspect` + `turns` (continuous `turns·180°` tilt sweep).
- `aspect_ramp` — `tilt_deg`, `a0`, `a1` (linear aspect ramp at fixed tilt).

### Impact-T eval (`impact_eval_tracking.py` / `run_impact_tracking_eval.zsh`)

- `--which-traj {staircase, tilt_rotation, aspect_ramp, all}` — held-out schedule(s).
- `--fidelity {lofi, hifi}` — lofi = `DEFAULT_LHS_CONSTANTS` (2k particles, 8³ mesh);
  hifi = `configs/sweep/lhs_train_hifi.yaml` constants (20k particles, 32³ mesh).
- `--parity-check` — run the flow-vs-Impact `(s1, s2)` parity pre-check first
  (`--parity-points`, default 6) → `figures/impact_parity_<fid>.png`.
- `--start-knobs` / `--distgen` (default 0.5) — fixed start point of the closed loop.
- `--shac` / `--bptt` / `--ppo` — point to the trained run dir (one per invocation).
- Batch driver flags: `--fidelity {lofi, hifi, both}` (default `both`),
  `--episode-length`, `--parity` (parity on the first job of each phase),
  `--dry-run`.

### Runtimes

A moving-target Impact-T rollout is **inherently sequential** (~T+1 Impact-T jobs
back-to-back, because each step's knobs depend on the previous bunch):

- **lofi** ≈ 9 s/run → ~10 min per 64-step rollout.
- **hifi** ≈ 35 s/run → ~40 min per rollout.

The batch driver fans the 9 independent (model × schedule) rollouts out in parallel
per fidelity phase (`OMP_NUM_THREADS=1`, designed for a 128-core node), so each
phase's wall time is roughly one rollout (~10 min lofi, ~40 min hifi).
