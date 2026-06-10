# flow_surrogate

Conditional **affine-coupling normalizing flow** that generates the photoinjector
output bunch as a distribution, conditioned on the control knobs:

```
condition (11-D) = the 11 XOPT-sampled knobs (5 Impact + 6 distgen), in [0,1]
output           = PR10241 electron cloud, (P, 6) in (x,y,z,px,py,pz)
                   x,y,z [m]; px,py,pz [eV/c]; openPMD-beamphysics convention
base             = N(0, I) in 6-D
```

This is the full-phase-space sibling of `emittance_target` (the v1 scalar MLP that
maps the same 11 knobs to `log10(norm_emit_4d)`). Because it models the whole
output density, any beam statistic — `norm_emit_4d`, spot size, slice emittance —
can be computed from a sampled cloud, and via the **reparameterization trick** its
gradient flows back to the knobs. That is the intended substrate for first-order
**model-based RL** (rewards backprop through the flow to the control knobs).

The 11-knob order/bounds are the single source of truth in `emittance_target` and
are re-exported here, so every script agrees on column order.

## Files

| File | Role |
|---|---|
| `__init__.py` | Re-exports `SETTING_KEYS/SETTING_BOUNDS/N_INPUT` from `emittance_target`; adds `LATENT_DIM=6`, `COORD_KEYS`, `ELECTRON_MC2_EV`, `DEFAULT_P=1500`. |
| `preprocess.py` | Walk `archives/*/*.h5` → `processed/flow_surrogate.h5` (`settings (N,11)`, `particles (N,P,6)`, `norm_emit_4d (N,)`, `fingerprint`) + `_norm.json` (settings min-max, per-dim particle mean/std, `P`). Subsamples each run to a fixed `P`. Reuses `emittance_target.preprocess.extract_settings`. |
| `dataset.py` | `FlowDataset` (in-memory, normalizes at construction → `(particles_norm (P,6), cond_norm (11,))`) + `FlowDataModule` (90/10 split, seed 42, `batch_size=32`). |
| `model.py` | `ConditionalAffineFlow` (Lightning). Affine coupling, `condition_dim=11`, NLL + emittance/beam-matrix aux losses. Exposes `sample`, `sample_physical`, `log_prob`, `emittance_from_knobs` (differentiable), `forward` (diff_env drop-in). De-norm stats stored as buffers. |
| `train.py` | Training entry point. CSVLogger + ModelCheckpoint + EarlyStopping, `gradient_clip_val=1.0`. Held-out eval = Sliced-Wasserstein + emittance % error; writes `final_metrics.json` + `val_phase_space_overlay.png`. |

## Workflow

From the clean-repo root, inside the `slac-rl` conda env, with `PYTHONPATH=$PWD/src`.
Lofi data is reached via the `archives/train` symlink → sibling repo.

```bash
# 1. preprocess lofi  →  processed/flow_surrogate.h5  (+ _norm.json)
python -m photoinjector_rl.surrogates.flow.preprocess \
    --archives 'archives/train/*.h5' \
    --out processed/flow_surrogate.h5 \
    --target-particles 1500

# 2. train
python -m photoinjector_rl.surrogates.flow.train \
    --processed processed/flow_surrogate.h5 \
    --out-dir trained/flow_surrogate \
    --devices 1
# Outputs: trained/flow_surrogate/checkpoints/best-*.ckpt, last.ckpt,
#          csv_logs/, val_phase_space_overlay.png, final_metrics.json
```

Hi-fi later is the same with `--archives 'archives/train_hifi/*.h5' --target-particles 20000`
and a separate `--out`/`--out-dir`.

## Inference

```python
import torch
from photoinjector_rl.surrogates.flow.model import ConditionalAffineFlow

model = ConditionalAffineFlow.load_from_checkpoint(
    "trained/flow_surrogate/checkpoints/best-....ckpt", map_location="cpu").eval()

cond = torch.rand(1, 11)                         # 11 knobs in [0,1]
cloud = model.sample_physical(cond, n=1500)      # (1, 1500, 6) raw units (m, eV/c)
emit = model.emittance_from_knobs(cond)          # (1,) norm_emit_4d in m^2
```

## MBRL readiness (RL wiring is future work)

- `forward(x: [B,11]) -> [B,1]` returns **normalized log10(norm_emit_4d)** — a drop-in for the
  surrogate slot in `emittance_target/diff_env.py` via its `surrogate=<flow instance>`
  constructor arg (reward `= -y_norm`). The flow z-scores internally from its own buffers, so
  pass `target_mean=0, target_std=1` to the env. Note: the env's other path
  (`surrogate_ckpt=...` + `norm_json`) is hardcoded to `EmittanceMLP` and to a norm JSON with
  `target_mean/std`, which the flow's `_norm.json` does not carry — use the instance path.
- `emittance_from_knobs(x: [B,11], n) -> [B]` is differentiable w.r.t. the knobs via the
  reparameterization trick (`z ~ N(0,I)` sampled once as a constant; transform deterministic
  in the condition). Gradients flow knobs → sampled cloud → emittance.
- Both are **stochastic** across calls (fresh `z`); raise `n` / `forward_n_particles` to cut
  Monte-Carlo variance for RL.
- Wiring SHAC/BPTT (`diff_env.py`, `diffrl/`) to this surrogate is **not** done here.

## Design choices

- **Affine coupling** (not RQ-spline): stable, no extra deps (`normflows` not required),
  exactly invertible (round-trip tested). `tanh×0.5` scale clamp + `gradient_clip_val=1.0`
  guard against the flow's classic NaN-gradient failure mode.
- **Fixed P=1500** per run: lofi runs hold ~2000 macroparticles (sweep allows loss to ~1500),
  so P=1500 keeps every run (0 of 10,048 dropped) and lets us store a dense `(N,P,6)` array
  (no ragged storage). Subsamples down, never up.
- **Per-dim StandardScaler** on particles: raw coordinate scales span ~10 orders of
  magnitude (z std ~3e-4 m vs pz std ~7e4 eV/c).
- **`norm_emit_4d = sqrt(det Σ_4d) / (mc²)²`** (Σ_4d = cov of `(x,px,y,py)` raw, `mc²=0.511e6 eV`)
  reproduces `ParticleGroup.norm_emit_4d`, so the flow's emittance matches the v1 surrogate's
  target exactly.
- **Aux emittance losses** (on top of NLL): directly sharpen the second moments the downstream
  reward depends on (2D/4D/6D weighted 1.0/0.5/0.1). The **beam-matrix SMAPE term is OFF by
  default** (`--w-beam 0`) — it uses a per-bunch `torch.cov` Python loop (slow) and can
  destabilize early training; enable it with `--w-beam 0.005` if desired. Set all weights to 0
  for pure NLL.
- **Lofi first**; hi-fi (20000 particles, ~4.8 GB processed) is a later swap.
