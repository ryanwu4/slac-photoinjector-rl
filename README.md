# photoinjector-rl

Stage 1 of the FACET-II photoinjector RL pipeline: an LHS-sampled Impact-T
parameter sweep that produces per-run HDF5 archives for downstream surrogate
training. Stages 2–5 (preprocessing, surrogate, env, RL) are not yet built.

**Conda env:** `slac-rl` (already has lume-impact, distgen, xopt 2.7, mpi4py,
h5py, pmd-beamphysics).

**Entry point:**

```bash
conda activate slac-rl
pip install -e .
bash scripts/run_sweep_local.sh                       # smoke (default)
bash scripts/run_sweep_local.sh configs/sweep/lhs_train.yaml
```

Sweep configs live in `configs/sweep/`; the Impact-T deck and distgen template
live in `configs/impact/`. Per-run archives land in `archives/{smoke,train,val}/`.
