"""
Conditional affine-coupling normalizing-flow surrogate.

Generates the PR10241 output particle bunch (6D phase space) conditioned on the
11 XOPT-sampled knobs (5 Impact + 6 distgen). Unlike the scalar v1 MLP in
``emittance_target`` (11 knobs -> log10(norm_emit_4d)), this models the full
output distribution, so a downstream reward (e.g. norm_emit_4d) can be computed
from the sampled cloud and -- via the reparameterization trick -- be
differentiated back to the knobs for first-order model-based RL.

The knob ordering, bounds, and input dimension are the single source of truth in
``photoinjector_rl.core.settings`` and are re-exported here so every script agrees.
"""

from photoinjector_rl.core.settings import (  # noqa: F401
    N_INPUT,
    SETTING_BOUNDS,
    SETTING_KEYS,
)

# Flow-specific constants (single source of truth for the rest of the package).
LATENT_DIM = 6
# Locks the (N, 6) column order used everywhere -- matches the order assumed by
# ``compute_emittance_torch`` in model.py and the ParticleGroup read in
# preprocess.py.
COORD_KEYS = ("x", "y", "z", "px", "py", "pz")
# Electron rest energy in eV; used to convert geometric 4D emittance
# sqrt(det Sigma_4d) into the normalized emittance norm_emit_4d = geo / (mc^2)^2,
# reproducing openpmd-beamphysics' ParticleGroup.norm_emit_4d.
ELECTRON_MC2_EV = 0.51099895e6
# Fixed number of macroparticles kept per run. Lofi runs hold ~2000 and the
# sweep allows loss down to ~1500, so P=1500 keeps every run (subsamples down,
# never up); hifi has 20000 -- override via preprocess --target-particles.
DEFAULT_P = 1500

__all__ = [
    "N_INPUT",
    "SETTING_BOUNDS",
    "SETTING_KEYS",
    "LATENT_DIM",
    "COORD_KEYS",
    "ELECTRON_MC2_EV",
    "DEFAULT_P",
]
