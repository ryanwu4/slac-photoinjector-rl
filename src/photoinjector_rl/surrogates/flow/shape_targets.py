"""
Target-trajectory generators + curriculum for the moving-target (aspect, tilt)
controller. Trajectories live in the normalized shape-vector space
(s1, s2) = ((σx²−σy²)/(σx²+σy²), 2·cov_xy/(σx²+σy²)); see properties.py. The
magnitude r=√(s1²+s2²) maps to the eigen aspect and ½·atan2(s2,s1) to the tilt.

A single `sample_shape_trajectory(rng, T, difficulty, cfg)` covers the curriculum:
  difficulty ~0  -> STATIC (constant target for the whole episode),
  mid           -> PIECEWISE-CONSTANT steps (hold k, jump; shorter holds as it rises),
  high          -> SMOOTH (tilt rotation / aspect ramp / OU walk; faster as it rises).
All targets stay inside the reachable disk r < r_max. Every magic number is in
`CurriculumConfig` (config-file driven via `diff_env.curriculum`); `CurriculumState`
holds the live `progress` (updated by the driver each epoch) + the config.

The deterministic held-out eval trajectories (staircase / tilt-rotation /
aspect-ramp) are likewise parameterized and assembled by `build_eval_trajectories`
from an `eval_trajectories` spec (config-file driven).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

# Reachable shape-vector disk: ~80% of (s1,s2) is covered in the dataset; keep
# targets a bit inside so they're achievable for most distgen contexts.
DEFAULT_R_MAX = 0.85


@dataclass
class CurriculumConfig:
    """All knobs of the training-time target-trajectory generator + curriculum
    ramp. Populated from the YAML `diff_env.curriculum` block via `from_dict`."""

    r_max: float = DEFAULT_R_MAX           # reachable disk radius for setpoints
    # regime mix vs difficulty: p_static = clip(base - slope*d, floor, 1); p_steps fixed.
    p_static_base: float = 0.75
    p_static_slope: float = 0.70
    p_static_floor: float = 0.05
    p_steps: float = 0.30
    # piecewise-constant step regime: hold length fraction of T (easy->hard).
    hold_frac_easy: float = 0.50
    hold_frac_hard: float = 0.125
    hold_min: int = 4
    # smooth regime — tilt rotation: radius range + turns over the episode (easy->hard).
    smooth_r_lo: float = 0.30
    tilt_turns_easy: float = 0.25
    tilt_turns_hard: float = 2.0
    # smooth regime — OU random walk: per-step noise (easy->hard) + mean-reversion.
    ou_sigma_easy: float = 0.02
    ou_sigma_hard: float = 0.10
    ou_decay: float = 0.92
    ou_init_frac: float = 0.70             # OU start within r_max*ou_init_frac
    # curriculum ramp: per-episode difficulty ~ progress * U(bias_lo, 1).
    difficulty_bias_lo: float = 0.40
    enabled: bool = True                   # ramp difficulty over training

    @classmethod
    def from_dict(cls, d: dict | None) -> "CurriculumConfig":
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _rand_point(rng: np.random.Generator, r_max: float) -> np.ndarray:
    """Uniform-ish point in the disk r<r_max (sqrt for area-uniform radius)."""
    r = r_max * np.sqrt(rng.uniform(0.0, 1.0))
    th = rng.uniform(-np.pi, np.pi)
    return np.array([r * np.cos(th), r * np.sin(th)], dtype=np.float64)


def _static(rng, T, cfg):
    return np.tile(_rand_point(rng, cfg.r_max), (T, 1))


def _steps(rng, T, cfg, difficulty):
    frac = cfg.hold_frac_easy + (cfg.hold_frac_hard - cfg.hold_frac_easy) * difficulty
    hold = max(cfg.hold_min, int(round(T * frac)))
    traj = np.zeros((T, 2))
    t = 0
    while t < T:
        traj[t:t + hold] = _rand_point(rng, cfg.r_max)
        t += hold
    return traj


def _smooth(rng, T, cfg, difficulty):
    kind = rng.integers(0, 3)
    ts = np.arange(T) / max(T - 1, 1)
    if kind == 0:  # tilt rotation at ~fixed aspect: r fixed, theta ramps
        r = rng.uniform(cfg.smooth_r_lo, cfg.r_max)
        n_turns = cfg.tilt_turns_easy + (cfg.tilt_turns_hard - cfg.tilt_turns_easy) * difficulty
        th0 = rng.uniform(-np.pi, np.pi)
        th = th0 + 2 * np.pi * n_turns * ts
        return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    if kind == 1:  # aspect ramp at ~fixed tilt: r ramps, theta fixed
        th = rng.uniform(-np.pi, np.pi)
        r = np.linspace(*sorted(rng.uniform(0.0, cfg.r_max, size=2)), T)
        return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    # OU random walk in the disk
    sigma = cfg.ou_sigma_easy + (cfg.ou_sigma_hard - cfg.ou_sigma_easy) * difficulty
    p = _rand_point(rng, cfg.r_max * cfg.ou_init_frac)
    out = np.zeros((T, 2))
    for t in range(T):
        p = cfg.ou_decay * p + sigma * rng.standard_normal(2)
        n = np.linalg.norm(p)
        if n > cfg.r_max:
            p *= cfg.r_max / n
        out[t] = p
    return out


def sample_shape_trajectory(rng: np.random.Generator, T: int, difficulty: float,
                            cfg: CurriculumConfig | None = None) -> np.ndarray:
    """(T, 2) target shape-vector trajectory for the given difficulty in [0,1]."""
    cfg = cfg or CurriculumConfig()
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    # difficulty shifts the regime mix: easy -> mostly static, no smooth; hard ->
    # mostly smooth.
    u = rng.uniform()
    p_static = max(cfg.p_static_floor, cfg.p_static_base - cfg.p_static_slope * difficulty)
    if u < p_static:
        return _static(rng, T, cfg)
    if u < p_static + cfg.p_steps:
        return _steps(rng, T, cfg, difficulty)
    return _smooth(rng, T, cfg, difficulty)


class CurriculumState:
    """Shared progress signal (0->1), updated by the training driver each epoch,
    plus the `CurriculumConfig`. `sample_trajectory` draws one episode's target."""

    def __init__(self, progress: float = 1.0, enabled: bool = True,
                 config: CurriculumConfig | None = None):
        self.config = config or CurriculumConfig()
        self.progress = float(progress)
        self.enabled = bool(enabled)

    def difficulty_for_episode(self, rng: np.random.Generator) -> float:
        if not self.enabled:
            return 1.0
        # Bias toward `progress`: early training easy (mostly static), late spans range.
        lo = self.config.difficulty_bias_lo
        return float(np.clip(self.progress * rng.uniform(lo, 1.0), 0.0, 1.0))

    def sample_trajectory(self, rng: np.random.Generator, T: int) -> np.ndarray:
        return sample_shape_trajectory(rng, T, self.difficulty_for_episode(rng), self.config)


# ----- deterministic held-out trajectories for evaluation --------------------

DEFAULT_STAIRCASE = [(2.0, 0.0), (2.0, 45.0), (3.0, -30.0), (1.5, 80.0)]


def eval_staircase(T: int = 64, segments=None) -> np.ndarray:
    """(aspect,tilt) step staircase with equal holds (settling / step-response)."""
    from .properties import aspect_tilt_to_s
    segs = [tuple(s) for s in (segments or DEFAULT_STAIRCASE)]
    hold = T // len(segs)
    out = np.zeros((T, 2))
    for i, (a, t) in enumerate(segs):
        s1, s2 = aspect_tilt_to_s(a, t)
        lo, hi = i * hold, (T if i == len(segs) - 1 else (i + 1) * hold)
        out[lo:hi] = [s1, s2]
    return out


def eval_tilt_rotation(T: int = 64, aspect: float = 2.0, turns: float = 1.0) -> np.ndarray:
    """Continuous tilt rotation at fixed aspect (the 'spin the beam' demo): the
    tilt sweeps `turns`*180°, i.e. the Stokes angle 2θ sweeps `turns`*360°."""
    from .properties import aspect_tilt_to_s
    r = abs(aspect_tilt_to_s(aspect, 0.0)[0])           # |s| (= r*) for this aspect
    phi = 2 * np.pi * turns * (np.arange(T) / max(T - 1, 1))   # Stokes angle = 2*tilt
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def eval_aspect_ramp(T: int = 64, tilt_deg: float = 0.0,
                     a0: float = 1.2, a1: float = 3.5) -> np.ndarray:
    """Aspect ramp at fixed tilt (a1=3.5 -> r≈0.85, the reachable-disk edge)."""
    from .properties import aspect_tilt_to_s
    aspects = np.linspace(a0, a1, T)
    return np.stack([aspect_tilt_to_s(a, tilt_deg) for a in aspects], axis=0)


# Default held-out eval spec; override any subset via a config file.
DEFAULT_EVAL_SPEC = {
    "staircase": {"segments": DEFAULT_STAIRCASE},
    "tilt_rotation": {"aspect": 2.0, "turns": 1.0},
    "aspect_ramp": {"tilt_deg": 20.0, "a0": 1.2, "a1": 3.5},
}

_EVAL_BUILDERS = {
    "staircase": eval_staircase,
    "tilt_rotation": eval_tilt_rotation,
    "aspect_ramp": eval_aspect_ramp,
}


def build_eval_trajectories(T: int, spec: dict | None = None) -> dict[str, np.ndarray]:
    """Assemble named held-out (T,2) target trajectories from a spec dict
    {name: {param: value, ...}}; falls back to DEFAULT_EVAL_SPEC."""
    spec = spec or DEFAULT_EVAL_SPEC
    out = {}
    for name, params in spec.items():
        if name not in _EVAL_BUILDERS:
            raise ValueError(f"unknown eval trajectory '{name}' "
                             f"(known: {sorted(_EVAL_BUILDERS)})")
        out[name] = _EVAL_BUILDERS[name](T, **(params or {}))
    return out
