"""
Vendored SHAC / BPTT from NVlabs/DiffRL, stripped of dflex and physics-engine
deps so they can drive an arbitrary differentiable torch env.

The DiffRL class skeletons are kept ~verbatim; only the env-instantiation
sites were modified to import a local env directly instead of `getattr(envs,
...)`. See `shac.py` and `bptt.py` for the changes.

Upstream: https://github.com/NVlabs/DiffRL (ICLR 2022, Apache 2.0-style header
preserved in each file).
"""
from .shac import SHAC
from .bptt import BPTT

__all__ = ["SHAC", "BPTT"]
