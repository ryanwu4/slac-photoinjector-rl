"""
LEGACY first-iteration surrogate: MLP from 11 XOPT-sampled knob values ->
scalar norm_emit_4d at PR10241.

Kept for provenance. The conditional-flow surrogate (``photoinjector_rl.surrogates.flow``)
is the maintained path; new work should target that. The knob ordering, bounds,
and input dimension now live in ``photoinjector_rl.core.settings`` and are
re-exported here for backwards compatibility.
"""

from photoinjector_rl.core.settings import (  # noqa: F401
    N_INPUT,
    SETTING_BOUNDS,
    SETTING_KEYS,
)

__all__ = ["N_INPUT", "SETTING_BOUNDS", "SETTING_KEYS"]
