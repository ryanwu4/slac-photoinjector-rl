"""
Unit tests for the conditional affine-coupling flow. These build a tiny model
in-process (no checkpoint) and always run. The gradient tests are the acceptance
spec for the MBRL contract: emittance/forward must be differentiable w.r.t. the
11-D knob condition via the reparameterization trick.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import pytest
import torch

from photoinjector_rl.flow_surrogate import N_INPUT
from photoinjector_rl.flow_surrogate.model import (
    ConditionalAffineFlow,
    compute_emittance_torch,
)

B, P = 4, 64


def _tiny_model() -> ConditionalAffineFlow:
    torch.manual_seed(0)
    # Small but non-trivial; aux losses off so _step is cheap if exercised.
    return ConditionalAffineFlow(
        condition_dim=N_INPUT, latent_dim=6, hidden_dim=16, n_layers=4,
        w_emit_2d=0.0, w_emit_4d=0.0, w_emit_6d=0.0, w_beam=0.0,
    ).eval()


def test_forward_inverse_roundtrip() -> None:
    """Affine coupling is exactly invertible for a fixed condition."""
    m = _tiny_model()
    x = torch.randn(B, 6)
    cond = torch.rand(B, N_INPUT)
    z, _ = m.inverse(x, cond)
    x2, _ = m.forward_flow(z, cond)
    torch.testing.assert_close(x, x2, atol=1e-4, rtol=1e-4)


def test_logdet_roundtrip_cancels() -> None:
    m = _tiny_model()
    x = torch.randn(B, 6)
    cond = torch.rand(B, N_INPUT)
    z, ld_inv = m.inverse(x, cond)
    _, ld_fwd = m.forward_flow(z, cond)
    torch.testing.assert_close(ld_inv + ld_fwd, torch.zeros(B), atol=1e-4, rtol=1e-4)


def test_log_prob_finite_and_shape() -> None:
    m = _tiny_model()
    lp = m.log_prob(torch.randn(B, 6), torch.rand(B, N_INPUT))
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()


def test_sample_shape_and_finite() -> None:
    m = _tiny_model()
    out = m.sample(torch.rand(B, N_INPUT), n=50)
    assert out.shape == (B, 50, 6)
    assert torch.isfinite(out).all()


def test_emittance_from_knobs_shape_positive() -> None:
    m = _tiny_model()
    emit = m.emittance_from_knobs(torch.rand(B, N_INPUT), n=128)
    assert emit.shape == (B,)
    assert (emit > 0).all() and torch.isfinite(emit).all()


def test_forward_returns_B1_normalized() -> None:
    """The diff_env drop-in contract: (B,11) -> (B,1), finite."""
    m = _tiny_model()
    y = m(torch.rand(B, N_INPUT))
    assert y.shape == (B, 1)
    assert torch.isfinite(y).all()


def test_emittance_from_knobs_is_differentiable() -> None:
    """KEY MBRL test: gradient flows knobs -> sampled cloud -> emittance."""
    m = _tiny_model()
    torch.manual_seed(1)
    x = torch.rand(B, N_INPUT, requires_grad=True)
    emit = m.emittance_from_knobs(x, n=256)
    emit.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_forward_reward_path_is_differentiable() -> None:
    """The exact path SHAC/BPTT would use: reward = -forward(x)."""
    m = _tiny_model()
    torch.manual_seed(2)
    x = torch.rand(B, N_INPUT, requires_grad=True)
    reward = -m(x).sum()
    reward.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_no_nan_param_grads_after_step() -> None:
    """Tripwire for flow NaN-gradients: NLL backward leaves all grads finite."""
    m = _tiny_model().train()
    torch.manual_seed(3)
    particles = torch.randn(B, P, 6)
    cond = torch.rand(B, N_INPUT)
    loss = m._step((particles, cond), "train")
    loss.backward()
    for name, prm in m.named_parameters():
        if prm.grad is not None:
            assert torch.isfinite(prm.grad).all(), f"non-finite grad in {name}"


def test_compute_emittance_matches_numpy() -> None:
    """compute_emittance_torch 4D == sqrt(det cov) on a Gaussian cloud."""
    torch.manual_seed(4)
    parts = torch.randn(1, 2000, 6)
    em = compute_emittance_torch(parts)["fourd"]
    cov = torch.cov(parts[0][:, [0, 3, 1, 4]].T)
    ref = torch.sqrt(torch.abs(torch.linalg.det(cov)))
    torch.testing.assert_close(em[0], ref, atol=1e-4, rtol=1e-3)
