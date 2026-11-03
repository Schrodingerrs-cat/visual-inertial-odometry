"""Unit tests for losses.

Run with:
    pytest tests/test_losses.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.losses import (  # noqa: E402
    DecomposedHomographyLoss, GeodesicRotationLoss, IMUPoseLoss, VILoss,
    hovering_weight,
)


def _identity_batch(B: int = 4) -> dict:
    eye = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    return {
        "R_pred": eye.clone(),
        "R_gt":   eye.clone(),
        "td_pred": torch.zeros(B, 3),
        "n_pred":  torch.tensor([[0., 0., 1.]]).expand(B, -1).clone(),
        "n_gt":    torch.tensor([[0., 0., 1.]]).expand(B, -1).clone(),
        "t_gt":    torch.zeros(B, 3),           # hovering
        "d_gt":    torch.full((B,), 4.0),
    }


def test_geodesic_identity() -> None:
    R = torch.eye(3).unsqueeze(0).expand(4, -1, -1).clone()
    g = GeodesicRotationLoss()
    out = g(R, R)
    # acos(1 - _EPS) ≈ sqrt(2·_EPS) ≈ 4.5e-4 with _EPS=1e-7 and float32.
    # This is the floor of the clamp-then-acos pipeline, not an error.
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-3)


def test_hovering_weight_monotone_and_endpoints() -> None:
    """Smooth hover mask: small at small dt, ~1 at large dt, monotone between."""
    dt = torch.tensor([0.001, 0.005, 0.010, 0.015, 0.020, 0.050, 0.100])
    w = hovering_weight(dt, center=0.01, width=0.005)
    # With width=0.005, sigmoid((0.001-0.01)/0.005) = sigmoid(-1.8) ≈ 0.142.
    # The mask isn't designed for hard zeroing — just soft down-weighting.
    assert float(w[0]) < 0.2
    assert float(w[-1]) > 0.95
    # Monotone non-decreasing.
    diffs = (w[1:] - w[:-1]).numpy()
    assert (diffs >= -1e-6).all()


def test_decomposed_loss_hovering_suppresses_td_and_n() -> None:
    b = _identity_batch()
    loss = DecomposedHomographyLoss(w_rot=1.0, w_td=1.0, w_n=1.0, w_warp=0.0,
                                    hover_center=0.01, hover_width=0.005)
    out = loss(b["R_pred"], b["td_pred"], b["n_pred"],
               R_gt=b["R_gt"], t_gt=b["t_gt"], n_gt=b["n_gt"], d_gt=b["d_gt"])
    # With t_gt = 0 everywhere, hover_w ≈ sigmoid(-2) ≈ 0.12 — small but
    # nonzero. l_td reduces to weighted zero-vs-zero, so ~0.
    # l_n has the same _EPS=1e-7 acos floor as geodesic rotation (≈4.5e-4).
    assert float(out["l_td"]) == pytest.approx(0.0, abs=1e-6)
    assert float(out["l_n"]) == pytest.approx(0.0, abs=1e-3)


def test_decomposed_loss_nonhover_uses_td() -> None:
    b = _identity_batch()
    b["t_gt"] = torch.tensor([[0.0, 0.0, 0.1]]).expand(4, -1).clone()  # 10 cm
    loss = DecomposedHomographyLoss(w_rot=0.0, w_td=1.0, w_n=0.0, w_warp=0.0)
    out = loss(b["R_pred"], b["td_pred"], b["n_pred"],
               R_gt=b["R_gt"], t_gt=b["t_gt"], n_gt=b["n_gt"], d_gt=b["d_gt"])
    assert float(out["l_td"]) > 0.0


def test_vi_loss_allzero_predictions_reasonable() -> None:
    B = 4
    eye = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    preds = {
        "R_v": eye.clone(), "td_v": torch.zeros(B, 3),
        "n_v": torch.tensor([[0., 0., 1.]]).expand(B, -1).clone(),
        "R_i": eye.clone(), "t_i": torch.zeros(B, 3),
        "d_final": torch.ones(B),
        "R": eye.clone(), "t": torch.zeros(B, 3),
    }
    loss = VILoss(w_rot=1.0, w_td=1.0, w_n=1.0, w_warp=0.0, w_trans=1.0,
                  w_imu_trans=1.0, w_rot_imu=1.0, w_d=1.0)
    out = loss(preds, R_gt=eye.clone(), t_gt=torch.zeros(B, 3),
               n_gt=torch.tensor([[0., 0., 1.]]).expand(B, -1).clone(),
               d_gt=torch.full((B,), 4.0))
    assert torch.isfinite(out["total"])


def test_imu_loss_zero_when_perfect() -> None:
    B = 4
    eye = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    t = torch.zeros(B, 3)
    loss = IMUPoseLoss(w_rot=1.0, w_trans=1.0)
    out = loss(eye.clone(), t.clone(), eye.clone(), t.clone())
    # Floor ≈ 4.5e-4 from w_rot·acos(1 - _EPS) — see geodesic identity test.
    assert float(out["total"]) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
