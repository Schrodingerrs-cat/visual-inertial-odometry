"""Tests for NetworkVI inference-time fallback flag (Tier-1 B.3)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from models.network_vi import NetworkVI  # noqa: E402


@pytest.fixture
def tiny_model():
    m = NetworkVI(feat_dim=32, lstm_hidden=16, fallback_threshold=0.05)
    return m


def _inputs(B: int = 2, H: int = 64, W: int = 64, T: int = 10):
    img_a = torch.rand(B, 3, H, W)
    img_b = torch.rand(B, 3, H, W)
    imu = torch.randn(B, T, 6)
    return img_a, img_b, imu


def test_training_mode_has_no_fallback_key(tiny_model) -> None:
    tiny_model.train()
    img_a, img_b, imu = _inputs()
    out = tiny_model(img_a, img_b, imu)
    assert "use_imu_fallback" not in out, \
        "fallback flag should not be present during training"


def test_eval_mode_adds_fallback_key(tiny_model) -> None:
    tiny_model.eval()
    img_a, img_b, imu = _inputs()
    with torch.no_grad():
        out = tiny_model(img_a, img_b, imu)
    assert "use_imu_fallback" in out
    assert out["use_imu_fallback"].dtype == torch.bool
    assert out["use_imu_fallback"].shape[0] == img_a.shape[0]


def test_fallback_true_when_td_norm_tiny(tiny_model) -> None:
    """Force td_v to near-zero by monkey-patching the vision head output."""
    tiny_model.eval()
    B = 3
    img_a, img_b, imu = _inputs(B=B)

    real_vision = tiny_model.vision

    class _ZeroTdVision(torch.nn.Module):
        def __init__(self, real):
            super().__init__()
            self.real = real
        def forward(self, a, b):
            R, _td, n, v = self.real(a, b)
            td0 = torch.zeros_like(_td) + 1e-4  # << 0.05
            return R, td0, n, v

    tiny_model.vision = _ZeroTdVision(real_vision)
    with torch.no_grad():
        out = tiny_model(img_a, img_b, imu)
    assert bool(out["use_imu_fallback"].all().item()), \
        f"expected all-True fallback mask, got {out['use_imu_fallback']}"
    # Restore.
    tiny_model.vision = real_vision


def test_fallback_false_when_td_norm_large(tiny_model) -> None:
    """Force td_v to be large; fallback flag should be all False."""
    tiny_model.eval()
    img_a, img_b, imu = _inputs(B=3)

    real_vision = tiny_model.vision

    class _BigTdVision(torch.nn.Module):
        def __init__(self, real):
            super().__init__()
            self.real = real
        def forward(self, a, b):
            R, _td, n, v = self.real(a, b)
            big = torch.ones_like(_td) * 0.5  # >> 0.05
            return R, big, n, v

    tiny_model.vision = _BigTdVision(real_vision)
    with torch.no_grad():
        out = tiny_model(img_a, img_b, imu)
    assert not bool(out["use_imu_fallback"].any().item())
    tiny_model.vision = real_vision


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
