"""Tests for PhotometricAugment."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from utils.dataloader import PhotometricAugment  # noqa: E402


def _rand_pair(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(3, 32, 32, generator=g)
    b = torch.rand(3, 32, 32, generator=g)
    return a, b


def test_same_transform_applied_to_both_frames() -> None:
    """Identical input -> identical output under the paired transform."""
    aug = PhotometricAugment(brightness=0.3, contrast=0.3, saturation=0.2,
                             hue=0.05, gaussian_noise_std=0.0, grayscale_prob=0.0)
    a, _b = _rand_pair(seed=1)
    torch.manual_seed(123)
    out_a, out_b = aug(a.clone(), a.clone())
    assert torch.allclose(out_a, out_b)


def test_output_shape_matches_input() -> None:
    aug = PhotometricAugment()
    a, b = _rand_pair(seed=2)
    out_a, out_b = aug(a, b)
    assert out_a.shape == a.shape
    assert out_b.shape == b.shape


def test_output_clamped_to_unit_range() -> None:
    aug = PhotometricAugment(brightness=0.5, contrast=0.5, saturation=0.3,
                             hue=0.05, gaussian_noise_std=0.1, grayscale_prob=0.0)
    a, b = _rand_pair(seed=3)
    out_a, out_b = aug(a, b)
    assert float(out_a.min()) >= 0.0 - 1e-6
    assert float(out_a.max()) <= 1.0 + 1e-6
    assert float(out_b.min()) >= 0.0 - 1e-6
    assert float(out_b.max()) <= 1.0 + 1e-6


def test_grayscale_applies_to_both_or_neither() -> None:
    """When the stochastic grayscale path triggers, both frames flatten."""
    aug = PhotometricAugment(brightness=0.0, contrast=0.0, saturation=0.0,
                             hue=0.0, gaussian_noise_std=0.0,
                             grayscale_prob=1.0)  # force grayscale every call
    a, b = _rand_pair(seed=4)
    out_a, out_b = aug(a, b)
    assert torch.allclose(out_a[0], out_a[1])
    assert torch.allclose(out_a[1], out_a[2])
    assert torch.allclose(out_b[0], out_b[1])
    assert torch.allclose(out_b[1], out_b[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
