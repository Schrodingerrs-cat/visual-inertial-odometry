"""Tests for the weighted motion-magnitude sampler (Tier-1 B.2)."""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from utils.stratified_sampler import build_weighted_motion_sampler  # noqa: E402


def _synth_three_bins(n_small=800, n_mid=150, n_large=50) -> np.ndarray:
    rng = np.random.default_rng(0)
    small = rng.uniform(0.0, 0.02, n_small)
    mid   = rng.uniform(0.04, 0.06, n_mid)
    large = rng.uniform(0.10, 0.15, n_large)
    return np.concatenate([small, mid, large])


def _bin_of(mag: float) -> int:
    if mag <= 0.03:
        return 0
    if mag <= 0.08:
        return 1
    return 2


def _bin_counts_after_resample(mags: np.ndarray, alpha: float,
                               num_samples: int = 3000) -> Counter:
    edges = [0.0, 0.03, 0.08, 1.0]
    sampler = build_weighted_motion_sampler(
        mags, alpha=alpha, num_bins=3, bin_edges=edges,
        num_samples=num_samples, seed=42)
    picks = list(iter(sampler))
    counts = Counter(_bin_of(mags[i]) for i in picks)
    # Ensure all three bins are keys even if zero (they won't be).
    for b in (0, 1, 2):
        counts.setdefault(b, 0)
    return counts


def test_alpha_zero_approximates_proportional() -> None:
    mags = _synth_three_bins()
    counts = _bin_counts_after_resample(mags, alpha=0.0, num_samples=3000)
    # Natural proportions: 800 / 1000 = 0.80, 150 / 1000 = 0.15, 50 / 1000 = 0.05.
    total = sum(counts.values())
    frac_small = counts[0] / total
    frac_large = counts[2] / total
    assert frac_small > 0.65 and frac_small < 0.90
    assert frac_large < 0.10


def test_alpha_one_approximates_balanced() -> None:
    mags = _synth_three_bins()
    counts = _bin_counts_after_resample(mags, alpha=1.0, num_samples=3000)
    total = sum(counts.values())
    # Every bin should be ~33% (generous band to avoid flakiness).
    for b in (0, 1, 2):
        frac = counts[b] / total
        assert 0.20 <= frac <= 0.45, \
            f"bin {b} frac {frac:.2f} out of balanced band"


def test_alpha_half_is_between_zero_and_one() -> None:
    mags = _synth_three_bins()
    c0 = _bin_counts_after_resample(mags, alpha=0.0, num_samples=3000)
    c5 = _bin_counts_after_resample(mags, alpha=0.5, num_samples=3000)
    c1 = _bin_counts_after_resample(mags, alpha=1.0, num_samples=3000)
    total = 3000
    # Fraction from the rare large-motion bin (bin 2).
    large0 = c0[2] / total
    large5 = c5[2] / total
    large1 = c1[2] / total
    assert large0 < large5 < large1


def test_empty_array_raises() -> None:
    with pytest.raises(ValueError):
        build_weighted_motion_sampler(np.array([]), alpha=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
