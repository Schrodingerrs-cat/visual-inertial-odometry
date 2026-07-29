"""Motion-magnitude samplers.

- ``build_weighted_motion_sampler`` — Tier-1 B.2 default. Returns a
  ``WeightedRandomSampler`` whose per-sample weight is
  ``(1 / bin_count[bin_of_sample]) ** alpha``. Controlled by a single knob
  ``alpha ∈ [0.0, 1.0]``:
    * 0.0 → uniform sampling (follows the dataset's natural motion distribution)
    * 0.5 → moderate rebalancing (default)
    * 1.0 → strict balancing (each bin contributes equally in expectation)

  ``alpha = 0.5`` gives a middle ground: boost the high-motion tail without
  starving the bulk of the data.

- ``build_small_only_sampler`` — draws only from the smallest |Δt| bin;
  used by the motion-magnitude study to reproduce the zero-predictor failure.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler, SubsetRandomSampler, WeightedRandomSampler


def _compute_bin_membership(motion: np.ndarray, num_bins: int = 5,
                            bin_edges: Optional[Sequence[float]] = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_of_sample, edges). Edges have length num_bins+1."""
    motion = np.asarray(motion, dtype=np.float64)
    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=np.float64)
        assert np.all(np.diff(edges) > 0), "bin_edges must be strictly increasing"
    else:
        qs = np.linspace(0.0, 1.0, num_bins + 1)
        edges = np.quantile(motion, qs)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9
    bin_of_sample = np.digitize(motion, edges[1:-1], right=False)
    return bin_of_sample.astype(np.int64), edges


def build_weighted_motion_sampler(motion_magnitudes: np.ndarray,
                                  alpha: float = 0.5,
                                  num_bins: int = 5,
                                  bin_edges: Optional[Sequence[float]] = None,
                                  num_samples: Optional[int] = None,
                                  seed: int = 0
                                  ) -> WeightedRandomSampler:
    """Build a PyTorch WeightedRandomSampler for motion-magnitude rebalancing.

    sample_weight[i] = (1 / bin_count[bin_of_sample[i]]) ** alpha
    Bins with zero count contribute no samples (weight 0) — handled safely.
    """
    motion = np.asarray(motion_magnitudes, dtype=np.float64)
    if motion.size == 0:
        raise ValueError("motion_magnitudes must be non-empty")
    alpha = float(alpha)
    bin_of, _edges = _compute_bin_membership(motion, num_bins=num_bins,
                                             bin_edges=bin_edges)
    bin_counts = np.bincount(bin_of, minlength=num_bins).astype(np.float64)
    safe_counts = np.where(bin_counts > 0, bin_counts, 1.0)
    bin_weight = (1.0 / safe_counts) ** alpha
    bin_weight[bin_counts == 0] = 0.0
    sample_weight = bin_weight[bin_of]

    if sample_weight.sum() > 0:
        sample_weight = sample_weight / sample_weight.sum() * sample_weight.size

    n = int(num_samples or motion.size)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weight, dtype=torch.double),
        num_samples=n, replacement=True, generator=generator)


def build_small_only_sampler(motion_magnitudes: np.ndarray,
                             threshold_m: Optional[float] = None,
                             quantile: float = 0.2,
                             seed: int = 0) -> Sampler:
    """Draw only from samples whose |Δt| is in the smallest bin / quantile."""
    mag = np.asarray(motion_magnitudes, dtype=np.float64)
    if threshold_m is None:
        threshold_m = float(np.quantile(mag, float(quantile)))
    mask = mag <= threshold_m
    if mask.sum() == 0:
        raise RuntimeError("small_only sampler: no samples pass threshold "
                           f"{threshold_m} m (max mag = {mag.max():.3f} m)")
    indices = np.where(mask)[0]
    rng = np.random.default_rng(int(seed))
    return SubsetRandomSampler(rng.permutation(indices).tolist())
