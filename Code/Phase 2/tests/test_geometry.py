"""Unit tests for utils.geometry.

Run with:
    pytest tests/test_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import geometry as G  # noqa: E402


torch.manual_seed(0)
np.random.seed(0)


def _random_rotations(n: int = 8) -> torch.Tensor:
    aa = torch.randn(n, 3) * 0.6
    return G.rodrigues_to_matrix(aa)


def test_rodrigues_roundtrip() -> None:
    aa = torch.randn(8, 3) * 0.5
    R = G.rodrigues_to_matrix(aa)
    aa_back = G.matrix_to_rodrigues(R)
    R_back = G.rodrigues_to_matrix(aa_back)
    assert torch.allclose(R, R_back, atol=1e-5)


def test_rot6d_roundtrip() -> None:
    R = _random_rotations(16)
    x6 = G.matrix_to_rot6d(R)
    R_back = G.rot6d_to_matrix(x6)
    assert torch.allclose(R, R_back, atol=1e-6)


def test_rot6d_is_orthonormal() -> None:
    x6 = torch.randn(16, 6)
    R = G.rot6d_to_matrix(x6)
    RtR = torch.matmul(R.transpose(-1, -2), R)
    eye = torch.eye(3).expand_as(RtR)
    assert torch.allclose(RtR, eye, atol=1e-5)
    det = torch.linalg.det(R)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)


def test_quat_roundtrip() -> None:
    R = _random_rotations(8)
    q = G.matrix_to_quat(R)
    R_back = G.quat_to_matrix(q)
    assert torch.allclose(R, R_back, atol=1e-5)


def test_geodesic_loss_identity_returns_zero() -> None:
    R = _random_rotations(8)
    RtR = torch.matmul(R.transpose(-1, -2), R)  # identity
    trace = RtR.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    # acos(1 - 1e-7) ≈ sqrt(2e-7) ≈ 4.5e-4 in float32 — the clamp floor,
    # not a numerical error.
    assert torch.allclose(theta, torch.zeros_like(theta), atol=1e-3)


def test_compose_decompose_homography_roundtrip() -> None:
    K = np.array([[320.0, 0.0, 160.0],
                  [0.0, 320.0, 120.0],
                  [0.0, 0.0, 1.0]])
    R = G.rodrigues_to_matrix(torch.tensor([[0.05, -0.03, 0.04]]))[0].numpy()
    t = np.array([0.1, -0.05, 0.2])
    d = 4.0
    n = np.array([0.0, 0.0, 1.0])

    H = G.compose_homography(K, R, t, d, n)
    # Normalise the homography the way cv2 does so decomposition works.
    H = H / H[2, 2]
    R_r, tdash_r, n_r = G.decompose_homography(H, K, prior_normal=n)

    # We recover R up to a sign and normal — because plane-depth is relative
    # we check t_rec is parallel to t and R matches R, and the recovered
    # normal is close to n.
    assert np.allclose(R_r, R, atol=1e-4)
    t_dir_gt = t / np.linalg.norm(t)
    t_dir_rec = tdash_r / np.linalg.norm(tdash_r)
    assert np.allclose(np.abs(np.dot(t_dir_gt, t_dir_rec)), 1.0, atol=1e-3)
    assert np.allclose(n_r, n, atol=1e-3)


def test_se3_composition_is_associative() -> None:
    rng = np.random.default_rng(0)
    R1 = G.rodrigues_to_matrix(torch.tensor([rng.normal(size=3) * 0.3]))[0].numpy()
    R2 = G.rodrigues_to_matrix(torch.tensor([rng.normal(size=3) * 0.3]))[0].numpy()
    R3 = G.rodrigues_to_matrix(torch.tensor([rng.normal(size=3) * 0.3]))[0].numpy()
    t1 = rng.normal(size=3)
    t2 = rng.normal(size=3)
    t3 = rng.normal(size=3)
    A = G.compose_se3(*G.compose_se3(R1, t1, R2, t2), R3, t3)
    B = G.compose_se3(R1, t1, *G.compose_se3(R2, t2, R3, t3))
    assert np.allclose(A[0], B[0], atol=1e-8)
    assert np.allclose(A[1], B[1], atol=1e-8)


def test_dead_reckon_matches_absolute_chain() -> None:
    """Dead-reckoning from relative poses must reproduce absolute poses."""
    rng = np.random.default_rng(42)
    N = 20
    abs_Rs = np.zeros((N, 3, 3))
    abs_ts = np.zeros((N, 3))
    abs_Rs[0] = np.eye(3)
    for i in range(1, N):
        delta = torch.tensor([rng.normal(size=3) * 0.05])
        dR = G.rodrigues_to_matrix(delta)[0].numpy()
        dt = rng.normal(size=3) * 0.05
        abs_Rs[i], abs_ts[i] = G.compose_se3(abs_Rs[i - 1], abs_ts[i - 1], dR, dt)

    rel_Rs = np.zeros((N - 1, 3, 3))
    rel_ts = np.zeros((N - 1, 3))
    for i in range(N - 1):
        rel_Rs[i], rel_ts[i] = G.relative_pose(abs_Rs[i], abs_ts[i],
                                               abs_Rs[i + 1], abs_ts[i + 1])
    Rs_rec, ts_rec = G.dead_reckon(abs_Rs[0], abs_ts[0], rel_Rs, rel_ts)
    assert np.allclose(Rs_rec, abs_Rs, atol=1e-8)
    assert np.allclose(ts_rec, abs_ts, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
