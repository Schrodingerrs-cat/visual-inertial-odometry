"""Smoke tests for Step K's RAFT teacher and distillation loss.

Designed to run on CPU in <60 s without downloading the full RAFT-large
checkpoint. Tests cover:

  * Forward-backward consistency residual is zero on identity flow.
  * FB residual ≈ |shift| on a globally-shifted flow pair.
  * Per-pair median σ reduction is monotone in input magnitude.
  * Homography → pose round-trip on a synthetic homography.
  * TDirectDistillLoss computes finite outputs and respects valid_mask
    (zero distill term when no pairs are valid).
  * NaN guards: σ = NaN, σ = 0, σ = inf all produce finite losses.

The teacher's actual `predict_full` (which downloads RAFT weights) is
*not* exercised here — that path is exercised on the cluster by
``scripts/precompute_raft.py``.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from models.losses_distill import TDirectDistillLoss  # noqa: E402
from models.raft_teacher import (  # noqa: E402
    _flow_to_correspondences_np, _homography_to_pose_blender,
    fb_inconsistency_pixel, per_pair_flow_magnitude_px, per_pair_sigma_px,
)
from utils.geometry import (  # noqa: E402
    compose_homography_blender_plane, rodrigues_to_matrix,
)


class FBConsistencyTest(unittest.TestCase):
    def test_identity_flow_is_consistent(self) -> None:
        """Zero flow forward and backward → zero FB residual everywhere."""
        B, H, W = 1, 64, 80
        zero = torch.zeros(B, 2, H, W)
        incons = fb_inconsistency_pixel(zero, zero)
        self.assertEqual(incons.shape, (B, 1, H, W))
        self.assertLess(float(incons.abs().max()), 1e-5)

    def test_global_shift_inconsistency(self) -> None:
        """A constant shift forward + zero backward → residual ≈ |shift|."""
        B, H, W = 1, 64, 80
        shift = 3.0  # pixels
        flow_fwd = torch.zeros(B, 2, H, W)
        flow_fwd[:, 0] = shift                                # x-direction
        flow_bwd = torch.zeros(B, 2, H, W)                    # B says no flow
        incons = fb_inconsistency_pixel(flow_fwd, flow_bwd)
        # Interior pixels should see residual ≈ shift.
        interior = incons[..., 8:-8, 8:-8]
        self.assertAlmostEqual(float(interior.median()), shift, places=2)

    def test_sigma_reduction_monotone(self) -> None:
        B, H, W = 2, 64, 80
        # Pair 0: zero residual; pair 1: 5 px residual.
        flow_fwd = torch.zeros(B, 2, H, W)
        flow_fwd[1, 0] = 5.0
        flow_bwd = torch.zeros(B, 2, H, W)
        incons = fb_inconsistency_pixel(flow_fwd, flow_bwd)
        sigma = per_pair_sigma_px(incons)
        self.assertEqual(sigma.shape, (B,))
        self.assertLess(float(sigma[0]), 1e-3)
        self.assertGreater(float(sigma[1]), 4.5)

    def test_flow_magnitude_median(self) -> None:
        B, H, W = 1, 32, 40
        flow = torch.zeros(B, 2, H, W)
        flow[:, 0] = 2.0; flow[:, 1] = 0.0                    # |v| = 2 px
        mag = per_pair_flow_magnitude_px(flow)
        self.assertAlmostEqual(float(mag[0]), 2.0, places=4)


class HomographyToPoseRoundTripTest(unittest.TestCase):
    """Build a known H from (R, t/d) and confirm the decomposer recovers it."""

    def test_round_trip_identity_pose(self) -> None:
        K = np.array([[640.0, 0.0, 320.0],
                      [0.0, 640.0, 240.0],
                      [0.0, 0.0, 1.0]])
        # Identity rotation, zero translation, depth=4 m.
        R_ba = np.eye(3)
        t_ba = np.zeros(3)
        H = compose_homography_blender_plane(K, R_ba, t_ba, altitude=4.0)
        R_rec, t_rec = _homography_to_pose_blender(H, K, depth_m=4.0)
        self.assertIsNotNone(R_rec)
        self.assertIsNotNone(t_rec)
        # Identity round-trip: R should be I and t should be ~0.
        np.testing.assert_allclose(R_rec, np.eye(3), atol=1e-4)
        self.assertLess(float(np.linalg.norm(t_rec)), 1e-3)

    def test_round_trip_small_translation(self) -> None:
        K = np.array([[640.0, 0.0, 320.0],
                      [0.0, 640.0, 240.0],
                      [0.0, 0.0, 1.0]])
        # Small purely-translational motion: t_ba in cam_b's frame.
        # We use the OpenGL/Blender convention; the helper handles the flip.
        R_ba = np.eye(3)
        t_ba = np.array([0.05, 0.0, 0.0])                    # 5 cm in x_cam
        depth = 4.0
        H = compose_homography_blender_plane(K, R_ba, t_ba, altitude=depth)
        # _homography_to_pose_blender returns (R_ab, t_ab) — pose of cam_b
        # *in cam_a's frame*. For pure translation this equals (-R_ba @ t_ba)
        # under standard SE(3) inversion = (R_ba^T @ t_ba) for R=I.
        # Round-trip should give a small ‖t‖ in the same order of magnitude.
        R_rec, t_rec = _homography_to_pose_blender(H, K, depth_m=depth)
        self.assertIsNotNone(R_rec)
        self.assertIsNotNone(t_rec)
        # t magnitude should be in the right ballpark (within 50%).
        recovered_norm = float(np.linalg.norm(t_rec))
        self.assertLess(abs(recovered_norm - 0.05), 0.025,
                        f"recovered ‖t‖={recovered_norm:.4f} m, expected ~0.05 m")


class FlowCorrespondencesTest(unittest.TestCase):
    def test_grid_sampling_count(self) -> None:
        flow = np.zeros((2, 480, 640), dtype=np.float32)
        src, dst = _flow_to_correspondences_np(flow, stride=16)
        # 30 × 40 = 1200 correspondences at stride 16.
        self.assertEqual(src.shape, (1200, 2))
        self.assertEqual(dst.shape, (1200, 2))
        # Zero flow → src == dst.
        np.testing.assert_allclose(dst, src, atol=1e-6)


class DistillLossTest(unittest.TestCase):
    """End-to-end smoke for TDirectDistillLoss with various edge cases."""

    def _dummy_batch(self, B: int = 4):
        torch.manual_seed(0)
        # Random small student rotations and translations.
        aa = torch.randn(B, 3) * 0.05
        R_pred = rodrigues_to_matrix(aa)
        t_pred = torch.randn(B, 3) * 0.05
        # GT and teacher are slightly noisier versions of the student.
        R_gt = rodrigues_to_matrix(aa + 0.01 * torch.randn(B, 3))
        t_gt = t_pred + 0.005 * torch.randn(B, 3)
        R_b = rodrigues_to_matrix(aa + 0.02 * torch.randn(B, 3))
        t_b = t_pred + 0.01 * torch.randn(B, 3)
        sigma_b_px = torch.full((B,), 2.0)                     # 2 px σ
        valid = torch.ones(B, dtype=torch.bool)
        depth_m = torch.full((B,), 4.0)
        fx_px = torch.tensor(640.0)
        return dict(
            R_pred=R_pred, t_pred=t_pred,
            R_gt=R_gt, t_gt=t_gt,
            R_b=R_b, t_b=t_b,
            sigma_b_px=sigma_b_px, valid=valid,
            depth_m=depth_m, fx_px=fx_px,
        )

    def test_loss_finite_on_normal_inputs(self) -> None:
        loss = TDirectDistillLoss(lambda_distill=1.0)
        out = loss(**self._dummy_batch())
        for k, v in out.items():
            self.assertTrue(torch.isfinite(v).all(),
                            f"non-finite value in {k}: {v}")
        self.assertGreater(float(out["total"]), 0.0)
        self.assertGreater(float(out["valid_frac"]), 0.99)

    def test_no_valid_pairs_zero_distill(self) -> None:
        """When all valid=False, distill term contributes 0 (only pose loss)."""
        loss = TDirectDistillLoss(lambda_distill=1.0)
        batch = self._dummy_batch()
        batch["valid"] = torch.zeros(batch["valid"].shape, dtype=torch.bool)
        out = loss(**batch)
        self.assertAlmostEqual(float(out["l_distill"]), 0.0, places=6)
        self.assertAlmostEqual(float(out["valid_frac"]), 0.0, places=6)
        # total should equal pose-only.
        self.assertAlmostEqual(float(out["total"]),
                               float(out["l_pose"]), places=5)

    def test_nan_sigma_does_not_propagate(self) -> None:
        """σ = NaN should be clipped + nan_to_num'd; loss stays finite."""
        loss = TDirectDistillLoss(lambda_distill=1.0)
        batch = self._dummy_batch()
        batch["sigma_b_px"] = torch.tensor([float("nan"), 2.0, 2.0, 2.0])
        out = loss(**batch)
        self.assertTrue(torch.isfinite(out["total"]))
        self.assertTrue(torch.isfinite(out["l_distill_t"]))
        self.assertTrue(torch.isfinite(out["l_distill_R"]))

    def test_zero_sigma_floored(self) -> None:
        """σ = 0 should be floored to SIGMA_PX_FLOOR; loss stays finite."""
        loss = TDirectDistillLoss(lambda_distill=1.0)
        batch = self._dummy_batch()
        batch["sigma_b_px"] = torch.zeros_like(batch["sigma_b_px"])
        out = loss(**batch)
        self.assertTrue(torch.isfinite(out["total"]))
        # mean_sigma_t should reflect the floor: 0.5 px * (4 m / 640) ≈ 3.1 mm
        self.assertAlmostEqual(float(out["mean_sigma_t"]),
                               0.5 * 4.0 / 640.0, places=4)

    def test_inf_sigma_zeroes_distill(self) -> None:
        """σ = inf → weights → 0 → distill term effectively 0."""
        loss = TDirectDistillLoss(lambda_distill=1.0)
        batch = self._dummy_batch()
        batch["sigma_b_px"] = torch.full(batch["sigma_b_px"].shape,
                                         float("inf"))
        out = loss(**batch)
        self.assertTrue(torch.isfinite(out["total"]))
        self.assertLess(float(out["l_distill"]), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
