"""RAFT-based teacher for Step K knowledge distillation.

Wraps ``torchvision.models.optical_flow.raft_large`` (frozen) as a per-pair
pose teacher:

    img_a, img_b
        |
    raft_large (12 iterations)  →  flow_fwd, flow_bwd
        |
    forward-backward consistency  →  σ_b (per-pair median pixel residual)
        |
    flow_fwd → grid correspondences → cv2.findHomography (USAC_MAGSAC)
        |
    decompose_homography(H, K, prior_normal) → (R_b, t_b/d, n)
        |
    t_b = (t_b/d) · depth_gt        (scale recovery from known altitude)

The output is per-pair (R_b, t_b, σ_b_pixels, valid_mask). Distillation in
``models/losses_distill.py`` converts σ_b to metric/angular units before
inverse-variance-weighting the residual against the student.

This module is **training-time only**. It is never imported by the eval
pipeline; ``scripts/precompute_raft.py`` runs it once to populate
``data/raw_g/traj_{N}/raft_predictions.npz``, and ``scripts/train_v_distill.py``
reads those caches. The student model itself is unchanged
(``NetworkV_tdirect``), so inference cost is identical to Step J.

References:
    * RAFT: Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for
      Optical Flow," ECCV 2020.
    * Forward-backward consistency: standard occlusion-aware uncertainty
      proxy used widely in self-supervised flow / depth literature.
    * Homography decomposition with prior normal: see
      ``utils/geometry.py::decompose_homography``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:  # cv2 is required by the homography path; surface failure only when used.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

from utils.geometry import decompose_homography


# ---------------------------------------------------------------------------
# Defaults — kept here so precompute scripts and training agree on settings.
# ---------------------------------------------------------------------------

DEFAULT_NUM_FLOW_UPDATES = 12
DEFAULT_CORRESPONDENCE_STRIDE = 16
DEFAULT_RANSAC_THRESHOLD_PX = 2.0
DEFAULT_FB_VALID_THRESHOLD_PX = 5.0
DEFAULT_MIN_FLOW_MAGNITUDE_PX = 1.0
DEFAULT_MAX_FB_RESIDUAL_PX = 30.0
DEFAULT_MAX_T_NORM_M = 1.0
DEFAULT_MAX_R_GEODESIC_RAD = float(np.pi / 4)
# Below this pixel σ we treat RAFT as over-confident — clip up to avoid
# division-by-zero amplification in the distillation loss.
SIGMA_PX_FLOOR = 0.5


# OpenGL → CV basis flip (project-wide convention; matches utils/geometry.py).
_OPENGL_TO_CV_NP = np.diag([1.0, -1.0, -1.0])


@dataclass
class TeacherPrediction:
    """Per-pair teacher output. All fields are per-batch-item arrays.

    Shapes:
        R_b           (B, 3, 3) float32 — teacher rotation (cam-a → cam-b).
        t_b           (B, 3)    float32 — teacher translation in cam-a frame
                                          (metres, scaled by GT depth).
        sigma_b_px    (B,)      float32 — median forward-backward inconsistency
                                          in pixels.
        valid_mask    (B,)      bool    — passed all sanity checks.
        flow_med_px   (B,)      float32 — median flow magnitude (diagnostic).
    """
    R_b: np.ndarray
    t_b: np.ndarray
    sigma_b_px: np.ndarray
    valid_mask: np.ndarray
    flow_med_px: np.ndarray


# ---------------------------------------------------------------------------
# Forward-backward consistency
# ---------------------------------------------------------------------------

def _grid_warp_flow(flow_bwd: torch.Tensor, flow_fwd: torch.Tensor
                    ) -> torch.Tensor:
    """Warp ``flow_bwd`` using ``flow_fwd``.

    Conceptually, for each source pixel x_a in image_a:
      x_b = x_a + flow_fwd[x_a]                  # implied destination in B
      sampled = flow_bwd at x_b
    Returns ``flow_bwd`` resampled at the locations implied by flow_fwd, so
    that the FB residual ``flow_fwd + flow_bwd_warped`` is comparable
    pixel-by-pixel.

    Both inputs are (B, 2, H, W) in pixel units.
    """
    B, _, H, W = flow_fwd.shape
    device = flow_fwd.device
    dtype = flow_fwd.dtype

    # Build an integer pixel grid (x_a) of shape (B, H, W, 2) with channel
    # order (x, y) to match flow channel ordering.
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1)                      # (H, W, 2)
    base = base.unsqueeze(0).expand(B, -1, -1, -1)            # (B, H, W, 2)

    # Implied destination pixels x_b = x_a + flow_fwd
    fwd_xy = flow_fwd.permute(0, 2, 3, 1)                     # (B, H, W, 2)
    x_b = base + fwd_xy                                       # (B, H, W, 2)

    # Convert to grid_sample's normalized coordinates in [-1, 1].
    norm_x = (2.0 * x_b[..., 0] / max(W - 1, 1)) - 1.0
    norm_y = (2.0 * x_b[..., 1] / max(H - 1, 1)) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)              # (B, H, W, 2)

    # Sample flow_bwd at x_b. ``align_corners=True`` matches the
    # normalization formula above.
    sampled = F.grid_sample(flow_bwd, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    return sampled                                             # (B, 2, H, W)


def fb_inconsistency_pixel(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor
                           ) -> torch.Tensor:
    """Per-pixel forward-backward inconsistency in pixels.

    For a perfectly invertible flow field, ``flow_fwd[x_a] + flow_bwd[x_a +
    flow_fwd[x_a]] == 0``. The L2 norm of this residual is a per-pixel
    confidence proxy; large values indicate occlusion or texture-poor
    regions where RAFT's estimate is unreliable.

    Returns shape (B, 1, H, W) in pixels.
    """
    bwd_warped = _grid_warp_flow(flow_bwd, flow_fwd)
    res = flow_fwd + bwd_warped                              # (B, 2, H, W)
    mag = res.norm(dim=1, keepdim=True)                       # (B, 1, H, W)
    return mag


def per_pair_sigma_px(inconsistency_pixel: torch.Tensor) -> torch.Tensor:
    """Reduce per-pixel inconsistency (B, 1, H, W) to per-pair σ_b in pixels.

    Uses the spatial *median* (more robust than mean to occluded regions).
    Returns shape (B,).
    """
    flat = inconsistency_pixel.flatten(start_dim=1)           # (B, H*W)
    return flat.median(dim=-1).values                         # (B,)


def per_pair_flow_magnitude_px(flow_fwd: torch.Tensor) -> torch.Tensor:
    """Median flow magnitude per pair, in pixels. Diagnostic / valid mask."""
    mag = flow_fwd.norm(dim=1, keepdim=True)                 # (B, 1, H, W)
    flat = mag.flatten(start_dim=1)
    return flat.median(dim=-1).values                         # (B,)


# ---------------------------------------------------------------------------
# Flow → correspondences → homography → pose
# ---------------------------------------------------------------------------

def _flow_to_correspondences_np(flow_fwd_np: np.ndarray,
                                stride: int = DEFAULT_CORRESPONDENCE_STRIDE
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """For one image (2, H, W), return (src_xy, dst_xy) pixel correspondences.

    Sampled on a regular stride grid. With H=480, W=640, stride=16 → 30×40
    = 1200 correspondences — plenty for cv2.findHomography RANSAC.
    """
    _, H, W = flow_fwd_np.shape
    ys = np.arange(0, H, stride, dtype=np.int64)
    xs = np.arange(0, W, stride, dtype=np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")               # (gy, gx)
    src = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    flow_at_src = flow_fwd_np[:, yy.ravel(), xx.ravel()]      # (2, N)
    dst = src + flow_at_src.T.astype(np.float32)              # (N, 2)
    return src, dst


def _homography_to_pose_blender(
    H: np.ndarray, K: np.ndarray, depth_m: float,
    prior_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Decompose ``H`` and recover (R_ab, t_ab) in Blender / OpenGL camera frame.

    ``decompose_homography`` returns the CV-frame ``(R_cv, (t/d)_cv, n_cv)``.
    For pose targets that match the dataloader's `R_rel`/`t_rel` convention
    (Blender / OpenGL) we apply the inverse basis flip ``S = diag(1, -1, -1)``
    (S = S^{-1}) on both sides:

        R_blender = S @ R_cv @ S
        (t/d)_blender = S @ (t/d)_cv

    Then scale by the per-frame depth to recover metric translation.
    """
    if H is None or not np.all(np.isfinite(H)):
        return None, None
    try:
        R_cv, td_cv, _n_cv = decompose_homography(H, K, prior_normal=prior_normal)
    except Exception:  # pragma: no cover — defensive
        return None, None
    if not (np.all(np.isfinite(R_cv)) and np.all(np.isfinite(td_cv))):
        return None, None

    # The Step F/G/H convention is: ``R_rel`` from the dataloader rotates a
    # vector expressed in cam_a's frame to cam_b's frame, and the camera frame
    # is OpenGL/Blender (+X right, +Y up, −Z forward). decompose_homography
    # returns the CV-frame analogue. Apply the basis flip to align.
    S = _OPENGL_TO_CV_NP
    R_blender = S @ R_cv @ S
    td_blender = S @ np.asarray(td_cv).reshape(3)
    t_blender = td_blender * float(depth_m)
    return R_blender.astype(np.float32), t_blender.astype(np.float32)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class RAFTTeacher(nn.Module):
    """Frozen RAFT-based pose teacher.

    Construction is lazy — the underlying ``raft_large`` is built on
    ``setup()`` so we don't pay the weight-download cost in unit tests that
    only exercise the math helpers.
    """

    def __init__(self,
                 weights: str = "Raft_Large_Weights.C_T_SKHT_V2",
                 num_flow_updates: int = DEFAULT_NUM_FLOW_UPDATES,
                 correspondence_stride: int = DEFAULT_CORRESPONDENCE_STRIDE,
                 ransac_threshold_px: float = DEFAULT_RANSAC_THRESHOLD_PX,
                 ) -> None:
        super().__init__()
        self.weights_name = str(weights)
        self.num_flow_updates = int(num_flow_updates)
        self.correspondence_stride = int(correspondence_stride)
        self.ransac_threshold_px = float(ransac_threshold_px)
        self._raft: Optional[nn.Module] = None
        self._tfm = None

    # ------------------------------------------------------------------
    # Setup / weights
    # ------------------------------------------------------------------

    def setup(self, device: torch.device) -> None:
        """Lazily download + load the pretrained RAFT model onto ``device``."""
        if self._raft is not None:
            return
        from torchvision.models.optical_flow import (
            raft_large, Raft_Large_Weights,
        )
        # Resolve a string like "Raft_Large_Weights.C_T_SKHT_V2" → enum.
        attr_name = self.weights_name.split(".")[-1]
        weights_enum = getattr(Raft_Large_Weights, attr_name,
                               Raft_Large_Weights.DEFAULT)
        m = raft_large(weights=weights_enum, progress=False)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        self._raft = m.to(device)
        self._tfm = weights_enum.transforms()

    # ------------------------------------------------------------------
    # Forward pass — flow only
    # ------------------------------------------------------------------

    @torch.no_grad()
    def flow(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """One-direction flow img_a → img_b. Returns (B, 2, H, W) pixels."""
        assert self._raft is not None, "call setup(device) first"
        a, b = self._tfm(img_a, img_b)
        flows = self._raft(a, b, num_flow_updates=self.num_flow_updates)
        # ``flows`` is a list of refinement iterates; final iterate is the
        # production flow.
        return flows[-1]

    @torch.no_grad()
    def bidirectional_flow(self, img_a: torch.Tensor, img_b: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward (a→b) and backward (b→a) flow."""
        flow_fwd = self.flow(img_a, img_b)
        flow_bwd = self.flow(img_b, img_a)
        # NaN guard (L3): zero out any NaNs / Infs in the flow output. RAFT
        # is generally stable but we don't trust unconditionally.
        flow_fwd = torch.nan_to_num(flow_fwd, nan=0.0, posinf=0.0, neginf=0.0)
        flow_bwd = torch.nan_to_num(flow_bwd, nan=0.0, posinf=0.0, neginf=0.0)
        return flow_fwd, flow_bwd

    # ------------------------------------------------------------------
    # Full per-pair prediction (R, t, σ, valid)
    # ------------------------------------------------------------------

    def predict_full(self, img_a: torch.Tensor, img_b: torch.Tensor,
                     K: torch.Tensor, depth_m: torch.Tensor,
                     fb_valid_threshold_px: float = DEFAULT_FB_VALID_THRESHOLD_PX,
                     min_flow_magnitude_px: float = DEFAULT_MIN_FLOW_MAGNITUDE_PX,
                     max_fb_residual_px: float = DEFAULT_MAX_FB_RESIDUAL_PX,
                     max_t_norm_m: float = DEFAULT_MAX_T_NORM_M,
                     max_R_geodesic_rad: float = DEFAULT_MAX_R_GEODESIC_RAD,
                     ) -> TeacherPrediction:
        """End-to-end teacher prediction for a batch of frame pairs.

        Parameters
        ----------
        img_a, img_b : (B, 3, H, W) float in [0, 1].
        K            : (3, 3) intrinsics in pixel units.
        depth_m      : (B,) per-frame altitude in metres (positive scalar).
        ``*_threshold`` arguments: see ``valid_mask`` composition below.

        Returns ``TeacherPrediction`` (numpy arrays, batch-first).
        """
        if cv2 is None:
            raise RuntimeError("cv2 is required for RAFTTeacher.predict_full")

        device = img_a.device
        self.setup(device)

        flow_fwd, flow_bwd = self.bidirectional_flow(img_a, img_b)
        incons = fb_inconsistency_pixel(flow_fwd, flow_bwd)        # (B, 1, H, W)
        sigma_px = per_pair_sigma_px(incons)                        # (B,)
        flow_med = per_pair_flow_magnitude_px(flow_fwd)             # (B,)

        sigma_px = torch.nan_to_num(sigma_px, nan=float("inf"))
        flow_med = torch.nan_to_num(flow_med, nan=0.0)

        # Move to numpy for cv2.
        flow_fwd_np = flow_fwd.detach().cpu().numpy()                # (B, 2, H, W)
        sigma_np = sigma_px.detach().cpu().numpy().astype(np.float32)
        flow_med_np = flow_med.detach().cpu().numpy().astype(np.float32)
        K_np = K.detach().cpu().numpy().astype(np.float64)
        depth_np = depth_m.detach().cpu().numpy().astype(np.float64)

        B = flow_fwd_np.shape[0]
        R_b = np.zeros((B, 3, 3), dtype=np.float32)
        t_b = np.zeros((B, 3), dtype=np.float32)
        valid = np.zeros((B,), dtype=bool)

        # Precompute global mask predicates that don't need cv2.
        flow_ok = (flow_med_np > float(min_flow_magnitude_px)) & np.isfinite(flow_med_np)
        sigma_ok = (sigma_np < float(max_fb_residual_px)) & np.isfinite(sigma_np)

        for i in range(B):
            # Default: identity rotation, zero translation, invalid.
            R_b[i] = np.eye(3, dtype=np.float32)
            t_b[i] = 0.0
            if not (flow_ok[i] and sigma_ok[i]):
                continue
            src, dst = _flow_to_correspondences_np(
                flow_fwd_np[i], stride=self.correspondence_stride)
            if len(src) < 4:
                continue
            try:
                H, mask = cv2.findHomography(
                    src, dst, cv2.USAC_MAGSAC, self.ransac_threshold_px)
            except cv2.error:
                continue
            if H is None or mask is None or int(mask.sum()) < 8:
                continue
            R_i, t_i = _homography_to_pose_blender(H, K_np, float(depth_np[i]))
            if R_i is None or t_i is None:
                continue
            t_norm = float(np.linalg.norm(t_i))
            tr = float(np.clip(0.5 * (np.trace(R_i) - 1.0), -1.0 + 1e-7,
                               1.0 - 1e-7))
            R_geo = float(np.arccos(tr))
            if (t_norm > max_t_norm_m) or (R_geo > max_R_geodesic_rad):
                continue
            R_b[i] = R_i
            t_b[i] = t_i
            valid[i] = True

        return TeacherPrediction(
            R_b=R_b, t_b=t_b,
            sigma_b_px=sigma_np, valid_mask=valid,
            flow_med_px=flow_med_np,
        )


__all__ = [
    "RAFTTeacher", "TeacherPrediction",
    "fb_inconsistency_pixel", "per_pair_sigma_px",
    "per_pair_flow_magnitude_px",
    "DEFAULT_NUM_FLOW_UPDATES", "DEFAULT_CORRESPONDENCE_STRIDE",
    "DEFAULT_RANSAC_THRESHOLD_PX", "DEFAULT_FB_VALID_THRESHOLD_PX",
    "SIGMA_PX_FLOOR",
]
