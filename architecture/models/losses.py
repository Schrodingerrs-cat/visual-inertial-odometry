"""Phase 2 losses.

- `GeodesicRotationLoss` — angular distance between predicted and GT rotations.
- `DecomposedHomographyLoss` — sum of (geodesic R) + (td L1) + (n angle) +
  optional (photometric warp residual via Kornia). Tier-1 mitigation B.1:
  L_td and L_n are multiplied by a SMOOTH hovering weight
  ``sigmoid((|Δt| - center) / width)``, not a hard threshold — so the
  gradient signal fades in/out continuously as a pair crosses the hover
  boundary. L_rot is never weighted because rotation is always observable.
- `IMUPoseLoss` — geodesic rotation + translation L1.
- `VILoss` — combined loss for the full VI network.
- `photometric_warp_residual` — homography-warped photometric residual.
- `hovering_weight` — public helper so sanity scripts and tests can reuse it.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

try:
    import kornia
except ImportError:  # pragma: no cover
    kornia = None  # type: ignore


_EPS = 1e-7


def geodesic_rotation_angle(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Returns angular error in radians (B,)."""
    M = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = M.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + _EPS, 1.0 - _EPS)
    return torch.acos(cos_theta)


class GeodesicRotationLoss(nn.Module):
    """Mean geodesic rotation error in radians."""

    def forward(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        return geodesic_rotation_angle(R_pred, R_gt).mean()


def _unit_vector_angle(n_pred: torch.Tensor, n_gt: torch.Tensor) -> torch.Tensor:
    dot = (F.normalize(n_pred, dim=-1, eps=_EPS)
           * F.normalize(n_gt, dim=-1, eps=_EPS)).sum(-1)
    dot = dot.clamp(-1.0 + _EPS, 1.0 - _EPS)
    return torch.acos(dot)


def hovering_weight(dt_magnitude: torch.Tensor,
                    center: float = 0.01,
                    width: float = 0.005) -> torch.Tensor:
    """Smooth sigmoid weight in [0, 1] over per-sample ``|Δt_gt|``.

    Weight ≈ 0 for dt << center, ≈ 1 for dt >> center. Intermediate dt values
    blend smoothly so the gradient is continuous across the hover boundary —
    replaces the hard ``(dt >= 0.01).float()`` threshold (Tier-1 B.1).
    """
    return torch.sigmoid((dt_magnitude - float(center)) / float(width))


# OpenGL → CV basis flip. Takes a vector in Blender camera frame
# (+X right, +Y up, −Z forward) to CV frame (+X right, +Y down, +Z forward).
# Its own inverse: S @ S = I.
_OPENGL_TO_CV_T = torch.tensor(
    [[1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0]]
)


def photometric_warp_residual(img_a: torch.Tensor, img_b: torch.Tensor,
                              K: torch.Tensor,
                              R_ba: torch.Tensor, t_ba: torch.Tensor,
                              d: torch.Tensor) -> torch.Tensor:
    """MAE between image_b and H-warped image_a (ground-plane Blender scene).

    Inputs are in Blender/OpenGL camera convention; the function applies the
    CV basis flip internally so the homography formula is consistent.

    Arguments
    ---------
    img_a, img_b : (B, 3, H, W)           images in image coordinates.
    K            : (3, 3) or (B, 3, 3)    camera intrinsics.
    R_ba         : (B, 3, 3)              rotation: cam_a vectors → cam_b frame.
    t_ba         : (B, 3)                 position of cam_a's origin in cam_b.
    d            : (B,)                   altitude (camera → plane, positive).

    The plane normal is always ``(0, 0, −1)`` in CV frame (ground below the
    camera), so it's hardcoded — no ``n`` input.
    """
    if kornia is None:
        raise RuntimeError("kornia is required for photometric_warp_residual")
    B = img_a.shape[0]
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)
    S = _OPENGL_TO_CV_T.to(R_ba.device, R_ba.dtype)
    R_cv = S @ R_ba @ S                                      # (B, 3, 3)
    t_cv = (S @ t_ba.unsqueeze(-1)).squeeze(-1)              # (B, 3)
    n_col = torch.tensor([0.0, 0.0, -1.0],
                         device=R_ba.device,
                         dtype=R_ba.dtype).view(1, 3, 1)
    t_col = t_cv.unsqueeze(-1)
    d_safe = d.clamp_min(1e-3).view(B, 1, 1)
    M = R_cv - (t_col @ n_col.transpose(-1, -2)) / d_safe
    K_inv = torch.linalg.inv(K)
    H = K @ M @ K_inv
    warped = kornia.geometry.transform.homography_warp(
        img_a, H, (img_a.shape[-2], img_a.shape[-1]),
        mode='bilinear', padding_mode='zeros', normalized_coordinates=False,
        align_corners=False)
    return (warped - img_b).abs().mean()


# ---------------------------------------------------------------------------
# DecomposedHomographyLoss — Network V / vision branch of VI
# ---------------------------------------------------------------------------

class DecomposedHomographyLoss(nn.Module):
    def __init__(self, w_rot: float = 10.0, w_td: float = 1.0, w_n: float = 0.1,
                 w_warp: float = 0.1,
                 hover_center: float = 0.01,
                 hover_width: float = 0.005) -> None:
        super().__init__()
        self.w_rot = float(w_rot)
        self.w_td = float(w_td)
        self.w_n = float(w_n)
        self.w_warp = float(w_warp)
        self.hover_center = float(hover_center)
        self.hover_width = float(hover_width)

    def forward(self, R_pred: torch.Tensor, td_pred: torch.Tensor,
                n_pred: torch.Tensor,
                R_gt: torch.Tensor, t_gt: torch.Tensor,
                n_gt: torch.Tensor, d_gt: torch.Tensor,
                img_a: Optional[torch.Tensor] = None,
                img_b: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        # Scale GT translation by the known plane depth.
        td_gt = t_gt / d_gt.clamp_min(1e-3).unsqueeze(-1)
        t_norm = t_gt.norm(dim=-1)
        hover_w = hovering_weight(t_norm, self.hover_center, self.hover_width)

        l_rot = geodesic_rotation_angle(R_pred, R_gt).mean()
        l_td = (td_pred - td_gt).abs().sum(-1)  # (B,)
        l_n = _unit_vector_angle(n_pred, n_gt)   # (B,)
        # Smooth hover weighting: sum(w * l) / sum(w), not a hard mean.
        w_sum = hover_w.sum().clamp_min(1e-6)
        l_td_m = (l_td * hover_w).sum() / w_sum
        l_n_m = (l_n * hover_w).sum() / w_sum

        total = self.w_rot * l_rot + self.w_td * l_td_m + self.w_n * l_n_m
        if self.w_warp > 0 and img_a is not None and img_b is not None and K is not None:
            try:
                # photometric_warp_residual wants the pose of cam_a in cam_b
                # (R_ba, t_ba). Network predicts R_ab, td_ab = t_ab/d — we
                # invert to get (R_ba, t_ba):
                #   R_ba = R_ab^T
                #   t_ba = −R_ab^T · t_ab = −R_ba · t_ab
                R_ba = R_pred.transpose(-1, -2)
                t_ab = td_pred * d_gt.unsqueeze(-1)
                t_ba = -torch.einsum("bij,bj->bi", R_ba, t_ab)
                l_warp = photometric_warp_residual(
                    img_a, img_b, K, R_ba, t_ba, d_gt)
                total = total + self.w_warp * l_warp
            except RuntimeError:
                l_warp = torch.tensor(0.0, device=total.device)
        else:
            l_warp = torch.tensor(0.0, device=total.device)

        return {
            "total": total,
            "l_rot": l_rot,
            "l_td": l_td_m,
            "l_n": l_n_m,
            "l_warp": l_warp,
            "hover_weight_mean": hover_w.mean(),
        }


# ---------------------------------------------------------------------------
# IMUPoseLoss — Network I
# ---------------------------------------------------------------------------

class IMUPoseLoss(nn.Module):
    def __init__(self, w_rot: float = 5.0, w_trans: float = 1.0) -> None:
        super().__init__()
        self.w_rot = float(w_rot)
        self.w_trans = float(w_trans)

    def forward(self, R_pred: torch.Tensor, t_pred: torch.Tensor,
                R_gt: torch.Tensor, t_gt: torch.Tensor
               ) -> Dict[str, torch.Tensor]:
        l_rot = geodesic_rotation_angle(R_pred, R_gt).mean()
        l_t = (t_pred - t_gt).abs().sum(-1).mean()
        total = self.w_rot * l_rot + self.w_trans * l_t
        return {"total": total, "l_rot": l_rot, "l_trans": l_t}


class IMUPoseLoss_hover(nn.Module):
    """IMU-branch loss with hover-weighting on translation.

    Mirrors TDirectPoseLoss's translation handling: down-weights samples
    with very small |t_gt| so that "predict zero" is no longer locally
    optimal. L_rot is unweighted because rotation is always observable.
    """

    def __init__(self, w_rot: float = 5.0, w_trans: float = 1.0,
                 hover_center: float = 0.01,
                 hover_width: float = 0.005) -> None:
        super().__init__()
        self.w_rot = float(w_rot)
        self.w_trans = float(w_trans)
        self.hover_center = float(hover_center)
        self.hover_width = float(hover_width)

    def forward(self, R_pred: torch.Tensor, t_pred: torch.Tensor,
                R_gt: torch.Tensor, t_gt: torch.Tensor
               ) -> Dict[str, torch.Tensor]:
        l_rot = geodesic_rotation_angle(R_pred, R_gt).mean()

        t_norm = t_gt.norm(dim=-1)
        hover_w = hovering_weight(t_norm, self.hover_center, self.hover_width)

        l_t_per_sample = (t_pred - t_gt).abs().sum(-1)
        w_sum = hover_w.sum().clamp_min(1e-6)
        l_t = (l_t_per_sample * hover_w).sum() / w_sum

        total = self.w_rot * l_rot + self.w_trans * l_t
        return {
            "total": total,
            "l_rot": l_rot,
            "l_trans": l_t,
            "hover_weight_mean": hover_w.mean(),
        }


# ---------------------------------------------------------------------------
# VILoss — full Network VI
# ---------------------------------------------------------------------------

class VILoss(nn.Module):
    """Combined loss for Network VI.

    Sub-losses:
        • Vision branch: DecomposedHomographyLoss on (R_v, td_v, n_v).
        • IMU branch:    IMUPoseLoss on (R_i, t_i).
        • Scale:         L1 on log(d_final) vs log(||t_gt||), smooth-weighted
                         by hovering_weight to suppress near-static pairs.
        • Fused pose:    geodesic on R, L1 on t (rotation always weighted 1).
    """

    def __init__(self, w_rot: float = 10.0, w_td: float = 1.0, w_n: float = 0.1,
                 w_warp: float = 0.0, w_trans: float = 1.0,
                 w_imu_trans: float = 0.5, w_rot_imu: float = 5.0,
                 w_d: float = 0.5,
                 hover_center: float = 0.01,
                 hover_width: float = 0.005) -> None:
        super().__init__()
        self.vision = DecomposedHomographyLoss(
            w_rot=w_rot, w_td=w_td, w_n=w_n, w_warp=w_warp,
            hover_center=hover_center, hover_width=hover_width)
        self.imu = IMUPoseLoss(w_rot=w_rot_imu, w_trans=w_imu_trans)
        self.w_trans = float(w_trans)
        self.w_d = float(w_d)
        self.hover_center = float(hover_center)
        self.hover_width = float(hover_width)

    def forward(self, preds: Dict[str, torch.Tensor],
                R_gt: torch.Tensor, t_gt: torch.Tensor,
                n_gt: torch.Tensor, d_gt: torch.Tensor,
                img_a: Optional[torch.Tensor] = None,
                img_b: Optional[torch.Tensor] = None,
                K: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
        vis = self.vision(preds["R_v"], preds["td_v"], preds["n_v"],
                          R_gt, t_gt, n_gt, d_gt, img_a, img_b, K)
        imu = self.imu(preds["R_i"], preds["t_i"], R_gt, t_gt)

        t_gt_norm = t_gt.norm(dim=-1)
        hover_w = hovering_weight(t_gt_norm, self.hover_center, self.hover_width)
        if "d_final" in preds:
            diff = (preds["d_final"].clamp_min(1e-6).log()
                    - t_gt_norm.clamp_min(1e-6).log()).abs()
            w_sum = hover_w.sum().clamp_min(1e-6)
            l_d = (diff * hover_w).sum() / w_sum
        else:
            l_d = torch.tensor(0.0, device=R_gt.device)

        l_t_fused = (preds["t"] - t_gt).abs().sum(-1).mean()
        l_rot_fused = geodesic_rotation_angle(preds["R"], R_gt).mean()

        total = (vis["total"] + imu["total"]
                 + self.w_trans * l_t_fused
                 + self.w_d * l_d
                 + self.vision.w_rot * l_rot_fused)
        return {
            "total": total,
            "l_vision": vis["total"],
            "l_imu": imu["total"],
            "l_rot_fused": l_rot_fused,
            "l_trans_fused": l_t_fused,
            "l_d": l_d,
            **{f"vis_{k}": v for k, v in vis.items() if k != "total"},
            **{f"imu_{k}": v for k, v in imu.items() if k != "total"},
        }
