"""Phase 2 losses — t-direct variant for Step H.

Drops the (t/d, n) decomposition supervision. The vision network now predicts
metric translation directly, so the loss is just rotation + translation:

    L_V  = w_rot · L_rot + w_t · L_t
    L_VI = L_V_branch + L_I_branch + w_rot · L_rot_fused + w_trans · L_t_fused

L_t is L1 over the 3 axes, hover-weighted by ``hovering_weight(|t_gt|, …)``
(the same smooth sigmoid as the (t/d) loss; it was always computed on |t_rel|
in metres, not |t/d|, so no recomputation was needed when switching).

Removed vs ``losses.py``:
    * L_n              — no plane-normal head.
    * L_warp           — photometric warp depended on (t/d, n) reconstruction.
    * L_d              — VI no longer has a learned scale; t comes out metric.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .losses import (
    geodesic_rotation_angle,
    hovering_weight,
    IMUPoseLoss,
)


class TDirectPoseLoss(nn.Module):
    """V-branch loss: w_rot · L_rot + w_t · L_t (hover-weighted)."""

    def __init__(self, w_rot: float = 10.0, w_t: float = 5.0,
                 hover_center: float = 0.01,
                 hover_width: float = 0.005) -> None:
        super().__init__()
        self.w_rot = float(w_rot)
        self.w_t = float(w_t)
        self.hover_center = float(hover_center)
        self.hover_width = float(hover_width)

    def forward(self, R_pred: torch.Tensor, t_pred: torch.Tensor,
                R_gt: torch.Tensor, t_gt: torch.Tensor
               ) -> Dict[str, torch.Tensor]:
        t_norm = t_gt.norm(dim=-1)
        hover_w = hovering_weight(t_norm, self.hover_center, self.hover_width)

        l_rot = geodesic_rotation_angle(R_pred, R_gt).mean()
        l_t = (t_pred - t_gt).abs().sum(-1)                      # (B,)
        w_sum = hover_w.sum().clamp_min(1e-6)
        l_t_m = (l_t * hover_w).sum() / w_sum

        total = self.w_rot * l_rot + self.w_t * l_t_m
        return {
            "total": total,
            "l_rot": l_rot,
            "l_t": l_t_m,
            "hover_weight_mean": hover_w.mean(),
        }


class VILoss_tdirect(nn.Module):
    """Full VI loss for the t-direct variant.

    Vision branch loss + IMU branch loss + fused-pose terms. No learned-scale
    term (l_d) because there is no scale head in NetworkVI_tdirect.
    """

    def __init__(self, w_rot: float = 10.0, w_t: float = 5.0,
                 w_trans: float = 5.0,
                 w_imu_trans: float = 2.0, w_rot_imu: float = 5.0,
                 hover_center: float = 0.01,
                 hover_width: float = 0.005) -> None:
        super().__init__()
        self.vision = TDirectPoseLoss(
            w_rot=w_rot, w_t=w_t,
            hover_center=hover_center, hover_width=hover_width)
        self.imu = IMUPoseLoss(w_rot=w_rot_imu, w_trans=w_imu_trans)
        self.w_trans = float(w_trans)
        self.w_rot = float(w_rot)

    def forward(self, preds: Dict[str, torch.Tensor],
                R_gt: torch.Tensor, t_gt: torch.Tensor
               ) -> Dict[str, torch.Tensor]:
        vis = self.vision(preds["R_v"], preds["t_v"], R_gt, t_gt)
        imu = self.imu(preds["R_i"], preds["t_i"], R_gt, t_gt)

        l_t_fused = (preds["t"] - t_gt).abs().sum(-1).mean()
        l_rot_fused = geodesic_rotation_angle(preds["R"], R_gt).mean()

        total = (vis["total"] + imu["total"]
                 + self.w_trans * l_t_fused
                 + self.w_rot * l_rot_fused)
        return {
            "total": total,
            "l_vision": vis["total"],
            "l_imu": imu["total"],
            "l_rot_fused": l_rot_fused,
            "l_trans_fused": l_t_fused,
            **{f"vis_{k}": v for k, v in vis.items() if k != "total"},
            **{f"imu_{k}": v for k, v in imu.items() if k != "total"},
        }
