"""Phase 2 losses — anchored VI variant for Step I.

Combines fused-pose supervision (the headline loss), weak per-branch
supervision (so V and I don't drift while the gates learn), and a binary
gate-supervision signal that tells the gates which branch is currently
closer to the ground truth.

Total loss:

    L = w_rot · L_rot_fused + w_t · L_t_fused                    # fused
      + 0.5 · (w_rot · L_rot_v + w_t · L_t_v)                    # weak V
      + 0.5 · (w_rot · L_rot_i + w_t · L_t_i)                    # weak I
      + w_gate · (L_gate_t + L_gate_R)                            # gates

Gate supervision targets:

    target_alpha_t = (||t_i − t_gt|| > ||t_v − t_gt||).float()
    target_alpha_R = (geodesic(R_i, R_gt) > geodesic(R_v, R_gt)).float()

I.e. the gate is asked to output 1 (trust V) when V's per-sample error is
strictly smaller than I's, and 0 (trust I) when I is closer. ``.detach()``
on the targets keeps the gate gradient from leaking back into V or I.

Without ``L_gate_supervision`` the gates would either collapse to 0.5 (no
preference) or learn the wrong direction (e.g. trusting V even when V is
wrong, because the fused-pose loss alone has many ways to be small). With
it, the gate has an explicit per-sample teaching signal.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .losses import geodesic_rotation_angle, hovering_weight


class GateSupervisionLoss(nn.Module):
    """Binary cross-entropy on the gate weights vs the closer-branch target.

    See module docstring for the target definition. Returns the **sum** of
    the translation-gate and rotation-gate BCE terms — the caller scales by
    ``w_gate``.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, alpha_t: torch.Tensor, alpha_R: torch.Tensor,
                t_v: torch.Tensor, t_i: torch.Tensor,
                R_v: torch.Tensor, R_i: torch.Tensor,
                t_gt: torch.Tensor, R_gt: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        err_t_v = (t_v - t_gt).norm(dim=-1)
        err_t_i = (t_i - t_gt).norm(dim=-1)
        target_alpha_t = (err_t_i > err_t_v).float().detach()        # (B,)

        err_R_v = geodesic_rotation_angle(R_v, R_gt)
        err_R_i = geodesic_rotation_angle(R_i, R_gt)
        target_alpha_R = (err_R_i > err_R_v).float().detach()        # (B,)

        a_t = alpha_t.clamp(self.eps, 1.0 - self.eps)
        a_R = alpha_R.clamp(self.eps, 1.0 - self.eps)
        l_gate_t = F.binary_cross_entropy(a_t, target_alpha_t)
        l_gate_R = F.binary_cross_entropy(a_R, target_alpha_R)

        # Diagnostics: how often is V actually the closer branch? Useful
        # in CHANGES.md for sanity-checking the gate's working set.
        with torch.no_grad():
            target_t_mean = target_alpha_t.mean()
            target_R_mean = target_alpha_R.mean()

        return {
            "total": l_gate_t + l_gate_R,
            "l_gate_t": l_gate_t,
            "l_gate_R": l_gate_R,
            "target_alpha_t_mean": target_t_mean,
            "target_alpha_R_mean": target_R_mean,
        }


class VILoss_anchored(nn.Module):
    """Full VI loss for the anchored variant.

    Components:
      * fused: w_rot · L_rot_fused + w_t · L_t_fused
      * weak V supervision at half weight
      * weak I supervision at half weight
      * w_gate · (L_gate_t + L_gate_R)

    Translation losses are L1 over the 3 axes, with the same smooth
    hover-weighting (``hovering_weight(|t_gt|, …)``) used by every other
    Phase 2 translation loss so near-static pairs don't dominate.
    """

    def __init__(self, w_rot: float = 10.0, w_t: float = 5.0,
                 w_gate: float = 1.0,
                 hover_center: float = 0.01,
                 hover_width: float = 0.005) -> None:
        super().__init__()
        self.w_rot = float(w_rot)
        self.w_t = float(w_t)
        self.w_gate = float(w_gate)
        self.hover_center = float(hover_center)
        self.hover_width = float(hover_width)
        self.gate_loss = GateSupervisionLoss()

    def _t_l1_hovered(self, t_pred: torch.Tensor, t_gt: torch.Tensor,
                      hover_w: torch.Tensor) -> torch.Tensor:
        l_t = (t_pred - t_gt).abs().sum(-1)                          # (B,)
        w_sum = hover_w.sum().clamp_min(1e-6)
        return (l_t * hover_w).sum() / w_sum

    def forward(self, preds: Dict[str, torch.Tensor],
                R_gt: torch.Tensor, t_gt: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        t_norm = t_gt.norm(dim=-1)
        hover_w = hovering_weight(t_norm, self.hover_center, self.hover_width)

        # Fused pose
        l_rot_fused = geodesic_rotation_angle(preds["R"], R_gt).mean()
        l_t_fused = self._t_l1_hovered(preds["t"], t_gt, hover_w)

        # Per-branch (weak) supervision
        l_rot_v = geodesic_rotation_angle(preds["R_v"], R_gt).mean()
        l_t_v = self._t_l1_hovered(preds["t_v"], t_gt, hover_w)
        l_rot_i = geodesic_rotation_angle(preds["R_i"], R_gt).mean()
        l_t_i = self._t_l1_hovered(preds["t_i"], t_gt, hover_w)

        # Gate supervision
        gate = self.gate_loss(
            alpha_t=preds["alpha_t"], alpha_R=preds["alpha_R"],
            t_v=preds["t_v"], t_i=preds["t_i"],
            R_v=preds["R_v"], R_i=preds["R_i"],
            t_gt=t_gt, R_gt=R_gt,
        )

        total = (self.w_rot * l_rot_fused + self.w_t * l_t_fused
                 + 0.5 * (self.w_rot * l_rot_v + self.w_t * l_t_v)
                 + 0.5 * (self.w_rot * l_rot_i + self.w_t * l_t_i)
                 + self.w_gate * gate["total"])

        return {
            "total": total,
            "l_rot_fused": l_rot_fused,
            "l_t_fused": l_t_fused,
            "l_rot_v": l_rot_v,
            "l_t_v": l_t_v,
            "l_rot_i": l_rot_i,
            "l_t_i": l_t_i,
            "l_gate_t": gate["l_gate_t"],
            "l_gate_R": gate["l_gate_R"],
            "target_alpha_t_mean": gate["target_alpha_t_mean"],
            "target_alpha_R_mean": gate["target_alpha_R_mean"],
            "alpha_t_mean": preds["alpha_t"].detach().mean(),
            "alpha_R_mean": preds["alpha_R"].detach().mean(),
            "rho_v_mean": preds["rho_v"].detach().mean(),
            "hover_weight_mean": hover_w.mean(),
        }


__all__ = ["VILoss_anchored", "GateSupervisionLoss"]
