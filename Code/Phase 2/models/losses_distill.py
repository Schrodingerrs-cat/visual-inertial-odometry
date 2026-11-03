"""Phase 2 losses — distillation variant for Step K.

Composes the existing ``TDirectPoseLoss`` (rotation + translation against GT)
with an inverse-variance distillation term against the cached RAFT teacher:

    L = L_pose + λ · valid_mask · (1/(σ_t²+ε_t²)) · ‖t_a − t_b‖²
              + λ · valid_mask · (1/(σ_R²+ε_R²)) · geodesic(R_a, R_b)²

Critical detail: RAFT's forward-backward consistency residual ``σ_b_px`` is in
**pixel units**, but ``t_a − t_b`` is in **metres**, and ``geodesic(R_a, R_b)``
is in **radians**. To make the inverse-variance weighting unit-correct we
project σ_b_px through the camera intrinsics and per-frame depth:

    σ_t  ≈ σ_b_px · (depth / fx)              # pixel error → metric error
    σ_R  ≈ σ_b_px / fx                         # pixel error → angular error

(``fx`` is the focal length in pixels, dataloader's intrinsics ``K[0, 0]``.)

The distillation is *masked* — only frame pairs where the teacher passed all
sanity checks (forward-backward consistency, plausible flow magnitude,
homography decomposition succeeded, plausible motion magnitude) contribute.
The ``valid_frac`` diagnostic is logged so we can spot teacher coverage drops
during training.

Cross-references:
    * ``models/losses_tdirect.py::TDirectPoseLoss`` — the wrapped pose loss.
    * ``models/raft_teacher.py``                    — produces R_b/t_b/σ.
    * ``scripts/precompute_raft.py``                — caches the targets.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .losses import geodesic_rotation_angle
from .losses_tdirect import TDirectPoseLoss


# Below this pixel σ we treat RAFT as over-confident (sub-pixel claim) and
# clip up — keeps the inverse-variance term from dominating the loss when
# RAFT happens to produce a near-zero FB residual on a static-looking pair.
SIGMA_PX_FLOOR = 0.5


class TDirectDistillLoss(nn.Module):
    """Pose loss + RAFT-teacher inverse-variance distillation.

    Parameters
    ----------
    w_rot, w_t, hover_center, hover_width
        Forwarded to the wrapped ``TDirectPoseLoss``. Identical defaults to
        Step H/J so the *pose* component is unchanged across steps and the
        Step J ↔ Step K diff is purely the distillation term.
    lambda_distill
        Multiplier on the combined distillation term (translation + rotation
        sub-terms summed before this multiplier). Default 1.0; sweep via
        config if results suggest the teacher is too dominant or too weak.
    sigma_eps_t
        ε for ``1/(σ_t² + ε²)``. Default 0.001 m — matches typical pixel-
        equivalent uncertainty floor of (0.5 px × 4 m / 640 px) ≈ 3 mm at
        the project's nominal altitude/intrinsics.
    sigma_eps_R
        ε for ``1/(σ_R² + ε²)``. Default 0.005 rad ≈ 0.29° — matches the
        per-pair rotation noise scale we expect from RAFT.
    sigma_px_floor
        Minimum trustworthy pixel σ; values below are clipped *up* to avoid
        over-weighting suspiciously-confident pairs. Default
        ``SIGMA_PX_FLOOR`` = 0.5 px.
    """

    def __init__(self,
                 w_rot: float = 10.0,
                 w_t: float = 5.0,
                 hover_center: float = 0.01,
                 hover_width: float = 0.005,
                 lambda_distill: float = 1.0,
                 sigma_eps_t: float = 0.001,
                 sigma_eps_R: float = 0.005,
                 sigma_px_floor: float = SIGMA_PX_FLOOR) -> None:
        super().__init__()
        self.pose = TDirectPoseLoss(
            w_rot=w_rot, w_t=w_t,
            hover_center=hover_center, hover_width=hover_width)
        self.lambda_distill = float(lambda_distill)
        self.sigma_eps_t = float(sigma_eps_t)
        self.sigma_eps_R = float(sigma_eps_R)
        self.sigma_px_floor = float(sigma_px_floor)

    # ------------------------------------------------------------------
    # Distillation term
    # ------------------------------------------------------------------

    def _distill_term(self,
                      R_a: torch.Tensor, t_a: torch.Tensor,
                      R_b: torch.Tensor, t_b: torch.Tensor,
                      sigma_b_px: torch.Tensor, valid: torch.Tensor,
                      depth_m: torch.Tensor, fx_px: torch.Tensor,
                      ) -> Dict[str, torch.Tensor]:
        """Per-pair inverse-variance-weighted distillation residual.

        Inputs are all batched (B, …). ``valid`` is bool / 0-1 float;
        masked-out pairs contribute zero to both translation and rotation
        sub-terms (and to ``valid_frac``).
        """
        device = R_a.device
        dtype = R_a.dtype
        B = R_a.shape[0]

        valid_f = valid.to(dtype=dtype)                            # (B,)
        # If no valid pairs, every term is a zero scalar with grad-on so the
        # graph stays connected. Avoids NaN from div by 0 in valid_f.sum().
        n_valid = valid_f.sum().clamp_min(1.0)

        # Per-pair σ in pixel units, NaN-guarded and floored.
        sigma_px = torch.nan_to_num(sigma_b_px.to(dtype=dtype),
                                    nan=float("inf"),
                                    posinf=float("inf"),
                                    neginf=float("inf"))
        sigma_px = sigma_px.clamp_min(self.sigma_px_floor)         # (B,)

        # Convert to metric / angular σ via the camera intrinsics and per-
        # frame depth. fx is constant across the batch but we broadcast for
        # uniform treatment. ε is added in *quadrature* so the resulting
        # weight w = 1/(σ²+ε²) is finite even when σ → 0.
        depth = depth_m.to(dtype=dtype).clamp_min(1e-3)            # (B,)
        fx = fx_px.to(dtype=dtype).clamp_min(1e-3)                 # scalar

        sigma_t = sigma_px * (depth / fx)                          # (B,)
        sigma_R = sigma_px / fx                                    # (B,)

        w_t = 1.0 / (sigma_t * sigma_t + self.sigma_eps_t ** 2)    # (B,)
        w_R = 1.0 / (sigma_R * sigma_R + self.sigma_eps_R ** 2)    # (B,)

        # NaN-guard the weights themselves — defensive against extreme σ.
        w_t = torch.nan_to_num(w_t, nan=0.0, posinf=0.0, neginf=0.0)
        w_R = torch.nan_to_num(w_R, nan=0.0, posinf=0.0, neginf=0.0)

        # Residuals.
        err_t_sq = (t_a - t_b).pow(2).sum(dim=-1)                  # (B,) m²
        err_R = geodesic_rotation_angle(R_a, R_b)                  # (B,) rad
        err_R_sq = err_R * err_R

        # Apply mask + average over valid pairs.
        l_t = (valid_f * w_t * err_t_sq).sum() / n_valid
        l_R = (valid_f * w_R * err_R_sq).sum() / n_valid

        # Final NaN-guard on outputs (L3) — graph still connected to inputs.
        l_t = torch.nan_to_num(l_t, nan=0.0, posinf=0.0, neginf=0.0)
        l_R = torch.nan_to_num(l_R, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            valid_frac = valid_f.mean()
            mean_sigma_t = (valid_f * sigma_t).sum() / n_valid
            mean_sigma_R = (valid_f * sigma_R).sum() / n_valid

        return {
            "l_distill_t": l_t,
            "l_distill_R": l_R,
            "valid_frac": valid_frac,
            "mean_sigma_t": mean_sigma_t,
            "mean_sigma_R": mean_sigma_R,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                R_pred: torch.Tensor, t_pred: torch.Tensor,
                R_gt: torch.Tensor, t_gt: torch.Tensor,
                R_b: torch.Tensor, t_b: torch.Tensor,
                sigma_b_px: torch.Tensor, valid: torch.Tensor,
                depth_m: torch.Tensor, fx_px: torch.Tensor,
                ) -> Dict[str, torch.Tensor]:
        pose = self.pose(R_pred, t_pred, R_gt, t_gt)
        distill = self._distill_term(
            R_a=R_pred, t_a=t_pred,
            R_b=R_b, t_b=t_b,
            sigma_b_px=sigma_b_px, valid=valid,
            depth_m=depth_m, fx_px=fx_px,
        )

        l_distill = distill["l_distill_t"] + distill["l_distill_R"]
        total = pose["total"] + self.lambda_distill * l_distill
        # Final NaN-guard on the scalar total. If, somehow, every per-term
        # NaN guard let something through, we'd rather log a 0 loss than
        # crash the train run — the train.log row will surface the issue.
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "total": total,
            "l_pose": pose["total"],
            "l_rot": pose["l_rot"],
            "l_t": pose["l_t"],
            "l_distill": l_distill,
            "l_distill_t": distill["l_distill_t"],
            "l_distill_R": distill["l_distill_R"],
            "valid_frac": distill["valid_frac"],
            "mean_sigma_t": distill["mean_sigma_t"],
            "mean_sigma_R": distill["mean_sigma_R"],
            "hover_weight_mean": pose["hover_weight_mean"],
        }


__all__ = ["TDirectDistillLoss", "SIGMA_PX_FLOOR"]
