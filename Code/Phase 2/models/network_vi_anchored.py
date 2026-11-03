"""Network VI — anchored variant for Step I.

Soft IMU-anchored fusion with photometric self-validation. Vision branch is
the unchanged ``NetworkV_tdirect``; IMU branch is the unchanged ``NetworkI``.
The fusion is performed by two small gating MLPs that produce per-sample
weights ``alpha_t, alpha_R ∈ [0, 1]``:

    t_fused = alpha_t * t_v + (1 - alpha_t) * t_i
    R_fused = SLERP(R_i, R_v, alpha_R)        # alpha_R = 1 → R_v

Gate inputs include ``rho_v`` — a free photometric self-confidence that
warps img_a → img_b under the network's predicted (R_v, t_v) and a known
ground-plane assumption, then squashes the per-sample mean residual into a
``[0, 1]`` confidence (1 = perfect match, 0 = bad match).

Step I question: with this design the gate has a free signal (``rho_v``) for
when V is unreliable. Does VI then beat I on simple trajectories (traj 300)
and at least match I on the harder traj 3? See ``outputs/step_i/CHANGES.md``
for the decision tree.

Cross-references:
    * ``models/network_v_tdirect.py`` — vision branch.
    * ``models/network_i.py``         — IMU branch.
    * ``models/losses.py::photometric_warp_residual`` — the same homography-
      warp pipeline, but it returns a scalar averaged over the batch and
      pixels. Here we need a per-sample residual, so the same maths is
      reproduced inline (no edits to ``losses.py``).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from utils.geometry import (
    matrix_to_rodrigues, rodrigues_to_matrix, slerp_matrices,
)

from .network_i import NetworkI
from .network_v_tdirect import NetworkV_tdirect

try:  # kornia is required at training time; eval-time we fall back gracefully.
    import kornia
except ImportError:  # pragma: no cover
    kornia = None  # type: ignore


# Same OpenGL → CV basis flip used by ``losses.py::photometric_warp_residual``.
_OPENGL_TO_CV_T = torch.tensor(
    [[1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0]]
)


class NetworkVI_anchored(nn.Module):
    """VI fusion with photometric self-validation and IMU-anchored gates.

    Parameters
    ----------
    feat_dim, lstm_hidden, cnn_channels
        Forwarded to the V and I branches; identical defaults to
        ``NetworkVI_tdirect``.
    fx, fy, cx, cy
        Phase 2 default intrinsics (640×480, 53° HFOV). Used only by the
        photometric self-validation block.
    default_altitude_m
        Eval-time fallback altitude for the photometric warp when no GT depth
        is supplied. The Step I eval trajectory (300) is at constant 4.0 m;
        traj 3 sinusoid varies in 1.5–3.0 m, but rho_v is a confidence — a
        constant offset just shifts every frame's residual together and is
        absorbed by the gate MLP, so a single scalar default is fine.
    rho_decay
        Slope of the residual → confidence map: ``rho = exp(-decay·MAE)``.
        With image pixels in [0, 1] the typical MAE on a perfect warp is
        ≲ 0.02 (artifacts at borders), so decay=8 gives rho ≈ 0.85 at MAE
        0.02 and rho ≈ 0.20 at MAE 0.20 — a useful dynamic range.
    """

    def __init__(self, feat_dim: int = 512, lstm_hidden: int = 128,
                 cnn_channels: tuple | None = None,
                 fx: float = 640.0, fy: float = 640.0,
                 cx: float = 320.0, cy: float = 240.0,
                 default_altitude_m: float = 4.0,
                 rho_decay: float = 8.0) -> None:
        super().__init__()
        self.vision = NetworkV_tdirect(feat_dim=feat_dim,
                                       cnn_channels=cnn_channels)
        self.imu = NetworkI(hidden_dim=lstm_hidden)

        # Translation gate: 4 inputs — [rho_v, t-direction agreement,
        #                              ||t_v||, ||t_i||].
        self.gate_t = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        # Rotation gate: 2 inputs — [rho_v, axis-angle direction agreement].
        self.gate_R = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        # Neutral init: zero the final-layer weight AND bias so every sample
        # outputs sigmoid(0) = 0.5 at step 0 regardless of input. Backprop
        # under L_gate_supervision then carves the dependence from scratch.
        # Without this the random init would lock the gate before the
        # supervision signal has had time to teach it.
        nn.init.zeros_(self.gate_t[-1].weight)
        nn.init.zeros_(self.gate_t[-1].bias)
        nn.init.zeros_(self.gate_R[-1].weight)
        nn.init.zeros_(self.gate_R[-1].bias)

        self.register_buffer(
            "K",
            torch.tensor([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=torch.float32),
        )
        self.default_altitude_m = float(default_altitude_m)
        self.rho_decay = float(rho_decay)

    # ------------------------------------------------------------------
    # Pretrained branch loaders (mirror NetworkVI_tdirect.load_pretrained)
    # ------------------------------------------------------------------

    def load_pretrained(self, v_state: dict | None,
                        i_state: dict | None) -> None:
        """Optionally warm-start vision / IMU branches from V / I checkpoints.

        Step I always loads V from Step H and I from Step G via symlink, so
        ``v_state`` / ``i_state`` come from the corresponding ``best.pt``s.
        ``strict=False`` because the Step H / G checkpoints contain only the
        sub-network state dicts (no anchored gates).
        """
        if v_state is not None:
            self.vision.load_state_dict(v_state, strict=False)
        if i_state is not None:
            self.imu.load_state_dict(i_state, strict=False)

    # ------------------------------------------------------------------
    # Photometric self-validation
    # ------------------------------------------------------------------

    def _photometric_residual(self, img_a: torch.Tensor, img_b: torch.Tensor,
                              R_v: torch.Tensor, t_v: torch.Tensor,
                              d: Optional[torch.Tensor]) -> torch.Tensor:
        """Per-sample photometric confidence ``rho ∈ [0, 1]``.

        Warps ``img_a`` to ``img_b`` under the network's predicted relative
        pose (which is cam_a → cam_b, same convention as ``t_rel``), assuming
        a single ground plane at altitude ``d``. Returns
        ``rho = exp(-decay · MAE)`` where MAE is reduced per-sample (mean
        over channels and pixels). ``rho ≈ 1`` for a clean warp; ``rho`` near
        0 indicates large photometric mismatch — the free "is V trustworthy"
        signal.

        If ``d`` is None (eval time, no GT depth in the loader output) the
        ``default_altitude_m`` is used. If kornia is missing we fall back to
        a neutral 0.5 so the gate has no information to act on.
        """
        B = img_a.shape[0]
        device = img_a.device
        dtype = img_a.dtype

        if kornia is None:
            return torch.full((B,), 0.5, device=device, dtype=dtype)

        if d is None:
            d_full = torch.full((B,), self.default_altitude_m,
                                device=device, dtype=dtype)
        else:
            d_full = d.to(device=device, dtype=dtype).reshape(-1)
            if d_full.shape[0] != B:
                d_full = d_full.expand(B)

        # Invert (R_v, t_v) so we have (R_ba, t_ba) = pose of cam_a in cam_b.
        R_ba = R_v.transpose(-1, -2)
        t_ba = -torch.einsum("bij,bj->bi", R_ba, t_v)

        # Build H = K · (R_cv − t_cv · n_cv^T / d) · K^{-1} in CV camera
        # frame; same formula as ``losses.py::photometric_warp_residual``.
        S = _OPENGL_TO_CV_T.to(device=device, dtype=dtype)
        R_cv = S @ R_ba @ S
        t_cv = (S @ t_ba.unsqueeze(-1)).squeeze(-1)
        n_col = torch.tensor([0.0, 0.0, -1.0],
                             device=device, dtype=dtype).view(1, 3, 1)
        t_col = t_cv.unsqueeze(-1)
        d_safe = d_full.clamp_min(1e-3).view(B, 1, 1)
        M = R_cv - (t_col @ n_col.transpose(-1, -2)) / d_safe
        K = self.K.to(device=device, dtype=dtype)
        K_b = K.unsqueeze(0).expand(B, -1, -1)
        K_inv = torch.linalg.inv(K_b)
        H = K_b @ M @ K_inv

        try:
            warped = kornia.geometry.transform.homography_warp(
                img_a, H,
                (img_a.shape[-2], img_a.shape[-1]),
                mode="bilinear", padding_mode="zeros",
                normalized_coordinates=False, align_corners=False,
            )
        except RuntimeError:
            return torch.full((B,), 0.5, device=device, dtype=dtype)

        # Per-sample MAE over (C, H, W).
        mae = (warped - img_b).abs().mean(dim=(1, 2, 3))           # (B,)
        rho = torch.exp(-self.rho_decay * mae).clamp(0.0, 1.0)
        # NaN guard: clamp() does not replace NaN, only out-of-range finites.
        # If kornia returns NaN for a degenerate H (or any silent upstream
        # NaN), substituting 0.5 keeps the gate at its neutral input rather
        # than poisoning the loss.
        rho = torch.nan_to_num(rho, nan=0.5)
        return rho

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor,
                imu: torch.Tensor,
                depth_gt: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        R_v, t_v, vfeat = self.vision(img_a, img_b)
        R_i, t_i, ifeat = self.imu(imu)

        rho_v = self._photometric_residual(img_a, img_b, R_v, t_v, depth_gt)

        # ---------- translation gate ----------
        t_v_dir = F.normalize(t_v, dim=-1, eps=1e-6)
        t_i_dir = F.normalize(t_i, dim=-1, eps=1e-6)
        agreement_t = (t_v_dir * t_i_dir).sum(-1, keepdim=True)        # (B, 1)
        gate_t_in = torch.cat([
            rho_v.unsqueeze(-1),
            agreement_t,
            t_v.norm(dim=-1, keepdim=True),
            t_i.norm(dim=-1, keepdim=True),
        ], dim=-1)                                                       # (B, 4)
        alpha_t = torch.sigmoid(self.gate_t(gate_t_in))                  # (B, 1)
        t_fused = alpha_t * t_v + (1.0 - alpha_t) * t_i

        # ---------- rotation gate ----------
        aa_v = matrix_to_rodrigues(R_v)
        aa_i = matrix_to_rodrigues(R_i)
        agreement_R = (
            F.normalize(aa_v, dim=-1, eps=1e-6)
            * F.normalize(aa_i, dim=-1, eps=1e-6)
        ).sum(-1, keepdim=True)                                          # (B, 1)
        gate_R_in = torch.cat([rho_v.unsqueeze(-1), agreement_R], dim=-1)
        alpha_R = torch.sigmoid(self.gate_R(gate_R_in))                  # (B, 1)
        R_fused = slerp_matrices(R_i, R_v, alpha_R)                      # α=1 → R_v

        return {
            "R": R_fused, "t": t_fused,
            "R_v": R_v, "t_v": t_v,
            "R_i": R_i, "t_i": t_i,
            "rho_v": rho_v,
            "alpha_t": alpha_t.squeeze(-1),
            "alpha_R": alpha_R.squeeze(-1),
            "agreement_t": agreement_t.squeeze(-1),
            "agreement_R": agreement_R.squeeze(-1),
            "vfeat": vfeat, "ifeat": ifeat,
        }


__all__ = ["NetworkVI_anchored"]
