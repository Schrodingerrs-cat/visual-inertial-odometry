"""Network VI (t-direct variant) — visual-inertial fusion for Step H.

Vision branch = ``NetworkV_tdirect`` (predicts R, t directly).
IMU branch    = ``NetworkI`` (axis-angle + metric Δt, unchanged).

Fusion: simple residual refiner only. No scale head, no scale mixer, no
analytic ``d_imu = ||t_i|| / ||t_v||`` — those only made sense when the vision
branch produced a scale-free direction (t/d). With t-direct, the vision
branch already gives metric translation, so we just add a residual:

    R_final = dR @ R_v
    t_final = t_v + dt(vfeat, ifeat)

The residual refiner is the same ``_ResidualRefiner`` used in the (t/d) VI
network, kept tanh-bounded to ±0.1 rad / ±0.2 m so it can't dominate.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .network_v_tdirect import NetworkV_tdirect
from .network_i import NetworkI
from .network_vi import _ResidualRefiner


class NetworkVI_tdirect(nn.Module):
    def __init__(self, feat_dim: int = 512, lstm_hidden: int = 128,
                 cnn_channels: tuple | None = None) -> None:
        super().__init__()
        self.vision = NetworkV_tdirect(feat_dim=feat_dim,
                                       cnn_channels=cnn_channels)
        self.imu = NetworkI(hidden_dim=lstm_hidden)
        # Step S2: NetworkI.out_dim (256) is the actual ifeat width — was
        # silently == lstm_hidden (128) before the bidirectional encoder.
        self.refiner = _ResidualRefiner(vision_dim=feat_dim,
                                        imu_dim=self.imu.out_dim)

    def load_pretrained(self, v_state: dict | None, i_state: dict | None
                       ) -> None:
        """Optionally warm-start vision / IMU branches from V / I checkpoints."""
        if v_state is not None:
            self.vision.load_state_dict(v_state, strict=False)
        if i_state is not None:
            self.imu.load_state_dict(i_state, strict=False)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor,
                imu: torch.Tensor) -> Dict[str, torch.Tensor]:
        R_v, t_v, vfeat = self.vision(img_a, img_b)
        R_i, t_i, ifeat = self.imu(imu)

        dR, dt = self.refiner(vfeat, ifeat)
        R_final = dR @ R_v
        t_final = t_v + dt

        return {
            "R_v": R_v, "t_v": t_v, "vfeat": vfeat,
            "R_i": R_i, "t_i": t_i, "ifeat": ifeat,
            "R": R_final, "t": t_final,
        }
