"""Network VI variants — visual-inertial relative pose estimators.

Three architectures live here:

- `NetworkVI` — full "Option A+" design. Vision branch = Network V (decomposed
  homography heads). IMU branch = Network I (axis-angle + metric Δt). A scale
  mixer MLP blends `d_imu = ||Δt_imu|| / max(||td_v||, ε)` with a direct
  learned scale `d_learned` from the vision feature. A residual refiner MLP
  produces a small (δR, δt) correction.

- `NetworkVIConcat` — baseline: concatenates the CNN and LSTM features and
  regresses 6-DoF relative pose directly (axis-angle + Δt) via an MLP.

- `NetworkVINoDecomp` — uses the full-VI pipeline but strips the IMU-derived
  scale and relies solely on `d_learned` (i.e. vision-only scale head, no
  IMU fusion for translation magnitude). Lets us attribute the benefit of
  IMU-derived scale in the ablation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .network_v import NetworkV
from .network_i import NetworkI
from .backbones import VGG8Backbone, IMUEncoder
from utils.geometry import rodrigues_to_matrix, rot6d_to_matrix


_EPS = 1e-3


def _log_safe(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


class _ScaleMixer(nn.Module):
    """Blend IMU-derived scale and learned direct scale via softmax over [log d]."""

    def __init__(self, vision_dim: int, imu_dim: int, hidden: int = 64) -> None:
        super().__init__()
        in_dim = vision_dim + imu_dim + 2  # + [log(d_imu), log(d_learned)]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),  # softmax weights over two sources
        )

    def forward(self, vfeat: torch.Tensor, ifeat: torch.Tensor,
                d_imu: torch.Tensor, d_learned: torch.Tensor) -> torch.Tensor:
        x = torch.cat([vfeat, ifeat,
                       _log_safe(d_imu).unsqueeze(-1),
                       _log_safe(d_learned).unsqueeze(-1)], dim=-1)
        logits = self.net(x)
        w = F.softmax(logits, dim=-1)
        log_d_final = w[..., 0] * _log_safe(d_imu) + w[..., 1] * _log_safe(d_learned)
        return torch.exp(log_d_final)


class _ResidualRefiner(nn.Module):
    """Tiny MLP producing a small (δR, δt) correction.

    Axis-angle δR is tanh-squashed to ±0.1 rad; δt is tanh-squashed to ±0.2 m.
    """

    def __init__(self, vision_dim: int, imu_dim: int, hidden: int = 64,
                 max_d_rot: float = 0.1, max_d_trans: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim + imu_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 6),
        )
        self.max_d_rot = float(max_d_rot)
        self.max_d_trans = float(max_d_trans)

    def forward(self, vfeat: torch.Tensor, ifeat: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat([vfeat, ifeat], dim=-1))
        d_aa = torch.tanh(raw[..., :3]) * self.max_d_rot
        d_t = torch.tanh(raw[..., 3:]) * self.max_d_trans
        dR = rodrigues_to_matrix(d_aa)
        return dR, d_t


# ---------------------------------------------------------------------------
# Full VI network (Option A+)
# ---------------------------------------------------------------------------

class NetworkVI(nn.Module):
    def __init__(self, feat_dim: int = 512, lstm_hidden: int = 128,
                 cnn_channels: tuple | None = None,
                 fallback_threshold: float = 0.05) -> None:
        super().__init__()
        self.vision = NetworkV(feat_dim=feat_dim, cnn_channels=cnn_channels)
        self.imu = NetworkI(hidden_dim=lstm_hidden)
        # Step S2: NetworkI now exposes `.out_dim` (encoder projection dim,
        # default 256). Use it here so the fusion modules don't hardcode
        # `lstm_hidden` — that was the LSTM hidden width, not the feature
        # width, and the two diverged when the encoder gained a projection.
        imu_dim = self.imu.out_dim
        self.scale_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.mixer = _ScaleMixer(vision_dim=feat_dim, imu_dim=imu_dim)
        self.refiner = _ResidualRefiner(vision_dim=feat_dim, imu_dim=imu_dim)
        # Tier-1 B.3: inference-time fallback threshold on ||td_v||.
        self.fallback_threshold = float(fallback_threshold)

    def load_pretrained(self, v_state: dict | None, i_state: dict | None) -> None:
        """Optionally warm-start from Network V / Network I checkpoints."""
        if v_state is not None:
            self.vision.load_state_dict(v_state, strict=False)
        if i_state is not None:
            self.imu.load_state_dict(i_state, strict=False)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor,
                imu: torch.Tensor) -> Dict[str, torch.Tensor]:
        R_v, td_v, n_v, vfeat = self.vision(img_a, img_b)
        R_i, t_i, ifeat = self.imu(imu)

        td_norm = td_v.norm(dim=-1).clamp_min(_EPS)
        t_i_norm = t_i.norm(dim=-1)
        d_imu = t_i_norm / td_norm
        d_learned = F.softplus(self.scale_head(vfeat).squeeze(-1)) + 1e-3

        d_final = self.mixer(vfeat, ifeat, d_imu, d_learned)  # (B,)

        t_pre = td_v * d_final.unsqueeze(-1)                  # (B, 3)
        dR, dt = self.refiner(vfeat, ifeat)
        R_final = dR @ R_v
        t_final = t_pre + dt

        out = {
            "R_v": R_v, "td_v": td_v, "n_v": n_v, "vfeat": vfeat,
            "R_i": R_i, "t_i": t_i, "ifeat": ifeat,
            "d_imu": d_imu, "d_learned": d_learned, "d_final": d_final,
            "R": R_final, "t": t_final,
            "td_v_norm": td_v.norm(dim=-1),
        }
        # Tier-1 B.3: at inference only, flag samples whose vision direction is
        # near-zero so the eval pipeline can swap in the IMU branch. Training
        # ignores this flag.
        if not self.training:
            out["use_imu_fallback"] = (
                td_v.norm(dim=-1) < self.fallback_threshold)
        return out


# ---------------------------------------------------------------------------
# Baseline: concat fusion, vanilla 6-DoF head
# ---------------------------------------------------------------------------

class NetworkVIConcat(nn.Module):
    def __init__(self, feat_dim: int = 512, lstm_hidden: int = 128) -> None:
        super().__init__()
        self.cnn = VGG8Backbone(in_channels=6, feat_dim=feat_dim)
        # Step S2: bidirectional encoder with projection — `out_dim` (256)
        # replaces the prior single-direction `hidden_dim` (128) in the head.
        self.imu_enc = IMUEncoder(input_size=6, hidden_size=lstm_hidden,
                                  num_layers=2)
        imu_dim = self.imu_enc.out_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim + imu_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 6),
        )

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor,
                imu: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([img_a, img_b], dim=1)
        vfeat = self.cnn(x)
        ifeat = self.imu_enc(imu)
        raw = self.head(torch.cat([vfeat, ifeat], dim=-1))
        aa, t = raw[..., :3], raw[..., 3:]
        R = rodrigues_to_matrix(aa)
        return {"R": R, "t": t, "vfeat": vfeat, "ifeat": ifeat}


# ---------------------------------------------------------------------------
# Ablation: decomposed-homography vision + learned scale, no IMU scale
# ---------------------------------------------------------------------------

class NetworkVINoDecomp(nn.Module):
    """Same vision head (decomposed), IMU branch used only for rotation refinement."""

    def __init__(self, feat_dim: int = 512, lstm_hidden: int = 128) -> None:
        super().__init__()
        self.vision = NetworkV(feat_dim=feat_dim)
        self.imu = NetworkI(hidden_dim=lstm_hidden)
        imu_dim = self.imu.out_dim     # Step S2 — see NetworkVI.__init__ note
        self.scale_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.refiner = _ResidualRefiner(vision_dim=feat_dim, imu_dim=imu_dim)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor,
                imu: torch.Tensor) -> Dict[str, torch.Tensor]:
        R_v, td_v, n_v, vfeat = self.vision(img_a, img_b)
        R_i, t_i, ifeat = self.imu(imu)
        d_learned = F.softplus(self.scale_head(vfeat).squeeze(-1)) + 1e-3
        t_pre = td_v * d_learned.unsqueeze(-1)
        dR, dt = self.refiner(vfeat, ifeat)
        R_final = dR @ R_v
        t_final = t_pre + dt
        return {
            "R_v": R_v, "td_v": td_v, "n_v": n_v,
            "R_i": R_i, "t_i": t_i,
            "d_learned": d_learned, "R": R_final, "t": t_final,
            "vfeat": vfeat, "ifeat": ifeat,
        }


# ---------------------------------------------------------------------------
# Ablation: decomposed-homography vision + concat-MLP scale fusion
# ---------------------------------------------------------------------------

class NetworkVIConcatDecompose(nn.Module):
    """Decomposed vision head, IMU branch, but scale comes from a concat-MLP
    learned fusion (no analytic d_imu = ||t_i|| / ||td_v||). Isolates the
    contribution of the analytic fusion rule vs. just learning from both sources.
    """

    def __init__(self, feat_dim: int = 512, lstm_hidden: int = 128,
                 hidden: int = 128) -> None:
        super().__init__()
        self.vision = NetworkV(feat_dim=feat_dim)
        self.imu = NetworkI(hidden_dim=lstm_hidden)
        imu_dim = self.imu.out_dim     # Step S2 — see NetworkVI.__init__ note
        self.scale_head = nn.Sequential(
            nn.Linear(feat_dim + imu_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.refiner = _ResidualRefiner(vision_dim=feat_dim, imu_dim=imu_dim)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor,
                imu: torch.Tensor) -> Dict[str, torch.Tensor]:
        R_v, td_v, n_v, vfeat = self.vision(img_a, img_b)
        R_i, t_i, ifeat = self.imu(imu)
        d_learned = F.softplus(self.scale_head(
            torch.cat([vfeat, ifeat], dim=-1)).squeeze(-1)) + 1e-3
        t_pre = td_v * d_learned.unsqueeze(-1)
        dR, dt = self.refiner(vfeat, ifeat)
        R_final = dR @ R_v
        t_final = t_pre + dt
        return {
            "R_v": R_v, "td_v": td_v, "n_v": n_v,
            "R_i": R_i, "t_i": t_i,
            "d_learned": d_learned, "R": R_final, "t": t_final,
            "vfeat": vfeat, "ifeat": ifeat,
        }
