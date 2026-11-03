"""Shared backbones used by the three networks.

- `VGG8Backbone`: 8-layer VGG-style CNN mapping (B, 6, H, W) to a 512-d
  feature vector. Two stacked RGB frames are concatenated on the channel
  dimension before being fed in. Resolution-invariant: works for any
  H, W divisible by 16 (we target 480×640 after the 2026-04-24 overhaul,
  originally 240×320).
- `IMUEncoder`: bidirectional LSTM (2 layers, hidden=128 per direction by
  default) over (B, T, 6). All layers from both directions are concatenated
  and projected to a `feat_dim`-d vector (default 256). Lifted from
  `models/imu_model.py` (the architecture that actually trains well) and
  consolidated here so `NetworkI` and the `NetworkVI*` variants share it.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn


DEFAULT_VGG8_CHANNELS: Tuple[int, ...] = (32, 32, 64, 64, 128, 128, 256, 256)


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG8Backbone(nn.Module):
    """8 conv blocks + adaptive avg pool + linear → feature dim.

    Expects input (B, 6, H, W). Works for any H, W divisible by 16.
    Output shape: (B, feat_dim).

    Pooling: ``nn.AdaptiveAvgPool2d(1)`` replaces the prior
    ``torch.mean(h, dim=(-2, -1))``. Algebraically identical; the explicit
    module makes the resolution-invariance contract obvious and reports in
    ``print(model)`` / profiling.

    A 4×4 or 5×5 adaptive pool was considered but ruled out: the S6
    diagnostic (2026-04-24) confirmed capacity is not the bottleneck — V is
    undertrained, not undersized. Scaling head input from 512 → 8192 would
    blow head parameters from 66K → 1M each, which S6 explicitly advises
    against. Keeping H_pool=W_pool=1 preserves the pre-overhaul head param
    count exactly.
    """

    def __init__(self, in_channels: int = 6, feat_dim: int = 512,
                 channels: Sequence[int] = DEFAULT_VGG8_CHANNELS) -> None:
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if len(channels) != 8:
            raise ValueError(f"VGG8Backbone expects 8 channel entries, got {len(channels)}")
        c0, c1, c2, c3, c4, c5, c6, c7 = channels
        # 4 resolution stages of 2 conv blocks each; pool after every stage.
        self.stage1 = nn.Sequential(
            _conv_block(in_channels, c0),
            _conv_block(c0, c1),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            _conv_block(c1, c2),
            _conv_block(c2, c3),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            _conv_block(c3, c4),
            _conv_block(c4, c5),
            nn.MaxPool2d(2),
        )
        self.stage4 = nn.Sequential(
            _conv_block(c5, c6),
            _conv_block(c6, c7),
            nn.MaxPool2d(2),
        )
        # Project to feat_dim, then resolution-invariant adaptive pool to 1×1.
        self.proj = nn.Sequential(
            nn.Conv2d(c7, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = int(feat_dim)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.proj(h)
        h = self.pool(h)
        return h.flatten(1)   # (B, feat_dim)


class IMUEncoder(nn.Module):
    """Bidirectional LSTM encoder for an IMU sequence (Step S2).

    Lifted from `models/imu_model.py:IMUEncoder` and consolidated here so
    `NetworkI` and the `NetworkVI*` variants share the same architecture.

    Keyword arguments
    -----------------
    input_size  : 6 (acc + gyro)
    hidden_size : per-direction LSTM hidden units (was `hidden_dim`)
    num_layers  : stacked LSTM layers
    feat_dim    : output projection dimension — `out_dim` mirrors this
    dropout     : LSTM inter-layer dropout AND projection-tail dropout

    Output shape: `(B, feat_dim)`. Callers should read `.out_dim` rather
    than rely on a constructor argument so the dim flows through to fusion
    modules without hardcoding.

    The legacy positional names (`input_dim`, `hidden_dim`) are accepted as
    aliases — `NetworkVIConcat` and any external probe scripts are spared
    a signature change.
    """

    def __init__(self,
                 input_size: int = 6,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 feat_dim: int = 256,
                 dropout: float = 0.1,
                 *,
                 input_dim: int | None = None,
                 hidden_dim: int | None = None,
                 ) -> None:
        super().__init__()
        if input_dim is not None:
            input_size = int(input_dim)
        if hidden_dim is not None:
            hidden_size = int(hidden_dim)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # All layers, both directions: 2 * num_layers * hidden_size.
        lstm_out_dim = hidden_size * 2 * num_layers
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.feat_dim = int(feat_dim)
        self.out_dim = int(feat_dim)
        # Legacy alias retained so external code that read `hidden_dim` keeps
        # working (it now refers to per-direction hidden size).
        self.hidden_dim = int(hidden_size)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        # imu: (B, T, 6)
        _, (h_n, _) = self.lstm(imu)
        # h_n: (num_layers * 2, B, H) — fwd/bwd interleaved per layer.
        B = imu.shape[0]
        h = h_n.permute(1, 0, 2).reshape(B, -1)   # (B, num_layers*2*H)
        return self.proj(h)                         # (B, feat_dim)


if __name__ == "__main__":
    # Shape sanity check at 480×640 (6-channel stacked pair).
    import torch
    m = VGG8Backbone()
    x = torch.randn(2, 6, 480, 640)
    y = m(x)
    print(f"VGG8Backbone @ {tuple(x.shape)} -> {tuple(y.shape)}  "
          f"(expected (2, {m.feat_dim}))")
