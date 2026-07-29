"""Evaluation utilities for Phase 2.

Covers:
  - TUM-format trajectory I/O.
  - `run_evo_ape` — subprocess wrapper around `evo_ape tum … -a`.
  - `segment_drift` — per-window ATE (in-process, no subprocess).
  - `plot_3d_overlay`, `plot_2d_topdown`, `plot_per_axis`,
    `plot_per_axis_error` — single-method plots (GT vs pred).
  - `plot_xy_overlay_methods`, `plot_3d_overlay_methods`,
    `plot_per_axis_error_methods` — multi-method comparison plots.
  - `MethodTrace` — minimal dataclass for accumulating per-method
    predictions across the eval method loop, then handed to the
    `*_methods` plot helpers in one shot.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Color palette for V/I/VI overlays. Stable across plot helpers so the same
# method is always the same color in every figure.
_METHOD_COLORS: Dict[str, str] = {
    "v":  "#1f77b4",   # blue
    "i":  "#2ca02c",   # green
    "vi": "#d62728",   # red
}


def _color_for(method: str, fallback: str = "#7f7f7f") -> str:
    return _METHOD_COLORS.get(method.lower(), fallback)


def _format_ate_suffix(ate: Optional[float]) -> str:
    """Render ' — ATE = X.XX m' or '' if `ate` is None / NaN."""
    if ate is None:
        return ""
    try:
        if math.isnan(float(ate)):
            return ""
    except (TypeError, ValueError):
        return ""
    return f" — ATE = {float(ate):.2f} m"


@dataclass
class MethodTrace:
    """Per-method prediction trace for one trajectory.

    Used by ``scripts/evaluate_*.py`` to accumulate V / I / VI predictions
    across the method loop, then hand them to the multi-method comparison
    plot helpers. Translation arrays are aligned to the same length N so
    the per-axis error overlay subtracts cleanly.
    """
    method: str
    pred_ts: np.ndarray            # (N, 3) translations
    times: np.ndarray              # (N,) seconds
    ate: Optional[float] = None    # eval ATE for legend / title
    color: Optional[str] = None    # override default palette

    def display_color(self) -> str:
        return self.color or _color_for(self.method)

    def label(self) -> str:
        suffix = "" if self.ate is None else f" (ATE={self.ate:.2f} m)"
        return f"{self.method}{suffix}"


def write_tum(path: Path, times: np.ndarray, ts: np.ndarray, quats_xyzw: np.ndarray
             ) -> None:
    """Write a TUM-format trajectory file.

    `times` shape (N,), `ts` (N, 3) xyz, `quats_xyzw` (N, 4).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for k in range(len(times)):
            t = times[k]
            tx, ty, tz = ts[k]
            qx, qy, qz, qw = quats_xyzw[k]
            f.write(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def read_tum(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(str(path))
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


# ---------------------------------------------------------------------------
# ATE
# ---------------------------------------------------------------------------

@dataclass
class ATEResult:
    ate_aligned: float     # evo-reported RMSE in metres
    ate_endpoint: float    # Euclidean distance between last-pose translations
    raw_stats: Dict[str, float]


_EVO_RMSE_RE = re.compile(r"\s*rmse\s*\t?\s*([0-9.eE+-]+)", re.IGNORECASE)


def run_evo_ape(gt_tum: Path, pred_tum: Path, evo_bin: str = "evo_ape"
               ) -> ATEResult:
    """Call `evo_ape tum GT PRED -a` and parse the RMSE.

    Falls back to an in-process implementation (SE(3)-aligned ATE using
    Horn-style Umeyama) if the binary is missing.
    """
    try:
        out = subprocess.check_output(
            [evo_bin, "tum", str(gt_tum), str(pred_tum), "-a"],
            stderr=subprocess.STDOUT, text=True, timeout=120)
        m = _EVO_RMSE_RE.search(out)
        rmse = float(m.group(1)) if m else float("nan")
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired) as exc:
        print(f"[eval] evo_ape failed ({exc}); falling back to in-process ATE.")
        rmse = _ate_umeyama_rmse(gt_tum, pred_tum)
    _t_gt, ts_gt, _q = read_tum(gt_tum)
    _t_pr, ts_pr, _q = read_tum(pred_tum)
    endpoint = float(np.linalg.norm(ts_gt[-1] - ts_pr[-1]))
    return ATEResult(ate_aligned=float(rmse), ate_endpoint=endpoint, raw_stats={})


def _ate_umeyama_rmse(gt_tum: Path, pred_tum: Path) -> float:
    """SE(3) ATE (no scale) via Umeyama / Horn alignment on translations."""
    _t, ts_gt, _q = read_tum(gt_tum)
    _t, ts_pr, _q = read_tum(pred_tum)
    N = min(ts_gt.shape[0], ts_pr.shape[0])
    X = ts_pr[:N].T           # (3, N) prediction
    Y = ts_gt[:N].T           # (3, N) ground truth
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)
    Xc = X - mu_x
    Yc = Y - mu_y
    S = (Yc @ Xc.T) / N
    U, _sv, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    t = mu_y.flatten() - R @ mu_x.flatten()
    aligned = R @ X + t[:, None]
    rmse = float(np.sqrt(((aligned - Y) ** 2).sum(axis=0).mean()))
    return rmse


def segment_drift(gt_ts: np.ndarray, pred_ts: np.ndarray, times: np.ndarray,
                  window_s: float = 2.0) -> Tuple[float, float, np.ndarray]:
    """Per-`window_s` ATE (translation RMSE) median and 95th percentile.

    Parameters
    ----------
    gt_ts, pred_ts : (N, 3) translations.
    times          : (N,) timestamps in seconds.
    """
    N = gt_ts.shape[0]
    per = []
    k = 0
    while k < N:
        t0 = times[k]
        lo = k
        hi = lo
        while hi < N and (times[hi] - t0) < window_s:
            hi += 1
        seg_gt = gt_ts[lo:hi]
        seg_pr = pred_ts[lo:hi]
        if seg_gt.shape[0] < 2:
            break
        # Anchor both to their start (translation-only alignment is enough for
        # short windows and matches how drift-per-second is usually reported).
        dgt = seg_gt - seg_gt[0]
        dpr = seg_pr - seg_pr[0]
        err = np.linalg.norm(dgt - dpr, axis=-1)
        per.append(float(err.mean()))
        k = hi
    arr = np.array(per) if per else np.zeros(1)
    return float(np.median(arr)), float(np.quantile(arr, 0.95)), arr


# ---------------------------------------------------------------------------
# Plotting — single-method (GT vs one prediction)
# ---------------------------------------------------------------------------
#
# Conventions (Step J):
#   * Start point: green dot (size 50).
#   * End point:   red dot   (size 50).
#   * GT line:     red, thick.
#   * Pred line:   blue, thin.
#   * Title may carry " — ATE = X.XX m" via the optional ``ate`` argument.

_START_KW = dict(color="#00b050", s=55, zorder=5, edgecolor="black",
                 linewidth=0.6, label="start")
_END_KW = dict(color="#d62728", s=55, zorder=5, edgecolor="black",
               linewidth=0.6, label="end")


def plot_3d_overlay(gt_ts: np.ndarray, pred_ts: np.ndarray, path: Path,
                    title: str, ate: Optional[float] = None) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_ts[:, 0], gt_ts[:, 1], gt_ts[:, 2], "r-", linewidth=2.5,
            label="GT", alpha=0.9)
    ax.plot(pred_ts[:, 0], pred_ts[:, 1], pred_ts[:, 2], "b-", linewidth=1.0,
            label="Prediction", alpha=0.95)
    ax.scatter(*gt_ts[0], **_START_KW)
    ax.scatter(*gt_ts[-1], **_END_KW)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(loc="best"); ax.set_title(title + _format_ate_suffix(ate))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


def plot_2d_topdown(gt_ts: np.ndarray, pred_ts: np.ndarray, path: Path,
                    title: str, ate: Optional[float] = None) -> None:
    """Top-down (XY) overlay — GT solid red, pred thin blue, equal aspect."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(gt_ts[:, 0], gt_ts[:, 1], color="red", lw=1.5, label="GT")
    ax.plot(pred_ts[:, 0], pred_ts[:, 1], color="blue", lw=1.0, label="Prediction")
    ax.scatter([gt_ts[0, 0]], [gt_ts[0, 1]], **_START_KW)
    ax.scatter([gt_ts[-1, 0]], [gt_ts[-1, 1]], **_END_KW)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title + _format_ate_suffix(ate))
    ax.grid(alpha=0.3); ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_per_axis(times: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                  path: Path, ylabels: Iterable[str], title: str,
                  ate: Optional[float] = None) -> None:
    """Per-axis (3 stacked subplots) GT vs pred over time."""
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    for i, (ax, yl) in enumerate(zip(axs, ylabels)):
        ax.plot(times, gt[:, i], "r-", linewidth=2.0, label="GT")
        ax.plot(times, pred[:, i], "b-", linewidth=1.0, label="Prediction")
        ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
    axs[0].set_title(title + _format_ate_suffix(ate)); axs[0].legend(loc="best")
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


def plot_per_axis_error(times: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                        path: Path, title: str,
                        ylabels: Sequence[str] = ("x error (m)",
                                                  "y error (m)",
                                                  "z error (m)"),
                        ate: Optional[float] = None) -> None:
    """Per-axis error (pred − gt) over time, with a zero baseline.

    Diagnoses *where* errors accumulate — e.g. is the bias purely vertical
    (z drift due to altitude misestimation) or planar (xy drift)?
    """
    err = pred - gt
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    for i, (ax, yl) in enumerate(zip(axs, ylabels)):
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
        ax.plot(times, err[:, i], color="#d62728", lw=1.0,
                label="pred − GT")
        ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
    axs[0].set_title(title + _format_ate_suffix(ate)); axs[0].legend(loc="best")
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting — multi-method comparison (V vs I vs VI)
# ---------------------------------------------------------------------------

def plot_xy_overlay_methods(gt_ts: np.ndarray,
                            traces: Sequence[MethodTrace],
                            path: Path, title: str) -> None:
    """Top-down XY plot with GT in solid black and one line per method."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(gt_ts[:, 0], gt_ts[:, 1], color="black", lw=2.0, label="GT")
    for tr in traces:
        ax.plot(tr.pred_ts[:, 0], tr.pred_ts[:, 1],
                color=tr.display_color(), lw=1.2, alpha=0.95,
                label=tr.label())
    ax.scatter([gt_ts[0, 0]], [gt_ts[0, 1]], **_START_KW)
    ax.scatter([gt_ts[-1, 0]], [gt_ts[-1, 1]], **_END_KW)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_3d_overlay_methods(gt_ts: np.ndarray,
                            traces: Sequence[MethodTrace],
                            path: Path, title: str) -> None:
    """3D xyz overlay with GT (black) and one line per method."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_ts[:, 0], gt_ts[:, 1], gt_ts[:, 2],
            color="black", lw=2.2, label="GT", alpha=0.9)
    for tr in traces:
        ax.plot(tr.pred_ts[:, 0], tr.pred_ts[:, 1], tr.pred_ts[:, 2],
                color=tr.display_color(), lw=1.2, alpha=0.95,
                label=tr.label())
    ax.scatter(*gt_ts[0], **_START_KW)
    ax.scatter(*gt_ts[-1], **_END_KW)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(loc="best"); ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


def plot_per_axis_error_methods(gt_ts: np.ndarray,
                                traces: Sequence[MethodTrace],
                                path: Path, title: str,
                                ylabels: Sequence[str] = ("x error (m)",
                                                          "y error (m)",
                                                          "z error (m)")
                                ) -> None:
    """Per-axis error (pred − gt) over time, one line per method per axis."""
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    for i, (ax, yl) in enumerate(zip(axs, ylabels)):
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
        for tr in traces:
            N = min(tr.pred_ts.shape[0], gt_ts.shape[0], tr.times.shape[0])
            err = tr.pred_ts[:N, i] - gt_ts[:N, i]
            ax.plot(tr.times[:N], err, color=tr.display_color(),
                    lw=1.0, alpha=0.95, label=tr.label())
        ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
    axs[0].set_title(title); axs[0].legend(loc="best")
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Euler extraction (for per-axis rpy plots)
# ---------------------------------------------------------------------------

def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Extract XYZ intrinsic Euler angles (roll, pitch, yaw) from (N, 3, 3)."""
    sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = sy < 1e-6
    roll = np.where(singular, np.arctan2(-R[:, 1, 2], R[:, 1, 1]),
                    np.arctan2(R[:, 2, 1], R[:, 2, 2]))
    pitch = np.arctan2(-R[:, 2, 0], sy)
    yaw = np.where(singular, np.zeros_like(roll),
                   np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    return np.stack([roll, pitch, yaw], axis=-1)


# ---------------------------------------------------------------------------
# Dead-reckoning helpers (used by scripts/evaluate.py and the study scripts)
# ---------------------------------------------------------------------------

def dead_reckon_from_gt_start(traj, rels: List[Tuple[np.ndarray, np.ndarray]]
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a TrajectoryData and a list of relative (R, t) tuples,
    chain them onto the GT pose of frame 0 and return (times, ts, quats_xyzw).
    """
    import torch
    from .geometry import compose_se3, matrix_to_quat
    # Local import to avoid circular on module load.
    from .dataloader import _quat_to_R

    row0 = traj.poses_full[0]
    R = _quat_to_R(row0[3:])
    t = row0[:3].copy()
    N = len(rels) + 1
    ts = np.zeros((N, 3)); ts[0] = t
    Rs = np.zeros((N, 3, 3)); Rs[0] = R
    for i, (dR, dt) in enumerate(rels):
        R, t = compose_se3(Rs[i], ts[i], dR, dt)
        Rs[i + 1] = R; ts[i + 1] = t
    times = np.arange(N, dtype=np.float64) / traj.metadata["fps_image"]
    q = matrix_to_quat(torch.from_numpy(Rs)).numpy()
    return times, ts, q


def gt_trajectory_frames(traj) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample `poses_full` to one-per-kept-frame and return
    (times, ts, quats_xyzw).
    """
    stride = 10
    rows = traj.poses_full[::stride]
    times = np.arange(rows.shape[0], dtype=np.float64) / traj.metadata["fps_image"]
    return times, rows[:, :3], rows[:, 3:]
