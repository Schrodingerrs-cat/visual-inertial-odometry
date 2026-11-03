"""Exhaustive evaluation of trained Step K and Step S checkpoints.

Runs end-to-end on the Turing cluster, no flags required. Default behavior:
evaluates both Step K and Step S checkpoints on every trajectory in their
config (train + val + test) for every modality (V, I, VI) and writes a
fully-organized bundle of plots and metrics under ``<phase2>/full_results/``.

Folder layout (one tree per step):

    full_results/
      step_<k|s>/
        losses/                            # parsed from train.log per modality
          v_loss.png  i_loss.png  vi_loss.png  combined_loss.png
          v_val_ate.png  i_val_ate.png  vi_val_ate.png  combined_val_ate.png
        train/<shape>_<speed>mps/<modality>/   # one dir per (split, traj, model)
          gt_vs_pred_3d.png
          top_xy.png
          position_xyz.png
          position_error.png
          rpy.png
          rpy_error.png
          err_vs_distance.png
          err_vs_speed.png
          metrics.json
          gt.tum  pred.tum
        train/<shape>_<speed>mps/comparison/   # V/I/VI overlay per traj
          xy_overlay.png  3d_overlay.png  per_axis_error.png  rpy_error_overlay.png
        val/...    test/...                # same structure
        aggregate/
          ate_per_modality.png             # bar chart, all trajs
          ate_vs_speed.png                 # ATE on y, target speed on x, hue=modality
          rpe_vs_speed.png                 # same for RPE-trans
          err_vs_distance_all.png          # error-by-distance overlays
        summary.csv  summary.json          # one row per (split, traj, modality)

Every plot title, filename, and legend uses ``<shape>_<speed>mps`` (e.g.
``circle_0.5mps``, ``figure8_1.0mps``) — never opaque traj_<id>.

Usage on Turing:
    cd "Code/Phase 2"
    python3 scripts/full_results_eval.py
    # or override:
    python3 scripts/full_results_eval.py --steps s
    python3 scripts/full_results_eval.py --steps k s --output_root /scratch/full_results
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parents[1]   # Code/Phase 2/
sys.path.insert(0, str(HERE))

from models.network_i import NetworkI  # noqa: E402
from models.network_v_tdirect import NetworkV_tdirect  # noqa: E402
from models.network_vi_tdirect import NetworkVI_tdirect  # noqa: E402
from utils.dataloader import build_eval_loader  # noqa: E402
from utils.eval import (  # noqa: E402
    dead_reckon_from_gt_start,
    gt_trajectory_frames,
    matrix_to_rpy,
    write_tum,
)
from utils.geometry import quat_to_matrix  # noqa: E402
from utils.training import get_device, load_checkpoint, load_config, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS: Tuple[str, ...] = ("v", "i", "vi")

_METHOD_COLORS: Dict[str, str] = {
    "v":  "#1f77b4",
    "i":  "#2ca02c",
    "vi": "#d62728",
}
_METHOD_NAME: Dict[str, str] = {
    "v":  "NetworkV",
    "i":  "NetworkI",
    "vi": "NetworkVI",
}

# Map canonical sampler family → short label used in filenames + titles.
_FAMILY_SHORT: Dict[str, str] = {
    "simple_figure8":     "figure8",
    "simple_circle":      "circle",
    "simple_ellipse":     "ellipse",
    "simple_diamond":     "diamond",
    "simple_zigzag":      "zigzag",
    "simple_triangle":    "triangle",
    "simple_sinusoid_30s": "sinusoid30s",
    "sinusoid":           "sinusoid",
    "figure_8":           "figure8",
    "flat_spiral":        "spiral",
    "clover":             "clover",
}

# Default step → config & checkpoint directory.
_STEP_DEFAULTS: Dict[str, Dict[str, str]] = {
    "k": {"config": "configs/step_k.yaml", "checkpoints": "outputs/step_k"},
    "s": {"config": "configs/step_s.yaml", "checkpoints": "outputs/step_s"},
}

# Step S extra test trajectories — held-out simple_triangle at a speed sweep
# spanning OOD-slow → train ceiling → OOD-fast. Trained Step S never saw
# triangle motion; these probe shape generalisation across speed.
# Each entry: (traj_id, target_peak_speed_mps). Family is always
# simple_triangle. Texture pool/name is forced to tex_03.webp at render time.
# IDs 420-424 must already exist in blender/generate_data.py:TRAJ_ID_MAP
# and TEXTURE_ASSIGNMENT (added in the same change as this script).
_STEP_S_EXTRA_TEST: List[Tuple[int, float]] = [
    (420, 0.3),
    (421, 0.8),
    (422, 1.0),
    (423, 1.2),
    (424, 1.5),
]


# ---------------------------------------------------------------------------
# Trajectory labeling — humanise traj_id into shape_speed
# ---------------------------------------------------------------------------

@dataclass
class TrajLabel:
    """Human-readable identifier for a trajectory."""
    traj_id: int
    family_short: str          # e.g. "circle", "figure8"
    speed_mps: float           # peak speed in m/s
    speed_str: str             # "0.5" or "auto1.2" (legacy fallback)
    split: str                 # "train" / "val" / "test"

    @property
    def label(self) -> str:
        """e.g. ``circle_0.5mps``."""
        return f"{self.family_short}_{self.speed_str}mps"

    @property
    def long_label(self) -> str:
        """e.g. ``train / circle_0.5mps`` for plot titles."""
        return f"{self.split} / {self.label}"


def _label_for_traj(traj_dir: Path, split_hint: Optional[str] = None
                    ) -> TrajLabel:
    """Read metadata.json and produce a TrajLabel.

    ``target_peak_speed`` is preferred when present; falls back to the
    empirical peak |v_xy| on the GT positions for legacy trajectories.
    """
    md = json.loads((traj_dir / "metadata.json").read_text())
    fam = str(md.get("family", "unknown"))
    fam_short = _FAMILY_SHORT.get(fam, fam.replace("_", ""))
    split = str(md.get("split", split_hint or "?"))
    tgt = md.get("target_peak_speed", None)
    if tgt is None:
        pos = np.load(traj_dir / "poses_full.npy")[:, :3]
        fps = float(md.get("fps_imu", 1000))
        v = np.gradient(pos, 1.0 / fps, axis=0)
        speed = float(np.max(np.linalg.norm(v[:, :2], axis=-1)))
        speed_str = f"auto{speed:.1f}"
    else:
        speed = float(tgt)
        speed_str = f"{speed:.1f}"
    traj_id = int(md.get("trajectory_id", int(traj_dir.name.split("_")[-1])))
    return TrajLabel(traj_id=traj_id, family_short=fam_short,
                     speed_mps=speed, speed_str=speed_str, split=split)


# ---------------------------------------------------------------------------
# Model assembly — mirrors evaluate_tdirect.py
# ---------------------------------------------------------------------------

def _build_model(method: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    if method == "v":
        return NetworkV_tdirect(feat_dim=cfg["model"]["cnn_feat_dim"]).to(device)
    if method == "i":
        return NetworkI(hidden_dim=cfg["model"]["lstm_hidden"]).to(device)
    if method == "vi":
        return NetworkVI_tdirect(
            feat_dim=cfg["model"]["cnn_feat_dim"],
            lstm_hidden=cfg["model"]["lstm_hidden"]).to(device)
    raise ValueError(f"unknown method {method!r}")


def _predict(method: str, model: torch.nn.Module, batch: dict
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    if method == "v":
        R, t, _ = model(batch["img_a"], batch["img_b"])
    elif method == "i":
        imu_for_i = batch.get("imu_context", batch["imu"])
        R, t, _ = model(imu_for_i)
    elif method == "vi":
        out = model(batch["img_a"], batch["img_b"], batch["imu"])
        R, t = out["R"], out["t"]
    else:
        raise ValueError(method)
    return R[0], t[0]


def _ckpt_path(method: str, ckpt_root: Path) -> Path:
    return ckpt_root / method / "best.pt"


def _load_models(cfg: dict, device: torch.device, ckpt_root: Path
                 ) -> Dict[str, torch.nn.Module]:
    """Load V / I / VI from ``ckpt_root/<method>/best.pt``.

    Tolerates architecturally-incompatible checkpoints by skipping the
    modality with a clear log message instead of crashing. This matters
    for Step K's I and VI checkpoints, which were trained against the
    legacy unidirectional NetworkI (shipped with hidden_dim=128, single
    direction → LSTM layer-1 weight_ih shape (512, 128)). The current
    NetworkI is bidirectional (shape (512, 256)). ``strict=False`` does
    NOT help with size mismatches — only with missing/extra keys.
    """
    models: Dict[str, torch.nn.Module] = {}
    for m in METHODS:
        ckpt = _ckpt_path(m, ckpt_root)
        if not ckpt.exists():
            print(f"[full_eval] {m}: checkpoint missing at {ckpt} — skipping")
            continue
        net = _build_model(m, cfg, device)
        blob = load_checkpoint(ckpt, map_location=device)
        try:
            missing, unexpected = net.load_state_dict(blob["model"],
                                                     strict=False)
        except RuntimeError as exc:
            msg = str(exc)
            # Heuristic: a "size mismatch" RuntimeError means the
            # checkpoint was trained against a different architecture.
            if "size mismatch" in msg:
                print(f"[full_eval] {m}: checkpoint at {ckpt} is "
                      f"architecturally incompatible with the current model "
                      f"(size mismatch). Likely a pre-Step-S NetworkI "
                      f"checkpoint loaded against the post-upgrade "
                      f"bidirectional NetworkI. Skipping {m} for this step.")
            else:
                print(f"[full_eval] {m}: load_state_dict raised "
                      f"RuntimeError — skipping. Detail: {msg.splitlines()[0]}")
            continue
        if missing:
            # Only log if the missing keys would change behavior. Heads /
            # backbone weights are load-bearing; absence is suspicious.
            critical = [k for k in missing if not k.startswith("_")]
            if critical:
                print(f"[full_eval] {m}: WARNING — {len(critical)} keys "
                      f"missing from checkpoint (model has them but ckpt "
                      f"doesn't): first 3 = {critical[:3]}")
        if unexpected:
            print(f"[full_eval] {m}: note — {len(unexpected)} extra keys in "
                  f"checkpoint (silently ignored): first 3 = {list(unexpected)[:3]}")
        net.eval()
        models[m] = net
        print(f"[full_eval] {m}: loaded {ckpt}")
    return models


# ---------------------------------------------------------------------------
# Render-on-demand (Blender) — used to materialise extra triangle trajs
# that may not yet exist on disk.
# ---------------------------------------------------------------------------

def _traj_is_rendered(traj_dir: Path) -> bool:
    """A traj is "rendered" when the canonical four artifacts are present."""
    if not traj_dir.is_dir():
        return False
    if not (traj_dir / "metadata.json").exists():
        return False
    if not (traj_dir / "poses_full.npy").exists():
        return False
    if not (traj_dir / "imu.npy").exists():
        return False
    frames = traj_dir / "frames"
    if not frames.is_dir():
        return False
    # Heuristic: must have ≥ 100 frames (any 30 s, 100 Hz traj has 3000+).
    try:
        n_frames = sum(1 for _ in frames.iterdir())
    except OSError:
        return False
    return n_frames >= 100


def _resolve_blender_binary() -> Optional[Path]:
    """Find the Blender binary the SLURM scripts use ($BLENDER, then default)."""
    import os
    env_blender = os.environ.get("BLENDER", "").strip()
    if env_blender and Path(env_blender).is_file():
        return Path(env_blender)
    # Same fallback as slurm/_common.sh.
    default = HERE / "blender_app" / "current" / "blender"
    if default.is_file():
        return default
    # System blender on PATH.
    from shutil import which
    sys_blender = which("blender")
    if sys_blender:
        return Path(sys_blender)
    return None


def _render_traj(traj_id: int, raw_dir: Path,
                 force_texture: str = "tex_03.webp") -> bool:
    """Invoke Blender to render one trajectory. Returns True on success.

    Mirrors slurm/step_s_render.sh's per-traj invocation. The traj_id must
    already be present in blender/generate_data.py:TRAJ_ID_MAP and
    TEXTURE_ASSIGNMENT (this script's design adds 420-424 in the same change).
    """
    import subprocess
    blender = _resolve_blender_binary()
    if blender is None:
        print(f"[full_eval] BLENDER binary not found. Set $BLENDER, or "
              f"install at {HERE/'blender_app/current/blender'}.")
        return False
    gen_script = HERE / "blender" / "generate_data.py"
    if not gen_script.is_file():
        print(f"[full_eval] generate_data.py missing at {gen_script}")
        return False
    cmd = [
        str(blender), "--background", "--python", str(gen_script), "--",
        "--trajectory_id", str(traj_id),
        "--output_dir", str(raw_dir),
        "--force_texture", force_texture,
        "--skip_if_exists",
    ]
    print(f"[full_eval] rendering traj {traj_id}: "
          f"{' '.join(cmd[:3])} ... --trajectory_id {traj_id}")
    try:
        proc = subprocess.run(cmd, cwd=HERE, check=False)
    except Exception as exc:                                      # noqa: BLE001
        print(f"[full_eval] render traj {traj_id}: EXCEPTION {exc}")
        return False
    ok = proc.returncode == 0
    if not ok:
        print(f"[full_eval] render traj {traj_id}: FAILED rc={proc.returncode}")
    return ok


def _ensure_trajectories_rendered(traj_ids: Sequence[int], raw_dir: Path,
                                  attempt_render: bool = True,
                                  force_texture: str = "tex_03.webp"
                                  ) -> List[int]:
    """Make sure each traj_id has a complete render under raw_dir.

    Returns the list of traj_ids that are usable after the call. Missing
    trajs are skipped from the returned list when render fails or
    ``attempt_render`` is False.
    """
    usable: List[int] = []
    for tid in traj_ids:
        td = raw_dir / f"traj_{tid:04d}"
        if _traj_is_rendered(td):
            usable.append(tid); continue
        if not attempt_render:
            print(f"[full_eval] traj {tid}: not rendered (looked at {td}); "
                  f"skipping (attempt_render=False)")
            continue
        print(f"[full_eval] traj {tid}: not rendered at {td} — rendering now")
        if _render_traj(tid, raw_dir, force_texture=force_texture) \
                and _traj_is_rendered(td):
            usable.append(tid)
        else:
            print(f"[full_eval] traj {tid}: render did not produce a complete "
                  f"trajectory dir; skipping")
    return usable


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _umeyama_se3(pred_ts: np.ndarray, gt_ts: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SE(3) Umeyama alignment of `pred_ts` onto `gt_ts`. Returns
    (R_align, t_align, aligned_pred_ts)."""
    N = min(pred_ts.shape[0], gt_ts.shape[0])
    X = pred_ts[:N].T
    Y = gt_ts[:N].T
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
    aligned = (R @ X + t[:, None]).T
    return R, t, aligned


def _ate_rmse(aligned_pred: np.ndarray, gt_ts: np.ndarray) -> float:
    N = min(aligned_pred.shape[0], gt_ts.shape[0])
    err = aligned_pred[:N] - gt_ts[:N]
    return float(np.sqrt((err ** 2).sum(axis=1).mean()))


def _rpe_translation_rotation(gt_R: np.ndarray, gt_t: np.ndarray,
                              pred_R: np.ndarray, pred_t: np.ndarray,
                              delta_steps: int
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Relative pose error at fixed step delta.

    For each i with i+delta < N, compute T_gt = (T_gt_i)^-1 ∘ T_gt_{i+δ}
    and T_pr = (T_pr_i)^-1 ∘ T_pr_{i+δ}, then the residual T_err = T_gt^-1
    ∘ T_pr. Returns (per-pair translation norm, per-pair rotation angle in
    degrees).
    """
    N = min(gt_R.shape[0], pred_R.shape[0])
    if N <= delta_steps:
        return np.zeros(0), np.zeros(0)
    trans_err = np.zeros(N - delta_steps)
    rot_err_deg = np.zeros(N - delta_steps)
    for i in range(N - delta_steps):
        j = i + delta_steps
        # GT relative
        R_gt_rel = gt_R[i].T @ gt_R[j]
        t_gt_rel = gt_R[i].T @ (gt_t[j] - gt_t[i])
        R_pr_rel = pred_R[i].T @ pred_R[j]
        t_pr_rel = pred_R[i].T @ (pred_t[j] - pred_t[i])
        # residual
        R_diff = R_gt_rel.T @ R_pr_rel
        t_diff = t_pr_rel - t_gt_rel
        trans_err[i] = float(np.linalg.norm(t_diff))
        cos_theta = max(-1.0, min(1.0, (np.trace(R_diff) - 1.0) / 2.0))
        rot_err_deg[i] = float(np.degrees(np.arccos(cos_theta)))
    return trans_err, rot_err_deg


def _segment_drift(gt_ts: np.ndarray, pred_ts: np.ndarray, times: np.ndarray,
                   window_s: float = 2.0) -> Tuple[float, float, np.ndarray]:
    """Per-window drift (start-anchored). Returns (median, p95, raw)."""
    N = gt_ts.shape[0]
    per: List[float] = []
    k = 0
    while k < N:
        t0 = times[k]
        lo, hi = k, k
        while hi < N and (times[hi] - t0) < window_s:
            hi += 1
        if hi - lo < 2:
            break
        dgt = gt_ts[lo:hi] - gt_ts[lo]
        dpr = pred_ts[lo:hi] - pred_ts[lo]
        per.append(float(np.linalg.norm(dgt - dpr, axis=-1).mean()))
        k = hi
    arr = np.array(per) if per else np.zeros(1)
    return float(np.median(arr)), float(np.quantile(arr, 0.95)), arr


def _cumulative_distance(gt_ts: np.ndarray) -> np.ndarray:
    """Cumulative path length of GT positions, shape (N,)."""
    diffs = np.diff(gt_ts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    return cum


def _instantaneous_speed(gt_ts: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Finite-difference speed |dposition / dt|, shape (N,)."""
    if gt_ts.shape[0] < 2:
        return np.zeros(gt_ts.shape[0])
    dt = np.gradient(times)
    dt[dt < 1e-9] = 1e-9
    v = np.gradient(gt_ts, axis=0) / dt[:, None]
    return np.linalg.norm(v, axis=1)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_START_KW = dict(color="#00b050", s=55, zorder=5, edgecolor="black",
                 linewidth=0.6, label="start")
_END_KW = dict(color="#d62728", s=55, zorder=5, edgecolor="black",
               linewidth=0.6, label="end")


def _save(fig: plt.Figure, path: Path, dpi: int = 140) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_3d_overlay(gt: np.ndarray, pred: np.ndarray, path: Path,
                     title: str, ate: Optional[float] = None) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "r-", lw=2.5, label="GT", alpha=0.9)
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], "b-", lw=1.0,
            label="Prediction", alpha=0.95)
    ax.scatter(*gt[0], **_START_KW); ax.scatter(*gt[-1], **_END_KW)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(loc="best")
    ax.set_title(title + (f"\nATE = {ate:.3f} m" if ate is not None else ""))
    fig.tight_layout(); _save(fig, path)


def _plot_top_xy(gt: np.ndarray, pred: np.ndarray, path: Path,
                 title: str, ate: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(gt[:, 0], gt[:, 1], color="red", lw=1.5, label="GT")
    ax.plot(pred[:, 0], pred[:, 1], color="blue", lw=1.0, label="Prediction")
    ax.scatter([gt[0, 0]], [gt[0, 1]], **_START_KW)
    ax.scatter([gt[-1, 0]], [gt[-1, 1]], **_END_KW)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title + (f"\nATE = {ate:.3f} m" if ate is not None else ""))
    fig.tight_layout(); _save(fig, path)


def _plot_per_axis(times: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                   path: Path, ylabels: Sequence[str], title: str,
                   ate: Optional[float] = None) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
    for i, (ax, yl) in enumerate(zip(axs, ylabels)):
        ax.plot(times, gt[:, i], "r-", lw=2.0, label="GT")
        ax.plot(times, pred[:, i], "b-", lw=1.0, label="Prediction")
        ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
    axs[0].set_title(title + (f"   (ATE = {ate:.3f} m)" if ate is not None else ""))
    axs[0].legend(loc="best")
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout(); _save(fig, path)


def _plot_per_axis_error(times: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                         path: Path, ylabels: Sequence[str], title: str,
                         ate: Optional[float] = None) -> None:
    err = pred - gt
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
    for i, (ax, yl) in enumerate(zip(axs, ylabels)):
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
        ax.plot(times, err[:, i], color="#d62728", lw=1.0, label="pred − GT")
        ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
    axs[0].set_title(title + (f"   (ATE = {ate:.3f} m)" if ate is not None else ""))
    axs[0].legend(loc="best")
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout(); _save(fig, path)


def _plot_err_vs_distance(distance: np.ndarray, err_norm: np.ndarray,
                          path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(distance, err_norm, color="#d62728", lw=1.0)
    ax.set_xlabel("cumulative distance traveled (m)")
    ax.set_ylabel("position error magnitude (m)")
    ax.grid(True, alpha=0.3); ax.set_title(title)
    fig.tight_layout(); _save(fig, path)


def _plot_err_vs_speed(speed: np.ndarray, err_norm: np.ndarray,
                       path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(speed, err_norm, s=4, alpha=0.4, color="#1f77b4")
    # Bin and average for trend
    if speed.size > 10:
        nbins = 12
        edges = np.linspace(speed.min(), speed.max() + 1e-9, nbins + 1)
        idx = np.clip(np.digitize(speed, edges) - 1, 0, nbins - 1)
        means = np.array([err_norm[idx == k].mean() if (idx == k).any() else np.nan
                         for k in range(nbins)])
        centers = 0.5 * (edges[:-1] + edges[1:])
        mask = np.isfinite(means)
        ax.plot(centers[mask], means[mask], color="black", lw=1.5,
                label="binned mean")
        ax.legend(loc="best")
    ax.set_xlabel("instantaneous speed |v| (m/s)")
    ax.set_ylabel("position error magnitude (m)")
    ax.grid(True, alpha=0.3); ax.set_title(title)
    fig.tight_layout(); _save(fig, path)


# --- GT-only "trajectory shape" diagrams ----------------------------------
# These show the trajectory itself (no prediction) so the reader can see
# what the trajectory looks like at a glance.

def _plot_gt_top_diagram(gt: np.ndarray, path: Path, title: str) -> None:
    """Top-down (XY) GT trajectory diagram — start green, end red."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(gt[:, 0], gt[:, 1], color="black", lw=1.6, label="GT")
    ax.scatter([gt[0, 0]], [gt[0, 1]], **_START_KW)
    ax.scatter([gt[-1, 0]], [gt[-1, 1]], **_END_KW)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout(); _save(fig, path)


def _plot_gt_3d_diagram(gt: np.ndarray, path: Path, title: str) -> None:
    """3D GT trajectory diagram."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="black", lw=1.6, label="GT")
    ax.scatter(*gt[0], **_START_KW); ax.scatter(*gt[-1], **_END_KW)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(loc="best"); ax.set_title(title)
    fig.tight_layout(); _save(fig, path)


def _plot_gt_speed_profile(times: np.ndarray, speed: np.ndarray,
                           path: Path, title: str) -> None:
    """|v(t)| profile of the GT trajectory — shows the speed envelope the
    networks were asked to predict on."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(times, speed, color="#1f77b4", lw=1.2)
    ax.axhline(float(speed.mean()), color="gray", lw=0.8, ls="--",
               label=f"mean = {speed.mean():.2f} m/s")
    ax.axhline(float(speed.max()), color="red", lw=0.8, ls="--",
               label=f"peak = {speed.max():.2f} m/s")
    ax.set_xlabel("time (s)"); ax.set_ylabel("speed |v| (m/s)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout(); _save(fig, path)


# --- multi-method comparison overlays --------------------------------------

@dataclass
class _Trace:
    method: str
    aligned_pred: np.ndarray         # (N, 3)
    pred_R: np.ndarray               # (N, 3, 3)
    times: np.ndarray
    ate: float


def _plot_xy_overlay(gt: np.ndarray, traces: Sequence[_Trace], path: Path,
                     title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(gt[:, 0], gt[:, 1], color="black", lw=2.0, label="GT")
    for tr in traces:
        ax.plot(tr.aligned_pred[:, 0], tr.aligned_pred[:, 1],
                color=_METHOD_COLORS[tr.method], lw=1.2,
                label=f"{_METHOD_NAME[tr.method]} (ATE={tr.ate:.2f} m)")
    ax.scatter([gt[0, 0]], [gt[0, 1]], **_START_KW)
    ax.scatter([gt[-1, 0]], [gt[-1, 1]], **_END_KW)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); _save(fig, path)


def _plot_3d_overlay_methods(gt: np.ndarray, traces: Sequence[_Trace],
                             path: Path, title: str) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="black", lw=2.2, label="GT")
    for tr in traces:
        ax.plot(tr.aligned_pred[:, 0], tr.aligned_pred[:, 1], tr.aligned_pred[:, 2],
                color=_METHOD_COLORS[tr.method], lw=1.2,
                label=f"{_METHOD_NAME[tr.method]} (ATE={tr.ate:.2f} m)")
    ax.scatter(*gt[0], **_START_KW); ax.scatter(*gt[-1], **_END_KW)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(loc="best"); ax.set_title(title)
    fig.tight_layout(); _save(fig, path)


def _plot_per_axis_error_methods(gt: np.ndarray, traces: Sequence[_Trace],
                                 path: Path, title: str) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
    ylabels = ("x error (m)", "y error (m)", "z error (m)")
    for i, (ax, yl) in enumerate(zip(axs, ylabels)):
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
        for tr in traces:
            N = min(tr.aligned_pred.shape[0], gt.shape[0])
            err = tr.aligned_pred[:N, i] - gt[:N, i]
            ax.plot(tr.times[:N], err,
                    color=_METHOD_COLORS[tr.method], lw=1.0,
                    label=f"{_METHOD_NAME[tr.method]} (ATE={tr.ate:.2f} m)")
        ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
    axs[0].set_title(title); axs[0].legend(loc="best")
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout(); _save(fig, path)


def _plot_rpy_error_methods(gt_rpy: np.ndarray, traces_rpy: Dict[str, np.ndarray],
                            times: np.ndarray, path: Path, title: str
                            ) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
    ylabels = ("roll error (deg)", "pitch error (deg)", "yaw error (deg)")
    for i, (ax, yl) in enumerate(zip(axs, ylabels)):
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
        for m, pr_rpy in traces_rpy.items():
            N = min(pr_rpy.shape[0], gt_rpy.shape[0])
            err = pr_rpy[:N, i] - gt_rpy[:N, i]
            # Wrap to [-180, 180]
            err = (err + 180.0) % 360.0 - 180.0
            ax.plot(times[:N], err, color=_METHOD_COLORS[m], lw=1.0,
                    label=_METHOD_NAME[m])
        ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
    axs[0].set_title(title); axs[0].legend(loc="best")
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout(); _save(fig, path)


# ---------------------------------------------------------------------------
# Loss-curve plotting (parses train.log)
# ---------------------------------------------------------------------------

_TRAIN_LOG_HEADER = ("epoch", "stage", "train_loss", "val_loss",
                     "val_ATE_on_traj_0", "val_ATE_on_minitest", "lr")


def _parse_train_log(path: Path) -> Dict[str, np.ndarray]:
    """Parse train.log (TSV with header)."""
    if not path.exists():
        return {}
    rows: List[Dict[str, str]] = []
    with path.open("r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    out: Dict[str, np.ndarray] = {}

    def _to_float(s: str) -> float:
        try:
            return float(s)
        except ValueError:
            return float("nan")

    for col in header:
        if col == "stage":
            out[col] = np.array([r.get(col, "") for r in rows])
        else:
            out[col] = np.array([_to_float(r.get(col, "nan")) for r in rows])
    return out


def _plot_loss_curve(log: Dict[str, np.ndarray], modality: str, path: Path
                     ) -> None:
    if "epoch" not in log:
        return
    ep = log["epoch"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if "train_loss" in log:
        ax.plot(ep, log["train_loss"], color="#1f77b4", lw=1.3, label="train")
    if "val_loss" in log:
        ax.plot(ep, log["val_loss"], color="#d62728", lw=1.3, label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(f"{_METHOD_NAME[modality]} — train vs val loss")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); _save(fig, path)


def _plot_val_ate_curve(log: Dict[str, np.ndarray], modality: str, path: Path
                        ) -> None:
    if "epoch" not in log:
        return
    ep = log["epoch"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    has_data = False
    if "val_ATE_on_traj_0" in log and np.isfinite(log["val_ATE_on_traj_0"]).any():
        ax.plot(ep, log["val_ATE_on_traj_0"], color="#1f77b4", lw=1.2,
                label="val ATE (traj 0)")
        has_data = True
    if "val_ATE_on_minitest" in log and np.isfinite(log["val_ATE_on_minitest"]).any():
        ax.plot(ep, log["val_ATE_on_minitest"], color="#d62728", lw=1.2,
                label="val ATE (minitest)")
        has_data = True
    if not has_data:
        plt.close(fig); return
    ax.set_xlabel("epoch"); ax.set_ylabel("val ATE (m)")
    ax.set_title(f"{_METHOD_NAME[modality]} — validation ATE")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); _save(fig, path)


def _plot_combined_loss(logs: Dict[str, Dict[str, np.ndarray]], path: Path
                        ) -> None:
    if not logs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
    for m, log in logs.items():
        if "epoch" not in log:
            continue
        c = _METHOD_COLORS[m]
        if "train_loss" in log:
            axs[0].plot(log["epoch"], log["train_loss"], color=c, lw=1.2,
                        label=_METHOD_NAME[m])
        if "val_loss" in log:
            axs[1].plot(log["epoch"], log["val_loss"], color=c, lw=1.2,
                        label=_METHOD_NAME[m])
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("train loss")
    axs[0].set_title("Train loss"); axs[0].grid(alpha=0.3); axs[0].legend()
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("val loss")
    axs[1].set_title("Validation loss"); axs[1].grid(alpha=0.3); axs[1].legend()
    fig.tight_layout(); _save(fig, path)


def _plot_combined_val_ate(logs: Dict[str, Dict[str, np.ndarray]], path: Path
                           ) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    drew = False
    for m, log in logs.items():
        if "epoch" not in log: continue
        ate_curve = log.get("val_ATE_on_minitest", None)
        if ate_curve is None or not np.isfinite(ate_curve).any(): continue
        ax.plot(log["epoch"], ate_curve, color=_METHOD_COLORS[m], lw=1.3,
                label=_METHOD_NAME[m])
        drew = True
    if not drew:
        plt.close(fig); return
    ax.set_xlabel("epoch"); ax.set_ylabel("val ATE on minitest (m)")
    ax.set_title("Validation ATE — V vs I vs VI")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); _save(fig, path)


# ---------------------------------------------------------------------------
# Per-trajectory orchestration
# ---------------------------------------------------------------------------

@dataclass
class TrajResult:
    """Everything we record for one (split, traj, modality) cell."""
    step: str
    split: str
    label: str
    modality: str
    speed_mps: float
    ate_aligned: float
    ate_endpoint: float
    rpe_trans_1s_median: float
    rpe_trans_1s_p95: float
    rpe_rot_1s_median_deg: float
    rpe_rot_1s_p95_deg: float
    seg2s_median: float
    seg2s_p95: float
    n_frames: int
    duration_s: float


def _evaluate_trajectory(method: str, model: torch.nn.Module,
                         loader, traj, device: torch.device, fps: float
                         ) -> Dict[str, np.ndarray]:
    """Forward-pass + dead-reckoning for one trajectory.

    Returns a dict containing aligned predictions, GT, RPY, and intermediates.
    """
    rels: List[Tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            R, t = _predict(method, model, batch)
            rels.append((R.cpu().numpy(), t.cpu().numpy()))
    times, pred_ts, pred_q = dead_reckon_from_gt_start(traj, rels)
    gt_times, gt_ts, gt_q = gt_trajectory_frames(traj)

    N = min(times.shape[0], gt_times.shape[0])
    times = times[:N]; pred_ts = pred_ts[:N]; pred_q = pred_q[:N]
    gt_times = gt_times[:N]; gt_ts = gt_ts[:N]; gt_q = gt_q[:N]

    R_align, t_align, aligned_pred = _umeyama_se3(pred_ts, gt_ts)
    pred_R = quat_to_matrix(torch.from_numpy(pred_q)).numpy()
    gt_R = quat_to_matrix(torch.from_numpy(gt_q)).numpy()
    aligned_pred_R = R_align[None, :, :] @ pred_R   # rotate the rotations
    pred_rpy = np.degrees(matrix_to_rpy(aligned_pred_R))
    gt_rpy = np.degrees(matrix_to_rpy(gt_R))

    return {
        "times": times,
        "pred_ts_raw": pred_ts,
        "pred_q": pred_q,
        "aligned_pred": aligned_pred,
        "aligned_pred_R": aligned_pred_R,
        "gt_times": gt_times,
        "gt_ts": gt_ts,
        "gt_q": gt_q,
        "gt_R": gt_R,
        "pred_rpy": pred_rpy,
        "gt_rpy": gt_rpy,
        "fps": fps,
    }


def _process_traj(step: str, split: str, label: TrajLabel, cfg: dict,
                  models: Dict[str, torch.nn.Module], device: torch.device,
                  step_dir: Path, summary_rows: List[TrajResult]) -> None:
    """Evaluate every loaded modality on one traj, save per-method plots,
    then save the multi-method comparison plots."""
    raw_dir = HERE / cfg["data"]["raw_dir"] if not Path(cfg["data"]["raw_dir"]).is_absolute() \
              else Path(cfg["data"]["raw_dir"])
    img_size = tuple(cfg["data"]["img_size"])
    imu_ctx = cfg.get("data", {}).get("imu_context_length", None)

    print(f"[full_eval] {step} / {split} / {label.label} (traj {label.traj_id})")

    loader, ds = build_eval_loader(raw_dir=raw_dir, traj_ids=[label.traj_id],
                                   img_size=img_size, num_workers=0,
                                   imu_context_length=imu_ctx)
    traj = list(ds._trajectories.values())[0]
    fps = float(traj.metadata.get("fps_image", 100))

    out_split_dir = step_dir / split / label.label
    out_split_dir.mkdir(parents=True, exist_ok=True)

    # ---- "Normal trajectory diagram" — GT only, before any prediction. --
    # Lets the reader see what the trajectory looks like at a glance,
    # independent of any model output.
    diagram_dir = out_split_dir / "trajectory_diagram"
    gt_times_only, gt_ts_only, _gt_q_only = gt_trajectory_frames(traj)
    diagram_title = f"GT trajectory — {label.long_label}"
    _plot_gt_top_diagram(gt_ts_only, diagram_dir / "shape_top.png",
                         title=f"{diagram_title} (top-down)")
    _plot_gt_3d_diagram(gt_ts_only, diagram_dir / "shape_3d.png",
                        title=f"{diagram_title} (3D)")
    speed_profile = _instantaneous_speed(gt_ts_only, gt_times_only)
    _plot_gt_speed_profile(gt_times_only, speed_profile,
                           diagram_dir / "speed_profile.png",
                           title=f"{diagram_title} — speed profile")

    traces: List[_Trace] = []
    pred_rpy_per_method: Dict[str, np.ndarray] = {}
    gt_for_overlay: Optional[np.ndarray] = None
    gt_rpy_for_overlay: Optional[np.ndarray] = None
    times_for_overlay: Optional[np.ndarray] = None

    if not models:
        # Diagrams already saved; nothing else to do for this traj.
        print(f"[full_eval] {step}/{split}/{label.label}: no models loaded — "
              f"only GT diagram + speed profile saved.")
        return

    for method, model in models.items():
        out = _evaluate_trajectory(method, model, loader, traj, device, fps)
        gt_ts = out["gt_ts"]; gt_R = out["gt_R"]
        aligned_pred = out["aligned_pred"]; aligned_pred_R = out["aligned_pred_R"]
        times = out["times"]
        gt_rpy = out["gt_rpy"]; pred_rpy = out["pred_rpy"]

        ate = _ate_rmse(aligned_pred, gt_ts)
        endpoint = float(np.linalg.norm(aligned_pred[-1] - gt_ts[-1]))
        med2s, p95_2s, _ = _segment_drift(gt_ts, aligned_pred, times,
                                          window_s=cfg.get("eval", {}).get("segment_seconds", 2.0))
        # 1s RPE step count (image rate is 100 Hz → 100 steps per 1s, but
        # gt_trajectory_frames subsamples by 10, so kept-frame rate is fps/1
        # = 100 Hz/10 = 10 fps. 1 second = 10 kept frames).
        # We use fps (kept-frame rate) directly; default cfg has fps_image=100.
        # The dataloader emits 1 frame pair per kept frame, so a 1s RPE
        # window is `int(fps)` steps. Guard against fps=10 case:
        kept_fps = fps if fps <= 30 else fps / 10
        rpe_steps_1s = max(1, int(round(kept_fps)))
        rpe_t, rpe_r = _rpe_translation_rotation(gt_R, gt_ts,
                                                 aligned_pred_R, aligned_pred,
                                                 delta_steps=rpe_steps_1s)
        rpe_t_med = float(np.median(rpe_t)) if rpe_t.size else float("nan")
        rpe_t_p95 = float(np.quantile(rpe_t, 0.95)) if rpe_t.size else float("nan")
        rpe_r_med = float(np.median(rpe_r)) if rpe_r.size else float("nan")
        rpe_r_p95 = float(np.quantile(rpe_r, 0.95)) if rpe_r.size else float("nan")

        method_dir = out_split_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        # Plots
        plot_title = f"{_METHOD_NAME[method]} — {label.long_label}"
        _plot_3d_overlay(gt_ts, aligned_pred,
                         method_dir / "gt_vs_pred_3d.png",
                         title=plot_title, ate=ate)
        _plot_top_xy(gt_ts, aligned_pred,
                     method_dir / "top_xy.png",
                     title=plot_title, ate=ate)
        _plot_per_axis(times, gt_ts, aligned_pred,
                       method_dir / "position_xyz.png",
                       ylabels=("x (m)", "y (m)", "z (m)"),
                       title=f"{plot_title} — position vs time", ate=ate)
        _plot_per_axis_error(times, gt_ts, aligned_pred,
                             method_dir / "position_error.png",
                             ylabels=("x error (m)", "y error (m)", "z error (m)"),
                             title=f"{plot_title} — position error", ate=ate)
        _plot_per_axis(times, gt_rpy, pred_rpy,
                       method_dir / "rpy.png",
                       ylabels=("roll (deg)", "pitch (deg)", "yaw (deg)"),
                       title=f"{plot_title} — RPY vs time", ate=ate)
        # RPY error per axis (wrap to [-180, 180])
        rpy_err = pred_rpy - gt_rpy
        rpy_err = (rpy_err + 180.0) % 360.0 - 180.0
        _plot_per_axis_error(times, np.zeros_like(rpy_err), rpy_err,
                             method_dir / "rpy_error.png",
                             ylabels=("roll error (deg)", "pitch error (deg)",
                                      "yaw error (deg)"),
                             title=f"{plot_title} — RPY error",
                             ate=ate)

        # Error vs distance traveled
        cum_dist = _cumulative_distance(gt_ts)
        err_norm = np.linalg.norm(aligned_pred - gt_ts, axis=1)
        _plot_err_vs_distance(cum_dist, err_norm,
                              method_dir / "err_vs_distance.png",
                              title=f"{plot_title} — error vs distance traveled")

        # Error vs instantaneous speed
        speed = _instantaneous_speed(gt_ts, times)
        _plot_err_vs_speed(speed, err_norm,
                           method_dir / "err_vs_speed.png",
                           title=f"{plot_title} — error vs instantaneous speed")

        # TUM dumps for downstream analysis
        write_tum(method_dir / "gt.tum", times, gt_ts, out["gt_q"])
        write_tum(method_dir / "pred.tum", times, aligned_pred, out["pred_q"])

        # Per-cell metrics
        metrics = {
            "step": step,
            "split": split,
            "label": label.label,
            "modality": method,
            "traj_id": label.traj_id,
            "speed_mps": label.speed_mps,
            "ate_aligned": ate,
            "ate_endpoint": endpoint,
            "rpe_trans_1s_median": rpe_t_med,
            "rpe_trans_1s_p95": rpe_t_p95,
            "rpe_rot_1s_median_deg": rpe_r_med,
            "rpe_rot_1s_p95_deg": rpe_r_p95,
            "seg2s_median": med2s,
            "seg2s_p95": p95_2s,
            "n_frames": int(times.shape[0]),
            "duration_s": float(times[-1] - times[0]) if times.shape[0] > 1 else 0.0,
        }
        (method_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        summary_rows.append(TrajResult(
            step=step, split=split, label=label.label, modality=method,
            speed_mps=label.speed_mps, ate_aligned=ate, ate_endpoint=endpoint,
            rpe_trans_1s_median=rpe_t_med, rpe_trans_1s_p95=rpe_t_p95,
            rpe_rot_1s_median_deg=rpe_r_med, rpe_rot_1s_p95_deg=rpe_r_p95,
            seg2s_median=med2s, seg2s_p95=p95_2s,
            n_frames=metrics["n_frames"], duration_s=metrics["duration_s"],
        ))

        traces.append(_Trace(method=method, aligned_pred=aligned_pred,
                             pred_R=aligned_pred_R, times=times, ate=ate))
        pred_rpy_per_method[method] = pred_rpy
        if gt_for_overlay is None:
            gt_for_overlay = gt_ts.copy()
            gt_rpy_for_overlay = gt_rpy.copy()
            times_for_overlay = times.copy()

    # Multi-method comparison overlays (only if 2+ modalities loaded)
    if len(traces) >= 2 and gt_for_overlay is not None:
        comp_dir = out_split_dir / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        title = f"comparison — {label.long_label}"
        _plot_xy_overlay(gt_for_overlay, traces,
                         comp_dir / "xy_overlay.png", title=title)
        _plot_3d_overlay_methods(gt_for_overlay, traces,
                                 comp_dir / "3d_overlay.png", title=title)
        _plot_per_axis_error_methods(gt_for_overlay, traces,
                                     comp_dir / "per_axis_error.png",
                                     title=f"{title} — per-axis position error")
        if gt_rpy_for_overlay is not None and times_for_overlay is not None:
            _plot_rpy_error_methods(gt_rpy_for_overlay, pred_rpy_per_method,
                                    times_for_overlay,
                                    comp_dir / "rpy_error_overlay.png",
                                    title=f"{title} — RPY error")


# ---------------------------------------------------------------------------
# Aggregate plots across one step
# ---------------------------------------------------------------------------

def _plot_aggregate_ate(rows: List[TrajResult], path: Path) -> None:
    """Bar chart: x = trajectory label (sorted by split, speed), y = ATE,
    grouped by modality."""
    # group rows by (split, label, speed); each cell has up to 3 modalities
    by_traj: Dict[Tuple[str, str], Dict[str, float]] = {}
    speed_lookup: Dict[Tuple[str, str], float] = {}
    for r in rows:
        key = (r.split, r.label)
        by_traj.setdefault(key, {})[r.modality] = r.ate_aligned
        speed_lookup[key] = r.speed_mps
    # sort: split first (train < val < test alphabetically), then label
    split_order = {"train": 0, "val": 1, "test": 2}
    keys = sorted(by_traj.keys(),
                  key=lambda k: (split_order.get(k[0], 99),
                                 speed_lookup.get(k, 0.0), k[1]))
    if not keys:
        return
    n = len(keys)
    x = np.arange(n)
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), 5))
    for i, m in enumerate(METHODS):
        vals = np.array([by_traj[k].get(m, np.nan) for k in keys])
        ax.bar(x + (i - 1) * width, vals, width=width,
               color=_METHOD_COLORS[m], label=_METHOD_NAME[m])
    ax.set_xticks(x)
    labels = [f"{k[0]}/{k[1]}" for k in keys]
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("ATE-aligned (m)")
    ax.set_title("Per-trajectory ATE — V vs I vs VI")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); _save(fig, path)


def _plot_metric_vs_speed(rows: List[TrajResult], metric: str,
                          ylabel: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in METHODS:
        sub = [r for r in rows if r.modality == m]
        if not sub: continue
        # group by split for marker style
        for split, marker in (("train", "o"), ("val", "s"), ("test", "^")):
            sp = [r for r in sub if r.split == split]
            if not sp: continue
            xs = [r.speed_mps for r in sp]
            ys = [getattr(r, metric) for r in sp]
            labels = [r.label for r in sp]
            ax.scatter(xs, ys, color=_METHOD_COLORS[m],
                       marker=marker, s=42, alpha=0.85,
                       label=f"{_METHOD_NAME[m]} ({split})")
            for x_, y_, lab in zip(xs, ys, labels):
                ax.annotate(lab, (x_, y_), fontsize=6, alpha=0.65,
                            xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("target peak speed (m/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); _save(fig, path)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _resolve_split_ids(cfg: dict, step_key: str) -> Dict[str, List[int]]:
    """Read split IDs from the cfg. For Step S, append the triangle speed
    sweep (420-424) to the test split — these probe shape-generalisation
    across speed and were not part of the original Step S test design."""
    train_ids = list(cfg["data"].get("train_ids", []))
    val_id = cfg["data"].get("val_traj_id", None)
    test_ids = list(cfg["data"].get("test_ids", []))
    if step_key == "s":
        for tid, _speed in _STEP_S_EXTRA_TEST:
            if tid not in test_ids:
                test_ids.append(tid)
    splits = {"train": train_ids, "test": test_ids}
    if val_id is not None:
        splits["val"] = [int(val_id)]
    else:
        splits["val"] = []
    return splits


def _write_trajectories_table(step_dir: Path,
                              cfg_split_ids: Dict[str, List[int]],
                              raw_dir: Path) -> None:
    """Write a clean (markdown + CSV) traj_id ↔ shape ↔ speed ↔ split table.

    Sourced from on-disk metadata for every traj listed in the cfg splits —
    even ones that failed to evaluate (so the table is a complete inventory,
    not just successfully-evaluated trajs).
    """
    entries: List[Dict[str, object]] = []
    seen: set = set()
    split_order = {"train": 0, "val": 1, "test": 2}
    for split_name, ids in cfg_split_ids.items():
        for tid in ids:
            if tid in seen: continue
            seen.add(tid)
            traj_dir = raw_dir / f"traj_{tid:04d}"
            if not (traj_dir / "metadata.json").exists():
                entries.append({
                    "traj_id": tid, "split": split_name,
                    "family": "(missing)", "shape": "(missing)",
                    "target_speed_mps": "—", "label": "(missing)",
                })
                continue
            label = _label_for_traj(traj_dir, split_hint=split_name)
            label.split = split_name
            entries.append({
                "traj_id": tid, "split": split_name,
                "family": _canonical_family(traj_dir),
                "shape": label.family_short,
                "target_speed_mps": label.speed_mps,
                "label": label.label,
            })
    entries.sort(key=lambda e: (split_order.get(e["split"], 99),
                                e["target_speed_mps"]
                                if isinstance(e["target_speed_mps"], (int, float))
                                else 99.0,
                                e["traj_id"]))
    # CSV
    csv_path = step_dir / "trajectories_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj_id", "split", "family", "shape",
                    "target_speed_mps", "label"])
        for e in entries:
            w.writerow([e["traj_id"], e["split"], e["family"], e["shape"],
                        e["target_speed_mps"], e["label"]])
    # Markdown
    md_path = step_dir / "trajectories_table.md"
    lines = [
        f"# Trajectories — Step {step_dir.name.split('_')[-1].upper()}",
        "",
        "| traj_id | split | shape | target speed (m/s) | label |",
        "|--------:|:------|:------|------------------:|:------|",
    ]
    for e in entries:
        speed = e["target_speed_mps"]
        speed_str = f"{speed:.2f}" if isinstance(speed, (int, float)) else str(speed)
        lines.append(f"| {e['traj_id']:04d} | {e['split']} | {e['shape']} | "
                     f"{speed_str} | `{e['label']}` |")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"[full_eval] {csv_path}")
    print(f"[full_eval] {md_path}")


def _canonical_family(traj_dir: Path) -> str:
    """Read the un-shortened family name from metadata.json."""
    md = json.loads((traj_dir / "metadata.json").read_text())
    return str(md.get("family", "unknown"))


def _process_step(step_key: str, config_path: Path, ckpt_root: Path,
                  output_root: Path, device: torch.device,
                  attempt_render: bool = True) -> List[TrajResult]:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))
    step_dir = output_root / f"step_{step_key}"
    step_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== STEP {step_key.upper()} ==========")
    print(f"  config: {config_path}")
    print(f"  ckpts:  {ckpt_root}")
    print(f"  out:    {step_dir}")

    # --- Loss curves first (cheap, no GPU work) ---------------------------
    loss_dir = step_dir / "losses"
    logs: Dict[str, Dict[str, np.ndarray]] = {}
    for m in METHODS:
        log = _parse_train_log(ckpt_root / m / "train.log")
        if log:
            logs[m] = log
            _plot_loss_curve(log, m, loss_dir / f"{m}_loss.png")
            _plot_val_ate_curve(log, m, loss_dir / f"{m}_val_ate.png")
    _plot_combined_loss(logs, loss_dir / "combined_loss.png")
    _plot_combined_val_ate(logs, loss_dir / "combined_val_ate.png")

    # --- Load checkpoints (GPU work begins) -------------------------------
    models = _load_models(cfg, device, ckpt_root)
    if not models:
        print(f"[full_eval] step {step_key}: no checkpoints found — will "
              f"still emit GT-only trajectory diagrams and the table.")

    # --- Trajectory loop --------------------------------------------------
    splits = _resolve_split_ids(cfg, step_key)
    raw_dir = HERE / cfg["data"]["raw_dir"] if not Path(cfg["data"]["raw_dir"]).is_absolute() \
              else Path(cfg["data"]["raw_dir"])

    # --- Pre-flight: render any missing trajs before evaluation ----------
    # The Step S extra triangle test set (420-424) may not be on disk yet;
    # render it via Blender on first run. Other (existing) trajectories pass
    # through unchanged.
    all_required_ids = sorted({tid for ids in splits.values() for tid in ids})
    missing_ids = [tid for tid in all_required_ids
                   if not _traj_is_rendered(raw_dir / f"traj_{tid:04d}")]
    if missing_ids:
        print(f"[full_eval] step {step_key}: {len(missing_ids)} trajs missing "
              f"on disk: {missing_ids}")
        print(f"[full_eval] looking for trajectories under: {raw_dir}")
        print(f"[full_eval] (on Turing this is "
              f"~/vio/Code/Phase 2/data/raw_g/traj_NNNN/)")
        _ensure_trajectories_rendered(missing_ids, raw_dir,
                                      attempt_render=attempt_render,
                                      force_texture="tex_03.webp")

    # --- Trajectories table — write before eval so even a partial run -----
    # produces the inventory (the report-friendly traj ↔ speed table).
    _write_trajectories_table(step_dir, splits, raw_dir)

    summary_rows: List[TrajResult] = []
    for split_name, ids in splits.items():
        for tid in ids:
            traj_dir = raw_dir / f"traj_{tid:04d}"
            if not (traj_dir / "metadata.json").exists():
                print(f"[full_eval] {step_key} / {split_name}: traj {tid} "
                      f"missing on disk ({traj_dir}); skipping")
                continue
            label = _label_for_traj(traj_dir, split_hint=split_name)
            label.split = split_name   # honor the cfg split, not metadata
            try:
                _process_traj(step_key, split_name, label, cfg, models,
                              device, step_dir, summary_rows)
            except Exception as exc:                                       # noqa: BLE001
                print(f"[full_eval] FAILED on {step_key}/{split_name}/{label.label}: {exc}")
                import traceback; traceback.print_exc()

    # --- Aggregate plots --------------------------------------------------
    agg = step_dir / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    _plot_aggregate_ate(summary_rows, agg / "ate_per_modality.png")
    _plot_metric_vs_speed(summary_rows, "ate_aligned",
                          "ATE-aligned (m)",
                          agg / "ate_vs_speed.png",
                          title="ATE vs target peak speed")
    _plot_metric_vs_speed(summary_rows, "rpe_trans_1s_median",
                          "RPE-translation 1 s median (m)",
                          agg / "rpe_trans_vs_speed.png",
                          title="RPE-translation (1 s, median) vs target peak speed")
    _plot_metric_vs_speed(summary_rows, "rpe_rot_1s_median_deg",
                          "RPE-rotation 1 s median (deg)",
                          agg / "rpe_rot_vs_speed.png",
                          title="RPE-rotation (1 s, median) vs target peak speed")

    # --- Summary tables ---------------------------------------------------
    csv_path = step_dir / "summary.csv"
    json_path = step_dir / "summary.json"
    fields = ["step", "split", "label", "modality", "speed_mps",
              "ate_aligned", "ate_endpoint", "rpe_trans_1s_median",
              "rpe_trans_1s_p95", "rpe_rot_1s_median_deg", "rpe_rot_1s_p95_deg",
              "seg2s_median", "seg2s_p95", "n_frames", "duration_s"]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in summary_rows:
            w.writerow([getattr(r, k) for k in fields])
    json_path.write_text(json.dumps(
        [{k: getattr(r, k) for k in fields} for r in summary_rows],
        indent=2) + "\n")

    print(f"[full_eval] step {step_key}: wrote {len(summary_rows)} rows to {csv_path}")
    return summary_rows


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", nargs="+", default=["k", "s"],
                   choices=["k", "s"],
                   help="Which step checkpoints to evaluate.")
    p.add_argument("--output_root", type=Path,
                   default=HERE / "full_results",
                   help="Output root. Defaults to <Phase 2>/full_results/")
    p.add_argument("--config_k", type=Path,
                   default=HERE / _STEP_DEFAULTS["k"]["config"])
    p.add_argument("--config_s", type=Path,
                   default=HERE / _STEP_DEFAULTS["s"]["config"])
    p.add_argument("--ckpts_k", type=Path,
                   default=HERE / _STEP_DEFAULTS["k"]["checkpoints"])
    p.add_argument("--ckpts_s", type=Path,
                   default=HERE / _STEP_DEFAULTS["s"]["checkpoints"])
    p.add_argument("--no_render", action="store_true",
                   help="Skip the Blender render step entirely. Trajectories "
                        "missing on disk are skipped instead of generated.")
    return p.parse_args()


def main() -> int:
    args = _parse()
    args.output_root = Path(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"[full_eval] device: {device}")
    print(f"[full_eval] output_root: {args.output_root}")

    all_rows: List[TrajResult] = []
    for step in args.steps:
        cfg_path = args.config_k if step == "k" else args.config_s
        ckpts = args.ckpts_k if step == "k" else args.ckpts_s
        if not cfg_path.exists():
            print(f"[full_eval] step {step}: config not found at {cfg_path}, skipping")
            continue
        if not ckpts.exists():
            print(f"[full_eval] step {step}: ckpt dir not found at {ckpts}, skipping")
            continue
        rows = _process_step(step, cfg_path, ckpts, args.output_root, device,
                             attempt_render=not args.no_render)
        all_rows.extend(rows)

    # Top-level cross-step summary CSV
    if len(all_rows) > 0 and len(args.steps) > 1:
        master_csv = args.output_root / "summary_all_steps.csv"
        fields = ["step", "split", "label", "modality", "speed_mps",
                  "ate_aligned", "ate_endpoint", "rpe_trans_1s_median",
                  "rpe_trans_1s_p95", "rpe_rot_1s_median_deg",
                  "rpe_rot_1s_p95_deg", "seg2s_median", "seg2s_p95",
                  "n_frames", "duration_s"]
        with master_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            for r in all_rows:
                w.writerow([getattr(r, k) for k in fields])
        print(f"[full_eval] cross-step summary: {master_csv}")

    print("\n[full_eval] DONE.")
    print(f"[full_eval] open: {args.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
