"""Shared helpers for Phase 2 plotting scripts.

Matches Phase 1 style: GT red thick, estimate blue thin, grid alpha=0.3.

All Phase 2 artifacts are read from the expected locations (see
`scripts/evaluate.py` and `scripts/_train_common.py`). When an artifact is
missing, loaders return ``None`` / empty containers and the caller warns and
skips — nothing hard-fails.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Matplotlib: headless, consistent style.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PHASE2_ROOT = Path(__file__).resolve().parents[1]           # Code/Phase 2/
REPO_ROOT = PHASE2_ROOT.parents[1]                          # repo root
RESULTS_ROOT = REPO_ROOT / "Results" / "Phase2"
OUTPUTS_ROOT = PHASE2_ROOT / "outputs"

DEFAULT_METHODS = ("v", "i", "vi")
ABLATION_METHODS = ("vi_concat", "vi_no_decomp", "vi_concat_decompose")
ALL_METHODS = DEFAULT_METHODS + ABLATION_METHODS

# Trajectory IDs live in configs/default.yaml; keep a fallback for loader-less use.
TRAIN_IDS = (0, 1, 2, 3)
TEST_IDS = (100, 101, 102)
TEST_NAMES = {100: "flat_spiral", 101: "figure_8", 102: "clover"}

METHOD_COLORS = {
    "v": "#1f77b4",                    # blue
    "i": "#2ca02c",                    # green
    "vi": "#d62728",                   # red
    "vi_concat": "#9467bd",            # purple
    "vi_no_decomp": "#8c564b",         # brown
    "vi_concat_decompose": "#e377c2",  # pink
}
METHOD_LABELS = {
    "v": "Vision-only",
    "i": "Inertial-only",
    "vi": "Visual-inertial",
    "vi_concat": "VI (concat)",
    "vi_no_decomp": "VI (no-decomp)",
    "vi_concat_decompose": "VI (concat+decomp)",
}

# Phase 1 convention: GT red, estimate blue.
GT_COLOR = "red"
EST_COLOR = "tab:blue"
GT_LW = 1.6
EST_LW = 1.1


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tum(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read a TUM file: returns (times, ts_xyz, quats_xyzw) or None if missing."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    arr = np.loadtxt(str(p))
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 8:
        return None
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


def load_train_log(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Parse a ``train.log`` TSV written by ``_train_common.TextLogger``.

    Returns dict with keys matching COLS, values as float arrays (stage left as
    str array). Returns None if file missing or only a header.

    De-duplicates by epoch (keeps the LAST occurrence). ``_train_common.TextLogger``
    appends without clearing, so two training runs against the same out_subdir
    concatenate into the log; without this dedupe, line plots draw a diagonal
    sweep from the last point of run 1 back to the first point of run 2.
    Keeping the last occurrence per epoch corresponds to the most recent run,
    which is the one that wrote the current ``best.pt`` and ``results.csv``.
    """
    p = Path(path)
    if not p.exists():
        return None
    rows: List[List[str]] = []
    header: Optional[List[str]] = None
    with p.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            rows.append(parts)
    if not rows or header is None:
        return None
    # Deduplicate by epoch (column 0 of the header is "epoch"). Keep last.
    try:
        epoch_col = header.index("epoch")
    except ValueError:
        epoch_col = 0
    by_epoch: Dict[int, List[str]] = {}
    for r in rows:
        try:
            e = int(r[epoch_col])
        except (ValueError, IndexError):
            continue
        by_epoch[e] = r
    rows = [by_epoch[e] for e in sorted(by_epoch)]
    out: Dict[str, np.ndarray] = {}
    for i, col in enumerate(header):
        col_vals = [r[i] if i < len(r) else "" for r in rows]
        if col == "stage":
            out[col] = np.array(col_vals, dtype=object)
        else:
            parsed: List[float] = []
            for v in col_vals:
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(np.nan)
            out[col] = np.asarray(parsed, dtype=float)
    return out


def load_results_csv(path: Path) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Long-form results.csv -> dict[(method, traj_id)][metric] = value."""
    p = Path(path)
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    if not p.exists():
        return out
    with p.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (row["method"], int(row["traj_id"]))
                out.setdefault(key, {})[row["metric_name"]] = float(row["value"])
            except (KeyError, ValueError):
                continue
    return out


def load_csv_table(path: Path) -> Optional[List[Dict[str, str]]]:
    """Simple CSV -> list-of-dicts. Used for the studies/ tables."""
    p = Path(path)
    if not p.exists():
        return None
    with p.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Math helpers (quaternion -> rotation / rpy / errors)
# ---------------------------------------------------------------------------

def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """Scalar-last quaternions (N,4) -> rotation matrices (N,3,3).

    Convention: body-to-world (active), matching ``utils/geometry.quat_to_matrix``.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def R_to_ypr_deg(R: np.ndarray) -> np.ndarray:
    """(N,3,3) -> (N,3) yaw/pitch/roll in degrees (ZYX intrinsic)."""
    sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    sing = sy < 1e-9
    yaw = np.where(sing, 0.0, np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    pitch = np.arctan2(-R[:, 2, 0], sy)
    roll = np.where(sing,
                    np.arctan2(-R[:, 1, 2], R[:, 1, 1]),
                    np.arctan2(R[:, 2, 1], R[:, 2, 2]))
    return np.degrees(np.stack([yaw, pitch, roll], axis=-1))


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SE(3) alignment (no scale). Returns (R, t) so R @ src.T + t ≈ dst.T.

    Follows the reference implementation in ``Code/Phase 1/evaluate.py``.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    sc = src - mu_src
    dc = dst - mu_dst
    cov = dc.T @ sc / n
    U, _D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    t = mu_dst - R @ mu_src
    return R, t


def cumulative_distance(P: np.ndarray) -> np.ndarray:
    """Arclength along an (N,3) position array, starts at 0."""
    d = np.zeros(len(P))
    if len(P) > 1:
        d[1:] = np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))
    return d


def relative_errors_at_distance(P_gt: np.ndarray, R_gt: np.ndarray,
                                P_est: np.ndarray, R_est: np.ndarray,
                                cum: np.ndarray, target_d: float,
                                stride: int = 5, tol: float = 0.1
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """RPE at sub-trajectory length ``target_d``.

    For every start i, find first j with cum[j]-cum[i] >= target_d; compute
    (||t_err||, |yaw_err|_deg) where T_err = inv(T_gt_ij) @ T_est_ij.
    Ported from Phase 1 ``plot_errors.relative_errors_at_distance``.
    """
    n = len(P_gt)
    trans_errs: List[float] = []
    yaw_errs: List[float] = []
    max_d = target_d * (1.0 + tol)
    j_start = 0
    for i in range(0, n - 1, stride):
        d0 = cum[i]
        j = max(j_start, i + 1)
        while j < n and cum[j] - d0 < target_d:
            j += 1
        if j >= n:
            break
        if cum[j] - d0 > max_d:
            continue
        j_start = max(j_start, i + 1)

        dR_gt = R_gt[i].T @ R_gt[j]
        dt_gt = R_gt[i].T @ (P_gt[j] - P_gt[i])
        dR_est = R_est[i].T @ R_est[j]
        dt_est = R_est[i].T @ (P_est[j] - P_est[i])

        R_err = dR_gt.T @ dR_est
        t_err = dR_gt.T @ (dt_est - dt_gt)
        yaw, _pitch, _roll = R_to_ypr_deg(R_err[None, :, :])[0]
        trans_errs.append(float(np.linalg.norm(t_err)))
        yaw_errs.append(abs(yaw))
    return np.asarray(trans_errs), np.asarray(yaw_errs)


def per_step_errors(gt_P: np.ndarray, pred_P: np.ndarray,
                    gt_R: np.ndarray, pred_R: np.ndarray
                    ) -> Dict[str, np.ndarray]:
    """Per-consecutive-frame errors used for histograms/CDFs."""
    d_gt_t = np.diff(gt_P, axis=0)
    d_pr_t = np.diff(pred_P, axis=0)
    trans_step = np.linalg.norm(d_pr_t - d_gt_t, axis=1)

    # per-step rotation error (angle of inv(dR_gt) @ dR_pred)
    dR_gt = np.einsum("nij,njk->nik", np.transpose(gt_R[:-1], (0, 2, 1)), gt_R[1:])
    dR_pr = np.einsum("nij,njk->nik", np.transpose(pred_R[:-1], (0, 2, 1)), pred_R[1:])
    R_err = np.einsum("nij,njk->nik", np.transpose(dR_gt, (0, 2, 1)), dR_pr)
    tr = np.clip((R_err[:, 0, 0] + R_err[:, 1, 1] + R_err[:, 2, 2] - 1.0) / 2.0,
                 -1.0, 1.0)
    rot_step_deg = np.degrees(np.arccos(tr))

    # per-step yaw error
    ypr_gt = R_to_ypr_deg(dR_gt)
    ypr_pr = R_to_ypr_deg(dR_pr)
    yaw_step = np.abs(ypr_pr[:, 0] - ypr_gt[:, 0])
    # wrap to [-180, 180] then abs
    yaw_step = np.minimum(yaw_step, 360.0 - yaw_step)
    return {"trans": trans_step, "rot": rot_step_deg, "yaw": yaw_step}


# ---------------------------------------------------------------------------
# Plot-layout helpers
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> Path:
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def savefig(fig, path: Path, dpi: int = 130) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def warn(msg: str) -> None:
    print(f"[plot_results] WARN: {msg}")


def info(msg: str) -> None:
    print(f"[plot_results] {msg}")


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "gray")


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


# ---------------------------------------------------------------------------
# Trajectory bundle — loaded once, reused across per-traj plots
# ---------------------------------------------------------------------------

class TrajBundle:
    """Holds GT + prediction trajectories aligned to the same length N.

    ``pred_aligned_P`` is the Umeyama-aligned prediction (positions), used for
    Phase-1-style error-vs-distance plots. ``pred_aligned_R`` rotates the raw
    prediction rotations by the alignment R on the left.
    """
    __slots__ = ("method", "traj_id", "times", "gt_P", "gt_Q", "gt_R",
                 "pred_P", "pred_Q", "pred_R", "pred_aligned_P",
                 "pred_aligned_R", "cum")

    def __init__(self, method: str, traj_id: int, times: np.ndarray,
                 gt_P: np.ndarray, gt_Q: np.ndarray,
                 pred_P: np.ndarray, pred_Q: np.ndarray) -> None:
        self.method = method
        self.traj_id = traj_id
        N = min(len(times), len(gt_P), len(pred_P))
        self.times = times[:N]
        self.gt_P = gt_P[:N]
        self.gt_Q = gt_Q[:N]
        self.pred_P = pred_P[:N]
        self.pred_Q = pred_Q[:N]
        self.gt_R = quat_xyzw_to_R(self.gt_Q)
        self.pred_R = quat_xyzw_to_R(self.pred_Q)
        R_a, t_a = umeyama_se3(self.pred_P, self.gt_P)
        self.pred_aligned_P = (R_a @ self.pred_P.T).T + t_a
        self.pred_aligned_R = np.einsum("ij,njk->nik", R_a, self.pred_R)
        self.cum = cumulative_distance(self.gt_P)


def load_bundle(method: str, traj_id: int,
                traj_dir: Path) -> Optional[TrajBundle]:
    """Load one (method, traj_id) bundle from outputs/trajectories/."""
    gt_path = Path(traj_dir) / f"gt_{traj_id}.tum"
    pred_path = Path(traj_dir) / f"{method}_{traj_id}.tum"
    gt = load_tum(gt_path)
    pr = load_tum(pred_path)
    if gt is None or pr is None:
        return None
    gt_t, gt_P, gt_Q = gt
    pr_t, pr_P, pr_Q = pr
    # Use the GT timestamps (more accurate; prediction dead-reckons at fps).
    N = min(len(gt_t), len(pr_t))
    return TrajBundle(method, traj_id, gt_t[:N], gt_P[:N], gt_Q[:N],
                      pr_P[:N], pr_Q[:N])


def discover_pairs(traj_dir: Path, methods: Iterable[str] = ALL_METHODS,
                   traj_ids: Iterable[int] = TEST_IDS
                   ) -> List[Tuple[str, int]]:
    """Return (method, traj_id) pairs where BOTH gt and pred TUM files exist."""
    out: List[Tuple[str, int]] = []
    td = Path(traj_dir)
    for m in methods:
        for t in traj_ids:
            if (td / f"gt_{t}.tum").exists() and (td / f"{m}_{t}.tum").exists():
                out.append((m, t))
    return out
