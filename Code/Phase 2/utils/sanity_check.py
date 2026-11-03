"""Pre-training sanity checks for Phase 2.

Each check returns a ``SanityCheckResult`` with a pass flag, a short human
message, and optional matplotlib figures. The orchestrator
(``scripts/run_sanity_checks.py``) saves figures under ``outputs/sanity/``.

Checks (eight total):
    1. check_pose_chaining         — GT pose chaining round-trip
    2. check_imu_integration       — IMU double-integration drift
    3. check_homography_warp       — GT homography warps I_t close to I_{t+1}
    4. check_motion_histogram      — global |Δt| histogram across training
    5. check_drone_dynamics        — thrust-vector alignment invariant
    6. check_motion_distribution   — per-traj |Δt_gt| distribution (Tier-1 B.5)
    7. check_footprint_confinement — pose + camera footprint stays on plane
    8. check_texture_assignment    — metadata.texture_path lookup
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

from .dataloader import _IMU_WINDOW, TrajectoryData
from .geometry import (compose_homography, compose_homography_blender_plane,
                       relative_pose)


GRAVITY = np.array([0.0, 0.0, -9.81])
_IMU_HZ = 1000
# Camera HFOV used by the footprint check (matches the K matrix in configs).
DEFAULT_HFOV_DEG = 53.0


@dataclass
class SanityCheckResult:
    name: str
    passed: bool
    message: str
    figures: List[plt.Figure] = field(default_factory=list)
    details: dict = field(default_factory=dict)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _altitude_from_meta(meta: dict) -> float:
    return float(meta.get("altitude_initial", meta.get("altitude_m", 4.0)))


# ---------------------------------------------------------------------------
# Check 1 — GT pose chaining
# ---------------------------------------------------------------------------

def check_pose_chaining(traj: TrajectoryData) -> SanityCheckResult:
    P = traj.poses_full
    N = P.shape[0]
    Rs = np.stack([_quat_to_R(P[i, 3:]) for i in range(N)])
    ts = P[:, :3]
    rel_R = np.zeros((N - 1, 3, 3))
    rel_t = np.zeros((N - 1, 3))
    for i in range(N - 1):
        rel_R[i], rel_t[i] = relative_pose(Rs[i], ts[i], Rs[i + 1], ts[i + 1])
    rebuilt_R = np.zeros_like(Rs); rebuilt_t = np.zeros_like(ts)
    rebuilt_R[0] = Rs[0]; rebuilt_t[0] = ts[0]
    for i in range(N - 1):
        rebuilt_R[i + 1] = rebuilt_R[i] @ rel_R[i]
        rebuilt_t[i + 1] = rebuilt_R[i] @ rel_t[i] + rebuilt_t[i]
    err_t = float(np.max(np.linalg.norm(rebuilt_t - ts, axis=-1)))
    err_R = float(np.max(np.linalg.norm(
        rebuilt_R.transpose(0, 2, 1) @ Rs - np.eye(3)[None], axis=(1, 2))))
    passed = (err_t < 1e-6) and (err_R < 1e-6)
    msg = f"max |Δt| err = {err_t:.2e} m, max |R err| = {err_R:.2e}"
    return SanityCheckResult(name="pose_chaining", passed=passed, message=msg,
                             details={"err_t": err_t, "err_R": err_R})


# ---------------------------------------------------------------------------
# Check 2 — IMU double-integration
# ---------------------------------------------------------------------------

def check_imu_integration(traj: TrajectoryData) -> SanityCheckResult:
    P = traj.poses_full
    imu = traj.imu
    N = P.shape[0]
    Rs = np.stack([_quat_to_R(P[i, 3:]) for i in range(N)])
    acc_body = imu[:, :3]
    acc_world = np.einsum("nij,nj->ni", Rs, acc_body) + GRAVITY
    dt = 1.0 / _IMU_HZ
    vel = np.zeros_like(acc_world)
    vel[0] = (P[1, :3] - P[0, :3]) / dt
    for i in range(N - 1):
        vel[i + 1] = vel[i] + acc_world[i] * dt
    pos = np.zeros_like(acc_world)
    pos[0] = P[0, :3]
    for i in range(N - 1):
        pos[i + 1] = pos[i] + vel[i] * dt + 0.5 * acc_world[i] * dt * dt

    drift = np.linalg.norm(pos - P[:, :3], axis=-1)
    passed = float(drift.max()) < 1.0
    msg = f"IMU integration max drift = {drift.max():.3f} m, end drift = {drift[-1]:.3f} m"

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(N) * dt, drift, 'r-')
    ax.set_xlabel("time (s)")
    ax.set_ylabel("integration drift (m)")
    ax.set_title(f"traj {traj.traj_id:04d} — IMU double-integration drift")
    fig.tight_layout()
    return SanityCheckResult(name="imu_integration", passed=passed, message=msg,
                             figures=[fig],
                             details={"max_drift_m": float(drift.max()),
                                      "end_drift_m": float(drift[-1])})


# ---------------------------------------------------------------------------
# Check 3 — Homography warping
# ---------------------------------------------------------------------------

def check_homography_warp(traj: TrajectoryData, frame_idx: int = 100,
                          border_pad: int = 20) -> SanityCheckResult:
    if cv2 is None:
        return SanityCheckResult(name="homography_warp", passed=False,
                                 message="opencv-python not installed")
    from PIL import Image as _PILImage

    K = np.asarray(traj.metadata["K"], dtype=np.float64)

    a = frame_idx * _IMU_WINDOW
    b = (frame_idx + 1) * _IMU_WINDOW
    # S5 fix (2026-04-24): plane depth is the camera's world-frame Z at frame
    # _a, not the scalar altitude_initial. Matches the dataloader change.
    d = float(traj.poses_full[a, 2])
    R_wa = _quat_to_R(traj.poses_full[a, 3:])
    t_wa = traj.poses_full[a, :3]
    R_wb = _quat_to_R(traj.poses_full[b, 3:])
    t_wb = traj.poses_full[b, :3]
    # The homography mapping pixel_a → pixel_b needs the pose of cam_a in
    # cam_b's frame (not cam_b in cam_a). Swap the relative_pose args.
    R_ba, t_ba = relative_pose(R_wb, t_wb, R_wa, t_wa)

    frame_a_path = traj.root / "frames" / f"{frame_idx:05d}.png"
    frame_b_path = traj.root / "frames" / f"{frame_idx + 1:05d}.png"
    img_a = np.asarray(_PILImage.open(frame_a_path).convert("RGB"))
    img_b = np.asarray(_PILImage.open(frame_b_path).convert("RGB"))
    # Use the Blender/OpenGL-aware helper: applies S = diag(1,-1,-1) internally
    # and uses n = (0, 0, -1) in CV frame. Verified MAE < 5 on GT frames.
    H_ab = compose_homography_blender_plane(K, R_ba, t_ba, altitude=d)
    warped = cv2.warpPerspective(img_a, H_ab, (img_a.shape[1], img_a.shape[0]))
    diff = np.abs(warped.astype(np.float32) - img_b.astype(np.float32)).mean(-1)
    inner = diff[border_pad:-border_pad, border_pad:-border_pad]
    mae = float(inner.mean())
    passed = mae < 12.0

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    axs[0].imshow(img_a); axs[0].set_title("I_t")
    axs[1].imshow(warped); axs[1].set_title("H·I_t (warped to t+1)")
    axs[2].imshow(img_b); axs[2].set_title("I_{t+1}")
    for a_ in axs: a_.axis("off")
    fig.suptitle(f"traj {traj.traj_id:04d} frame {frame_idx}: MAE(inner) = {mae:.2f}")
    fig.tight_layout()

    return SanityCheckResult(name="homography_warp", passed=passed,
                             message=f"photometric MAE (inner) = {mae:.2f}",
                             figures=[fig], details={"mae": mae})


# ---------------------------------------------------------------------------
# Check 4 — Motion-magnitude histogram (legacy, cross-trajectory)
# ---------------------------------------------------------------------------

def check_motion_histogram(trajs: Iterable[TrajectoryData],
                           median_bounds_cm: tuple[float, float] = (3.0, 8.0)
                          ) -> SanityCheckResult:
    dt_list: List[float] = []
    dR_list: List[float] = []
    for traj in trajs:
        P = traj.poses_full
        N = traj.n_frames
        for k in range(N - 1):
            a = k * _IMU_WINDOW; b = (k + 1) * _IMU_WINDOW
            R_wa = _quat_to_R(P[a, 3:])
            R_wb = _quat_to_R(P[b, 3:])
            t_wa = P[a, :3]; t_wb = P[b, :3]
            R_ab, t_ab = relative_pose(R_wa, t_wa, R_wb, t_wb)
            dt_list.append(float(np.linalg.norm(t_ab)))
            tr = float(np.trace(R_ab))
            cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
            dR_list.append(float(np.degrees(np.arccos(cos_theta))))
    dt_arr = np.array(dt_list) * 100.0
    dR_arr = np.array(dR_list)
    median_dt = float(np.median(dt_arr))
    lo, hi = median_bounds_cm
    passed = (lo <= median_dt <= hi)
    msg = (f"median |Δt| = {median_dt:.2f} cm (bounds [{lo}, {hi}]); "
           f"median |ΔR| = {float(np.median(dR_arr)):.2f} deg")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(dt_arr, bins=40, color='steelblue')
    axs[0].axvline(median_dt, color='red', label=f"median = {median_dt:.2f} cm")
    axs[0].set_xlabel("|Δt| (cm)"); axs[0].set_ylabel("count")
    axs[0].legend(); axs[0].set_title("Translation magnitude per frame-pair")
    axs[1].hist(dR_arr, bins=40, color='seagreen')
    axs[1].set_xlabel("|ΔR| (deg)"); axs[1].set_ylabel("count")
    axs[1].set_title("Rotation magnitude per frame-pair")
    fig.tight_layout()
    return SanityCheckResult(name="motion_histogram", passed=passed, message=msg,
                             figures=[fig], details={"median_dt_cm": median_dt,
                                                     "n_pairs": len(dt_arr)})


# ---------------------------------------------------------------------------
# Check 5 — Drone-dynamics thrust alignment
# ---------------------------------------------------------------------------

def check_drone_dynamics(traj: TrajectoryData,
                         g: float = 9.81,
                         tol: float = 1e-4) -> SanityCheckResult:
    attitudes_path = traj.root / "attitudes_full.npy"
    accel_path = traj.root / "accel_world.npy"
    if not attitudes_path.exists() or not accel_path.exists():
        return SanityCheckResult(
            name="drone_dynamics", passed=False,
            message=f"missing attitudes_full.npy / accel_world.npy in {traj.root}")

    rpy = np.load(attitudes_path)
    acc_world = np.load(accel_path)
    roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    bz_x = cy * sp * cr + sy * sr
    bz_y = sy * sp * cr - cy * sr
    bz_z = cp * cr
    thrust_pred = np.stack([bz_x, bz_y, bz_z], axis=-1)

    grav = np.array([0.0, 0.0, g])
    thrust_ref_raw = acc_world + grav
    mag = np.linalg.norm(thrust_ref_raw, axis=-1, keepdims=True)
    mask = (mag[:, 0] >= 0.5 * g)
    thrust_ref = thrust_ref_raw / np.maximum(mag, 1e-12)

    err = np.linalg.norm(thrust_pred - thrust_ref, axis=-1)
    err_masked = err[mask]
    max_err = float(err_masked.max()) if err_masked.size else float("nan")
    passed = bool(np.isfinite(max_err) and max_err < tol)
    msg = (f"max |R_wb·[0,0,1] - (a+[0,0,g])/||·||| = {max_err:.2e} "
           f"(tol {tol:.0e}); {int(mask.sum())}/{mask.size} samples kept")
    return SanityCheckResult(name="drone_dynamics", passed=passed, message=msg,
                             details={"max_err": max_err,
                                      "tol": tol,
                                      "n_kept": int(mask.sum()),
                                      "n_total": int(mask.size)})


# ---------------------------------------------------------------------------
# Check 6 — Motion distribution per training trajectory (Tier-1 B.5)
# ---------------------------------------------------------------------------

def check_motion_distribution(data_dir: Path,
                              train_ids: Sequence[int],
                              sanity_out: Path,
                              window: int = 10) -> SanityCheckResult:
    """|Δt_gt| distribution for each training trajectory.

    Asserts:
      < 5% of samples have |Δt_gt| < 1 cm
      > 80% of samples have |Δt_gt| ∈ [1 cm, 15 cm]
      0 samples with |Δt_gt| > 30 cm
    Writes a histogram PNG per trajectory under ``sanity_out``.
    """
    data_dir = Path(data_dir)
    sanity_out = Path(sanity_out)
    sanity_out.mkdir(parents=True, exist_ok=True)
    figs: List[plt.Figure] = []
    all_ok = True
    details: dict = {"per_traj": {}}
    msg_parts: List[str] = []

    for tid in train_ids:
        traj_dir = data_dir / f"traj_{int(tid):04d}"
        if not traj_dir.exists():
            all_ok = False
            msg_parts.append(f"traj {tid}: directory missing")
            continue
        poses = np.load(traj_dir / "poses_full.npy")
        ts = poses[:, :3]
        n = ts.shape[0]
        steps = n - window
        if steps <= 0:
            continue
        disp = ts[window:] - ts[:steps]
        mag = np.linalg.norm(disp, axis=-1)  # metres
        mag_cm = mag * 100.0

        frac_small = float((mag < 0.01).mean())
        frac_mid   = float(((mag >= 0.01) & (mag <= 0.15)).mean())
        n_over_30  = int((mag > 0.30).sum())

        traj_ok = (frac_small < 0.05) and (frac_mid > 0.80) and (n_over_30 == 0)
        details["per_traj"][str(int(tid))] = {
            "frac_lt_1cm": frac_small,
            "frac_in_1_to_15cm": frac_mid,
            "n_over_30cm": n_over_30,
            "passed": traj_ok,
        }
        msg_parts.append(
            f"traj {tid}: <1cm {frac_small*100:.1f}%, "
            f"1–15cm {frac_mid*100:.1f}%, >30cm {n_over_30}")
        all_ok = all_ok and traj_ok

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(mag_cm, bins=40, color="steelblue")
        ax.axvline(1.0, color="orange", linestyle="--", label="1 cm")
        ax.axvline(15.0, color="green",  linestyle="--", label="15 cm")
        ax.axvline(30.0, color="red",    linestyle="--", label="30 cm")
        ax.set_xlabel("|Δt_gt| over 10-sample windows (cm)")
        ax.set_ylabel("count")
        ax.set_title(f"traj {int(tid):04d} motion distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(sanity_out / f"motion_dist_traj{int(tid):04d}.png", dpi=120)
        figs.append(fig)

    msg = "; ".join(msg_parts) if msg_parts else "no trajectories found"
    return SanityCheckResult(name="motion_distribution", passed=all_ok,
                             message=msg, figures=figs, details=details)


# ---------------------------------------------------------------------------
# Check 7 — Footprint confinement
# ---------------------------------------------------------------------------

def check_footprint_confinement(data_dir: Path,
                                traj_ids: Sequence[int],
                                sanity_out: Path,
                                hfov_deg: float = DEFAULT_HFOV_DEG
                                ) -> SanityCheckResult:
    """Every pose's camera footprint must stay inside the trajectory's plane."""
    data_dir = Path(data_dir)
    sanity_out = Path(sanity_out)
    sanity_out.mkdir(parents=True, exist_ok=True)
    half_tan = float(np.tan(np.deg2rad(hfov_deg) / 2.0))
    figs: List[plt.Figure] = []
    all_ok = True
    msg_parts: List[str] = []
    details: dict = {"per_traj": {}}

    for tid in traj_ids:
        traj_dir = data_dir / f"traj_{int(tid):04d}"
        meta_path = traj_dir / "metadata.json"
        if not meta_path.exists():
            all_ok = False
            msg_parts.append(f"traj {tid}: metadata missing")
            continue
        meta = json.loads(meta_path.read_text())
        poses = np.load(traj_dir / "poses_full.npy")
        plane = meta.get("plane_dims_meters", None)
        if plane is None:
            msg_parts.append(f"traj {tid}: plane_dims_meters missing (legacy)")
            continue
        plane_x, plane_y = float(plane[0]), float(plane[1])
        eff_density = meta.get("effective_density_px_per_m")
        # Raised 40 → 100 px/m after the 2026-04-24 overhaul (stretched_per_image_v1
        # + 640×480 render). Carpets at ≥1500 px over an 18 m plane give ≥ ~83
        # px/m; a margin above the target confirms S1 HF content is preserved.
        if eff_density is not None and eff_density < 100.0:
            msg_parts.append(
                f"traj {tid}: texture will look blurry "
                f"(effective density {eff_density:.1f} px/m < 100)")

        ts = poses[:, :3]
        footprint_half = ts[:, 2] * half_tan  # altitude-dependent
        x_margin = (plane_x / 2.0) - (np.abs(ts[:, 0]) + footprint_half)
        y_margin = (plane_y / 2.0) - (np.abs(ts[:, 1]) + footprint_half)
        worst_x = float(x_margin.min())
        worst_y = float(y_margin.min())
        ok = (worst_x >= 0.0) and (worst_y >= 0.0)
        all_ok = all_ok and ok
        details["per_traj"][str(int(tid))] = {
            "plane_x": plane_x, "plane_y": plane_y,
            "worst_x_margin_m": worst_x,
            "worst_y_margin_m": worst_y,
            "passed": ok,
        }
        msg_parts.append(
            f"traj {tid}: min margin XY = ({worst_x:.2f}, {worst_y:.2f}) m")

        fig, ax = plt.subplots(figsize=(5, 5))
        rect = mpatches.Rectangle((-plane_x / 2, -plane_y / 2),
                                  plane_x, plane_y, fill=False,
                                  edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        ax.plot(ts[::50, 0], ts[::50, 1], color="blue", lw=0.7, label="trajectory")
        # Example footprint circles every N samples.
        stride = max(1, ts.shape[0] // 20)
        for i in range(0, ts.shape[0], stride):
            circ = mpatches.Circle((ts[i, 0], ts[i, 1]),
                                   radius=float(footprint_half[i]),
                                   fill=False, edgecolor="orange", alpha=0.5)
            ax.add_patch(circ)
        ax.set_xlim(-plane_x, plane_x)
        ax.set_ylim(-plane_y, plane_y)
        ax.set_aspect("equal")
        ax.set_title(f"traj {int(tid):04d} footprint confinement")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(sanity_out / f"footprint_traj{int(tid):04d}.png", dpi=120)
        figs.append(fig)

    msg = "; ".join(msg_parts) if msg_parts else "no trajectories found"
    return SanityCheckResult(name="footprint_confinement", passed=all_ok,
                             message=msg, figures=figs, details=details)


# ---------------------------------------------------------------------------
# Check 8 — Texture assignment
# ---------------------------------------------------------------------------

EXPECTED_TRAIN_TEX = {
    0: "tex_00.webp", 1: "tex_01.webp",
    2: "tex_02.webp", 3: "tex_03.webp",
}
EXPECTED_TEST_TEX = {
    # 4-train/1-test carpet split: tex_04 is the sole held-out texture,
    # shared across traj 100 and 101 (both "unseen-texture, unseen-shape").
    # Traj 102 reuses train tex_00 (the intentional seen-texture / unseen-
    # shape isolation case).
    100: "tex_04.webp", 101: "tex_04.webp",
    102: "tex_00.webp",
}
# Minitest uses the same held-out texture (tex_04) as the main test trajs.
EXPECTED_VAL_TEX = {200: "tex_04.webp"}


def check_texture_assignment(data_dir: Path,
                             project_root: Optional[Path] = None,
                             traj_ids: Optional[Sequence[int]] = None
                             ) -> SanityCheckResult:
    data_dir = Path(data_dir)
    project_root = Path(project_root) if project_root is not None else data_dir.parent
    ids = (list(traj_ids) if traj_ids is not None else
           list(EXPECTED_TRAIN_TEX) + list(EXPECTED_TEST_TEX) + list(EXPECTED_VAL_TEX))

    all_ok = True
    parts: List[str] = []
    details: dict = {"per_traj": {}}

    for tid in ids:
        traj_dir = data_dir / f"traj_{int(tid):04d}"
        meta_path = traj_dir / "metadata.json"
        if not meta_path.exists():
            parts.append(f"traj {tid}: metadata.json missing")
            all_ok = False
            continue
        meta = json.loads(meta_path.read_text())
        tex_name = meta.get("texture_name", None)
        tex_pool = meta.get("texture_pool", None)
        tex_rel  = meta.get("texture_path", None)

        # Presence.
        if tex_name is None or tex_pool is None or tex_rel is None:
            parts.append(f"traj {tid}: texture fields missing in metadata")
            all_ok = False
            continue

        # File existence.
        tex_path = (project_root / tex_rel).resolve()
        if not tex_path.exists():
            parts.append(f"traj {tid}: texture file missing: {tex_rel}")
            all_ok = False
            continue

        # Expected assignment.
        expected_name: Optional[str] = None
        expected_pool: Optional[str] = None
        if tid in EXPECTED_TRAIN_TEX:
            expected_name = EXPECTED_TRAIN_TEX[tid]
            expected_pool = "train"
        elif tid == 102:
            expected_name = EXPECTED_TEST_TEX[tid]; expected_pool = "train"
        elif tid in EXPECTED_TEST_TEX:
            expected_name = EXPECTED_TEST_TEX[tid]; expected_pool = "test"
        elif tid in EXPECTED_VAL_TEX:
            expected_name = EXPECTED_VAL_TEX[tid]; expected_pool = "test"

        ok = True
        if expected_name is not None and tex_name != expected_name:
            parts.append(f"traj {tid}: name {tex_name} != expected {expected_name}")
            ok = False
        if expected_pool is not None and tex_pool != expected_pool:
            parts.append(f"traj {tid}: pool {tex_pool} != expected {expected_pool}")
            ok = False
        all_ok = all_ok and ok
        details["per_traj"][str(int(tid))] = {
            "texture_name": tex_name, "texture_pool": tex_pool,
            "texture_path": tex_rel, "passed": ok,
        }

    msg = "; ".join(parts) if parts else "all expected textures present"
    return SanityCheckResult(name="texture_assignment", passed=all_ok,
                             message=msg, details=details)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_figures(fig_list: List[plt.Figure], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fig in enumerate(fig_list):
        fig.savefig(out_dir / f"{prefix}_{i}.png", dpi=120)
        plt.close(fig)


def dump_summary(results: List[SanityCheckResult], out_path: Path) -> None:
    blob = [{"name": r.name, "passed": r.passed, "message": r.message,
             "details": r.details} for r in results]
    out_path.write_text(json.dumps(blob, indent=2))
