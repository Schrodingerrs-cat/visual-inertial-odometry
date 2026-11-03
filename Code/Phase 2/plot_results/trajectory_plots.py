"""Per-trajectory plots (Phase 2), Phase-1-style naming.

For each (method, traj_id) whose TUM files exist under
``outputs/trajectories/``, writes the following into
``Results/Phase2/<method>_<traj>_plots/``:

  fig1_trans_error.png    — x/y/z translation drift (mm) vs distance
  fig2_rot_error.png      — yaw/pitch/roll error (deg) vs distance
  fig3_rel_trans_error.png — RPE translation boxplot across 5 sub-traj lengths
  fig4_rel_yaw_error.png  — RPE yaw boxplot
  fig5_traj_side.png      — (x, z) side view, GT vs pred
  fig6_traj_top.png       — (x, y) top view, GT vs pred
  fig7_pos_norm.png       — ‖position error‖ vs time
  fig8_yaw_drift.png      — cumulative yaw drift vs time
  fig9_3d.png             — 3D overlay

Additionally, per test trajectory writes ``overlay_<tid>/`` with V+I+VI on the
same axes (top, side, 3D, per-axis error norm) and an "all-trajectories"
ground-truth overview plot under ``Results/Phase2/trajectories_overview/``.

Also computes the per-step error arrays that ``plot_distributions`` consumes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import common as C
from .common import plt, TrajBundle


# ---------------------------------------------------------------------------
# Per-bundle plots (fig1..9)
# ---------------------------------------------------------------------------

def _fig1_trans_error(b: TrajBundle, out: Path) -> None:
    trans_err_mm = (b.pred_aligned_P - b.gt_P) * 1000.0
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.plot(b.cum, trans_err_mm[:, 0], color="red", lw=1.0, label="x")
    ax.plot(b.cum, trans_err_mm[:, 1], color="green", lw=1.0, label="y")
    ax.plot(b.cum, trans_err_mm[:, 2], color="blue", lw=1.0, label="z")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Position drift [mm]")
    ax.set_title(f"{b.method} — traj {b.traj_id} — translation error")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    C.savefig(fig, out / "fig1_trans_error.png")


def _fig2_rot_error(b: TrajBundle, out: Path) -> None:
    N = len(b.gt_R)
    R_err = np.einsum("nij,njk->nik", np.transpose(b.gt_R, (0, 2, 1)),
                      b.pred_aligned_R)
    ypr = C.R_to_ypr_deg(R_err)   # (N, 3): yaw, pitch, roll
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.plot(b.cum, ypr[:, 0], color="red", lw=1.0, label="yaw")
    ax.plot(b.cum, ypr[:, 1], color="green", lw=1.0, label="pitch")
    ax.plot(b.cum, ypr[:, 2], color="blue", lw=1.0, label="roll")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Orientation error [deg]")
    ax.set_title(f"{b.method} — traj {b.traj_id} — rotation error")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    C.savefig(fig, out / "fig2_rot_error.png")


def _fig34_rpe_boxplots(b: TrajBundle, out: Path) -> None:
    total_len = float(b.cum[-1]) if len(b.cum) else 0.0
    if total_len <= 0:
        return
    fracs = [0.10, 0.20, 0.30, 0.40, 0.50]
    dists = [round(total_len * f, 2) for f in fracs]

    trans_box: List[np.ndarray] = []
    yaw_box: List[np.ndarray] = []
    for d in dists:
        tr, yr = C.relative_errors_at_distance(
            b.gt_P, b.gt_R, b.pred_aligned_P, b.pred_aligned_R,
            b.cum, target_d=d, stride=5, tol=0.1)
        trans_box.append(tr if len(tr) else np.array([0.0]))
        yaw_box.append(yr if len(yr) else np.array([0.0]))

    labels = [f"{d:.2f}" for d in dists]
    positions = np.arange(1, len(dists) + 1)

    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.boxplot(trans_box, positions=positions, widths=0.6,
               showfliers=False, patch_artist=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Distance traveled [m]")
    ax.set_ylabel("Translation error [m]")
    ax.set_title(f"{b.method} — traj {b.traj_id} — relative translation error")
    ax.grid(True, alpha=0.3)
    C.savefig(fig, out / "fig3_rel_trans_error.png")

    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.boxplot(yaw_box, positions=positions, widths=0.6,
               showfliers=False, patch_artist=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Distance traveled [m]")
    ax.set_ylabel("Yaw error [deg]")
    ax.set_title(f"{b.method} — traj {b.traj_id} — relative yaw error")
    ax.grid(True, alpha=0.3)
    C.savefig(fig, out / "fig4_rel_yaw_error.png")


def _fig5_side(b: TrajBundle, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(b.gt_P[:, 0], b.gt_P[:, 2],
            color=C.GT_COLOR, lw=C.GT_LW, label="GT")
    ax.plot(b.pred_aligned_P[:, 0], b.pred_aligned_P[:, 2],
            color=C.EST_COLOR, lw=C.EST_LW, label=C.method_label(b.method))
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.set_title(f"{b.method} — traj {b.traj_id} — side view (x, z)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    C.savefig(fig, out / "fig5_traj_side.png")


def _fig6_top(b: TrajBundle, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(b.gt_P[:, 0], b.gt_P[:, 1],
            color=C.GT_COLOR, lw=C.GT_LW, label="GT")
    ax.plot(b.pred_aligned_P[:, 0], b.pred_aligned_P[:, 1],
            color=C.EST_COLOR, lw=C.EST_LW, label=C.method_label(b.method))
    ax.scatter([b.gt_P[0, 0]], [b.gt_P[0, 1]], color="black", marker="o",
               s=30, zorder=5, label="start")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"{b.method} — traj {b.traj_id} — top view (x, y)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    C.savefig(fig, out / "fig6_traj_top.png")


def _fig7_pos_norm(b: TrajBundle, out: Path) -> None:
    err = np.linalg.norm(b.pred_aligned_P - b.gt_P, axis=1)
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.plot(b.times, err, color=C.EST_COLOR, lw=1.2, label="‖pos error‖")
    ax.fill_between(b.times, 0, err, color=C.EST_COLOR, alpha=0.15)
    ax.set_xlabel("time [s]"); ax.set_ylabel("‖error‖ [m]")
    ax.set_title(f"{b.method} — traj {b.traj_id} — position-error norm")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
    C.savefig(fig, out / "fig7_pos_norm.png")


def _fig8_yaw_drift(b: TrajBundle, out: Path) -> None:
    R_err = np.einsum("nij,njk->nik", np.transpose(b.gt_R, (0, 2, 1)),
                      b.pred_aligned_R)
    yaw_deg = C.R_to_ypr_deg(R_err)[:, 0]  # signed yaw
    # unwrap to track cumulative drift
    yaw_rad = np.unwrap(np.radians(yaw_deg))
    yaw_deg = np.degrees(yaw_rad)
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.plot(b.times, yaw_deg, color=C.EST_COLOR, lw=1.2)
    ax.axhline(0, color="red", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel("time [s]"); ax.set_ylabel("yaw drift [deg]")
    ax.set_title(f"{b.method} — traj {b.traj_id} — cumulative yaw drift")
    ax.grid(True, alpha=0.3)
    C.savefig(fig, out / "fig8_yaw_drift.png")


def _fig9_3d(b: TrajBundle, out: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(b.gt_P[:, 0], b.gt_P[:, 1], b.gt_P[:, 2],
            color=C.GT_COLOR, lw=C.GT_LW, label="GT")
    ax.plot(b.pred_aligned_P[:, 0], b.pred_aligned_P[:, 1],
            b.pred_aligned_P[:, 2],
            color=C.EST_COLOR, lw=C.EST_LW, label=C.method_label(b.method))
    ax.scatter(*b.gt_P[0], color="black", s=30, label="start")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"{b.method} — traj {b.traj_id} — 3D overlay")
    C.savefig(fig, out / "fig9_3d.png")


def plot_bundle(b: TrajBundle, results_dir: Path) -> Path:
    """Render all per-bundle figures; return the output dir."""
    out = C.ensure_dir(results_dir / f"{b.method}_{b.traj_id}_plots")
    _fig1_trans_error(b, out)
    _fig2_rot_error(b, out)
    _fig34_rpe_boxplots(b, out)
    _fig5_side(b, out)
    _fig6_top(b, out)
    _fig7_pos_norm(b, out)
    _fig8_yaw_drift(b, out)
    _fig9_3d(b, out)
    return out


# ---------------------------------------------------------------------------
# Multi-method overlay per traj_id
# ---------------------------------------------------------------------------

def _overlay_top(bundles: List[TrajBundle], traj_id: int, out: Path) -> None:
    gt = bundles[0].gt_P
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(gt[:, 0], gt[:, 1], color=C.GT_COLOR, lw=C.GT_LW, label="GT")
    for b in bundles:
        ax.plot(b.pred_aligned_P[:, 0], b.pred_aligned_P[:, 1],
                color=C.method_color(b.method), lw=1.1,
                label=C.method_label(b.method))
    ax.scatter([gt[0, 0]], [gt[0, 1]], color="black", s=30, zorder=5,
               label="start")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"traj {traj_id} — all methods (top view)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    C.savefig(fig, out / "top.png")


def _overlay_side(bundles: List[TrajBundle], traj_id: int, out: Path) -> None:
    gt = bundles[0].gt_P
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(gt[:, 0], gt[:, 2], color=C.GT_COLOR, lw=C.GT_LW, label="GT")
    for b in bundles:
        ax.plot(b.pred_aligned_P[:, 0], b.pred_aligned_P[:, 2],
                color=C.method_color(b.method), lw=1.1,
                label=C.method_label(b.method))
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.set_title(f"traj {traj_id} — all methods (side view)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    C.savefig(fig, out / "side.png")


def _overlay_3d(bundles: List[TrajBundle], traj_id: int, out: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    gt = bundles[0].gt_P
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2],
            color=C.GT_COLOR, lw=C.GT_LW, label="GT")
    for b in bundles:
        ax.plot(b.pred_aligned_P[:, 0], b.pred_aligned_P[:, 1],
                b.pred_aligned_P[:, 2], color=C.method_color(b.method),
                lw=1.1, label=C.method_label(b.method))
    ax.scatter(*gt[0], color="black", s=30, label="start")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"traj {traj_id} — all methods (3D)")
    ax.legend(loc="best", fontsize=9)
    C.savefig(fig, out / "3d.png")


def _overlay_err_norm(bundles: List[TrajBundle], traj_id: int,
                      out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for b in bundles:
        err = np.linalg.norm(b.pred_aligned_P - b.gt_P, axis=1)
        ax.plot(b.times, err, color=C.method_color(b.method), lw=1.1,
                label=C.method_label(b.method))
    ax.set_xlabel("time [s]"); ax.set_ylabel("‖pos error‖ [m]")
    ax.set_title(f"traj {traj_id} — position-error norm, all methods")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
    C.savefig(fig, out / "err_norm.png")


def plot_overlay(traj_id: int, bundles: List[TrajBundle],
                 results_dir: Path) -> None:
    out = C.ensure_dir(results_dir / f"overlay_{traj_id}")
    _overlay_top(bundles, traj_id, out)
    _overlay_side(bundles, traj_id, out)
    _overlay_3d(bundles, traj_id, out)
    _overlay_err_norm(bundles, traj_id, out)


# ---------------------------------------------------------------------------
# Global train/test GT overview (BlackBird-Fig-6 style)
# ---------------------------------------------------------------------------

def plot_train_test_gt_overview(traj_dir: Path, results_dir: Path,
                                train_ids: Iterable[int],
                                test_ids: Iterable[int]) -> None:
    """Overview plot: all train GT trajectories + all test GT trajectories.

    Two side-by-side top-down panels (train | test), one 3D panel. Uses the
    gt_<tid>.tum files that ``scripts/evaluate.py`` writes.
    """
    train_gts: Dict[int, np.ndarray] = {}
    test_gts: Dict[int, np.ndarray] = {}
    for tid in train_ids:
        t = C.load_tum(Path(traj_dir) / f"gt_{tid}.tum")
        if t is not None:
            train_gts[tid] = t[1]
    for tid in test_ids:
        t = C.load_tum(Path(traj_dir) / f"gt_{tid}.tum")
        if t is not None:
            test_gts[tid] = t[1]

    if not train_gts and not test_gts:
        C.warn("no GT TUM files found — skipping train/test overview")
        return
    out = C.ensure_dir(results_dir / "trajectories_overview")

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.0))
    for k, (tid, P) in enumerate(train_gts.items()):
        axs[0].plot(P[:, 0], P[:, 1], lw=1.3, label=f"traj {tid}")
    axs[0].set_title("Train trajectories (GT)")
    axs[0].set_xlabel("x [m]"); axs[0].set_ylabel("y [m]")
    axs[0].grid(True, alpha=0.3)
    axs[0].set_aspect("equal", adjustable="datalim")
    if train_gts:
        axs[0].legend(loc="best", fontsize=8)

    for k, (tid, P) in enumerate(test_gts.items()):
        name = C.TEST_NAMES.get(tid, f"traj {tid}")
        axs[1].plot(P[:, 0], P[:, 1], lw=1.3, label=f"{tid} ({name})")
    axs[1].set_title("Test trajectories (GT)")
    axs[1].set_xlabel("x [m]"); axs[1].set_ylabel("y [m]")
    axs[1].grid(True, alpha=0.3)
    axs[1].set_aspect("equal", adjustable="datalim")
    if test_gts:
        axs[1].legend(loc="best", fontsize=8)

    C.savefig(fig, out / "train_vs_test_top.png")

    # 3D side panel combining both
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    for tid, P in train_gts.items():
        ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=1.0, alpha=0.7,
                label=f"train {tid}")
    for tid, P in test_gts.items():
        name = C.TEST_NAMES.get(tid, f"{tid}")
        ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=1.5, label=f"test {name}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Train + test GT trajectories (3D)")
    ax.legend(loc="best", fontsize=8)
    C.savefig(fig, out / "train_vs_test_3d.png")

    C.info(f"wrote train/test GT overview → {out}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def plot_all_trajectories(traj_dir: Path, results_dir: Path,
                          methods: Iterable[str] = C.ALL_METHODS,
                          traj_ids: Iterable[int] = C.TEST_IDS
                          ) -> Dict[Tuple[str, int], TrajBundle]:
    """Render per-bundle + overlay plots. Returns loaded bundles so distribution
    plots can reuse them without re-parsing TUM files."""
    bundles: Dict[Tuple[str, int], TrajBundle] = {}
    pairs = C.discover_pairs(Path(traj_dir), methods, traj_ids)
    if not pairs:
        C.warn(f"no (method, traj) TUM pairs found under {traj_dir}; "
               "trajectory plots skipped")
        return bundles
    for method, tid in pairs:
        b = C.load_bundle(method, tid, Path(traj_dir))
        if b is None:
            continue
        bundles[(method, tid)] = b
        plot_bundle(b, results_dir)
        C.info(f"wrote per-traj plots for {method} / traj {tid}")

    # overlays per traj_id
    tid_to_bundles: Dict[int, List[TrajBundle]] = {}
    for (m, t), b in bundles.items():
        tid_to_bundles.setdefault(t, []).append(b)
    for tid, blist in tid_to_bundles.items():
        if len(blist) >= 2:
            plot_overlay(tid, blist, results_dir)
            C.info(f"wrote method-overlay plots for traj {tid} "
                   f"({len(blist)} methods)")
    return bundles
