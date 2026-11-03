"""IMU diagnostic plotter (Step S2).

Walks `data/raw_g/traj_*/` and writes per-trajectory IMU diagnostic figures
(accel / gyro / attitude / position panels + first-1s zoom) plus a summary
comparison figure. Output layout matches the original `Results/Phase2/imu_diag/`
so the post-fix plots can be diffed directly against the pre-fix ones.

Labeling: `{split}_{family_short}_{speed}mps` per the Step S2 spec — split
and family are read from each trajectory's metadata.json; speed comes from
`target_peak_speed` if present, else falls back to the empirical peak |v_xy|
computed from poses_full (annotated as `auto…mps` in that case).

Usage:
    python3 scripts/plot_imu_diag.py
    python3 scripts/plot_imu_diag.py --data_dir data/raw_g \\
        --out_dir Results/Phase2/imu_diag_postfix
    python3 scripts/plot_imu_diag.py --trajs traj_0303 traj_0307
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parents[1]                 # Code/Phase 2/
REPO_ROOT = HERE.parents[1]
DEFAULT_DATA_DIR = HERE / "data" / "raw_g"
DEFAULT_OUT_DIR = REPO_ROOT / "Results" / "Phase2" / "imu_diag_postfix"

# Short-name map for filename generation (avoids underscores per Step S2 spec).
_FAMILY_SHORT = {
    "simple_figure8": "figure8",
    "simple_circle":  "circle",
    "simple_ellipse": "ellipse",
    "simple_diamond": "diamond",
    "simple_zigzag":  "zigzag",
    "simple_triangle": "triangle",
    "simple_sinusoid_30s": "sinusoid30s",
}


def _label_for_traj(traj_dir: Path) -> Tuple[str, str, str, float, bool]:
    """Return (label, split, family_short, speed_mps, speed_is_target).

    `label` = "{split}_{family_short}_{speed:.1f}mps"; `speed_is_target` is
    True when the speed comes from metadata's target_peak_speed (i.e. the
    Step S2 regenerated data), False when computed empirically (legacy data
    that predates target_peak_speed plumbing — annotated as autoXmps).
    """
    md = json.loads((traj_dir / "metadata.json").read_text())
    fam = str(md.get("family", "unknown"))
    fam_short = _FAMILY_SHORT.get(fam, fam.replace("_", ""))
    split = str(md.get("split", "?"))
    tgt = md.get("target_peak_speed", None)
    if tgt is None:
        # Legacy entry — empirical peak from poses_full. np.gradient = 2nd-
        # order central diff with one-sided endpoints; good enough for a
        # diagnostic label.
        pos = np.load(traj_dir / "poses_full.npy")[:, :3]
        fps = float(md.get("fps_imu", 1000))
        v = np.gradient(pos, 1.0 / fps, axis=0)
        speed = float(np.max(np.linalg.norm(v[:, :2], axis=-1)))
        speed_str = f"auto{speed:.1f}"
        is_target = False
    else:
        speed = float(tgt)
        speed_str = f"{speed:.1f}"
        is_target = True
    label = f"{split}_{fam_short}_{speed_str}mps"
    return label, split, fam_short, speed, is_target


def _plot_full_panel(traj_dir: Path, out_path: Path, label: str,
                     md: dict, imu: np.ndarray, pos: np.ndarray,
                     att: np.ndarray) -> None:
    fps = float(md.get("fps_imu", 1000))
    dt = 1.0 / fps
    t = np.arange(imu.shape[0]) * dt
    sigma_acc = md.get("noise_params", {}).get("sigma_acc", "?")
    sigma_gyro = md.get("noise_params", {}).get("sigma_gyro", "?")

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axs[0].set_title(f"{label}  family={md.get('family','?')}  "
                     f"seed={md.get('seed','?')}  "
                     f"sigma_acc={sigma_acc}  sigma_gyro={sigma_gyro}")
    axs[0].plot(t, imu[:, 0], 'r', lw=0.6, label='ax (body)')
    axs[0].plot(t, imu[:, 1], 'g', lw=0.6, label='ay (body)')
    axs[0].plot(t, imu[:, 2], 'b', lw=0.6, label='az (body)')
    axs[0].axhline(9.81, color='k', ls=':', lw=0.5, label='g')
    axs[0].set_ylabel('accel body [m/s^2]')
    axs[0].legend(loc='upper right', fontsize=8); axs[0].grid(alpha=0.3)

    gmax = float(np.max(np.abs(imu[:, 3:])))
    axs[1].plot(t, imu[:, 3], 'r', lw=0.6, label='gx (body)')
    axs[1].plot(t, imu[:, 4], 'g', lw=0.6, label='gy (body)')
    axs[1].plot(t, imu[:, 5], 'b', lw=0.6, label='gz (body)')
    axs[1].set_ylabel('gyro body [rad/s]')
    axs[1].legend(loc='upper right', fontsize=8); axs[1].grid(alpha=0.3)
    axs[1].set_title(f'gyro range +/- {gmax:.4f} rad/s '
                     f'({np.rad2deg(gmax):.2f} deg/s)', fontsize=9)

    axs[2].plot(t, np.rad2deg(att[:, 0]), 'r', lw=0.6, label='roll')
    axs[2].plot(t, np.rad2deg(att[:, 1]), 'g', lw=0.6, label='pitch')
    axs[2].plot(t, np.rad2deg(att[:, 2]), 'b', lw=0.6, label='yaw')
    axs[2].set_ylabel('attitude [deg]')
    axs[2].legend(loc='upper right', fontsize=8); axs[2].grid(alpha=0.3)

    axs[3].plot(t, pos[:, 0], 'r', lw=0.6, label='px')
    axs[3].plot(t, pos[:, 1], 'g', lw=0.6, label='py')
    axs[3].plot(t, pos[:, 2], 'b', lw=0.6, label='pz')
    axs[3].set_xlabel('time [s]'); axs[3].set_ylabel('pos [m]')
    axs[3].legend(loc='upper right', fontsize=8); axs[3].grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_first_second(traj_dir: Path, out_path: Path, label: str,
                       md: dict, imu: np.ndarray) -> None:
    fps = float(md.get("fps_imu", 1000))
    dt = 1.0 / fps
    t = np.arange(imu.shape[0]) * dt
    mask = t < 1.0
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(t[mask], imu[mask, 0], 'r', lw=0.8, label='ax')
    axs[0].plot(t[mask], imu[mask, 1], 'g', lw=0.8, label='ay')
    axs[0].plot(t[mask], imu[mask, 2] - 9.81, 'b', lw=0.8, label='az - g')
    axs[0].axhline(0, color='k', ls=':', lw=0.5)
    axs[0].set_title(f'{label} - first 1.0s (gravity-removed for az)')
    axs[0].set_ylabel('accel [m/s^2]'); axs[0].legend(); axs[0].grid(alpha=0.3)
    axs[1].plot(t[mask], imu[mask, 3], 'r', lw=0.8, label='gx')
    axs[1].plot(t[mask], imu[mask, 4], 'g', lw=0.8, label='gy')
    axs[1].plot(t[mask], imu[mask, 5], 'b', lw=0.8, label='gz')
    axs[1].set_xlabel('time [s]'); axs[1].set_ylabel('gyro [rad/s]')
    axs[1].legend(); axs[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_summary(per_traj: List[dict], out_path: Path) -> None:
    """4-column figure-8 vs ellipse comparison (matches the original layout).

    Picks up to 2 figure8 trajectories and 2 ellipse trajectories from the
    per_traj list. If fewer are present (e.g. user passed --trajs to filter),
    the figure shrinks to whatever's available.
    """
    fig8 = [d for d in per_traj if d["family_short"] == "figure8"][:2]
    ell  = [d for d in per_traj if d["family_short"] == "ellipse"][:2]
    cases = fig8 + ell
    if not cases:
        return
    fig, axs = plt.subplots(2, len(cases), figsize=(5 * len(cases), 8),
                            squeeze=False)
    fig.suptitle('IMU sanity: figure8 vs ellipse', fontsize=14)
    for col, d in enumerate(cases):
        imu = d["imu"]; t = d["t"]
        ax = axs[0, col]
        ax.plot(t, imu[:, 3], 'r', lw=0.5, label='gx')
        ax.plot(t, imu[:, 4], 'g', lw=0.5, label='gy')
        ax.plot(t, imu[:, 5], 'b', lw=0.5, label='gz')
        gmax = np.max(np.abs(imu[:, 3:]))
        ax.set_title(f"{d['label']}\n"
                     f"full seq, gyro range +-{gmax:.3f} rad/s")
        ax.set_xlabel('t [s]'); ax.set_ylabel('rad/s')
        ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)

        ax = axs[1, col]
        n = 50
        ax.plot(t[:n], imu[:n, 3], 'r-o', ms=3, lw=0.8, label='gx')
        ax.plot(t[:n], imu[:n, 4], 'g-o', ms=3, lw=0.8, label='gy')
        ax.plot(t[:n], imu[:n, 5], 'b-o', ms=3, lw=0.8, label='gz')
        ax.set_title(f"first 50 samples (50 ms) - gy[0]={imu[0,4]:+.3f}")
        ax.set_xlabel('t [s]'); ax.set_ylabel('rad/s')
        ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_imu_diag(data_dir: Path, out_dir: Path,
                  trajs: Optional[List[str]] = None) -> Dict[str, dict]:
    """Run the diagnostic plotter; returns a per-trajectory result dict."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if trajs:
        traj_dirs = [data_dir / t for t in trajs]
    else:
        traj_dirs = sorted(p for p in data_dir.iterdir()
                           if p.is_dir() and p.name.startswith("traj_"))
    if not traj_dirs:
        print(f"[plot_imu_diag] no traj_* directories under {data_dir}")
        return {}

    per_traj: List[dict] = []
    print(f"[plot_imu_diag] {len(traj_dirs)} trajectory(ies) -> {out_dir}")
    print(f"{'TRAJ_DIR':<14} {'LABEL':<32} {'gyro_max':>9} {'gyro_med':>9} "
          f"{'spike?':>7}")
    for td in traj_dirs:
        if not (td / "imu.npy").exists():
            print(f"  skip {td.name}: no imu.npy")
            continue
        label, split, fam_short, speed, is_target = _label_for_traj(td)
        md = json.loads((td / "metadata.json").read_text())
        imu = np.load(td / "imu.npy")
        pos = np.load(td / "poses_full.npy")[:, :3]
        att_path = td / "attitudes_full.npy"
        att = (np.load(att_path) if att_path.exists()
               else np.zeros((imu.shape[0], 3)))
        fps = float(md.get("fps_imu", 1000))
        t_axis = np.arange(imu.shape[0]) / fps

        _plot_full_panel(td, out_dir / f"{label}.png", label, md, imu, pos, att)
        _plot_first_second(td, out_dir / f"{label}_first1s.png", label, md, imu)

        gnorm = np.linalg.norm(imu[:, 3:], axis=1)
        gmax = float(gnorm.max()); gmed = float(np.median(gnorm))
        edge = np.concatenate([gnorm[:5], gnorm[-5:]])
        spike = bool((edge > 10.0 * max(gmed, 1e-6)).any())
        print(f"  {td.name:<14} {label:<32} {gmax:>9.4f} {gmed:>9.5f} "
              f"{('YES' if spike else 'no'):>7}")

        per_traj.append({
            "traj_dir": td.name, "label": label, "split": split,
            "family_short": fam_short, "speed": speed,
            "speed_is_target": is_target,
            "gyro_max": gmax, "gyro_median": gmed, "boundary_spike": spike,
            "imu": imu, "t": t_axis,
        })

    _plot_summary(per_traj, out_dir / "SUMMARY_figure8_vs_ellipse.png")
    print(f"[plot_imu_diag] wrote {len(per_traj)} traj plots + summary "
          f"to {out_dir}")
    # Drop the heavy IMU arrays from the returned dict so downstream callers
    # (orchestrator) don't carry them around.
    summary: Dict[str, dict] = {}
    for d in per_traj:
        s = {k: v for k, v in d.items() if k not in ("imu", "t")}
        summary[d["traj_dir"]] = s
    return summary


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--trajs", nargs="*", default=None,
                   help="Optional traj subdir names (e.g. traj_0303). "
                        "Default: all traj_* under --data_dir.")
    return p.parse_args()


def main() -> int:
    args = _parse()
    plot_imu_diag(args.data_dir, args.out_dir, trajs=args.trajs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
