"""Animated flight-path video: drone moving along a Phase 2 test trajectory.

Renders GT (black) plus one or more predicted trajectories (V/I/VI, using the
same colors as the rest of ``plot_results``) as an animated top-down flight
path with a moving marker per method, alongside a running Umeyama-aligned
position-error curve. No external ffmpeg binary is required (cv2.VideoWriter,
mp4v fourcc) — matches the Phase 1 ``make_video.py`` approach.

Usage (from ``Code/Phase 2/``):
    python -m plot_results.render_flight_video --traj_id 412 --case BEST \
        --traj_dir ../../Results/Phase2/step_s/eval/trajectories \
        --out ../../Results/Phase2/videos/phase2_best_traj412.mp4

    python -m plot_results.render_flight_video --traj_id 410 --case WORST \
        --traj_dir ../../Results/Phase2/step_s/eval/trajectories \
        --out ../../Results/Phase2/videos/phase2_worst_traj410.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import common as C

GT_LABEL = "Ground truth"
GT_PLOT_COLOR = "black"


def _fig_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_id", type=int, required=True)
    ap.add_argument("--traj_dir", required=True,
                     help="Directory holding gt_<id>.tum / <method>_<id>.tum")
    ap.add_argument("--methods", nargs="+", default=list(C.DEFAULT_METHODS),
                     help="Default: v i vi")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=450)
    ap.add_argument("--case", default="",
                     help='Free-text case label, e.g. "BEST" or "WORST"')
    ap.add_argument("--subtitle", default="",
                     help="Extra context line, e.g. speed / trajectory shape")
    ap.add_argument("--results_csv", default="",
                     help="Optional results.csv to pull ate_aligned for the legend")
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    bundles = {}
    for m in args.methods:
        b = C.load_bundle(m, args.traj_id, traj_dir)
        if b is None:
            C.warn(f"missing tum pair for method={m} traj={args.traj_id}, skipping")
            continue
        bundles[m] = b
    if not bundles:
        raise RuntimeError(f"no method trajectories found for traj_id={args.traj_id}")

    # ATE numbers for the legend, either from results.csv or computed here.
    ate: Dict[str, float] = {}
    if args.results_csv and Path(args.results_csv).exists():
        import csv as _csv
        with open(args.results_csv) as f:
            for row in _csv.DictReader(f):
                if (int(row["traj_id"]) == args.traj_id
                        and row["metric_name"] == "ate_aligned"
                        and row["method"] in bundles):
                    ate[row["method"]] = float(row["value"])
    for m, b in bundles.items():
        if m not in ate:
            ate[m] = float(np.sqrt(((b.pred_aligned_P - b.gt_P) ** 2).sum(axis=1).mean()))

    any_b = next(iter(bundles.values()))
    gt_P = any_b.gt_P
    n = len(gt_P)
    idx = np.linspace(0, n - 1, min(args.frames, n)).astype(int)

    fig, (axXY, axE) = plt.subplots(1, 2, figsize=(13, 6), dpi=110)

    header = args.case
    if args.subtitle:
        header = f"{header} — {args.subtitle}" if header else args.subtitle
    ate_str = "   ".join(f"{C.method_label(m)}: {ate[m]:.3f} m"
                          for m in args.methods if m in ate)
    title = f"Phase 2 — trajectory {args.traj_id}"
    if header:
        title += f"  [{header}]"
    fig.suptitle(f"{title}\nATE (aligned) — {ate_str}", fontsize=11)

    # Full-track context, faint.
    axXY.plot(gt_P[:, 0], gt_P[:, 1], color=GT_PLOT_COLOR, lw=2.2, alpha=0.25)
    for m, b in bundles.items():
        axXY.plot(b.pred_aligned_P[:, 0], b.pred_aligned_P[:, 1],
                   color=C.method_color(m), lw=1.2, alpha=0.20)

    (gt_line,) = axXY.plot([], [], color=GT_PLOT_COLOR, lw=2.4, label=GT_LABEL)
    (gt_head,) = axXY.plot([], [], "o", color=GT_PLOT_COLOR, markersize=9,
                            markeredgecolor="white", markeredgewidth=1.0)
    pred_lines, pred_heads = {}, {}
    for m in bundles:
        (pl,) = axXY.plot([], [], color=C.method_color(m), lw=1.6,
                           label=C.method_label(m))
        (ph,) = axXY.plot([], [], "o", color=C.method_color(m), markersize=7,
                           markeredgecolor="white", markeredgewidth=0.8)
        pred_lines[m] = pl
        pred_heads[m] = ph

    axXY.set_xlabel("X [m]"); axXY.set_ylabel("Y [m]")
    axXY.set_title("top-down flight path")
    axXY.set_aspect("equal", adjustable="datalim")
    axXY.grid(True, alpha=0.3)
    axXY.legend(loc="upper right", fontsize=9)

    cum = any_b.cum
    err_curves: Dict[str, np.ndarray] = {
        m: np.linalg.norm(b.pred_aligned_P - b.gt_P, axis=1) for m, b in bundles.items()
    }
    max_err = max(float(e.max()) for e in err_curves.values()) * 1.05
    for m, e in err_curves.items():
        axE.plot(cum, e, color=C.method_color(m), lw=0.7, alpha=0.35)
    err_lines = {}
    for m in bundles:
        (el,) = axE.plot([], [], color=C.method_color(m), lw=1.8,
                          label=f"{C.method_label(m)} (RMSE {ate[m]:.3f} m)")
        err_lines[m] = el
    axE.set_xlabel("Distance flown [m]")
    axE.set_ylabel("‖position error‖ [m]")
    axE.set_ylim(0, max(max_err, 1e-3))
    axE.set_title("aligned position error vs. distance")
    axE.grid(True, alpha=0.3)
    axE.legend(loc="upper left", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    first = _fig_to_bgr(fig)
    h, w = first.shape[:2]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    for k in idx:
        gt_line.set_data(gt_P[:k + 1, 0], gt_P[:k + 1, 1])
        gt_head.set_data([gt_P[k, 0]], [gt_P[k, 1]])
        for m, b in bundles.items():
            pred_lines[m].set_data(b.pred_aligned_P[:k + 1, 0], b.pred_aligned_P[:k + 1, 1])
            pred_heads[m].set_data([b.pred_aligned_P[k, 0]], [b.pred_aligned_P[k, 1]])
            err_lines[m].set_data(cum[:k + 1], err_curves[m][:k + 1])
        writer.write(_fig_to_bgr(fig))

    writer.release()
    plt.close(fig)
    print(f"wrote {out_path}  ({len(idx)} frames @ {args.fps} fps)")
    for m in bundles:
        print(f"  {m:3s} ATE (aligned) = {ate[m]:.4f} m")


if __name__ == "__main__":
    main()
