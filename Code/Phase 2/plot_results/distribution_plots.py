"""Per-step error distributions (histograms + CDFs), Phase 2.

For every loaded bundle, writes four PNGs into
``Results/Phase2/<method>_<traj>_plots/``:

  hist_trans.png   — per-step translation-error histogram (mm)
  hist_rot.png     — per-step rotation-error histogram (deg)
  hist_yaw.png     — per-step yaw-error histogram (deg)
  cdf_errors.png   — CDF panel: trans / rot / yaw on a 1x3 grid

Also produces a cross-method comparison CDF under
``Results/Phase2/distributions/cdf_<metric>_traj<tid>.png`` when >1 method
exists for a traj_id.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import common as C
from .common import plt, TrajBundle


def _hist(vals: np.ndarray, xlabel: str, title: str, path: Path,
          color: str = C.EST_COLOR) -> None:
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.hist(vals, bins=40, color=color, alpha=0.75, edgecolor="black",
            linewidth=0.4)
    med = float(np.median(vals))
    p95 = float(np.quantile(vals, 0.95))
    ax.axvline(med, color="red", lw=1.2, ls="--", label=f"median = {med:.3g}")
    ax.axvline(p95, color="orange", lw=1.2, ls="--", label=f"p95 = {p95:.3g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    C.savefig(fig, path)


def _cdf_panel(stats: Dict[str, np.ndarray], title: str, path: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    specs = [
        ("trans", "translation error [m]", "Trans"),
        ("rot", "rotation error [deg]", "Rot"),
        ("yaw", "yaw error [deg]", "Yaw"),
    ]
    for ax, (key, xlab, sub) in zip(axs, specs):
        vals = stats.get(key)
        if vals is None or vals.size == 0:
            ax.axis("off")
            continue
        xs = np.sort(vals)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.plot(xs, ys, color=C.EST_COLOR, lw=1.4)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.axhline(0.95, color="gray", lw=0.8, ls=":")
        ax.set_xlabel(xlab); ax.set_ylabel("CDF")
        ax.set_title(sub)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=10)
    C.savefig(fig, path)


def plot_bundle_distributions(b: TrajBundle, results_dir: Path) -> None:
    stats = C.per_step_errors(b.gt_P, b.pred_aligned_P, b.gt_R, b.pred_aligned_R)
    out = C.ensure_dir(results_dir / f"{b.method}_{b.traj_id}_plots")

    _hist(stats["trans"] * 1000.0,
          xlabel="per-step translation error [mm]",
          title=f"{b.method} — traj {b.traj_id} — trans step error",
          path=out / "hist_trans.png")
    _hist(stats["rot"],
          xlabel="per-step rotation error [deg]",
          title=f"{b.method} — traj {b.traj_id} — rot step error",
          path=out / "hist_rot.png")
    _hist(stats["yaw"],
          xlabel="per-step yaw error [deg]",
          title=f"{b.method} — traj {b.traj_id} — yaw step error",
          path=out / "hist_yaw.png")

    _cdf_panel(stats,
               title=f"{b.method} — traj {b.traj_id} — per-step error CDFs",
               path=out / "cdf_errors.png")


def plot_cross_method_cdfs(bundles: Dict[Tuple[str, int], TrajBundle],
                           results_dir: Path) -> None:
    out = C.ensure_dir(results_dir / "distributions")
    tid_to_bundles: Dict[int, List[TrajBundle]] = {}
    for (_m, t), b in bundles.items():
        tid_to_bundles.setdefault(t, []).append(b)

    for tid, blist in tid_to_bundles.items():
        if len(blist) < 2:
            continue
        per_method_stats: Dict[str, Dict[str, np.ndarray]] = {
            b.method: C.per_step_errors(b.gt_P, b.pred_aligned_P,
                                        b.gt_R, b.pred_aligned_R)
            for b in blist
        }
        for metric, xlab, scale in (
            ("trans", "per-step translation error [m]", 1.0),
            ("rot",   "per-step rotation error [deg]",  1.0),
            ("yaw",   "per-step yaw error [deg]",       1.0),
        ):
            fig, ax = plt.subplots(figsize=(6.5, 3.4))
            any_plotted = False
            for m, stats in per_method_stats.items():
                vals = stats.get(metric)
                if vals is None or vals.size == 0:
                    continue
                xs = np.sort(vals * scale)
                ys = np.arange(1, xs.size + 1) / xs.size
                ax.plot(xs, ys, color=C.method_color(m), lw=1.3,
                        label=C.method_label(m))
                any_plotted = True
            if not any_plotted:
                plt.close(fig)
                continue
            ax.axhline(0.5, color="gray", lw=0.8, ls=":")
            ax.axhline(0.95, color="gray", lw=0.8, ls=":")
            ax.set_xlabel(xlab); ax.set_ylabel("CDF")
            ax.set_title(f"traj {tid} — cross-method CDF ({metric})")
            ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
            C.savefig(fig, out / f"cdf_{metric}_traj{tid}.png")
        C.info(f"wrote cross-method CDFs for traj {tid}")


def plot_all_distributions(bundles: Dict[Tuple[str, int], TrajBundle],
                           results_dir: Path) -> None:
    if not bundles:
        return
    for b in bundles.values():
        plot_bundle_distributions(b, results_dir)
    plot_cross_method_cdfs(bundles, results_dir)
