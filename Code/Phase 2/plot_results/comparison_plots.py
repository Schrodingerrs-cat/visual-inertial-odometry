"""Cross-method comparison bar charts (Phase 2).

Reads ``outputs/results.csv`` (long-form: method, traj_id, metric_name, value)
and writes to ``Results/Phase2/comparison/``:

  ate_aligned_bar.png    — aligned ATE RMSE, grouped by traj_id × method
  ate_endpoint_bar.png   — endpoint drift
  seg2s_bar.png          — per-2s drift median with p95 errorbar
  fallback_bar.png       — IMU fallback fraction (VI-family only)
  summary_table.png      — all metrics as a heatmap-style table
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from . import common as C
from .common import plt


def _filter(results: Dict[Tuple[str, int], Dict[str, float]],
            metric: str, methods: Sequence[str], traj_ids: Sequence[int]
            ) -> np.ndarray:
    out = np.full((len(methods), len(traj_ids)), np.nan)
    for i, m in enumerate(methods):
        for j, t in enumerate(traj_ids):
            v = results.get((m, t), {}).get(metric)
            if v is not None:
                out[i, j] = float(v)
    return out


def _grouped_bar(matrix: np.ndarray, methods: Sequence[str],
                 traj_ids: Sequence[int], ylabel: str, title: str,
                 path: Path, yerr: np.ndarray = None) -> None:
    if np.all(np.isnan(matrix)):
        return
    n_methods = len(methods)
    n_traj = len(traj_ids)
    fig_w = max(6.5, 1.4 * n_traj * n_methods / 3)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))
    width = 0.8 / max(1, n_methods)
    xs = np.arange(n_traj)
    for i, m in enumerate(methods):
        vals = matrix[i]
        mask = np.isfinite(vals)
        if not np.any(mask):
            continue
        offs = xs + (i - (n_methods - 1) / 2) * width
        color = C.method_color(m)
        if yerr is not None:
            ax.bar(offs[mask], vals[mask], width=width * 0.95,
                   yerr=yerr[i, mask], capsize=3, color=color,
                   label=C.method_label(m), edgecolor="black", lw=0.3)
        else:
            ax.bar(offs[mask], vals[mask], width=width * 0.95, color=color,
                   label=C.method_label(m), edgecolor="black", lw=0.3)

    labels = [f"{t}\n({C.TEST_NAMES.get(t, '')})" if t in C.TEST_NAMES
              else str(t) for t in traj_ids]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=min(3, n_methods))
    C.savefig(fig, path)


def _summary_table(results: Dict[Tuple[str, int], Dict[str, float]],
                   methods: Sequence[str], traj_ids: Sequence[int],
                   metrics: Sequence[str], path: Path) -> None:
    """Render a text-based summary table as a figure."""
    if not results:
        return
    n_rows = len(methods) * len(traj_ids)
    n_cols = len(metrics) + 2  # method, traj
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * n_cols), 0.28 * n_rows + 0.8))
    ax.axis("off")

    header = ["method", "traj"] + list(metrics)
    cells: List[List[str]] = []
    for m in methods:
        for t in traj_ids:
            row = [C.method_label(m), str(t)]
            for metric in metrics:
                v = results.get((m, t), {}).get(metric)
                row.append("—" if v is None else f"{float(v):.3g}")
            cells.append(row)
    if not cells:
        plt.close(fig)
        return

    tbl = ax.table(cellText=cells, colLabels=header, loc="center",
                   cellLoc="center", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.3)
    for k, _ in enumerate(header):
        cell = tbl[(0, k)]
        cell.set_facecolor("#dddddd")
        cell.set_text_props(weight="bold")
    ax.set_title("Phase 2 — per-method / per-trajectory summary", fontsize=10)
    C.savefig(fig, path)


def plot_all_comparisons(results_csv: Path, results_dir: Path,
                         methods: Sequence[str] = C.ALL_METHODS,
                         traj_ids: Sequence[int] = C.TEST_IDS) -> None:
    results = C.load_results_csv(Path(results_csv))
    if not results:
        C.warn(f"no rows in {results_csv}; comparison plots skipped")
        return

    present_methods = [m for m in methods
                       if any((m, t) in results for t in traj_ids)]
    if not present_methods:
        C.warn("results.csv has rows but none match known methods; "
               "comparison skipped")
        return

    out = C.ensure_dir(results_dir / "comparison")

    ate = _filter(results, "ate_aligned", present_methods, traj_ids)
    _grouped_bar(ate, present_methods, traj_ids,
                 ylabel="ATE RMSE (aligned) [m]",
                 title="Aligned ATE — methods × test trajectories",
                 path=out / "ate_aligned_bar.png")

    endpoint = _filter(results, "ate_endpoint", present_methods, traj_ids)
    _grouped_bar(endpoint, present_methods, traj_ids,
                 ylabel="Endpoint drift [m]",
                 title="Endpoint drift — methods × test trajectories",
                 path=out / "ate_endpoint_bar.png")

    med = _filter(results, "seg2s_median", present_methods, traj_ids)
    p95 = _filter(results, "seg2s_p95", present_methods, traj_ids)
    # yerr represents the band from median up to p95 (lower = 0 so errorbars
    # point up-only).
    yerr_up = np.where(np.isfinite(p95) & np.isfinite(med),
                       np.clip(p95 - med, 0, None), 0.0)
    yerr_lo = np.zeros_like(yerr_up)
    yerr = np.stack([yerr_lo, yerr_up], axis=0)  # (2, methods, traj) → unused
    # matplotlib expects yerr shape (2, N) per bar; simplest: plot median only,
    # add p95 as a separate marker.
    _grouped_bar(med, present_methods, traj_ids,
                 ylabel="2-sec drift median [m]",
                 title="Segment-2s drift median — methods × test trajectories",
                 path=out / "seg2s_median_bar.png")
    _grouped_bar(p95, present_methods, traj_ids,
                 ylabel="2-sec drift p95 [m]",
                 title="Segment-2s drift p95 — methods × test trajectories",
                 path=out / "seg2s_p95_bar.png")

    # Fallback fraction — only VI-family rows carry this; show just those.
    vi_methods = [m for m in present_methods if m.startswith("vi")]
    if vi_methods:
        fb = _filter(results, "fallback_fraction", vi_methods, traj_ids)
        if np.any(np.isfinite(fb)):
            _grouped_bar(fb, vi_methods, traj_ids,
                         ylabel="IMU-fallback fraction",
                         title="Fallback fraction (VI only)",
                         path=out / "fallback_bar.png")

    _summary_table(results, present_methods, traj_ids,
                   metrics=("ate_aligned", "ate_endpoint",
                            "seg2s_median", "seg2s_p95",
                            "fallback_fraction"),
                   path=out / "summary_table.png")

    C.info(f"wrote comparison plots → {out}")
