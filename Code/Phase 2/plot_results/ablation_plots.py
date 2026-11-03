"""Ablation / study plots (Phase 2).

Reads CSVs from ``outputs/studies/``:

  motion_magnitude.csv    columns: mode, traj_id, ate_m
  ablation_long.csv       columns: method, traj_id, metric_name, value
  ablation_ate.csv        pivot:   method, traj_100, traj_101, traj_102, mean
  capacity.csv            columns: capacity, traj_id, ate_m
  capacity_ate.csv        pivot:   capacity, traj_100, traj_101, traj_102, mean

Writes to ``Results/Phase2/ablations/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from . import common as C
from .common import plt


def _pivot_mode_traj(rows: List[Dict[str, str]], key_col: str,
                     traj_col: str = "traj_id",
                     val_col: str = "ate_m"
                     ) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for r in rows:
        try:
            out.setdefault(r[key_col], {})[int(r[traj_col])] = float(r[val_col])
        except (KeyError, ValueError):
            continue
    return out


def _bar_grouped_by_traj(table: Dict[str, Dict[int, float]],
                         traj_ids: Sequence[int], title: str, ylabel: str,
                         path: Path) -> None:
    if not table:
        return
    keys = list(table.keys())
    fig_w = max(7, 1.3 * len(traj_ids) * len(keys) / 4)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))
    width = 0.8 / max(1, len(keys))
    xs = np.arange(len(traj_ids))
    for i, k in enumerate(keys):
        offs = xs + (i - (len(keys) - 1) / 2) * width
        vals = [table[k].get(t, np.nan) for t in traj_ids]
        mask = np.isfinite(vals)
        ax.bar(np.asarray(offs)[mask], np.asarray(vals)[mask],
               width=width * 0.95, label=k, edgecolor="black", lw=0.3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t}\n({C.TEST_NAMES.get(t, '')})"
                        for t in traj_ids], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=min(3, len(keys)))
    C.savefig(fig, path)


def _bar_mean_per_key(table: Dict[str, Dict[int, float]],
                      title: str, ylabel: str, path: Path) -> None:
    if not table:
        return
    keys = list(table.keys())
    means = []
    stds = []
    for k in keys:
        vals = np.array(list(table[k].values()), dtype=float)
        if vals.size == 0:
            means.append(np.nan)
            stds.append(0.0)
        else:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(keys)), 3.4))
    xs = np.arange(len(keys))
    ax.bar(xs, means, yerr=stds, capsize=4, color="tab:blue",
           edgecolor="black", lw=0.3, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    C.savefig(fig, path)


def plot_motion_magnitude(studies_dir: Path, out_dir: Path,
                          traj_ids: Sequence[int] = C.TEST_IDS) -> None:
    rows = C.load_csv_table(Path(studies_dir) / "motion_magnitude.csv")
    if not rows:
        return
    table = _pivot_mode_traj(rows, key_col="mode")
    _bar_grouped_by_traj(
        table, traj_ids,
        title="Motion-magnitude sampling study — test ATE per mode",
        ylabel="ATE RMSE (m)",
        path=out_dir / "motion_magnitude.png")
    _bar_mean_per_key(
        table,
        title="Motion-magnitude sampling — mean test ATE",
        ylabel="ATE RMSE (m), mean over trajs",
        path=out_dir / "motion_magnitude_mean.png")
    C.info("wrote motion-magnitude study plots")


def plot_ablation(studies_dir: Path, out_dir: Path,
                  traj_ids: Sequence[int] = C.TEST_IDS) -> None:
    long_csv = Path(studies_dir) / "ablation_long.csv"
    rows = C.load_csv_table(long_csv)
    if not rows:
        # fall back to the pivot table
        pivot = C.load_csv_table(Path(studies_dir) / "ablation_ate.csv")
        if not pivot:
            return
        table: Dict[str, Dict[int, float]] = {}
        for r in pivot:
            k = r.get("method") or r.get("variant") or list(r.values())[0]
            for tid in traj_ids:
                col = f"traj_{tid}"
                if col in r:
                    try:
                        table.setdefault(k, {})[tid] = float(r[col])
                    except ValueError:
                        continue
        _bar_grouped_by_traj(
            table, traj_ids,
            title="VI architecture ablation — ATE per trajectory",
            ylabel="ATE RMSE (m)",
            path=out_dir / "architecture_ate.png")
        C.info("wrote architecture ablation (from pivot CSV)")
        return

    # Long form: one plot per metric
    metrics = sorted({r["metric_name"] for r in rows})
    for metric in metrics:
        table: Dict[str, Dict[int, float]] = {}
        for r in rows:
            if r.get("metric_name") != metric:
                continue
            try:
                table.setdefault(r["method"], {})[int(r["traj_id"])] = \
                    float(r["value"])
            except ValueError:
                continue
        _bar_grouped_by_traj(
            table, traj_ids,
            title=f"VI architecture ablation — {metric}",
            ylabel=metric,
            path=out_dir / f"architecture_{metric}.png")
    C.info(f"wrote architecture ablation across {len(metrics)} metrics")


def plot_capacity(studies_dir: Path, out_dir: Path,
                  traj_ids: Sequence[int] = C.TEST_IDS) -> None:
    rows = C.load_csv_table(Path(studies_dir) / "capacity.csv")
    if not rows:
        return
    table = _pivot_mode_traj(rows, key_col="capacity")
    _bar_grouped_by_traj(
        table, traj_ids,
        title="Capacity ablation — test ATE per capacity",
        ylabel="ATE RMSE (m)",
        path=out_dir / "capacity_ate.png")
    _bar_mean_per_key(
        table,
        title="Capacity ablation — mean test ATE",
        ylabel="ATE RMSE (m), mean over trajs",
        path=out_dir / "capacity_ate_mean.png")
    C.info("wrote capacity ablation plots")


def plot_all_ablations(studies_dir: Path, results_dir: Path,
                       traj_ids: Sequence[int] = C.TEST_IDS) -> None:
    studies = Path(studies_dir)
    if not studies.exists():
        C.warn(f"no studies dir at {studies}; ablation plots skipped")
        return
    out = C.ensure_dir(results_dir / "ablations")
    plot_motion_magnitude(studies, out, traj_ids)
    plot_ablation(studies, out, traj_ids)
    plot_capacity(studies, out, traj_ids)
