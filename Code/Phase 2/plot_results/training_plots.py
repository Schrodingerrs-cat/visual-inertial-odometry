"""Training-curve plots (Phase 2).

Reads ``outputs/<net>/train.log`` (tab-separated, format pinned in
``scripts/_train_common.TextLogger``) and ``outputs/{v,i,vi}_multiseed_summary.csv``
if present, and writes to ``Results/Phase2/training/``.

Produces:
  * ``<net>_loss.png``            — train vs val loss over epochs
  * ``<net>_val_ate.png``         — val ATE on traj 0 & minitest over epochs
  * ``<net>_lr.png``              — learning-rate schedule
  * ``combined_loss.png``         — all three networks' train vs val loss on one axis
  * ``combined_val_ate.png``      — all three networks' minitest val ATE
  * ``multiseed_<net>.png``       — mean ± std bands across seeds (if CSV present)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from . import common as C
from .common import plt


def _plot_one_net_loss(log: Dict[str, np.ndarray], name: str,
                       out_dir: Path) -> None:
    ep = log["epoch"]
    train = log.get("train_loss")
    val = log.get("val_loss")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if train is not None:
        ax.plot(ep, train, color=C.EST_COLOR, lw=1.3, label="train")
    if val is not None:
        ax.plot(ep, val, color=C.GT_COLOR, lw=1.3, label="val")
    # stage transition markers (VI-only)
    stages = log.get("stage")
    if stages is not None:
        last = None
        for k in range(len(ep)):
            s = stages[k]
            if last is not None and s != last:
                ax.axvline(ep[k], color="gray", lw=0.8, ls="--", alpha=0.5)
            last = s
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{name} — train vs val loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    C.savefig(fig, out_dir / f"{name}_loss.png")


def _plot_one_net_val_ate(log: Dict[str, np.ndarray], name: str,
                          out_dir: Path) -> None:
    ep = log["epoch"]
    a0 = log.get("val_ATE_on_traj_0")
    am = log.get("val_ATE_on_minitest")
    if a0 is None and am is None:
        return
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if a0 is not None:
        ax.plot(ep, a0, color="tab:blue", lw=1.2, label="val ATE (traj 0)")
    if am is not None:
        ax.plot(ep, am, color="tab:red", lw=1.2, label="val ATE (minitest)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Unaligned ATE RMSE (m)")
    ax.set_title(f"{name} — validation ATE over training")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    C.savefig(fig, out_dir / f"{name}_val_ate.png")


def _plot_one_net_lr(log: Dict[str, np.ndarray], name: str,
                     out_dir: Path) -> None:
    lr = log.get("lr")
    if lr is None or not np.any(np.isfinite(lr)):
        return
    ep = log["epoch"]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(ep, lr, color="black", lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_yscale("log")
    ax.set_title(f"{name} — LR schedule")
    ax.grid(True, which="both", alpha=0.3)
    C.savefig(fig, out_dir / f"{name}_lr.png")


def _plot_combined_loss(logs: Dict[str, Dict[str, np.ndarray]],
                        out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    plotted = False
    for name, log in logs.items():
        if log is None:
            continue
        ep = log["epoch"]
        color = C.method_color(name)
        label = C.method_label(name)
        if "train_loss" in log:
            ax.plot(ep, log["train_loss"], color=color, lw=1.1,
                    label=f"{label} — train")
            plotted = True
        if "val_loss" in log:
            ax.plot(ep, log["val_loss"], color=color, lw=1.1, ls="--",
                    label=f"{label} — val")
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs val loss — all networks")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    C.savefig(fig, out_dir / "combined_loss.png")


def _plot_combined_val_ate(logs: Dict[str, Dict[str, np.ndarray]],
                           out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    plotted = False
    for name, log in logs.items():
        if log is None:
            continue
        ep = log["epoch"]
        am = log.get("val_ATE_on_minitest")
        if am is None:
            continue
        color = C.method_color(name)
        ax.plot(ep, am, color=color, lw=1.3, label=C.method_label(name))
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ATE RMSE on minitest (m)")
    ax.set_title("Validation (OOD) ATE — all networks")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    C.savefig(fig, out_dir / "combined_val_ate.png")


def _plot_multiseed(csv_path: Path, name: str, out_dir: Path) -> None:
    rows = C.load_csv_table(csv_path)
    if not rows:
        return
    # Expected columns (from train_multi_seed.py): seed, metric, mean, std, or
    # wide form with columns like ate_aligned_mean, ate_aligned_std, etc.
    # Two-mode plotter:
    cols = rows[0].keys()
    if {"metric", "mean", "std"}.issubset(cols):
        # long-form
        metrics = sorted({r["metric"] for r in rows})
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(metrics)), 3.4))
        xs = np.arange(len(metrics))
        means = [float(next(r for r in rows if r["metric"] == m)["mean"])
                 for m in metrics]
        stds = [float(next(r for r in rows if r["metric"] == m)["std"])
                for m in metrics]
        ax.bar(xs, means, yerr=stds, capsize=4,
               color=C.method_color(name), alpha=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("value")
        ax.set_title(f"{name} — multi-seed mean ± std")
        ax.grid(True, axis="y", alpha=0.3)
        C.savefig(fig, out_dir / f"multiseed_{name}.png")
    else:
        # wide form — try to extract epoch-indexed columns if present, else
        # just fall back to a table-style scatter.
        epochs, means, stds = [], [], []
        for r in rows:
            if "epoch" in r and "val_ate_minitest_mean" in r:
                epochs.append(float(r["epoch"]))
                means.append(float(r["val_ate_minitest_mean"]))
                stds.append(float(r.get("val_ate_minitest_std", 0.0)))
        if epochs:
            epochs = np.asarray(epochs)
            means = np.asarray(means)
            stds = np.asarray(stds)
            fig, ax = plt.subplots(figsize=(8, 3.4))
            ax.plot(epochs, means, color=C.method_color(name), lw=1.5,
                    label=f"{C.method_label(name)} mean")
            ax.fill_between(epochs, means - stds, means + stds,
                            color=C.method_color(name), alpha=0.25,
                            label="± 1 std")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("val ATE (minitest, m)")
            ax.set_title(f"{name} — multi-seed ATE across epochs")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            C.savefig(fig, out_dir / f"multiseed_{name}.png")
        else:
            C.warn(f"multiseed CSV {csv_path.name} has unexpected schema; "
                   "skipping.")


def plot_all_training(outputs_root: Path, results_dir: Path,
                      nets: Iterable[str] = ("v", "i", "vi")) -> None:
    out_dir = C.ensure_dir(results_dir / "training")
    logs: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
    for name in nets:
        log_path = outputs_root / name / "train.log"
        log = C.load_train_log(log_path)
        logs[name] = log
        if log is None:
            C.warn(f"no training log at {log_path.relative_to(outputs_root.parent)}"
                   f"; skipping {name}")
            continue
        _plot_one_net_loss(log, name, out_dir)
        _plot_one_net_val_ate(log, name, out_dir)
        _plot_one_net_lr(log, name, out_dir)
        C.info(f"wrote training curves for {name} → {out_dir}")

    _plot_combined_loss({k: v for k, v in logs.items() if v is not None}, out_dir)
    _plot_combined_val_ate({k: v for k, v in logs.items() if v is not None},
                           out_dir)

    for name in nets:
        csv_path = outputs_root / f"{name}_multiseed_summary.csv"
        if csv_path.exists():
            _plot_multiseed(csv_path, name, out_dir)
            C.info(f"wrote multi-seed plot for {name}")
