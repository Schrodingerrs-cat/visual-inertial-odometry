"""Shared training utilities for Phase 2 train_v / train_i / train_vi.

Provides:
  * TextLogger — append-only plaintext training log (epoch-level rows).
  * build_val_loader_single_traj — sequential DataLoader on one held-out trajectory.
  * compute_val_loss — run a forward-only pass on the val loader.
  * compute_val_ate_on_trajectory — dead-reckon predicted relative poses on a
    held-out trajectory and return unaligned trajectory ATE.
  * compute_val_ate_traj0 — back-compat wrapper calling compute_val_ate_on_trajectory.

Dual validation metrics (Tier-1 B.6):
  val_ATE_on_traj_0     : in-distribution — seen texture, seen shape.
  val_ATE_on_minitest   : OOD shape — seen texture, novel trajectory (traj_id 200).
Best-checkpoint selection should prefer ``val_ATE_on_minitest`` because it is
more predictive of held-out generalization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataloader import FramePairDataset
from utils.geometry import dead_reckon


class TextLogger:
    """Append-only plaintext log.

    Writes a header on first use and one row per call of ``log_epoch``. Column
    order is pinned so downstream scripts can parse reliably. Includes both
    in-distribution val ATE and the minitest OOD val ATE.
    """

    COLS = ("epoch", "stage", "train_loss", "val_loss",
            "val_ATE_on_traj_0", "val_ATE_on_minitest", "lr")

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.write_text("\t".join(self.COLS) + "\n")

    def log_epoch(self, **fields) -> None:
        row = []
        for k in self.COLS:
            v = fields.get(k, "")
            if isinstance(v, float):
                row.append(f"{v:.6f}")
            else:
                row.append(str(v))
        with self.path.open("a") as f:
            f.write("\t".join(row) + "\n")


# ---------------------------------------------------------------------------
# Val loader over a single trajectory
# ---------------------------------------------------------------------------

def build_val_loader_single_traj(raw_dir: Path, traj_id: int,
                                 img_size: Sequence[int],
                                 num_workers: int = 2,
                                 raft_cache: bool = False,
                                 imu_context_length: int | None = None,
                                 ) -> Tuple[DataLoader, FramePairDataset]:
    """Sequential, batch_size=1 loader over a single trajectory (no augment).

    ``raft_cache=True`` (Step K only) loads the per-traj RAFT cache so the
    val-loss computation, which uses the same loss function as training,
    has access to ``raft_{R, t, sigma, valid}`` keys.

    ``imu_context_length`` (Phase 2 only) sets the length of the optional
    causal context window emitted as ``batch["imu_context"]``. Default
    ``None`` → falls back to legacy length-10 behavior (imu_context aliases
    imu).
    """
    ds = FramePairDataset(raw_dir=Path(raw_dir), traj_ids=[int(traj_id)],
                          img_size=tuple(img_size), mode="val",
                          imu_context_length=imu_context_length,
                          raft_cache=raft_cache,
                          raft_cache_strict=raft_cache)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, ds


# ---------------------------------------------------------------------------
# Val-loss / val-ATE
# ---------------------------------------------------------------------------

def compute_val_loss(model: torch.nn.Module, loader: DataLoader,
                     loss_compute: Callable[[torch.nn.Module, dict], dict],
                     device: torch.device) -> float:
    """Run a forward pass on the val loader; return mean total loss."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            losses = loss_compute(model, batch)
            total += float(losses["total"].item())
            n += 1
    return total / max(1, n)


def compute_val_ate_on_trajectory(model: torch.nn.Module,
                                  loader: DataLoader,
                                  val_ds: FramePairDataset,
                                  forward_rel: Callable[[torch.nn.Module, dict],
                                                        Tuple[torch.Tensor, torch.Tensor]],
                                  device: torch.device,
                                  traj_id: int | None = None) -> float:
    """Dead-reckon the specified held-out trajectory and score unaligned ATE.

    If ``traj_id`` is None, uses the single trajectory present in ``val_ds``.
    """
    model.eval()
    rel_Rs, rel_ts = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            R_pred, t_pred = forward_rel(model, batch)
            rel_Rs.append(R_pred[0].detach().cpu().numpy())
            rel_ts.append(t_pred[0].detach().cpu().numpy())
    rel_Rs = np.stack(rel_Rs) if rel_Rs else np.zeros((0, 3, 3))
    rel_ts = np.stack(rel_ts) if rel_ts else np.zeros((0, 3))

    if traj_id is not None and traj_id in val_ds._trajectories:
        traj = val_ds._trajectories[traj_id]
    else:
        traj = next(iter(val_ds._trajectories.values()))
    poses = traj.poses_full
    stride = val_ds.imu_window
    n_pairs = rel_Rs.shape[0]
    n_frames = n_pairs + 1
    gt_ts = np.stack([poses[k * stride, :3] for k in range(n_frames)])
    init_R_row = poses[0]
    from utils.dataloader import _quat_to_R
    init_R = _quat_to_R(init_R_row[3:])
    init_t = init_R_row[:3]

    pred_Rs, pred_ts = dead_reckon(init_R, init_t,
                                   rel_Rs.astype(np.float64),
                                   rel_ts.astype(np.float64))
    errs = np.linalg.norm(pred_ts - gt_ts, axis=-1)
    return float(np.sqrt(np.mean(errs ** 2)))


def compute_val_ate_traj0(model, loader, val_ds, forward_rel, device) -> float:
    """Back-compat shim: calls compute_val_ate_on_trajectory on the sole val traj."""
    return compute_val_ate_on_trajectory(model, loader, val_ds, forward_rel,
                                         device, traj_id=None)


# ---------------------------------------------------------------------------
# Checkpoint every-N + best-by-ATE
# ---------------------------------------------------------------------------

def maybe_save_periodic(save_fn, out_dir: Path, epoch: int, every: int = 5,
                        meta: Dict = None) -> None:
    """Save ``epoch_{epoch:03d}.pt`` every ``every`` epochs (1-indexed)."""
    if (epoch + 1) % every == 0:
        save_fn(out_dir / f"epoch_{epoch + 1:03d}.pt", meta=meta or {})


# ---------------------------------------------------------------------------
# End-of-training plot generation
# ---------------------------------------------------------------------------

def generate_training_plots(model: torch.nn.Module,
                            loaders: Dict[str, Tuple[DataLoader,
                                                     FramePairDataset]],
                            forward_rel: Callable[[torch.nn.Module, dict],
                                                  Tuple[torch.Tensor,
                                                        torch.Tensor]],
                            device: torch.device,
                            out_plots_dir: Path,
                            method: str,
                            tag: str = "best") -> None:
    """Render end-of-training trajectory plots for one or more held-out trajs.

    Called once at the end of ``train_v_tdirect.py`` / ``train_i.py`` /
    ``train_vi_tdirect.py`` so we can inspect the trained model's behaviour
    on its validation trajectories without re-running the full eval pipeline.

    Parameters
    ----------
    model
        Trained model (will be put into ``eval()`` mode).
    loaders
        Mapping from a short tag (e.g. ``"val"``, ``"minitest"``) to a
        ``(DataLoader, FramePairDataset)`` pair as produced by
        ``build_val_loader_single_traj``. Each entry produces 4 PNGs:
        ``{method}_{traj_id}_3d.png``, ``_topdown.png``, ``_xyz.png``,
        ``_xyz_error.png``.
    forward_rel
        Callable returning ``(R_pred, t_pred)`` from ``(model, batch)`` —
        same convention as ``compute_val_ate_on_trajectory``.
    device
        torch device.
    out_plots_dir
        Output directory; created if missing. Files are named
        ``{method}_{traj_id}_{type}.png``.
    method
        Short method name (``"v"``, ``"i"``, ``"vi"``) — appears in titles.
    tag
        Free-form tag for the title (e.g. ``"best"``, ``"final"``).

    Plot generation is wrapped in try/except so a matplotlib failure on the
    cluster does **not** kill the train job after a successful run — the
    train_log and best.pt are already on disk by the time this runs.
    """
    # Local imports keep the train scripts' module-load surface lean.
    from utils.dataloader import _quat_to_R
    from utils.eval import (
        plot_2d_topdown, plot_3d_overlay, plot_per_axis,
        plot_per_axis_error,
    )
    from utils.geometry import dead_reckon as _dead_reckon

    out_plots_dir = Path(out_plots_dir)
    out_plots_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for loader_tag, (loader, val_ds) in loaders.items():
        try:
            rel_Rs: list = []
            rel_ts: list = []
            with torch.no_grad():
                for batch in loader:
                    batch = {k: v.to(device, non_blocking=True)
                             if torch.is_tensor(v) else v
                             for k, v in batch.items()}
                    R_pred, t_pred = forward_rel(model, batch)
                    rel_Rs.append(R_pred[0].detach().cpu().numpy())
                    rel_ts.append(t_pred[0].detach().cpu().numpy())
            if not rel_Rs:
                print(f"[{method}] generate_training_plots: empty loader "
                      f"for {loader_tag} — skipping.")
                continue
            rel_Rs_arr = np.stack(rel_Rs).astype(np.float64)
            rel_ts_arr = np.stack(rel_ts).astype(np.float64)

            traj = next(iter(val_ds._trajectories.values()))
            poses = traj.poses_full
            stride = val_ds.imu_window
            n_pairs = rel_Rs_arr.shape[0]
            n_frames = n_pairs + 1
            gt_ts = np.stack([poses[k * stride, :3] for k in range(n_frames)])
            init_R = _quat_to_R(poses[0, 3:])
            init_t = poses[0, :3]
            _pred_Rs, pred_ts = _dead_reckon(init_R, init_t,
                                             rel_Rs_arr, rel_ts_arr)
            fps = traj.metadata.get("fps_image", 100.0)
            times = np.arange(n_frames, dtype=np.float64) / float(fps)

            # Unaligned RMSE for the title (cheap; matches
            # compute_val_ate_on_trajectory).
            errs = np.linalg.norm(pred_ts - gt_ts, axis=-1)
            ate = float(np.sqrt(np.mean(errs ** 2)))

            tid = int(next(iter(val_ds._trajectories.keys())))
            title_root = (f"{method} ({tag}) — traj {tid} "
                          f"[{loader_tag}]")
            plot_3d_overlay(gt_ts, pred_ts,
                            out_plots_dir / f"{method}_{tid}_3d.png",
                            title=f"{title_root} (3D overlay)", ate=ate)
            plot_2d_topdown(gt_ts, pred_ts,
                            out_plots_dir / f"{method}_{tid}_topdown.png",
                            title=f"{title_root} (top-down XY)", ate=ate)
            plot_per_axis(times, gt_ts, pred_ts,
                          out_plots_dir / f"{method}_{tid}_xyz.png",
                          ylabels=("x (m)", "y (m)", "z (m)"),
                          title=f"{title_root} (position vs time)",
                          ate=ate)
            plot_per_axis_error(times, gt_ts, pred_ts,
                                out_plots_dir / f"{method}_{tid}_xyz_error.png",
                                title=f"{title_root} (position error)",
                                ate=ate)
            print(f"[{method}] training plots written to "
                  f"{out_plots_dir} for traj {tid} "
                  f"({loader_tag}, ATE={ate:.3f} m)")
        except Exception as exc:  # noqa: BLE001  — never fail the train job
            print(f"[{method}] generate_training_plots({loader_tag}) "
                  f"FAILED ({exc}); training artifacts still saved.")
