"""Motion-magnitude sampling study (Tier-1 B.2).

Trains Network V four times with identical hyperparameters but different
sampling strategies:
    alpha_0.0  — uniform (naive)          [weighted sampler, alpha=0.0]
    alpha_0.5  — soft rebalancing (default)
    alpha_1.0  — strict balancing
    small_only — smallest |Δt| bin only

Headline finding is the gap between training-loss curves (~identical) and
test ATE (~wide) — training loss is misleading for this failure mode.

Writes:
    outputs/studies/motion_magnitude.csv       per-(mode, traj) ATE
    outputs/studies/motion_magnitude.png       test ATE bar chart
    outputs/studies/motion_magnitude_loss.png  training-loss overlay
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from models.network_v import NetworkV  # noqa: E402
from utils.dataloader import build_eval_loader  # noqa: E402
from utils.eval import dead_reckon_from_gt_start, gt_trajectory_frames  # noqa: E402
from utils.training import get_device, load_config, set_seed  # noqa: E402


# Per the B.2 study: three alpha values + small_only.
MODES: tuple = ("alpha_0.0", "alpha_0.5", "alpha_1.0", "small_only")


def _mode_to_args(mode: str) -> list[str]:
    """Map display mode to extra CLI args for scripts/train_v.py."""
    if mode == "small_only":
        return ["--sampling_mode", "small_only"]
    alpha = mode.removeprefix("alpha_")
    return ["--sampling_mode", "weighted", "--alpha", alpha]


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--skip_train", action="store_true")
    return p.parse_args()


def _run(cmd: List[str]) -> None:
    print(f"[mm-study] $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _train_all_modes(cfg_path: str) -> None:
    for mode in MODES:
        out_sub = f"v_mm_{mode.replace('.', '_')}"
        extra = _mode_to_args(mode)
        _run([sys.executable, "scripts/train_v.py",
              "--config", cfg_path,
              "--out_subdir", out_sub,
              *extra])


def _evaluate_mode(mode: str, cfg: Dict, device: torch.device) -> List[float]:
    ckpt = Path(cfg["paths"]["outputs_dir"]) / f"v_mm_{mode.replace('.', '_')}" / "best.pt"
    if not ckpt.exists():
        print(f"[mm-study] WARN: missing checkpoint {ckpt}")
        return []
    model = NetworkV(feat_dim=cfg["model"]["cnn_feat_dim"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"],
                          strict=False)
    model.eval()
    ates: List[float] = []
    for tid in cfg["data"]["test_ids"]:
        loader, ds = build_eval_loader(
            raw_dir=Path(cfg["data"]["raw_dir"]), traj_ids=[tid],
            img_size=tuple(cfg["data"]["img_size"]), num_workers=0)
        traj = list(ds._trajectories.values())[0]
        altitude = float(traj.metadata.get("altitude_initial",
                                           traj.metadata.get("altitude_m", 4.0)))
        rels = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                R, td, _n, _f = model(batch["img_a"], batch["img_b"])
                t = td * altitude
                rels.append((R[0].cpu().numpy(), t[0].cpu().numpy()))
        _t, pred_ts, _q = dead_reckon_from_gt_start(traj, rels)
        _g, gt_ts, _gq = gt_trajectory_frames(traj)
        N = min(len(pred_ts), len(gt_ts))
        rmse = float(np.sqrt(((pred_ts[:N] - gt_ts[:N]) ** 2).sum(-1).mean()))
        ates.append(rmse)
        print(f"[mm-study] mode={mode:<10} traj={tid}  ATE={rmse:.4f} m")
    return ates


def _write_csv(csv_path: Path, per_mode: Dict[str, List[float]],
               test_ids: List[int]) -> None:
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "traj_id", "ate_m"])
        for mode in MODES:
            vals = per_mode.get(mode, [])
            for tid, ate in zip(test_ids, vals):
                w.writerow([mode, tid, f"{ate:.6f}"])


def _plot(csv_path: Path, out_path: Path) -> None:
    per_mode: Dict[str, List[float]] = {m: [] for m in MODES}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            per_mode.setdefault(row["mode"], []).append(float(row["ate_m"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(MODES))
    means = [np.mean(per_mode[m]) if per_mode[m] else 0.0 for m in MODES]
    errs = [np.std(per_mode[m]) if per_mode[m] else 0.0 for m in MODES]
    ax.bar(positions, means, yerr=errs, tick_label=list(MODES),
           color=["#CC5555", "#EAA94C", "#2E6E31", "#7D9CC9"])
    ax.set_ylabel("ATE (m)")
    ax.set_title("Motion-magnitude sampling — mean ATE on test set\n"
                 "(training loss is NOT predictive of this gap)")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def _read_train_log(log_path: Path) -> List[float]:
    if not log_path.exists():
        return []
    losses: List[float] = []
    with log_path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            col = header.index("train_loss")
        except ValueError:
            return []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= col: continue
            try:
                losses.append(float(parts[col]))
            except ValueError:
                continue
    return losses


def _plot_loss_overlay(cfg: Dict, out_path: Path) -> None:
    colors = {"alpha_0.0": "#CC5555", "alpha_0.5": "#EAA94C",
              "alpha_1.0": "#2E6E31", "small_only": "#7D9CC9"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in MODES:
        log_path = (Path(cfg["paths"]["outputs_dir"])
                    / f"v_mm_{mode.replace('.', '_')}" / "train.log")
        losses = _read_train_log(log_path)
        if not losses:
            print(f"[mm-study] no train.log for {mode}: {log_path}")
            continue
        ax.plot(np.arange(1, len(losses) + 1), losses, label=mode,
                color=colors.get(mode, None), lw=1.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss (total)")
    ax.set_title("Training-loss overlay — all modes look similar\n"
                 "(diagnostic: the gap is invisible at train time)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    studies = Path(cfg["paths"]["studies_dir"])
    studies.mkdir(parents=True, exist_ok=True)
    csv_path = studies / "motion_magnitude.csv"

    if not args.skip_train:
        _train_all_modes(args.config)

    per_mode: Dict[str, List[float]] = {}
    for mode in MODES:
        per_mode[mode] = _evaluate_mode(mode, cfg, device)

    _write_csv(csv_path, per_mode, cfg["data"]["test_ids"])
    _plot(csv_path, studies / "motion_magnitude.png")
    _plot_loss_overlay(cfg, studies / "motion_magnitude_loss.png")
    print(f"[mm-study] wrote {csv_path}, motion_magnitude.png, "
          f"motion_magnitude_loss.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
