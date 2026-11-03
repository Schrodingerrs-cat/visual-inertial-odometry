"""Capacity ablation for VI-full.

Trains the full VI-full network at three CNN+LSTM capacities under the
identical training protocol used everywhere else:

    small  : CNN channels [32, 32, 64, 64, 128, 128, 256, 256], LSTM hidden 64
    medium : CNN channels [64, 64, 128, 128, 256, 256, 512, 512], LSTM hidden 128
    large  : CNN channels [64, 128, 128, 256, 256, 512, 512, 512], LSTM hidden 256

Tests the group10 claim that multi-trajectory failure is a capacity issue:
if all three reach similar test ATE, the blocker was parameterization (our
decomposed-homography + analytic-fusion fix), not capacity.

Writes:
    outputs/studies/capacity.csv
    outputs/studies/capacity_ate.csv        3x3 ATE-aligned pivot
    outputs/studies/capacity.png            ATE bar chart per capacity
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from tqdm import tqdm  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from _train_common import (  # noqa: E402
    TextLogger, build_val_loader_single_traj, compute_val_ate_traj0,
    maybe_save_periodic,
)
from models.losses import VILoss  # noqa: E402
from models.network_vi import NetworkVI  # noqa: E402
from utils.dataloader import build_eval_loader, build_train_loader  # noqa: E402
from utils.eval import dead_reckon_from_gt_start, gt_trajectory_frames  # noqa: E402
from utils.training import (  # noqa: E402
    cosine_warmup_lr, get_device, load_config, save_checkpoint, set_seed,
)


CAPACITIES: Dict[str, Dict] = {
    "small":  {"channels": (32, 32, 64, 64, 128, 128, 256, 256),
               "lstm_hidden": 64,  "feat_dim": 512},
    "medium": {"channels": (64, 64, 128, 128, 256, 256, 512, 512),
               "lstm_hidden": 128, "feat_dim": 512},
    "large":  {"channels": (64, 128, 128, 256, 256, 512, 512, 512),
               "lstm_hidden": 256, "feat_dim": 512},
}


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--only", type=str, nargs="*", default=None,
                   help="Subset of capacities to train (default: all).")
    return p.parse_args()


def _forward_rel(model, batch):
    preds = model(batch["img_a"], batch["img_b"], batch["imu"])
    return preds["R"], preds["t"]


def _train_capacity(name: str, spec: Dict, cfg: Dict, device: torch.device,
                    out_root: Path) -> Path:
    out_dir = out_root / f"vi_cap_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    logger = TextLogger(out_dir / "train.log")

    all_train = list(cfg["data"]["train_ids"])
    val_tid = all_train[0]
    train_tids = all_train[1:] if len(all_train) > 1 else all_train

    loader, ds = build_train_loader(
        raw_dir=Path(cfg["data"]["raw_dir"]),
        traj_ids=train_tids,
        img_size=tuple(cfg["data"]["img_size"]),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        sampling_mode=cfg["sampling"]["mode"],
        num_bins=cfg["sampling"]["num_bins"],
        bin_edges=cfg["sampling"]["bin_edges_m"],
        alpha=cfg["sampling"].get("alpha", 0.5),
        seed=cfg["seed"],
    )
    val_loader, val_ds = build_val_loader_single_traj(
        raw_dir=Path(cfg["data"]["raw_dir"]),
        traj_id=val_tid, img_size=tuple(cfg["data"]["img_size"]),
        num_workers=max(1, cfg["data"]["num_workers"] // 2),
    )

    model = NetworkVI(feat_dim=spec["feat_dim"], lstm_hidden=spec["lstm_hidden"],
                      cnn_channels=spec["channels"]).to(device)
    loss_fn = VILoss(w_rot=cfg["loss"]["w_rot"], w_td=cfg["loss"]["w_td"],
                     w_n=cfg["loss"]["w_n"], w_warp=cfg["loss"]["w_warp"],
                     w_trans=cfg["loss"]["w_trans"],
                     w_imu_trans=cfg["loss"]["w_imu_trans"],
                     w_rot_imu=cfg["loss"]["w_rot_imu"],
                     w_d=cfg["loss"]["w_d"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    n_epochs = (cfg["train"]["epochs_vi_stage1"]
                + cfg["train"]["epochs_vi_stage2"])
    K = torch.tensor(next(iter(ds._trajectories.values())).metadata["K"],
                     dtype=torch.float32, device=device)
    best_ate = float("inf")
    for epoch in range(n_epochs):
        cosine_warmup_lr(opt, epoch, n_epochs, cfg["train"]["warmup_epochs"],
                         cfg["train"]["lr"])
        model.train()
        pbar = tqdm(loader, desc=f"cap-{name} epoch {epoch+1}/{n_epochs}")
        total, m = 0.0, 0
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            preds = model(batch["img_a"], batch["img_b"], batch["imu"])
            l = loss_fn(preds, batch["R_rel"], batch["t_rel"],
                        batch["normal"], batch["depth"],
                        batch["img_a"], batch["img_b"], K)
            opt.zero_grad(set_to_none=True)
            l["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["train"]["grad_clip"])
            opt.step()
            total += float(l["total"]); m += 1
        mean_total = total / max(1, m)
        val_ate = compute_val_ate_traj0(model, val_loader, val_ds,
                                        _forward_rel, device)
        lr_now = opt.param_groups[0]["lr"]
        writer.add_scalar("epoch/train_total", mean_total, epoch)
        writer.add_scalar("epoch/val_ATE", val_ate, epoch)
        logger.log_epoch(epoch=epoch, stage="single", train_loss=mean_total,
                         val_loss="", val_ATE_on_traj_0=val_ate, lr=lr_now)
        save_checkpoint(out_dir / "last.pt", model, opt, epoch,
                        {"train_loss": mean_total, "val_ATE": val_ate})
        maybe_save_periodic(lambda p, meta: save_checkpoint(p, model, opt, epoch, meta),
                            out_dir, epoch, every=5,
                            meta={"train_loss": mean_total, "val_ATE": val_ate})
        if val_ate < best_ate:
            best_ate = val_ate
            save_checkpoint(out_dir / "best.pt", model, opt, epoch,
                            {"val_ATE": val_ate, "capacity": name,
                             "note": "best-by-val-ATE"})
    writer.close()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[capacity] {name} done. params={n_params:,} best_val_ATE={best_ate:.4f}")
    return out_dir / "best.pt"


def _evaluate(name: str, spec: Dict, cfg: Dict, device: torch.device,
              ckpt: Path) -> List[Tuple[int, float]]:
    model = NetworkVI(feat_dim=spec["feat_dim"], lstm_hidden=spec["lstm_hidden"],
                      cnn_channels=spec["channels"]).to(device)
    blob = torch.load(ckpt, map_location=device)
    model.load_state_dict(blob["model"], strict=False)
    model.eval()
    out: List[Tuple[int, float]] = []
    for tid in cfg["data"]["test_ids"]:
        loader, ds = build_eval_loader(
            raw_dir=Path(cfg["data"]["raw_dir"]), traj_ids=[tid],
            img_size=tuple(cfg["data"]["img_size"]), num_workers=0)
        traj = list(ds._trajectories.values())[0]
        rels = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                preds = model(batch["img_a"], batch["img_b"], batch["imu"])
                rels.append((preds["R"][0].cpu().numpy(),
                             preds["t"][0].cpu().numpy()))
        _t, pred_ts, _q = dead_reckon_from_gt_start(traj, rels)
        _g, gt_ts, _gq = gt_trajectory_frames(traj)
        N = min(len(pred_ts), len(gt_ts))
        rmse = float(np.sqrt(((pred_ts[:N] - gt_ts[:N]) ** 2).sum(-1).mean()))
        out.append((tid, rmse))
        print(f"[capacity] {name:<6} traj {tid}: ATE={rmse:.4f} m")
    return out


def _write_reports(per_cap: Dict[str, List[Tuple[int, float]]],
                   studies: Path, test_ids: List[int]) -> None:
    long_path = studies / "capacity.csv"
    with long_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capacity", "traj_id", "ate_m"])
        for name, rows in per_cap.items():
            for tid, ate in rows:
                w.writerow([name, tid, f"{ate:.6f}"])
    # 3xT pivot.
    pivot_path = studies / "capacity_ate.csv"
    with pivot_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capacity"] + [f"traj_{t}" for t in test_ids] + ["mean"])
        for name, rows in per_cap.items():
            d = dict(rows)
            vals = [d.get(t, float("nan")) for t in test_ids]
            valid = [v for v in vals if not np.isnan(v)]
            mean = float(np.mean(valid)) if valid else float("nan")
            w.writerow([name] + [f"{v:.6f}" for v in vals] + [f"{mean:.6f}"])
    # Bar chart.
    names = list(per_cap.keys())
    means = [float(np.mean([ate for _, ate in per_cap[n]]) if per_cap[n] else np.nan)
             for n in names]
    errs = [float(np.std([ate for _, ate in per_cap[n]]) if per_cap[n] else np.nan)
            for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(len(names)), means, yerr=errs, tick_label=names,
           color=["#CC5555", "#EAA94C", "#7D9CC9"])
    ax.set_ylabel("ATE-aligned (m), mean over test trajectories")
    ax.set_title("VI-full capacity ablation\n"
                 "(if flat across capacities: problem was parameterization)")
    fig.tight_layout(); fig.savefig(studies / "capacity.png", dpi=140); plt.close(fig)


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    out_root = Path(cfg["paths"]["outputs_dir"])
    studies = Path(cfg["paths"]["studies_dir"]); studies.mkdir(parents=True, exist_ok=True)

    selected = args.only or list(CAPACITIES.keys())
    for name in selected:
        if name not in CAPACITIES:
            raise ValueError(f"unknown capacity: {name}")

    if not args.skip_train:
        for name in selected:
            _train_capacity(name, CAPACITIES[name], cfg, device, out_root)

    per_cap: Dict[str, List[Tuple[int, float]]] = {}
    for name in selected:
        ckpt = out_root / f"vi_cap_{name}" / "best.pt"
        if not ckpt.exists():
            print(f"[capacity] WARN: missing {ckpt}")
            per_cap[name] = []
            continue
        per_cap[name] = _evaluate(name, CAPACITIES[name], cfg, device, ckpt)

    _write_reports(per_cap, studies, list(cfg["data"]["test_ids"]))
    print(f"[capacity] wrote {studies/'capacity.csv'}, "
          f"{studies/'capacity_ate.csv'}, {studies/'capacity.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
