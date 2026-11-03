"""Train Network V (vision-only decomposed-homography regressor).

Usage:
    python scripts/train_v.py --config configs/default.yaml [--out_subdir v]

Protocol:
    * Training set = config.data.train_ids.
    * Validation: (a) in-distribution val_ATE_on_traj_0 (first training traj),
                  (b) OOD   val_ATE_on_minitest (trajectory_id 200; clover,
                      reduced amplitude, seen texture, novel shape).
    * Best-checkpoint selection uses val_ATE_on_minitest (Tier-1 B.6).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from _train_common import (  # noqa: E402
    TextLogger, build_val_loader_single_traj, compute_val_ate_on_trajectory,
    compute_val_loss, maybe_save_periodic,
)
from models.losses import DecomposedHomographyLoss  # noqa: E402
from models.network_v import NetworkV  # noqa: E402
from utils.dataloader import PhotometricAugment, build_train_loader  # noqa: E402
from utils.training import (  # noqa: E402
    cosine_warmup_lr, get_device, load_config, save_checkpoint, set_seed,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--out_subdir", type=str, default="v")
    p.add_argument("--sampling_mode", type=str, default=None,
                   help="Override config sampling.mode.")
    p.add_argument("--alpha", type=float, default=None,
                   help="Override config sampling.alpha (weighted mode only).")
    p.add_argument("--seed", type=int, default=None,
                   help="Override config seed (for multi-seed training).")
    return p.parse_args()


def _build_K(ds, device) -> torch.Tensor:
    first_tid = next(iter(ds._trajectories.keys()))
    K_np = ds._trajectories[first_tid].metadata["K"]
    return torch.tensor(K_np, dtype=torch.float32, device=device)


def _loss_compute(loss_fn, K):
    def fn(model, batch):
        R_pred, td_pred, n_pred, _ = model(batch["img_a"], batch["img_b"])
        return loss_fn(R_pred, td_pred, n_pred,
                       R_gt=batch["R_rel"], t_gt=batch["t_rel"],
                       n_gt=batch["normal"], d_gt=batch["depth"],
                       img_a=batch["img_a"], img_b=batch["img_b"], K=K)
    return fn


def _forward_rel(model, batch):
    R_pred, td_pred, _n, _ = model(batch["img_a"], batch["img_b"])
    d_gt = batch["depth"].clamp_min(1e-3)
    t_pred = td_pred * d_gt.unsqueeze(-1)
    return R_pred, t_pred


def _maybe_build_augment(cfg: dict) -> PhotometricAugment | None:
    aug_cfg = cfg.get("augmentation", {}).get("photometric", {})
    if not aug_cfg.get("enabled", False):
        return None
    return PhotometricAugment(
        brightness=aug_cfg.get("brightness", 0.3),
        contrast=aug_cfg.get("contrast", 0.3),
        saturation=aug_cfg.get("saturation", 0.2),
        hue=aug_cfg.get("hue", 0.05),
        gaussian_noise_std=aug_cfg.get("gaussian_noise_std", 0.01),
        grayscale_prob=aug_cfg.get("grayscale_prob", 0.1))


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = get_device()

    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["paths"]["outputs_dir"]) / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    logger = TextLogger(out_dir / "train.log")

    all_train = list(cfg["data"]["train_ids"])
    val_tid = cfg["data"].get("val_traj_id", all_train[0])
    minitest_tid = cfg["data"].get("minitest_traj_id", 200)
    # With multi-texture protocol, training no longer excludes traj 0 — all
    # training trajectories carry distinct textures so we need every pair.
    train_tids = list(all_train)
    print(f"[train_v] train={train_tids} val={val_tid} minitest={minitest_tid} seed={seed}")

    sampling_mode = args.sampling_mode or cfg["sampling"]["mode"]
    alpha = args.alpha if args.alpha is not None else cfg["sampling"].get("alpha", 0.5)
    augment = _maybe_build_augment(cfg)
    loader, ds = build_train_loader(
        raw_dir=raw_dir, traj_ids=train_tids,
        img_size=tuple(cfg["data"]["img_size"]),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        sampling_mode=sampling_mode,
        num_bins=cfg["sampling"].get("num_bins", 5),
        bin_edges=cfg["sampling"].get("bin_edges_m"),
        alpha=alpha,
        seed=seed,
        photometric_augment=augment,
    )
    val_loader, val_ds = build_val_loader_single_traj(
        raw_dir=raw_dir, traj_id=val_tid,
        img_size=tuple(cfg["data"]["img_size"]),
        num_workers=max(1, cfg["data"]["num_workers"] // 2),
    )
    minitest_loader, minitest_ds = build_val_loader_single_traj(
        raw_dir=raw_dir, traj_id=minitest_tid,
        img_size=tuple(cfg["data"]["img_size"]),
        num_workers=max(1, cfg["data"]["num_workers"] // 2),
    )

    model = NetworkV(feat_dim=cfg["model"]["cnn_feat_dim"],
                     normal_init_bias=cfg["model"]["normal_init_bias"]).to(device)
    loss_fn = DecomposedHomographyLoss(
        w_rot=cfg["loss"]["w_rot"], w_td=cfg["loss"]["w_td"],
        w_n=cfg["loss"]["w_n"], w_warp=cfg["loss"]["w_warp"],
        hover_center=cfg["loss"].get("hover_mask_center", 0.01),
        hover_width=cfg["loss"].get("hover_mask_width", 0.005)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])

    K_train = _build_K(ds, device)
    K_val = _build_K(val_ds, device)
    loss_train = _loss_compute(loss_fn, K_train)
    loss_val = _loss_compute(loss_fn, K_val)

    best_ate = float("inf")
    total_epochs = cfg["train"]["epochs_v"]
    for epoch in range(total_epochs):
        cosine_warmup_lr(opt, epoch, total_epochs,
                         cfg["train"]["warmup_epochs"], cfg["train"]["lr"])
        model.train()
        pbar = tqdm(loader, desc=f"V epoch {epoch+1}/{total_epochs}")
        epoch_total, n = 0.0, 0
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            losses = loss_train(model, batch)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["train"]["grad_clip"])
            opt.step()
            epoch_total += losses["total"].item(); n += 1
            if i % cfg["train"]["log_every"] == 0:
                step = epoch * len(loader) + i
                for k, v in losses.items():
                    if torch.is_tensor(v):
                        writer.add_scalar(f"train/{k}", float(v.item()), step)
                pbar.set_postfix(total=float(losses['total']))
        mean_train = epoch_total / max(1, n)

        val_loss = compute_val_loss(model, val_loader, loss_val, device)
        val_ate_traj0 = compute_val_ate_on_trajectory(
            model, val_loader, val_ds, _forward_rel, device)
        val_ate_mini = compute_val_ate_on_trajectory(
            model, minitest_loader, minitest_ds, _forward_rel, device)
        lr_now = opt.param_groups[0]["lr"]
        writer.add_scalar("epoch/train_total", mean_train, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/val_ATE_traj0", val_ate_traj0, epoch)
        writer.add_scalar("epoch/val_ATE_minitest", val_ate_mini, epoch)
        logger.log_epoch(epoch=epoch, stage="single", train_loss=mean_train,
                         val_loss=val_loss,
                         val_ATE_on_traj_0=val_ate_traj0,
                         val_ATE_on_minitest=val_ate_mini, lr=lr_now)

        save_checkpoint(out_dir / "last.pt", model, opt, epoch,
                        {"train_loss": mean_train, "val_loss": val_loss,
                         "val_ATE_traj0": val_ate_traj0,
                         "val_ATE_minitest": val_ate_mini})
        maybe_save_periodic(lambda p, meta: save_checkpoint(p, model, opt, epoch, meta),
                            out_dir, epoch, every=5,
                            meta={"train_loss": mean_train,
                                  "val_ATE_minitest": val_ate_mini})
        # Tier-1 B.6: select best by OOD minitest ATE.
        if val_ate_mini < best_ate:
            best_ate = val_ate_mini
            save_checkpoint(out_dir / "best.pt", model, opt, epoch,
                            {"val_ATE_minitest": val_ate_mini,
                             "val_ATE_traj0": val_ate_traj0,
                             "note": "best-by-val-ATE-minitest"})
            print(f"[train_v] epoch {epoch}: new best val_ATE_minitest = {val_ate_mini:.4f}")
    writer.close()
    # Final log line (multi-seed summariser parses this).
    print(f"[train_v] done. best val_ATE_on_minitest = {best_ate:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
