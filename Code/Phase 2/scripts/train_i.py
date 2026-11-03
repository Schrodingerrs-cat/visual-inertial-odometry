"""Train Network I (IMU-only relative pose estimator).

Usage:
    python scripts/train_i.py --config configs/default.yaml

Protocol: same dual validation as train_v.py (val_ATE_on_traj_0 +
val_ATE_on_minitest; best-checkpoint by the OOD minitest ATE).
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
    compute_val_loss, generate_training_plots, maybe_save_periodic,
)
from models.losses import IMUPoseLoss, IMUPoseLoss_hover  # noqa: E402
from models.network_i import NetworkI  # noqa: E402
from utils.dataloader import PhotometricAugment, build_train_loader  # noqa: E402
from utils.training import (  # noqa: E402
    cosine_warmup_lr, get_device, load_config, save_checkpoint, set_seed,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--out_subdir", type=str, default="i")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _loss_compute(loss_fn):
    def fn(model, batch):
        # Phase 2: I trains on the longer causal context window
        # (batch["imu_context"]). Falls back to batch["imu"] when the
        # dataloader didn't emit a context tensor (Phase 0 / 1 behavior).
        imu_for_i = batch.get("imu_context", batch["imu"])
        R_pred, t_pred, _ = model(imu_for_i)
        return loss_fn(R_pred, t_pred, batch["R_rel"], batch["t_rel"])
    return fn


def _forward_rel(model, batch):
    imu_for_i = batch.get("imu_context", batch["imu"])
    R_pred, t_pred, _ = model(imu_for_i)
    return R_pred, t_pred


def _maybe_build_augment(cfg: dict) -> PhotometricAugment | None:
    # Augmentation doesn't affect an IMU-only model; kept here only so the
    # dataset-mode bookkeeping is consistent across train scripts.
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
    train_tids = list(all_train)
    print(f"[train_i] train={train_tids} val={val_tid} minitest={minitest_tid} seed={seed}")

    augment = _maybe_build_augment(cfg)
    imu_ctx_len = cfg["data"].get("imu_context_length", None)
    if imu_ctx_len is not None:
        print(f"[train_i] imu_context_length = {imu_ctx_len} "
              f"(Phase 2 causal IMU window for I branch)")
    loader, _ds = build_train_loader(
        raw_dir=raw_dir, traj_ids=train_tids,
        img_size=tuple(cfg["data"]["img_size"]),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        sampling_mode=cfg["sampling"]["mode"],
        num_bins=cfg["sampling"].get("num_bins", 5),
        bin_edges=cfg["sampling"].get("bin_edges_m"),
        alpha=cfg["sampling"].get("alpha", 0.5),
        seed=seed,
        photometric_augment=augment,
        imu_context_length=imu_ctx_len,
    )
    val_loader, val_ds = build_val_loader_single_traj(
        raw_dir=raw_dir, traj_id=val_tid,
        img_size=tuple(cfg["data"]["img_size"]),
        num_workers=max(1, cfg["data"]["num_workers"] // 2),
        imu_context_length=imu_ctx_len,
    )
    minitest_loader, minitest_ds = build_val_loader_single_traj(
        raw_dir=raw_dir, traj_id=minitest_tid,
        img_size=tuple(cfg["data"]["img_size"]),
        num_workers=max(1, cfg["data"]["num_workers"] // 2),
        imu_context_length=imu_ctx_len,
    )

    model = NetworkI(hidden_dim=cfg["model"]["lstm_hidden"]).to(device)
    # Step S: when loss.imu_use_hover is False, fall back to the original
    # IMUPoseLoss (no hover masking). Default True preserves Step K Phase 1
    # behavior so existing step_k_train_i.sh runs are unchanged.
    use_hover = cfg.get("loss", {}).get("imu_use_hover", True)
    if use_hover:
        loss_fn = IMUPoseLoss_hover(
            w_rot=cfg["loss"]["w_rot_imu"],
            w_trans=cfg["loss"]["w_imu_trans"],
            hover_center=cfg["loss"]["hover_mask_center"],
            hover_width=cfg["loss"]["hover_mask_width"],
        ).to(device)
    else:
        loss_fn = IMUPoseLoss(
            w_rot=cfg["loss"]["w_rot_imu"],
            w_trans=cfg["loss"]["w_imu_trans"],
        ).to(device)
    print(f"[train_i] loss class = {type(loss_fn).__name__} "
          f"(loss.imu_use_hover={use_hover})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    loss_compute = _loss_compute(loss_fn)

    best_ate = float("inf")
    total_epochs = cfg["train"]["epochs_i"]
    for epoch in range(total_epochs):
        cosine_warmup_lr(opt, epoch, total_epochs,
                         cfg["train"]["warmup_epochs"], cfg["train"]["lr"])
        model.train()
        pbar = tqdm(loader, desc=f"I epoch {epoch+1}/{total_epochs}")
        epoch_total, n = 0.0, 0
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            losses = loss_compute(model, batch)
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

        val_loss = compute_val_loss(model, val_loader, loss_compute, device)
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
        if val_ate_mini < best_ate:
            best_ate = val_ate_mini
            save_checkpoint(out_dir / "best.pt", model, opt, epoch,
                            {"val_ATE_minitest": val_ate_mini,
                             "val_ATE_traj0": val_ate_traj0,
                             "note": "best-by-val-ATE-minitest"})
            print(f"[train_i] epoch {epoch}: new best val_ATE_minitest = {val_ate_mini:.4f}")
    writer.close()
    print(f"[train_i] done. best val_ATE_on_minitest = {best_ate:.4f}")

    generate_training_plots(
        model=model,
        loaders={"val": (val_loader, val_ds),
                 "minitest": (minitest_loader, minitest_ds)},
        forward_rel=_forward_rel,
        device=device,
        out_plots_dir=out_dir / "plots",
        method="i",
        tag="final",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
