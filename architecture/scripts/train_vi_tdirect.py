"""Train Network VI — t-direct variant (Step H), two-stage.

Mirrors ``scripts/train_vi.py`` but uses ``NetworkVI_tdirect`` and
``VILoss_tdirect``. The fusion has no scale head and no scale mixer (only a
residual refiner), so the loss has no ``l_d`` term and there is no
``photometric warp residual`` (no ``K`` argument needed).

Stage 1: freeze vision and IMU backbones, train only the residual refiner.
Stage 2: unfreeze; backbone LR scaled by ``train.stage2_lr_mult``.

Best checkpoint selected by ``val_ATE_on_minitest`` (= traj 3 here).

Usage:
    python scripts/train_vi_tdirect.py --config configs/step_h.yaml \\
        --pretrained_v outputs/step_h/v/best.pt \\
        --pretrained_i outputs/step_h/i/best.pt   # symlink to step_g/i
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
from models.losses_tdirect import VILoss_tdirect  # noqa: E402
from models.network_vi_tdirect import NetworkVI_tdirect  # noqa: E402
from utils.dataloader import PhotometricAugment, build_train_loader  # noqa: E402
from utils.training import (  # noqa: E402
    cosine_warmup_lr, get_device, load_config, load_checkpoint,
    save_checkpoint, set_seed,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/step_h.yaml")
    p.add_argument("--out_subdir", type=str, default="vi")
    p.add_argument("--pretrained_v", type=str, default="")
    p.add_argument("--pretrained_i", type=str, default="")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _set_backbone_requires_grad(model: NetworkVI_tdirect, flag: bool) -> None:
    for p in model.vision.parameters(): p.requires_grad = flag
    for p in model.imu.parameters():    p.requires_grad = flag


def _loss_compute(loss_fn):
    def fn(model, batch):
        preds = model(batch["img_a"], batch["img_b"], batch["imu"])
        return loss_fn(preds, R_gt=batch["R_rel"], t_gt=batch["t_rel"])
    return fn


def _forward_rel(model, batch):
    preds = model(batch["img_a"], batch["img_b"], batch["imu"])
    return preds["R"], preds["t"]


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


def _run_epoch(model, loader, loss_compute, opt, device, cfg, writer,
               epoch: int, stage: str, global_step: int) -> tuple[float, int]:
    model.train()
    pbar = tqdm(loader, desc=f"VI[tdirect][{stage}] epoch {epoch+1}")
    total, n = 0.0, 0
    for i, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        losses = loss_compute(model, batch)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad,
                                              model.parameters()),
                                       cfg["train"]["grad_clip"])
        opt.step()
        total += losses["total"].item(); n += 1
        if i % cfg["train"]["log_every"] == 0:
            for k, v in losses.items():
                if torch.is_tensor(v):
                    writer.add_scalar(f"{stage}/{k}", float(v.item()), global_step)
            pbar.set_postfix(total=float(losses['total']))
        global_step += 1
    return total / max(1, n), global_step


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    variant = cfg.get("model", {}).get("variant", None)
    assert variant == "tdirect", (
        f"train_vi_tdirect.py requires model.variant=='tdirect' "
        f"in config {args.config}, got {variant!r}")
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
    print(f"[train_vi_tdirect] train={train_tids} val={val_tid} "
          f"minitest={minitest_tid} seed={seed}")

    augment = _maybe_build_augment(cfg)
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

    model = NetworkVI_tdirect(feat_dim=cfg["model"]["cnn_feat_dim"],
                              lstm_hidden=cfg["model"]["lstm_hidden"]).to(device)
    v_state = (load_checkpoint(Path(args.pretrained_v))["model"]
               if args.pretrained_v else None)
    i_state = (load_checkpoint(Path(args.pretrained_i))["model"]
               if args.pretrained_i else None)
    model.load_pretrained(v_state, i_state)

    loss_fn = VILoss_tdirect(
        w_rot=cfg["loss"]["w_rot"], w_t=cfg["loss"]["w_t"],
        w_trans=cfg["loss"]["w_trans"],
        w_imu_trans=cfg["loss"]["w_imu_trans"],
        w_rot_imu=cfg["loss"]["w_rot_imu"],
        hover_center=cfg["loss"].get("hover_mask_center", 0.01),
        hover_width=cfg["loss"].get("hover_mask_width", 0.005)
    ).to(device)

    loss_train = _loss_compute(loss_fn)
    loss_val = _loss_compute(loss_fn)

    best_ate = float("inf")
    step = 0
    global_epoch = 0

    def _checkpoint_and_log(epoch_local: int, stage: str, train_loss: float,
                            val_loss: float,
                            val_ate_traj0: float, val_ate_mini: float,
                            opt) -> None:
        nonlocal best_ate, global_epoch
        lr_now = opt.param_groups[0]["lr"]
        writer.add_scalar(f"epoch/{stage}/train_total", train_loss, epoch_local)
        writer.add_scalar(f"epoch/{stage}/val_loss", val_loss, epoch_local)
        writer.add_scalar(f"epoch/{stage}/val_ATE_traj0", val_ate_traj0, epoch_local)
        writer.add_scalar(f"epoch/{stage}/val_ATE_minitest", val_ate_mini, epoch_local)
        logger.log_epoch(epoch=global_epoch, stage=stage, train_loss=train_loss,
                         val_loss=val_loss,
                         val_ATE_on_traj_0=val_ate_traj0,
                         val_ATE_on_minitest=val_ate_mini, lr=lr_now)
        save_checkpoint(out_dir / f"{stage}_last.pt", model, opt, epoch_local,
                        {"train_loss": train_loss,
                         "val_ATE_traj0": val_ate_traj0,
                         "val_ATE_minitest": val_ate_mini,
                         "stage": stage})
        save_checkpoint(out_dir / "last.pt", model, opt, epoch_local,
                        {"train_loss": train_loss,
                         "val_ATE_traj0": val_ate_traj0,
                         "val_ATE_minitest": val_ate_mini,
                         "stage": stage})
        maybe_save_periodic(lambda p, meta: save_checkpoint(p, model, opt,
                                                             epoch_local, meta),
                            out_dir, global_epoch, every=5,
                            meta={"stage": stage,
                                  "val_ATE_minitest": val_ate_mini})
        if val_ate_mini < best_ate:
            best_ate = val_ate_mini
            save_checkpoint(out_dir / "best.pt", model, opt, epoch_local,
                            {"val_ATE_minitest": val_ate_mini,
                             "val_ATE_traj0": val_ate_traj0,
                             "stage": stage,
                             "note": "best-by-val-ATE-minitest"})
            print(f"[train_vi_tdirect] {stage} epoch {epoch_local}: "
                  f"new best val_ATE_minitest = {val_ate_mini:.4f}")
        global_epoch += 1

    # --- Stage 1: freeze backbones ---
    _set_backbone_requires_grad(model, False)
    stage1_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    opt = torch.optim.AdamW(stage1_params, lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    for epoch in range(cfg["train"]["epochs_vi_stage1"]):
        cosine_warmup_lr(opt, epoch, cfg["train"]["epochs_vi_stage1"],
                         cfg["train"]["warmup_epochs"], cfg["train"]["lr"])
        mean_total, step = _run_epoch(model, loader, loss_train, opt, device,
                                      cfg, writer, epoch, "stage1", step)
        val_loss = compute_val_loss(model, val_loader, loss_val, device)
        val_ate_traj0 = compute_val_ate_on_trajectory(
            model, val_loader, val_ds, _forward_rel, device)
        val_ate_mini = compute_val_ate_on_trajectory(
            model, minitest_loader, minitest_ds, _forward_rel, device)
        _checkpoint_and_log(epoch, "stage1", mean_total, val_loss,
                            val_ate_traj0, val_ate_mini, opt)

    # --- Stage 2: unfreeze, lower backbone lr ---
    _set_backbone_requires_grad(model, True)
    mult = cfg["train"].get("stage2_lr_mult", 0.25)
    head_params = list(model.refiner.parameters())
    backbone_params = (list(model.vision.parameters())
                       + list(model.imu.parameters()))
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["train"]["lr"] * mult, "lr_mult": mult},
        {"params": head_params,     "lr": cfg["train"]["lr"],        "lr_mult": 1.0},
    ], weight_decay=cfg["train"]["weight_decay"])
    for epoch in range(cfg["train"]["epochs_vi_stage2"]):
        cosine_warmup_lr(opt, epoch, cfg["train"]["epochs_vi_stage2"],
                         cfg["train"]["warmup_epochs"], cfg["train"]["lr"])
        mean_total, step = _run_epoch(model, loader, loss_train, opt, device,
                                      cfg, writer, epoch, "stage2", step)
        val_loss = compute_val_loss(model, val_loader, loss_val, device)
        val_ate_traj0 = compute_val_ate_on_trajectory(
            model, val_loader, val_ds, _forward_rel, device)
        val_ate_mini = compute_val_ate_on_trajectory(
            model, minitest_loader, minitest_ds, _forward_rel, device)
        _checkpoint_and_log(epoch, "stage2", mean_total, val_loss,
                            val_ate_traj0, val_ate_mini, opt)

    writer.close()
    print(f"[train_vi_tdirect] done. best val_ATE_on_minitest = {best_ate:.4f}")

    generate_training_plots(
        model=model,
        loaders={"val": (val_loader, val_ds),
                 "minitest": (minitest_loader, minitest_ds)},
        forward_rel=_forward_rel,
        device=device,
        out_plots_dir=out_dir / "plots",
        method="vi",
        tag="final",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
