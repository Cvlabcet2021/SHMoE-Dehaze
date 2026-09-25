import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml

from models.shmoe_dehaze import build_model_from_config
from losses.total_loss import TotalSHMoELoss
from utils.checkpoint import save_checkpoint
from utils.metrics import psnr


def unpack_batch(batch, device):
    if isinstance(batch, dict):
        hazy = batch.get("hazy", batch.get("input"))
        gt = batch.get("gt", batch.get("target", batch.get("clean")))
    else:
        hazy, gt = batch[:2]
    return hazy.to(device), gt.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/shmoe_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["project"]["seed"])

    device = torch.device(
        cfg["project"].get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )

    from datasets.reside_its import RESIDEITS

    train_set = RESIDEITS(
        cfg["dataset"]["train"]["hazy_dir"],
        cfg["dataset"]["train"]["gt_dir"],
        crop_size=cfg["dataset"]["crop_size"],
    )
    loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["stage2"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=cfg["dataset"]["pin_memory"],
    )

    # Replace this with your official SAM-3 loader.
    sam3_model = None

    model = build_model_from_config(
        cfg,
        sam3_model=sam3_model,
    ).to(device)

    # This check gives a clear message instead of silently training
    # without the semantic branch.
    if cfg["sam3"]["enabled"] and sam3_model is None:
        raise RuntimeError(
            "SAM-3 is enabled but no official SAM-3 model was supplied. "
            "Load the checkpoint/API and pass the model to build_model_from_config()."
        )

    # Optional stage-1 initialization.
    mh_path = Path("checkpoints/mhformer_pretrained.pth")
    if mh_path.exists():
        model.mhformer.load_state_dict(
            torch.load(mh_path, map_location="cpu")
        )

    diff_path = Path("checkpoints/hazediff_pretrained.pth")
    if diff_path.exists():
        model.hazediff.load_state_dict(
            torch.load(diff_path, map_location="cpu"),
            strict=False,
        )

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["stage2"]["learning_rate"],
        betas=tuple(cfg["training"]["optimizer"]["betas"]),
        weight_decay=cfg["training"]["optimizer"]["weight_decay"],
    )

    epochs = cfg["training"]["stage2"]["epochs"]
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg["training"]["mixed_precision"]["enabled"]
        and device.type == "cuda",
    )

    criterion = TotalSHMoELoss(
        reconstruction_weight=cfg["loss"]["reconstruction"]["weight"],
        perceptual_weight=cfg["loss"]["perceptual"]["weight"],
        physics_weight=cfg["loss"]["physics"]["weight"],
        routing_weight=cfg["loss"]["routing"]["weight"],
        diffusion_weight=cfg["loss"]["diffusion"]["weight"],
    ).to(device)

    best_psnr = -float("inf")

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for batch in loader:
            hazy, gt = unpack_batch(batch, device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(
                    hazy,
                    target=gt,
                    diffusion_sample=False,
                )
                losses = criterion(outputs, hazy, gt)
                loss = losses["total"]

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["training"]["gradient_clipping"]["max_norm"],
            )
            scaler.step(opt)
            scaler.update()

            total += loss.item()

        avg_loss = total / max(len(loader), 1)
        print(
            f"Epoch {epoch+1}/{epochs} "
            f"loss={avg_loss:.6f}"
        )

        if (epoch + 1) % cfg["checkpoint"]["save_every"] == 0:
            save_checkpoint(
                f'{cfg["checkpoint"]["directory"]}/epoch_{epoch+1}.pth',
                model,
                optimizer=opt,
                epoch=epoch + 1,
                best_metric=best_psnr,
            )

    save_checkpoint(
        f'{cfg["checkpoint"]["directory"]}/final.pth',
        model,
        optimizer=opt,
        epoch=epochs,
        best_metric=best_psnr,
    )


if __name__ == "__main__":
    main()
