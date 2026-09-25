import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml

from models.mhformer import MHFormer
from losses.reconstruction import ReconstructionLoss


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
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["project"].get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )

    # Import your dataset implementation here.
    from datasets.reside_its import RESIDEITS

    train_set = RESIDEITS(
        cfg["dataset"]["train"]["hazy_dir"],
        cfg["dataset"]["train"]["gt_dir"],
        crop_size=cfg["dataset"]["crop_size"],
    )
    loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["stage1"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=cfg["dataset"]["pin_memory"],
    )

    mcfg = cfg["mhformer"]
    model = MHFormer(
        in_channels=3,
        embed_dim=mcfg["embed_dim"],
        num_heads=mcfg["num_heads"],
        num_blocks=mcfg["num_blocks"],
        mlp_ratio=mcfg["mlp_ratio"],
        dropout=mcfg["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["stage1"]["learning_rate"],
        betas=tuple(cfg["training"]["optimizer"]["betas"]),
        weight_decay=cfg["training"]["optimizer"]["weight_decay"],
    )
    criterion = ReconstructionLoss()

    epochs = args.epochs or cfg["training"]["stage1"]["epochs"]

    for epoch in range(epochs):
        model.train()
        running = 0.0

        for batch in loader:
            hazy, gt = unpack_batch(batch, device)
            out = model(hazy)
            loss = criterion(out["restored"], gt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["training"]["gradient_clipping"]["max_norm"],
            )
            opt.step()
            running += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"loss={running/max(len(loader),1):.6f}"
        )

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(
        model.state_dict(),
        "checkpoints/mhformer_pretrained.pth",
    )


if __name__ == "__main__":
    main()
