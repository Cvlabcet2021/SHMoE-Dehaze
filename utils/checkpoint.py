from pathlib import Path
import torch


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    epoch=0,
    best_metric=None,
    scaler=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "best_metric": best_metric,
    }

    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    torch.save(state, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
):
    ckpt = torch.load(
        path,
        map_location=map_location,
    )
    model.load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt
