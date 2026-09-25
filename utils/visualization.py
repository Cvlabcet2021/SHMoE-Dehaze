from pathlib import Path
import torch
from torchvision.utils import save_image


@torch.no_grad()
def save_comparison(
    hazy,
    restored,
    target,
    path,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = torch.cat(
        [
            hazy[:1].cpu(),
            restored[:1].cpu(),
            target[:1].cpu(),
        ],
        dim=0,
    )
    save_image(
        images,
        path,
        nrow=3,
        normalize=False,
    )


@torch.no_grad()
def save_routing_map(probabilities, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the three routing channels independently.
    for k in range(probabilities.shape[1]):
        save_image(
            probabilities[:, k:k+1].cpu(),
            path.with_name(
                f"{path.stem}_expert{k+1}{path.suffix}"
            ),
            normalize=True,
        )
