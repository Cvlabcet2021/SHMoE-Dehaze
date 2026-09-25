import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import yaml

from models.shmoe_dehaze import build_model_from_config
from utils.checkpoint import load_checkpoint


def load_image(path, size=512):
    image = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return tfm(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/shmoe_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/dehazed.png")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["project"].get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )

    # Load your official SAM-3 model here.
    sam3_model = None

    if cfg["sam3"]["enabled"] and sam3_model is None:
        raise RuntimeError(
            "SAM-3 is enabled. Connect the official SAM-3 loader "
            "before inference."
        )

    model = build_model_from_config(
        cfg,
        sam3_model=sam3_model,
    ).to(device)

    load_checkpoint(
        args.checkpoint,
        model,
        map_location=device,
    )

    model.eval()

    image = load_image(
        args.input,
        cfg["inference"]["image_size"],
    ).to(device)

    with torch.no_grad():
        output = model(
            image,
            target=None,
            diffusion_sample=True,
        )["restored"]

    Path(args.output).parent.mkdir(
        parents=True, exist_ok=True
    )
    transforms.ToPILImage()(output[0].cpu().clamp(0, 1)).save(
        args.output
    )

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
