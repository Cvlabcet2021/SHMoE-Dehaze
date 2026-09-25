import math
import torch
import torch.nn.functional as F


def _to_y(x):
    return (
        0.299 * x[:, 0:1]
        + 0.587 * x[:, 1:2]
        + 0.114 * x[:, 2:3]
    )


@torch.no_grad()
def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


@torch.no_grad()
def ssim(pred, target):
    # Same compact SSIM formulation used by ReconstructionLoss.
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu1 = F.avg_pool2d(pred, 11, 1, 5)
    mu2 = F.avg_pool2d(target, 11, 1, 5)
    s1 = F.avg_pool2d(pred * pred, 11, 1, 5) - mu1 * mu1
    s2 = F.avg_pool2d(target * target, 11, 1, 5) - mu2 * mu2
    s12 = F.avg_pool2d(pred * target, 11, 1, 5) - mu1 * mu2
    value = (
        (2 * mu1 * mu2 + c1)
        * (2 * s12 + c2)
        /
        ((mu1 * mu1 + mu2 * mu2 + c1)
         * (s1 + s2 + c2))
    )
    return value.mean().item()


def batch_metrics(pred, target):
    return {
        "PSNR": psnr(pred, target),
        "SSIM": ssim(pred, target),
    }
