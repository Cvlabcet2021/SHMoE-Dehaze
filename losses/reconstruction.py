import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(
            torch.sqrt((pred - target) ** 2 + self.eps ** 2)
        )


class ReconstructionLoss(nn.Module):
    def __init__(self, charbonnier_weight=1.0, ssim_weight=0.2):
        super().__init__()
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.charbonnier = CharbonnierLoss()

    @staticmethod
    def _ssim_loss(pred, target, window=11):
        # Compact differentiable SSIM implementation.
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu1 = F.avg_pool2d(
            pred, window, 1, window // 2
        )
        mu2 = F.avg_pool2d(
            target, window, 1, window // 2
        )
        sigma1 = F.avg_pool2d(
            pred * pred, window, 1, window // 2
        ) - mu1 * mu1
        sigma2 = F.avg_pool2d(
            target * target, window, 1, window // 2
        ) - mu2 * mu2
        sigma12 = F.avg_pool2d(
            pred * target, window, 1, window // 2
        ) - mu1 * mu2

        ssim = (
            (2 * mu1 * mu2 + c1)
            * (2 * sigma12 + c2)
            /
            ((mu1 * mu1 + mu2 * mu2 + c1)
             * (sigma1 + sigma2 + c2))
        )
        return 1.0 - ssim.mean()

    def forward(self, pred, target):
        l1 = self.charbonnier(pred, target)
        ssim = self._ssim_loss(pred, target)
        return (
            self.charbonnier_weight * l1
            + self.ssim_weight * ssim
        )
