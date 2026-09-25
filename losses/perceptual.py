import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    """
    LPIPS wrapper. Requires the 'lpips' package.
    """

    def __init__(self, net="alex"):
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "Install lpips with: pip install lpips"
            ) from exc

        self.loss_fn = lpips.LPIPS(net=net)
        for p in self.loss_fn.parameters():
            p.requires_grad = False
        self.loss_fn.eval()

    def forward(self, pred, target):
        # LPIPS expects [-1,1].
        pred = pred * 2.0 - 1.0
        target = target * 2.0 - 1.0
        return self.loss_fn(pred, target).mean()
