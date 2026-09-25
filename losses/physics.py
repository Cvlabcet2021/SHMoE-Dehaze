import torch
import torch.nn as nn
import torch.nn.functional as F


class AtmosphericConsistencyLoss(nn.Module):
    """
    Enforces:
        I ≈ J*t + A*(1-t)
    """

    def forward(self, hazy, restored, transmission, atmospheric):
        if atmospheric.ndim == 2:
            atmospheric = atmospheric[:, :, None, None]
        reconstruction = (
            restored * transmission
            + atmospheric * (1.0 - transmission)
        )
        return F.l1_loss(reconstruction, hazy)


class RoutingBalanceLoss(nn.Module):
    """
    L_balance = sum_k (mean(P_k) - 1/K)^2
    """

    def __init__(self, num_experts=3):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, probabilities):
        mean_p = probabilities.mean(dim=(0, 2, 3))
        target = torch.full_like(
            mean_p, 1.0 / self.num_experts
        )
        return torch.sum((mean_p - target) ** 2)


class RoutingSmoothnessLoss(nn.Module):
    """
    Spatial routing smoothness.
    """

    def forward(self, probabilities):
        dx = probabilities[:, :, :, 1:] - probabilities[:, :, :, :-1]
        dy = probabilities[:, :, 1:, :] - probabilities[:, :, :-1, :]
        return dx.abs().mean() + dy.abs().mean()
