import torch
import torch.nn as nn
import torch.nn.functional as F

from .reconstruction import ReconstructionLoss
from .physics import (
    AtmosphericConsistencyLoss,
    RoutingBalanceLoss,
    RoutingSmoothnessLoss,
)


class TotalSHMoELoss(nn.Module):
    """
    Practical implementation of the paper's combined objective.

    Main terms:
      reconstruction
      perceptual
      physics
      diffusion epsilon loss
      routing balance
      routing smoothness
    """

    def __init__(
        self,
        reconstruction_weight=1.0,
        perceptual_weight=0.1,
        physics_weight=0.1,
        routing_weight=0.01,
        diffusion_weight=1.0,
        use_perceptual=True,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.physics_weight = physics_weight
        self.routing_weight = routing_weight
        self.diffusion_weight = diffusion_weight

        self.reconstruction = ReconstructionLoss()
        self.physics = AtmosphericConsistencyLoss()
        self.balance = RoutingBalanceLoss(3)
        self.smoothness = RoutingSmoothnessLoss()

        self.perceptual = None
        if use_perceptual:
            try:
                from .perceptual import PerceptualLoss
                self.perceptual = PerceptualLoss()
            except Exception:
                self.perceptual = None

    def forward(self, outputs, hazy, target):
        pred = outputs["restored"]

        l_recon = self.reconstruction(pred, target)

        if self.perceptual is not None:
            l_perc = self.perceptual(pred, target)
        else:
            l_perc = torch.zeros(
                (), device=pred.device, dtype=pred.dtype
            )

        asm = outputs["asm"]
        l_phys = self.physics(
            hazy,
            pred,
            asm["transmission"],
            asm["atmospheric_light"],
        )

        p = outputs["routing_probs"]
        l_balance = self.balance(p)
        l_smooth = self.smoothness(p)
        l_route = l_balance + l_smooth

        diff = outputs["hazediff"]
        if (
            diff.get("noise_prediction") is not None
            and diff.get("noise_target") is not None
        ):
            l_diff = F.mse_loss(
                diff["noise_prediction"],
                diff["noise_target"],
            )
        else:
            l_diff = torch.zeros(
                (), device=pred.device, dtype=pred.dtype
            )

        total = (
            self.reconstruction_weight * l_recon
            + self.perceptual_weight * l_perc
            + self.physics_weight * l_phys
            + self.routing_weight * l_route
            + self.diffusion_weight * l_diff
        )

        return {
            "total": total,
            "reconstruction": l_recon.detach(),
            "perceptual": l_perc.detach(),
            "physics": l_phys.detach(),
            "routing": l_route.detach(),
            "diffusion": l_diff.detach(),
        }
