import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMoEFusion(nn.Module):
    """
    J = P1*J1 + P2*J2 + P3*J3
    """

    def __init__(
        self,
        num_experts=3,
        output_channels=3,
        refine=True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.refine = (
            nn.Sequential(
                nn.Conv2d(output_channels, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, output_channels, 3, padding=1),
            )
            if refine else nn.Identity()
        )

    def forward(self, expert_outputs, probabilities):
        if len(expert_outputs) != self.num_experts:
            raise ValueError(
                f"Expected {self.num_experts} expert outputs."
            )

        target_size = expert_outputs[0].shape[-2:]
        outputs = [
            F.interpolate(
                x, size=target_size,
                mode="bilinear", align_corners=False
            ) if x.shape[-2:] != target_size else x
            for x in expert_outputs
        ]

        if probabilities.shape[-2:] != target_size:
            probabilities = F.interpolate(
                probabilities,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        fused = torch.zeros_like(outputs[0])
        for k, output in enumerate(outputs):
            fused = fused + probabilities[:, k:k+1] * output

        refined = torch.clamp(
            fused + self.refine(fused),
            0.0, 1.0
        )

        return {
            "restored": refined,
            "weighted_sum": fused,
        }
