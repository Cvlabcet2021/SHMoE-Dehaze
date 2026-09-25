import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelWiseGating(nn.Module):
    """
    Produces three spatial routing probabilities:
      P1 -> ASM
      P2 -> MH-Former
      P3 -> HazeDiff-CPG
    """

    def __init__(
        self,
        in_channels=256,
        hidden_dim=128,
        num_experts=3,
        temperature=1.0,
        use_softmax=True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.use_softmax = use_softmax

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_experts, 1),
        )

    def forward(self, features):
        logits = self.net(features)

        if self.use_softmax:
            probs = F.softmax(
                logits / max(self.temperature, 1e-6),
                dim=1,
            )
        else:
            probs = torch.sigmoid(logits)

        return logits, probs


# Alias matching common naming.
GatingNetwork = PixelWiseGating
