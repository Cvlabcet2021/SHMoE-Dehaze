import torch
import torch.nn as nn
import torch.nn.functional as F


class ASMFeatureBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.block(x)


class TransmissionEstimator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_blocks=6):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
        ]
        for _ in range(num_blocks):
            layers.append(ASMFeatureBlock(base_channels))
        layers += [
            nn.Conv2d(base_channels, 1, 3, padding=1),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AtmosphericLightEstimator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Global atmospheric-light estimate.
        a_map = self.net(x)
        return F.adaptive_avg_pool2d(a_map, 1)


class ASMExpert(nn.Module):
    """
    Atmospheric Scattering Model expert.

    J = (I - A) / max(t, t0) + A
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        num_blocks=6,
        output_channels=3,
        t0=0.12,
    ):
        super().__init__()
        self.t0 = t0
        self.transmission = TransmissionEstimator(
            in_channels, base_channels, num_blocks
        )
        self.atmospheric = AtmosphericLightEstimator(
            in_channels, base_channels
        )
        self.output = nn.Conv2d(
            in_channels, output_channels, 1
        )

    def forward(self, x):
        t = self.transmission(x)
        A = self.atmospheric(x)
        J = (x - A) / torch.clamp(t, min=self.t0) + A
        J = torch.clamp(self.output(J), 0.0, 1.0)
        return {
            "restored": J,
            "transmission": t,
            "atmospheric_light": A,
        }
