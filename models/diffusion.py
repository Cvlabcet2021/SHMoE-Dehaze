import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(
            torch.arange(half, device=t.device) * -scale
        )
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class FiLM(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.to_scale_shift = nn.Linear(
            cond_dim, channels * 2
        )

    def forward(self, x, cond):
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1.0 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1
        )
        self.norm1 = nn.GroupNorm(
            min(8, out_channels), out_channels
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )
        self.norm2 = nn.GroupNorm(
            min(8, out_channels), out_channels
        )
        self.film = FiLM(out_channels, cond_dim)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, cond):
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.silu(y)
        y = self.film(y, cond)
        y = self.conv2(y)
        y = self.norm2(y)
        y = F.silu(y)
        return y + self.skip(x)


class ConditionalDiffusionUNet(nn.Module):
    """
    Compact conditional epsilon predictor.

    Input:
        x_t: noisy clean image
        cond: concatenated conditioning channels
        t: diffusion timestep
    """

    def __init__(
        self,
        image_channels=3,
        cond_channels=3,
        base_channels=64,
        cond_dim=256,
        num_res_blocks=4,
    ):
        super().__init__()
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

        self.in_conv = nn.Conv2d(
            image_channels + cond_channels,
            base_channels,
            3, padding=1
        )

        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(
                ResBlock(
                    base_channels,
                    base_channels,
                    cond_dim,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.down = nn.Conv2d(
            base_channels, base_channels * 2, 4, stride=2, padding=1
        )
        self.mid = ResBlock(
            base_channels * 2,
            base_channels * 2,
            cond_dim,
        )
        self.up = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            4, stride=2, padding=1
        )
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, 3, padding=1),
        )

    def forward(self, x_t, cond, t):
        if cond.shape[-2:] != x_t.shape[-2:]:
            cond = F.interpolate(
                cond, size=x_t.shape[-2:],
                mode="bilinear", align_corners=False
            )

        emb = self.time(t)
        x = torch.cat([x_t, cond], dim=1)
        x = self.in_conv(x)

        for block in self.blocks:
            x = block(x, emb)

        x = self.down(x)
        x = self.mid(x, emb)
        x = self.up(x)

        return self.out(x)
