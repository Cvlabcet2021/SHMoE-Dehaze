import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConvBlock(nn.Module):
    def __init__(self, channels, expansion=2, dropout=0.0):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class WindowMHSA(nn.Module):
    """
    Windowed multi-head self-attention.
    Windows are partitioned locally, avoiding global 512x512 attention.
    """

    def __init__(self, dim, num_heads=8, window_size=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        ph = (ws - h % ws) % ws
        pw = (ws - w % ws) % ws
        xp = F.pad(x, (0, pw, 0, ph))
        hp, wp = xp.shape[-2:]

        windows = xp.unfold(2, ws, ws).unfold(3, ws, ws)
        windows = windows.permute(0, 2, 3, 4, 5, 1)
        windows = windows.reshape(-1, ws * ws, c)

        q = self.norm(windows)
        y, _ = self.attn(q, q, q, need_weights=False)
        y = windows + self.proj(y)

        y = y.reshape(b, hp // ws, wp // ws, ws, ws, c)
        y = y.permute(0, 5, 1, 3, 2, 4)
        y = y.reshape(b, c, hp, wp)

        return y[:, :, :h, :w]


class SelectiveKernelFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.a = nn.Conv2d(channels, channels, 3, padding=1)
        self.b = nn.Conv2d(channels, channels, 5, padding=2)
        self.attn = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        weights = self.attn(torch.cat([a, b], dim=1))
        return a * weights[:, 0:1] + b * weights[:, 1:2]


class MHFormerBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        window_size=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.dw = DWConvBlock(channels, 2, dropout)
        self.attn = WindowMHSA(
            channels, num_heads, window_size, dropout
        )
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1),
        )
        self.sk = SelectiveKernelFusion(channels)

    def forward(self, x):
        x = self.dw(x)
        x = x + self.attn(x)
        x = x + self.sk(x)
        x = x + self.mlp(x)
        return x


class MHFormer(nn.Module):
    """
    Hierarchical medium-haze restoration expert.

    The K/B parameterization follows:
        J = K * I + B
    """

    def __init__(
        self,
        in_channels=3,
        embed_dim=256,
        num_heads=8,
        num_blocks=4,
        mlp_ratio=4.0,
        dropout=0.1,
        output_channels=3,
    ):
        super().__init__()
        self.stem = nn.Conv2d(
            in_channels, embed_dim, 3, padding=1
        )
        self.blocks = nn.Sequential(*[
            MHFormerBlock(
                embed_dim,
                num_heads,
                window_size=8,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.GroupNorm(16, embed_dim)
        self.k_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, output_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.b_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, output_channels, 3, padding=1),
        )

    def forward(self, x):
        feat = self.stem(x)
        feat = self.blocks(feat)
        feat = self.out_norm(feat)
        K = self.k_head(feat)
        B = self.b_head(feat)
        J = torch.clamp(K * x + B, 0.0, 1.0)
        return {
            "restored": J,
            "K": K,
            "B": B,
            "features": feat,
        }
