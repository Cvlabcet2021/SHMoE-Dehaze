import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """Semantic-query / haze-key-value cross attention."""

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, q, kv):
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        q = q + out
        q = q + self.ffn(q)
        return q


class SemanticHazeRouting(nn.Module):
    """
    Cross-attention adapter.

    SAM-3 features are expected at 256 channels.
    The supplied HazeEncoder produces 128 channels, so haze features
    are projected to 256 internally.

    Attention is performed on a compact grid to avoid O((HW)^2)
    memory at 512x512 resolution.
    """

    def __init__(
        self,
        semantic_dim=256,
        haze_dim=128,
        embed_dim=256,
        num_heads=8,
        dropout=0.1,
        attention_size=32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_size = attention_size

        self.semantic_proj = (
            nn.Identity()
            if semantic_dim == embed_dim
            else nn.Conv2d(semantic_dim, embed_dim, 1)
        )
        self.haze_proj = (
            nn.Identity()
            if haze_dim == embed_dim
            else nn.Sequential(
                nn.Conv2d(haze_dim, embed_dim, 1, bias=False),
                nn.GroupNorm(8, embed_dim),
                nn.GELU(),
            )
        )

        self.cross_attn = CrossAttentionBlock(
            embed_dim, num_heads, dropout
        )
        self.out_norm = nn.GroupNorm(8, embed_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )

    def forward(self, semantic_features, haze_features):
        target_size = haze_features.shape[-2:]

        semantic_features = self.semantic_proj(semantic_features)
        haze_features = self.haze_proj(haze_features)

        semantic_features = F.interpolate(
            semantic_features,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # Compact attention grid.
        ah = min(self.attention_size, target_size[0])
        aw = min(self.attention_size, target_size[1])
        sem_small = F.adaptive_avg_pool2d(
            semantic_features, (ah, aw)
        )
        haze_small = F.adaptive_avg_pool2d(
            haze_features, (ah, aw)
        )

        b, c, h, w = sem_small.shape
        q = sem_small.flatten(2).transpose(1, 2)
        kv = haze_small.flatten(2).transpose(1, 2)

        fused = self.cross_attn(q, kv)
        fused = fused.transpose(1, 2).reshape(b, c, h, w)

        fused = F.interpolate(
            fused,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        fused = self.out_norm(fused)
        fused = fused + self.refine(fused)

        return fused
