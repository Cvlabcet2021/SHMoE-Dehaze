import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam3_encoder import SAM3Encoder
from .haze_encoder import HazeEncoder
from .routing import SemanticHazeRouting
from .gating import PixelWiseGating
from .asm_expert import ASMExpert
from .mhformer import MHFormer
from .hazediff_cpg import HazeDiffCPG
from .moe_fusion import SoftMoEFusion


class SHMoEDehaze(nn.Module):
    """
    Complete Semantic and Haze-aware Mixture-of-Experts model.

    SAM-3 is frozen.
    HazeEncoder, routing, gating and all experts are trainable.
    """

    def __init__(
        self,
        sam3_model=None,
        sam3_feature_dim=256,
        haze_base_channels=32,
        haze_embed_dim=128,
        cross_embed_dim=256,
        cross_heads=8,
        cross_dropout=0.1,
        routing_hidden_dim=128,
        routing_temperature=1.0,
        asm_base_channels=64,
        asm_blocks=6,
        mh_embed_dim=256,
        mh_heads=8,
        mh_blocks=4,
        mh_mlp_ratio=4.0,
        mh_dropout=0.1,
        diffusion_base_channels=64,
        diffusion_res_blocks=4,
        diffusion_timesteps=1000,
        diffusion_sampling_steps=50,
        guidance_scale=1.5,
    ):
        super().__init__()

        self.sam3_encoder = SAM3Encoder(
            sam3_model=sam3_model,
            feature_dim=sam3_feature_dim,
            output_dim=cross_embed_dim,
        )

        self.haze_encoder = HazeEncoder(
            in_channels=3,
            base_channels=haze_base_channels,
            embed_dim=haze_embed_dim,
        )

        self.routing = SemanticHazeRouting(
            semantic_dim=cross_embed_dim,
            haze_dim=haze_embed_dim,
            embed_dim=cross_embed_dim,
            num_heads=cross_heads,
            dropout=cross_dropout,
        )

        self.gating = PixelWiseGating(
            in_channels=cross_embed_dim,
            hidden_dim=routing_hidden_dim,
            num_experts=3,
            temperature=routing_temperature,
            use_softmax=True,
        )

        self.asm = ASMExpert(
            in_channels=3,
            base_channels=asm_base_channels,
            num_blocks=asm_blocks,
        )

        self.mhformer = MHFormer(
            in_channels=3,
            embed_dim=mh_embed_dim,
            num_heads=mh_heads,
            num_blocks=mh_blocks,
            mlp_ratio=mh_mlp_ratio,
            dropout=mh_dropout,
        )

        self.hazediff = HazeDiffCPG(
            image_channels=3,
            base_channels=diffusion_base_channels,
            num_res_blocks=diffusion_res_blocks,
            diffusion_timesteps=diffusion_timesteps,
            sampling_steps=diffusion_sampling_steps,
            guidance_scale=guidance_scale,
        )

        self.fusion = SoftMoEFusion(
            num_experts=3,
            output_channels=3,
        )

    def forward(self, hazy, target=None, diffusion_sample=False):
        semantic = self.sam3_encoder(hazy)
        haze = self.haze_encoder(hazy)

        routed = self.routing(semantic, haze)
        logits, probabilities = self.gating(routed)

        asm_out = self.asm(hazy)
        mh_out = self.mhformer(hazy)

        if self.training and target is not None and not diffusion_sample:
            diff_out = self.hazediff(
                hazy, target=target, sample=False
            )
            # A clean image is required for diffusion training.
            # For MoE fusion during training, use the current
            # HazeDiff reconstruction only when available.
            diff_image = (
                diff_out["restored"]
                if diff_out["restored"] is not None
                else hazy
            )
        else:
            diff_out = self.hazediff(
                hazy, sample=True
            )
            diff_image = diff_out["restored"]

        expert_images = [
            asm_out["restored"],
            mh_out["restored"],
            diff_image,
        ]

        fused = self.fusion(
            expert_images,
            probabilities,
        )

        return {
            "restored": fused["restored"],
            "weighted_sum": fused["weighted_sum"],
            "semantic_features": semantic,
            "haze_features": haze,
            "routed_features": routed,
            "routing_logits": logits,
            "routing_probs": probabilities,
            "asm": asm_out,
            "mhformer": mh_out,
            "hazediff": diff_out,
        }


def build_model_from_config(cfg, sam3_model=None):
    s = cfg["sam3"]
    h = cfg["haze_encoder"]
    ca = cfg["cross_attention"]
    r = cfg["routing"]
    a = cfg["asm"]
    m = cfg["mhformer"]
    d = cfg["hazediff_cpg"]

    return SHMoEDehaze(
        sam3_model=sam3_model,
        sam3_feature_dim=s.get("feature_dim", 256),
        haze_base_channels=h.get("base_channels", 32),
        haze_embed_dim=128,
        cross_embed_dim=ca.get("embed_dim", 256),
        cross_heads=ca.get("num_heads", 8),
        cross_dropout=ca.get("dropout", 0.1),
        routing_hidden_dim=r.get("hidden_dim", 128),
        routing_temperature=r.get("temperature", 1.0),
        asm_base_channels=a.get("base_channels", 64),
        asm_blocks=a.get("num_blocks", 6),
        mh_embed_dim=m.get("embed_dim", 256),
        mh_heads=m.get("num_heads", 8),
        mh_blocks=m.get("num_blocks", 4),
        mh_mlp_ratio=m.get("mlp_ratio", 4.0),
        mh_dropout=m.get("dropout", 0.1),
        diffusion_base_channels=d.get("base_channels", 64),
        diffusion_res_blocks=d.get("num_res_blocks", 4),
        diffusion_timesteps=d.get("diffusion_timesteps", 1000),
        diffusion_sampling_steps=d.get("sampling_steps", 50),
        guidance_scale=d.get("guidance_scale", 1.5),
    )
