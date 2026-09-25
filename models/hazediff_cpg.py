import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import ConditionalDiffusionUNet


class FrequencyGuidance(nn.Module):
    def forward(self, x):
        fft = torch.fft.rfft2(x, norm="ortho")
        return torch.abs(fft)


class MaskGuidance(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class PromptGuidance(nn.Module):
    def __init__(self, in_channels=3, prompt_dim=32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim),
        )

    def forward(self, x):
        pooled = self.pool(x).flatten(1)
        return self.proj(pooled)


class HazeDiffCPG(nn.Module):
    """
    Heavy-haze diffusion expert.

    Training returns epsilon prediction and auxiliary guidance.
    Inference uses DDIM-like deterministic sampling.
    """

    def __init__(
        self,
        image_channels=3,
        base_channels=64,
        num_res_blocks=4,
        diffusion_timesteps=1000,
        sampling_steps=50,
        sampler="ddim",
        classifier_free_guidance=True,
        guidance_scale=1.5,
        frequency_guidance=True,
        mask_guidance=True,
        prompt_guidance=True,
        physics_guidance=True,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.T = diffusion_timesteps
        self.sampling_steps = sampling_steps
        self.sampler = sampler
        self.classifier_free_guidance = classifier_free_guidance
        self.guidance_scale = guidance_scale
        self.use_frequency = frequency_guidance
        self.use_mask = mask_guidance
        self.use_prompt = prompt_guidance
        self.physics_guidance = physics_guidance

        cond_channels = image_channels
        if self.use_mask:
            cond_channels += 1

        self.mask_net = MaskGuidance(image_channels)
        self.prompt_net = PromptGuidance(image_channels)

        # The diffusion network conditions on image + optional mask.
        self.diffusion = ConditionalDiffusionUNet(
            image_channels=image_channels,
            cond_channels=cond_channels,
            base_channels=base_channels,
            cond_dim=256,
            num_res_blocks=num_res_blocks,
        )

        betas = torch.linspace(
            1e-4, 0.02, self.T
        )
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
        return xt, noise

    def _conditioning(self, hazy):
        mask = self.mask_net(hazy) if self.use_mask else None

        cond_parts = [hazy]
        if mask is not None:
            cond_parts.append(mask)

        cond = torch.cat(cond_parts, dim=1)
        prompt = self.prompt_net(hazy) if self.use_prompt else None
        freq = FrequencyGuidance()(hazy) if self.use_frequency else None

        return cond, mask, prompt, freq

    def forward(self, hazy, target=None, sample=False):
        if sample or target is None:
            return self.sample(hazy)

        cond, mask, prompt, freq = self._conditioning(hazy)

        b = hazy.shape[0]
        t = torch.randint(
            0, self.T, (b,), device=hazy.device
        )
        xt, noise = self.q_sample(target, t)

        eps = self.diffusion(xt, cond, t)

        # Auxiliary physics estimate.
        transmission = torch.sigmoid(
            2.0 * (1.0 - mask)
        ) if mask is not None else None

        return {
            "restored": None,
            "noise_prediction": eps,
            "noise_target": noise,
            "timestep": t,
            "mask": mask,
            "frequency": freq,
            "prompt": prompt,
            "transmission": transmission,
        }

    @torch.no_grad()
    def sample(self, hazy):
        b, _, h, w = hazy.shape
        cond, mask, prompt, freq = self._conditioning(hazy)

        x = torch.randn_like(hazy)

        # Uniformly spaced reverse schedule.
        steps = torch.linspace(
            self.T - 1,
            0,
            self.sampling_steps,
            device=hazy.device,
        ).long()

        for i, t_scalar in enumerate(steps):
            t = torch.full(
                (b,), int(t_scalar.item()),
                device=hazy.device,
                dtype=torch.long,
            )

            eps = self.diffusion(x, cond, t)

            if self.classifier_free_guidance:
                # The model has no separately trained null branch here.
                # Guidance is therefore kept as a bounded residual scaling.
                eps = eps * min(self.guidance_scale, 2.0)

            ab_t = self.alpha_bar[t].view(-1, 1, 1, 1)
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0 = torch.clamp(x0, 0.0, 1.0)

            if i == len(steps) - 1:
                x = x0
                break

            t_next = torch.full(
                (b,), int(steps[i + 1].item()),
                device=hazy.device,
                dtype=torch.long,
            )
            ab_next = self.alpha_bar[t_next].view(
                -1, 1, 1, 1
            )

            # Deterministic DDIM update (eta=0).
            x = (
                ab_next.sqrt() * x0
                + (1.0 - ab_next).sqrt() * eps
            )

        return {
            "restored": torch.clamp(x, 0.0, 1.0),
            "mask": mask,
            "frequency": freq,
            "prompt": prompt,
        }
