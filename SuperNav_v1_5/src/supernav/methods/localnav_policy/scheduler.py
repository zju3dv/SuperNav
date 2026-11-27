"""Square-cosine DDPM scheduler (matches diffusers' squaredcos_cap_v2 betas).

Kept dependency-light on purpose: the training loop and the serving path must
share the exact same schedule, and vendoring ~60 lines beats dragging the
whole diffusers package into the bridge environment.
"""

from __future__ import annotations

import math

import torch


class SquareCosineNoiseScheduler:
    def __init__(self, num_train_timesteps: int = 10, max_beta: float = 0.999) -> None:
        if num_train_timesteps < 1:
            raise ValueError("num_train_timesteps must be >= 1")
        self.num_train_timesteps = int(num_train_timesteps)

        def f(t: float) -> float:
            return math.cos((t / self.num_train_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = [
            min(1.0 - f(t + 1) / f(t), max_beta) for t in range(self.num_train_timesteps)
        ]
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @property
    def timesteps(self) -> torch.Tensor:
        """Denoising order: T−1 … 0."""
        return torch.arange(self.num_train_timesteps - 1, -1, -1, dtype=torch.long)

    def add_noise(
        self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod.to(sample.device)[timesteps]
        while alpha_bar.dim() < sample.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)
        return alpha_bar.sqrt() * sample + (1.0 - alpha_bar).sqrt() * noise

    def step(
        self,
        model_eps: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: "torch.Generator | None" = None,
    ) -> torch.Tensor:
        """One ancestral DDPM step x_t → x_{t−1} with epsilon prediction."""
        t = int(timestep)
        alpha_t = self.alphas[t].to(sample.device)
        alpha_bar_t = self.alphas_cumprod[t].to(sample.device)
        alpha_bar_prev = (
            self.alphas_cumprod[t - 1].to(sample.device)
            if t > 0
            else torch.tensor(1.0, device=sample.device)
        )
        beta_t = self.betas[t].to(sample.device)

        x0 = (sample - (1.0 - alpha_bar_t).sqrt() * model_eps) / alpha_bar_t.sqrt()
        x0 = x0.clamp(-1.0, 1.0)  # actions are trained in normalized [-1, 1]

        mean = (
            alpha_bar_prev.sqrt() * beta_t / (1.0 - alpha_bar_t) * x0
            + alpha_t.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * sample
        )
        if t == 0:
            return mean
        variance = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
        noise = torch.randn(
            sample.shape, device=sample.device, dtype=sample.dtype, generator=generator
        )
        return mean + variance.clamp(min=1e-20).sqrt() * noise
