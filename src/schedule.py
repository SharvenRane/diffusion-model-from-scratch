"""Noise schedules and the precomputed quantities a DDPM needs.

A diffusion process gradually corrupts a clean sample x_0 into pure noise over
T steps. The amount of noise added at each step is controlled by a variance
schedule beta_1 .. beta_T. From those betas we precompute the running products
of alpha = 1 - beta, which let us jump directly to any timestep with a closed
form instead of looping.
"""

import math

import torch


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """The original DDPM linear schedule of betas from beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64).float()


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """The improved DDPM cosine schedule (Nichol and Dhariwal).

    We build the cumulative product of alphas from a cosine curve and then
    recover the per step betas by taking ratios. Betas are clamped to keep the
    process numerically stable.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


def _extract(values: torch.Tensor, t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Gather per sample entries from a 1D buffer and broadcast to image shape.

    values has shape (T,). t has shape (batch,) and holds the timestep for each
    sample. The result has shape (batch, 1, 1, 1) so it multiplies cleanly
    against a (batch, C, H, W) tensor.
    """
    batch = t.shape[0]
    out = values.to(t.device).gather(0, t)
    return out.reshape(batch, *([1] * (len(target_shape) - 1)))


class NoiseSchedule:
    """Holds the betas and every derived quantity used by the forward process.

    All buffers have shape (T,). Indexing them by a timestep t gives the scalar
    constants for the closed form q(x_t | x_0):

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    """

    def __init__(self, betas: torch.Tensor):
        if betas.ndim != 1:
            raise ValueError("betas must be a 1D tensor")
        if torch.any(betas <= 0) or torch.any(betas >= 1):
            raise ValueError("betas must lie strictly in (0, 1)")

        self.betas = betas
        self.timesteps = betas.shape[0]

        alphas = 1.0 - betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas.dtype), self.alphas_cumprod[:-1]]
        )

        # Coefficients for the forward sample.
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Coefficients for the reverse (posterior) step used in sampling.
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device) -> "NoiseSchedule":
        for name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas",
            "posterior_variance",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Draw x_t from q(x_t | x_0) in closed form, no looping over steps."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_ab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_ab * x_0 + sqrt_one_minus_ab * noise
