"""The DDPM itself: training objective and ancestral sampling.

Training follows the simple noise prediction objective from Ho et al. We pick a
random timestep for each image, corrupt it with known noise using the closed
form, ask the network to predict that noise, and minimise the mean squared
error. Sampling runs the learned reverse process from pure noise back to a
clean image one step at a time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedule import NoiseSchedule, _extract


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, schedule: NoiseSchedule):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.timesteps = schedule.timesteps

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        device = None
        for arg in args:
            if isinstance(arg, (str, torch.device)):
                device = arg
        if device is not None:
            out.schedule.to(torch.device(device))
        return out

    def sample_timesteps(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch,), device=device, dtype=torch.long)

    def loss(self, x_0: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Simple noise prediction MSE objective."""
        device = x_0.device
        if t is None:
            t = self.sample_timesteps(x_0.shape[0], device)
        noise = torch.randn_like(x_0)
        x_t = self.schedule.q_sample(x_0, t, noise)
        predicted = self.model(x_t, t)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One reverse step: estimate x_{t-1} from x_t."""
        sch = self.schedule
        betas_t = _extract(sch.betas, t, x_t.shape)
        sqrt_one_minus_ab = _extract(sch.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas = _extract(sch.sqrt_recip_alphas, t, x_t.shape)

        predicted_noise = self.model(x_t, t)
        mean = sqrt_recip_alphas * (x_t - betas_t / sqrt_one_minus_ab * predicted_noise)

        posterior_var = _extract(sch.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        # No noise is added at the final step (t == 0).
        nonzero = (t != 0).float().reshape(-1, *([1] * (x_t.ndim - 1)))
        return mean + nonzero * torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(self, shape: tuple[int, ...], device: torch.device | None = None) -> torch.Tensor:
        """Ancestral sampling from pure noise to a clean image."""
        if device is None:
            device = next(self.model.parameters()).device
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x
