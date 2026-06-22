"""Behavior checks for the DDPM training objective and sampling."""

import torch

from src.schedule import NoiseSchedule, linear_beta_schedule
from src.unet import UNet
from src.ddpm import DDPM


def build(T=50):
    sch = NoiseSchedule(linear_beta_schedule(T))
    net = UNet(in_channels=1, base_channels=16, channel_mults=(1, 2))
    return DDPM(net, sch)


def test_loss_is_scalar_and_finite():
    ddpm = build()
    x0 = torch.randn(4, 1, 16, 16)
    loss = ddpm.loss(x0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_decreases_when_overfitting_one_batch():
    """Train on a single fixed tiny batch and confirm the loss falls.

    We hold the timesteps and target noise fixed across iterations so the only
    thing changing is the network, which gives a clean signal that learning is
    happening rather than schedule sampling noise.
    """
    torch.manual_seed(0)
    ddpm = build(T=50)
    x0 = torch.randn(4, 1, 16, 16)
    t = torch.tensor([1, 10, 25, 40])
    noise = torch.randn_like(x0)
    x_t = ddpm.schedule.q_sample(x0, t, noise)

    opt = torch.optim.Adam(ddpm.parameters(), lr=1e-3)

    def step():
        opt.zero_grad()
        pred = ddpm.model(x_t, t)
        loss = torch.nn.functional.mse_loss(pred, noise)
        loss.backward()
        opt.step()
        return loss.item()

    first = step()
    last = first
    for _ in range(150):
        last = step()

    assert last < first
    # On a fixed batch a working model drives the MSE down substantially.
    assert last < 0.5 * first


def test_sample_returns_correct_shape():
    torch.manual_seed(0)
    ddpm = build(T=10)
    out = ddpm.sample((2, 1, 16, 16))
    assert out.shape == (2, 1, 16, 16)
    assert torch.all(torch.isfinite(out))


def test_p_sample_single_step_shape():
    ddpm = build(T=10)
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([5, 5])
    out = ddpm.p_sample(x, t)
    assert out.shape == x.shape


def test_explicit_timestep_loss_path():
    ddpm = build()
    x0 = torch.randn(3, 1, 16, 16)
    t = torch.tensor([0, 1, 2])
    loss = ddpm.loss(x0, t)
    assert torch.isfinite(loss)
