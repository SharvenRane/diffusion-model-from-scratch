"""Behavior checks for the noise schedule and the closed form forward process."""

import torch

from src.schedule import (
    NoiseSchedule,
    linear_beta_schedule,
    cosine_beta_schedule,
)


def make_schedule(T=200):
    return NoiseSchedule(linear_beta_schedule(T))


def test_betas_in_valid_range():
    for betas in (linear_beta_schedule(100), cosine_beta_schedule(100)):
        assert torch.all(betas > 0)
        assert torch.all(betas < 1)
        assert betas.shape == (100,)


def test_alphas_cumprod_is_decreasing():
    sch = make_schedule()
    ac = sch.alphas_cumprod
    # The cumulative product of alphas must be monotonically non increasing.
    assert torch.all(ac[1:] <= ac[:-1] + 1e-6)
    assert ac[0] <= 1.0
    assert ac[-1] > 0.0


def test_q_sample_shape_and_determinism():
    sch = make_schedule()
    x0 = torch.randn(4, 1, 8, 8)
    t = torch.tensor([0, 5, 50, 199])
    noise = torch.randn_like(x0)
    x_t = sch.q_sample(x0, t, noise)
    assert x_t.shape == x0.shape
    # Same noise must give the same corrupted sample.
    again = sch.q_sample(x0, t, noise)
    assert torch.allclose(x_t, again)


def test_q_sample_at_t_zero_is_almost_clean():
    sch = make_schedule()
    x0 = torch.randn(8, 1, 8, 8)
    t = torch.zeros(8, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = sch.q_sample(x0, t, noise)
    # At t = 0 the signal coefficient is close to 1 and noise coefficient close
    # to 0, so x_t should track x0 far more closely than it tracks the noise.
    assert (x_t - x0).pow(2).mean() < (x_t - noise).pow(2).mean()


def test_q_sample_empirical_mean_and_variance():
    """The closed form is x_t = sqrt(ab) x0 + sqrt(1 - ab) eps.

    For a fixed x0 and random eps, over many draws the per pixel mean should
    approach sqrt(ab) * x0 and the per pixel variance should approach
    (1 - ab). We verify both at several timesteps.
    """
    torch.manual_seed(0)
    T = 200
    sch = make_schedule(T)
    x0 = torch.randn(1, 1, 4, 4)

    for ti in (0, 25, 100, 199):
        N = 20000
        x0_rep = x0.expand(N, -1, -1, -1)
        t = torch.full((N,), ti, dtype=torch.long)
        noise = torch.randn_like(x0_rep)
        x_t = sch.q_sample(x0_rep, t, noise)

        ab = sch.alphas_cumprod[ti].item()
        expected_mean = (ab ** 0.5) * x0
        expected_var = 1.0 - ab

        emp_mean = x_t.mean(dim=0, keepdim=True)
        emp_var = x_t.var(dim=0, unbiased=True).mean().item()

        assert torch.allclose(emp_mean, expected_mean, atol=0.05), (
            f"mean mismatch at t={ti}"
        )
        assert abs(emp_var - expected_var) < 0.05, (
            f"var mismatch at t={ti}: {emp_var} vs {expected_var}"
        )


def test_rejects_invalid_betas():
    import pytest

    with pytest.raises(ValueError):
        NoiseSchedule(torch.tensor([0.1, 1.5]))
    with pytest.raises(ValueError):
        NoiseSchedule(torch.zeros(3))
