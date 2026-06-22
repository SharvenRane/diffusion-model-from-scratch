"""Shape and conditioning checks for the UNet noise predictor."""

import torch

from src.unet import UNet, SinusoidalTimeEmbedding


def test_unet_output_shape_matches_input():
    net = UNet(in_channels=1, base_channels=16, channel_mults=(1, 2))
    x = torch.randn(3, 1, 16, 16)
    t = torch.randint(0, 100, (3,))
    out = net(x, t)
    assert out.shape == x.shape


def test_unet_multichannel():
    net = UNet(in_channels=3, base_channels=16, channel_mults=(1, 2))
    x = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 100, (2,))
    assert net(x, t).shape == x.shape


def test_time_embedding_shape_and_distinct():
    emb = SinusoidalTimeEmbedding(32)
    t = torch.arange(10)
    e = emb(t)
    assert e.shape == (10, 32)
    # Different timesteps produce different embeddings.
    assert not torch.allclose(e[0], e[1])


def test_unet_uses_the_timestep():
    """Feeding the same image at two different timesteps should change output."""
    torch.manual_seed(0)
    net = UNet(in_channels=1, base_channels=16, channel_mults=(1, 2))
    net.eval()
    x = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        out_a = net(x, torch.tensor([0]))
        out_b = net(x, torch.tensor([50]))
    assert not torch.allclose(out_a, out_b)
