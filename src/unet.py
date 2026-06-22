"""A small UNet that predicts the noise eps added at a given timestep.

The network sees a noisy image x_t and the timestep t, and outputs a tensor the
same shape as the image that estimates the noise. The timestep enters through a
sinusoidal embedding that is projected and added into each residual block, so
the same weights behave differently at different noise levels.

The architecture is deliberately compact: two downsampling stages, a
bottleneck, two upsampling stages, with skip connections. It is large enough to
overfit a tiny batch (which the tests rely on) yet cheap on CPU.
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Maps an integer timestep to a fixed sinusoidal feature vector."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / (half - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """Two conv layers plus a timestep bias, with a residual connection."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2),
        time_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling path.
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        chans = [base_channels]
        ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.down_blocks.append(ResidualBlock(ch, out_ch, time_dim))
            chans.append(out_ch)
            self.downsamples.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            ch = out_ch

        # Bottleneck.
        self.mid_block = ResidualBlock(ch, ch, time_dim)

        # Upsampling path. Each stage consumes the matching skip connection.
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsamples.append(
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            skip_ch = chans.pop()
            self.up_blocks.append(ResidualBlock(out_ch + skip_ch, out_ch, time_dim))
            ch = out_ch

        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        h = self.init_conv(x)
        skips = [h]
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid_block(h, t_emb)

        for up, block in zip(self.upsamples, self.up_blocks):
            h = up(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        return self.out_conv(self.act(self.out_norm(h)))
