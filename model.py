"""SR3-style U-Net: residual blocks with GroupNorm, noise-level (gamma) conditioning, self-attention at low resolution."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(c):
    g = min(32, c)
    while c % g:
        g -= 1
    return nn.GroupNorm(g, c)


class NoiseLevelEmbedding(nn.Module):
    """WaveGrad/SR3-style sinusoidal embedding of sqrt(gamma), followed by an MLP."""

    def __init__(self, dim, scale=5000.0):
        super().__init__()
        self.dim, self.scale = dim, scale
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, gamma):                        # gamma: (B,) in (0, 1]
        level = gamma.float().sqrt() * self.scale
        half = self.dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=gamma.device, dtype=torch.float32) / half)
        args = level[:, None] * freqs[None]
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=-1))


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> conv, add noise embedding, GroupNorm -> SiLU -> dropout -> conv, residual / sqrt(2)."""

    def __init__(self, cin, cout, emb_dim, dropout=0.0):
        super().__init__()
        self.n1, self.c1 = group_norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.n2, self.drop, self.c2 = group_norm(cout), nn.Dropout(dropout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return (h + self.skip(x)) / math.sqrt(2)     # SR3 rescales residual/skip sums by 1/sqrt(2)


class AttnBlock(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.norm = group_norm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)            # (B, H*W, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2).reshape(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Level(nn.Module):
    """One ResBlock plus optional attention."""

    def __init__(self, cin, cout, emb_dim, dropout, attn):
        super().__init__()
        self.res = ResBlock(cin, cout, emb_dim, dropout)
        self.attn = AttnBlock(cout) if attn else nn.Identity()

    def forward(self, x, emb):
        return self.attn(self.res(x, emb))


class UNet(nn.Module):
    """
    Input: concat([bicubic-upsampled LR, noisy HR]) -> 2*C channels. Output: predicted noise, C channels.
    Image sides must be divisible by 2 ** (len(channel_mults) - 1).
    """

    def __init__(self, channels=1, base_channels=64, channel_mults=(1, 2, 4, 8), num_res_blocks=2,
                 attn_levels=(3,), dropout=0.0):
        super().__init__()
        emb_dim = base_channels * 4
        self.embed = NoiseLevelEmbedding(base_channels)
        self.inp = nn.Conv2d(2 * channels, base_channels, 3, padding=1)
        self.factor = 2 ** (len(channel_mults) - 1)

        self.down = nn.ModuleList()
        skip_ch, ch = [base_channels], base_channels
        for i, m in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                self.down.append(Level(ch, base_channels * m, emb_dim, dropout, i in attn_levels))
                ch = base_channels * m
                skip_ch.append(ch)
            if i != len(channel_mults) - 1:
                self.down.append(Downsample(ch))
                skip_ch.append(ch)

        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(channel_mults))):
            for _ in range(num_res_blocks + 1):
                self.up.append(Level(ch + skip_ch.pop(), base_channels * m, emb_dim, dropout, i in attn_levels))
                ch = base_channels * m
            if i != 0:
                self.up.append(Upsample(ch))

        self.out = nn.Sequential(group_norm(ch), nn.SiLU(), nn.Conv2d(ch, channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, gamma):
        assert x.shape[-1] % self.factor == 0 and x.shape[-2] % self.factor == 0, \
            f"image sides must be divisible by {self.factor}, got {tuple(x.shape[-2:])}"
        emb = self.embed(gamma)
        h = self.inp(x)
        skips = [h]
        for layer in self.down:
            h = layer(h) if isinstance(layer, Downsample) else layer(h, emb)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for layer in self.up:
            if isinstance(layer, Upsample):
                h = layer(h)
            else:
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
        return self.out(h)


def build_model(cfg):
    m = cfg.model
    return UNet(channels=cfg.data.channels, base_channels=m.base_channels, channel_mults=tuple(m.channel_mults),
                num_res_blocks=m.num_res_blocks, attn_levels=tuple(m.attn_levels), dropout=m.dropout)
