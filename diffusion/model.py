"""
Conditional DDPM for 64x64 face generation.

Components:
    NoiseSchedule  — cosine noise schedule (Nichol & Dhariwal, 2021).
    TimeEncoding   — sinusoidal embedding of the diffusion step.
    Network        — conditional U-Net (64 -> 32 -> 16 -> 8) with residual
                     blocks, self-attention at 16x16 and 8x8, and the three
                     attributes injected at every resolution.
    ddpm_sample    — ancestral DDPM sampling with classifier-free guidance
                     and static thresholding.

Attributes are encoded as -1 / +1 ([Male, Smiling, Young]); the all-zero
vector is the "null" condition used for classifier-free guidance.
"""

import math

import torch
from torch import nn

IMAGE_SIZE = 64
DIFFUSION_STEPS = 1000
TIME_ENCODING_SIZE = 256
COND_SIZE = 3


# ---------------------------------------------------------------------------
# Diffusion process
# ---------------------------------------------------------------------------

class NoiseSchedule:
    """Cosine noise schedule with precomputed coefficients."""

    def __init__(self, L=DIFFUSION_STEPS, s=0.008, device="cpu"):
        self.L = L
        t = torch.linspace(0.0, L, L + 1, device=device) / L
        a = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        a = a / a[0]
        self.beta = (1 - a[1:] / a[:-1]).clip(0.0, 0.99)
        self.alpha = torch.cumprod(1.0 - self.beta, dim=0)  # cumulative product (alpha bar)
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_beta = torch.sqrt(self.beta)
        self.sqrt_1_alpha = torch.sqrt(1 - self.alpha)
        self.sqrt_1_beta = torch.sqrt(1 - self.beta)


def time_encoding_table(L=DIFFUSION_STEPS, dim=TIME_ENCODING_SIZE):
    """Sinusoidal encoding of the diffusion steps: one row of size dim per step."""
    dim2 = dim // 2
    encoding = torch.zeros(L, dim)
    ang = torch.linspace(0.0, math.pi / 2, L)
    logmul = torch.linspace(0.0, math.log(40), dim2)
    mul = torch.exp(logmul)
    for i in range(dim2):
        a = ang * mul[i]
        encoding[:, 2 * i] = torch.sin(a)
        encoding[:, 2 * i + 1] = torch.cos(a)
    return encoding


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Multi-head self-attention over the spatial positions, with a feed-forward block."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0] * size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size[0], size[1])


class ResBlockPro(nn.Module):
    """Residual block (GroupNorm -> SiLU -> Conv) x2 with the time embedding added in between."""

    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class BottomBlock(nn.Module):
    """Bottleneck at 8x8: residual block, self-attention, residual block."""

    def __init__(self, channels, time_channels):
        super().__init__()
        self.res1 = ResBlockPro(channels, channels, time_channels)
        self.attn = SelfAttention(channels)
        self.res2 = ResBlockPro(channels, channels, time_channels)

    def forward(self, x, t_emb, cond):
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x


class UNetBlock(nn.Module):
    """
    One U-Net level, defined recursively: downsample, process, call the inner
    level, upsample, merge the skip connection, process. The condition is
    projected and added to the feature maps on both paths.
    """

    def __init__(self, size, in_ch, out_ch, time_ch, cond_ch, inner_block=None):
        super().__init__()
        self.size = size
        self.inner = inner_block

        # Encoder path (down)
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1) if in_ch != out_ch else nn.Identity()
        self.cond_proj_down = nn.Linear(cond_ch, out_ch)
        self.res_down = ResBlockPro(out_ch, out_ch, time_ch)
        self.attn_down = SelfAttention(out_ch) if size in [16, 8] else None  # attention only at low resolution

        # Decoder path (up)
        self.up = nn.ConvTranspose2d(out_ch, in_ch, 4, stride=2, padding=1) if in_ch != out_ch else nn.Identity()
        self.combiner = nn.Conv2d(in_ch * 2, in_ch, 1)
        self.cond_proj_up = nn.Linear(cond_ch, in_ch)
        self.res_up = ResBlockPro(in_ch, in_ch, time_ch)
        self.attn_up = SelfAttention(in_ch) if size in [16, 8] else None

    def forward(self, x, t_emb, cond):
        h_down = self.down(x)
        h_down = h_down + self.cond_proj_down(cond)[:, :, None, None]
        h_down = self.res_down(h_down, t_emb)
        if self.attn_down:
            h_down = self.attn_down(h_down)

        h_inner = self.inner(h_down, t_emb, cond) if self.inner else h_down

        h_up = self.up(h_inner)
        h_up = self.combiner(torch.cat([x, h_up], dim=1))  # skip connection
        h_up = h_up + self.cond_proj_up(cond)[:, :, None, None]
        h_up = self.res_up(h_up, t_emb)
        if self.attn_up:
            h_up = self.attn_up(h_up)
        return h_up


class Network(nn.Module):
    """Conditional U-Net that predicts the noise eps(x_t, t, cond)."""

    def __init__(self):
        super().__init__()
        # Not a trainable parameter and not saved in the state dict.
        self.register_buffer("time_encoding", time_encoding_table(), persistent=False)

        self.pre = nn.Conv2d(3, 64, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(TIME_ENCODING_SIZE, 4 * TIME_ENCODING_SIZE),
            nn.SiLU(),
            nn.Linear(4 * TIME_ENCODING_SIZE, TIME_ENCODING_SIZE),
        )
        self.unet = self.build_unet(64, [64, 128, 256, 512])  # 64 -> 32 -> 16 -> 8
        self.post = nn.Sequential(nn.GroupNorm(32, 64), nn.SiLU(), nn.Conv2d(64, 3, 3, padding=1))

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(self.time_encoding[t])
        x = self.pre(x)
        y = self.unet(x, t_emb, cond)
        return self.post(y)

    def build_unet(self, size, feat_list):
        if len(feat_list) > 1:
            inner_block = self.build_unet(size // 2, feat_list[1:])
            return UNetBlock(size, feat_list[0], feat_list[1], TIME_ENCODING_SIZE, COND_SIZE, inner_block)
        return BottomBlock(feat_list[0], TIME_ENCODING_SIZE)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample(model, schedule, cond, guidance_scale=1.0, generator=None):
    """
    Generates one image per row of cond with ancestral DDPM sampling.

    Classifier-free guidance combines the conditional and unconditional
    noise predictions: eps = eps_uncond + w * (eps_cond - eps_uncond).
    Static thresholding clamps the predicted x0 to [-1, 1] at every step
    to keep the trajectory in the data range.

    Args:
        model:          trained Network.
        schedule:       NoiseSchedule on the same device as the model.
        cond:           [N, 3] tensor with values in {-1, +1}.
        guidance_scale: classifier-free guidance weight w.
        generator:      optional torch.Generator for reproducible sampling.

    Returns:
        [N, 3, 64, 64] tensor with values in [0, 1].
    """
    device = cond.device
    n = cond.shape[0]
    uncond = torch.zeros_like(cond)
    x = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE, device=device, generator=generator)

    for i in reversed(range(schedule.L)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        g1 = model(x, t, cond)
        g0 = model(x, t, uncond)
        eps = g0 + guidance_scale * (g1 - g0)

        beta = schedule.beta[t].view(-1, 1, 1, 1)
        sqrt_alpha_bar = schedule.sqrt_alpha[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = schedule.sqrt_1_alpha[t].view(-1, 1, 1, 1)

        # Static thresholding: clamp the predicted clean image, then recompute eps.
        pred_x0 = ((x - sqrt_one_minus_alpha * eps) / sqrt_alpha_bar).clamp(-1.0, 1.0)
        eps = (x - sqrt_alpha_bar * pred_x0) / sqrt_one_minus_alpha

        noise = torch.randn(x.shape, device=device, generator=generator) if i > 0 else 0
        x = (x - (beta / sqrt_one_minus_alpha) * eps) / torch.sqrt(1 - beta) + torch.sqrt(beta) * noise

    return x.clamp(-1, 1) * 0.5 + 0.5
