"""
ViT-based conditional generator.

The noise vector and the attributes are concatenated (early fusion) and
projected into a grid of 8x8 tokens; a stack of Transformer blocks refines
the tokens and a transposed-convolution head upsamples them to a 64x64 image.
"""

import torch
from torch import nn

EMBEDDING_DIMENSION = 512
ATTENTION_HEADS = 8
DECODER_LAYERS = 6
PATCH_SIZE = 8   # 64 / 8 = 8x8 grid of tokens
IMAGE_SIZE = 64
NUM_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2


class ViTGenerator(nn.Module):
    def __init__(self, latent_dim=128, num_attrs=3):
        super().__init__()
        self.latent_dim = latent_dim

        # Early fusion: the projection takes noise and attributes together,
        # so every token depends on the labels from the start.
        input_dim = latent_dim + num_attrs
        self.latent_to_grid = nn.Linear(input_dim, NUM_TOKENS * EMBEDDING_DIMENSION)

        # Fixed sinusoidal positional encoding (not saved in the state dict).
        self.register_buffer(
            "pos_encoding", self.build_positional_encoding(NUM_TOKENS, EMBEDDING_DIMENSION), persistent=False
        )

        # Transformer blocks for feature refinement
        self.blocks = nn.ModuleList([
            AttentionBlock(EMBEDDING_DIMENSION, ATTENTION_HEADS) for _ in range(DECODER_LAYERS)
        ])

        # Upsamples the token grid into the final image
        self.decoder_head = DecoderLayerCustom(IMAGE_SIZE, PATCH_SIZE, EMBEDDING_DIMENSION)

    @staticmethod
    def build_positional_encoding(max_len, dim):
        """Standard sinusoidal positional encoding."""
        n = torch.arange(0, max_len)[:, None]
        i = torch.arange(0, dim, 2)[None, :]
        ang = n / (max_len ** (i / dim))
        R = torch.zeros(max_len, dim)
        R[:, ::2] = torch.sin(ang)
        R[:, 1::2] = torch.cos(ang)
        return R

    def forward(self, z, attrs):
        # z: [B, latent_dim], attrs: [B, 3] with values in {0, 1}
        z_input = torch.cat([z, attrs], dim=1)                                          # [B, latent_dim + 3]
        x = self.latent_to_grid(z_input).view(-1, NUM_TOKENS, EMBEDDING_DIMENSION)      # [B, 64, 512]
        x = x + self.pos_encoding

        for block in self.blocks:
            x = block(x)

        return self.decoder_head(x)                                                     # [B, 3, 64, 64]


class AttentionBlock(nn.Module):
    """Pre-norm Transformer block: multi-head self-attention and MLP, both residual."""

    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderLayerCustom(nn.Module):
    """Upsampling head: reshapes the tokens into a feature map and upsamples 8 -> 64."""

    def __init__(self, image_size, patch_size, dim):
        super().__init__()
        self.grid_size = image_size // patch_size
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(dim, 256, 4, 2, 1),   # 8x8 -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 16x16 -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 32x32 -> 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),          # final RGB refinement
            nn.Tanh(),                               # output range [-1, 1]
        )

    def forward(self, x):
        B, L, D = x.shape
        # Tokens back into a spatial grid: [B, D, H, W]
        x = x.permute(0, 2, 1).reshape(B, D, self.grid_size, self.grid_size)
        return self.layers(x)
