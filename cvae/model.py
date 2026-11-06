"""
Conditional VAE for 64x64 face generation.

The attributes condition both networks: they are broadcast to three extra
input channels for the encoder and concatenated to the latent vector z for
the decoder. Attributes are encoded as 0 / 1 ([Male, Smiling, Young]).
"""

import torch
import torch.nn as nn

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=128, num_labels=3):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_labels = num_labels

        # --- ENCODER ---
        # The input will be the image (3 channels) + the labels (3 expanded channels)
        # Total input channels = 3 + 3 = 6
        self.encoder = nn.Sequential(
            # Start from 6x64x64
            # size = (input_size - K_size + 2*padding) / stride + 1
            nn.Conv2d(3 + num_labels, 32, kernel_size=4, stride=2, padding=1), # -> 32x32x32
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 64x16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 128x8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> 256x4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Flatten() # 256*4*4 = 4096, flatten dimension
        )

        # Two heads: mean and log-variance of q(z | x, c).
        # Predicting the log-variance keeps the variance positive (via exp) and is numerically stable.
        self.linear_mu = nn.Linear(4096, latent_dim)
        self.linear_log_sigma = nn.Linear(4096, latent_dim)

        # --- DECODER ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_labels, 4096), # latent vector z + classes
            nn.Unflatten(1, (256, 4, 4)), # 256x4x4 (1 skips the batch dim)
            # Start from 256x4x4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 128x8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 64x16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 32x32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # 3x64x64 (RGB only)
            nn.Tanh() # output in [-1, 1], matching Normalize(0.5, 0.5) on the inputs
        )

    def encode(self, x, labels):
        # To condition the encoder, we expand the labels to have the same spatial dimension as the image (64x64)
        # labels: [Batch, 3] -> [Batch, 3, 1, 1] -> [Batch, 3, 64, 64]
        labels = labels[:,:,None,None].expand(-1, -1, x.shape[2], x.shape[3])

        # Concatenate image and labels along the channels
        x = torch.cat((x, labels), dim=1)

        out = self.encoder(x)
        mu = self.linear_mu(out)
        log_sigma = self.linear_log_sigma(out)
        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        # Reparameterization trick: z = mu + std * epsilon
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        # The decoder is conditioned too: concatenate z and labels
        z = torch.cat((z, labels), dim=1)

        y = self.decoder(z)
        return y

    def forward(self, x, labels):
        mu, log_sigma = self.encode(x, labels)
        z = self.reparameterize(mu, log_sigma)
        y = self.decode(z, labels)
        return y, mu, log_sigma
