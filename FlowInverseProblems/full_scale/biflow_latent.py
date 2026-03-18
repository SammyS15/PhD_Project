"""
BiFlow for high-dimensional latent space (Flux latents).

BiFlow (arXiv 2512.10953) learns a fast reverse mapping that approximates
TarFlow's slow autoregressive inverse. It uses bidirectional attention
Transformers for fully parallel computation.

For the full-scale pipeline:
    - TarFlow forward (z -> z'): FAST, parallel (used for log p(z))
    - TarFlow inverse (z' -> z): SLOW, autoregressive
    - BiFlow reverse (z' -> z):  FAST, parallel (replaces slow inverse)

Key uses of BiFlow in posterior sampling:
    1. Fast initialization of MCMC chains (generate prior samples quickly)
    2. Fast proposal generation for MH-style samplers
    3. NOTE: BiFlow does NOT help with log p(z) evaluation (that needs TarFlow forward)
"""

import torch
import torch.nn as nn


class BiFlowLatent(nn.Module):
    """
    BiFlow reverse model for Flux latent space.

    Architecture: lightweight U-Net or Transformer that maps
    z' (Gaussian, shape C x H x W) -> z (target, shape C x H x W).

    Trained by distillation: minimize ||BiFlow(z') - TarFlow.inverse(z')||^2
    """

    def __init__(self, in_channels=32, hidden_channels=64, n_blocks=4):
        """
        Args:
            in_channels: number of latent channels (32 for Flux)
            hidden_channels: hidden dimension
            n_blocks: number of residual blocks
        """
        super().__init__()

        # Simple convolutional architecture for latent-space mapping
        layers = [
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(n_blocks):
            layers.append(ResBlock(hidden_channels))
        layers.append(nn.Conv2d(hidden_channels, in_channels, 3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, z_prime):
        """
        Fast approximate inverse: z' -> z.

        Args:
            z_prime: (B, C, H, W) Gaussian noise

        Returns:
            z: (B, C, H, W) approximate latent from target distribution
        """
        return z_prime + self.net(z_prime)  # residual connection


class ResBlock(nn.Module):
    """Simple residual block for convolutional BiFlow."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


def train_biflow_latent(biflow, tarflow, n_samples=10000, n_epochs=100,
                        batch_size=16, lr=1e-4, device="cuda"):
    """
    Train BiFlow by distillation from TarFlow.

    For each batch:
        1. Sample z' ~ N(0, I)
        2. Compute z = TarFlow.inverse(z')  [slow, done once for training data]
        3. Train BiFlow to minimize ||BiFlow(z') - z||^2

    In practice, pre-compute all (z', z) pairs first to avoid repeated
    slow TarFlow inversion during training.

    Args:
        biflow: BiFlowLatent model
        tarflow: trained TarFlowWrapper
        n_samples: number of training pairs
        n_epochs: training epochs
        batch_size: batch size
        lr: learning rate
        device: torch device

    Returns:
        list of losses
    """
    # TODO: implement once TarFlow is available
    # Step 1: Pre-generate training pairs (expensive, one-time)
    # z_primes = []
    # z_targets = []
    # for i in range(0, n_samples, batch_size):
    #     z_prime = torch.randn(batch_size, 32, 128, 128, device=device)
    #     with torch.no_grad():
    #         z = tarflow.inverse(z_prime)
    #     z_primes.append(z_prime.cpu())
    #     z_targets.append(z.cpu())
    #
    # Step 2: Train BiFlow on the pairs
    # optimizer = torch.optim.AdamW(biflow.parameters(), lr=lr)
    # ...
    raise NotImplementedError("Requires trained TarFlow model")
