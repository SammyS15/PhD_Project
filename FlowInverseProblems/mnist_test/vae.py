"""
Convolutional VAE for MNIST.

Role in the NSPS pipeline:
    encoder:  x (784,) → z (latent_dim,)   [used only during training]
    decoder:  z (latent_dim,) → x (784,)   [used during MCMC — must be differentiable]

The decoder is the D in:
    log p(eps|y) = -0.5||eps||² - 0.5||y - A(D(G(eps)))||² / σ²

All decoder outputs are in [0, 1] via sigmoid (normalised pixel values).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Encoder ────────────────────────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """28×28 image → (mu, logvar) of shape (latent_dim,)."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        # Spatial: 1×28×28 → 32×14×14 → 64×7×7 → 128×1×1
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # → 14×14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # → 7×7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=7),                       # → 1×1
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        """x: (B, 1, 28, 28) → mu (B, latent_dim), logvar (B, latent_dim)."""
        h = self.conv(x).view(x.size(0), -1)   # (B, 128)
        return self.fc_mu(h), self.fc_logvar(h)


# ── Decoder ────────────────────────────────────────────────────────────────────

class ConvDecoder(nn.Module):
    """z (latent_dim,) → x_flat (784,) with values in [0, 1]."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128)
        # Spatial: 128×1×1 → 64×7×7 → 32×14×14 → 1×28×28
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=7),                       # → 7×7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # → 14×14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # → 28×28
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        z: (B, latent_dim) or (latent_dim,) → x_flat: (B, 784) or (784,).
        Handles both batched and single-sample calls (needed for MCMC).
        """
        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)                          # (1, latent_dim)
        h = self.fc(z).view(-1, 128, 1, 1)             # (B, 128, 1, 1)
        x = self.deconv(h).view(-1, 784)               # (B, 784)
        return x.squeeze(0) if squeeze else x           # back to (784,) if single


# ── VAE ────────────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = ConvEncoder(latent_dim, hidden_dim)
        self.decoder    = ConvDecoder(latent_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        """
        x: (B, 784) or (B, 1, 28, 28) → x_recon (B, 784), mu, logvar.
        """
        x_img = x.view(-1, 1, 28, 28)
        mu, logvar = self.encoder(x_img)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, x):
        """Deterministic encode: returns mu (no noise). Used to build flow training set."""
        x_img = x.view(-1, 1, 28, 28)
        mu, _ = self.encoder(x_img)
        return mu

    def decode(self, z):
        """Differentiable decode. This is D(·) in the MCMC chain."""
        return self.decoder(z)


# ── Loss ───────────────────────────────────────────────────────────────────────

def vae_loss(x_recon, x, mu, logvar, beta: float = 1.0):
    """
    ELBO loss: MSE reconstruction + β * KL divergence.

    MSE (not BCE) so that the decoder output is used as a continuous
    representation compatible with the Gaussian likelihood in the inverse problem.
    """
    x_flat = x.view(-1, 784)
    recon  = F.mse_loss(x_recon, x_flat, reduction='sum') / x.size(0)
    kl     = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
    return recon + beta * kl, recon, kl


# ── Training loop ──────────────────────────────────────────────────────────────

def train_vae(vae, data_loader, n_epochs: int = 60, lr: float = 1e-3,
              beta: float = 1.0, device: str = 'cpu'):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    vae.train()
    losses = []
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for x, _ in data_loader:
            x = x.to(device).view(-1, 784)
            x_recon, mu, logvar = vae(x)
            loss, _, _ = vae_loss(x_recon, x, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / len(data_loader)
        losses.append(avg)
        if epoch % 10 == 0:
            print(f'  VAE  Epoch {epoch:3d}/{n_epochs}  loss={avg:.4f}')
    return losses
