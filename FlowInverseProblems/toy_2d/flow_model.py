"""
RealNVP normalizing flow for the NSPS epsilon-space approach.

Key difference from the original toy_2d:
    The flow is trained on N(0,I) samples (the BASE distribution),
    NOT on samples from the target. The flow learns an identity-like
    mapping in the simple case, but the important thing is that it
    provides a DIFFERENTIABLE inverse: eps -> z.

For the full-scale pipeline, this is replaced by BiFlow (fast reverse
of TarFlow). The flow inverse is what gets composed with the decoder
and forward model during NUTS sampling.

Convention (matching NSPS):
    forward(z)  -> eps, log_det   (data -> noise, for density verification)
    inverse(eps) -> z             (noise -> data, used in MCMC gradient chain)
    log_prob(z) -> scalar         (exact density, for verification only)
"""

import torch
import torch.nn as nn
import numpy as np


class CouplingLayer(nn.Module):
    """Affine coupling layer for 2D data."""

    def __init__(self, dim, hidden_dim=64, transform_dim=0):
        super().__init__()
        self.transform_dim = transform_dim
        self.condition_dim = 1 - transform_dim

        # s and t networks: condition on one dim, output scale + translation
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [s, t]
        )
        # Initialize near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        """Forward: z -> eps (data to noise). Returns (eps, log_det)."""
        cond = x[:, self.condition_dim : self.condition_dim + 1]
        st = self.net(cond)
        s = torch.tanh(st[:, 0]) * 2.0  # bounded scale
        t = st[:, 1]

        out = x.clone()
        out[:, self.transform_dim] = x[:, self.transform_dim] * torch.exp(s) + t
        return out, s  # s is the log-det contribution

    def inverse(self, y):
        """Inverse: eps -> z (noise to data)."""
        cond = y[:, self.condition_dim : self.condition_dim + 1]
        st = self.net(cond)
        s = torch.tanh(st[:, 0]) * 2.0
        t = st[:, 1]

        out = y.clone()
        out[:, self.transform_dim] = (y[:, self.transform_dim] - t) * torch.exp(-s)
        return out


class RealNVP(nn.Module):
    """
    RealNVP normalizing flow.

    Follows NSPS convention:
        forward(z)   -> (eps, log_det)  data -> noise
        inverse(eps) -> z               noise -> data  (USED IN MCMC)
        log_prob(z)  -> scalar          density evaluation (VERIFICATION ONLY)

    For NSPS posterior sampling, only inverse() is called during MCMC.
    The prior in eps-space is trivially N(0,I), so no log_prob needed.
    """

    def __init__(self, dim=2, hidden_dim=64, n_layers=8):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                CouplingLayer(dim, hidden_dim, transform_dim=i % dim)
            )

    def forward(self, z):
        """Forward: z -> eps with log-det (data to noise)."""
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in self.layers:
            x, log_det = layer.forward(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, eps):
        """Inverse: eps -> z (noise to data). This is the key function for MCMC."""
        x = eps
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, z):
        """log p(z) via change of variables. For verification only."""
        eps, log_det = self.forward(z)
        log_prob_base = -0.5 * (self.dim * np.log(2 * np.pi) + (eps ** 2).sum(dim=-1))
        return log_prob_base + log_det

    def sample(self, n_samples):
        """Sample z from the learned distribution."""
        eps = torch.randn(n_samples, self.dim, device=next(self.parameters()).device)
        return self.inverse(eps)


def train_flow(flow, target_samples, n_epochs=2000, lr=1e-3, batch_size=512,
               verbose=True):
    """
    Train flow by maximum likelihood on target samples.

    This trains the flow so that flow.inverse maps N(0,I) -> target distribution.
    Equivalently, flow.forward maps target -> N(0,I).

    Args:
        flow: RealNVP model
        target_samples: (N, d) tensor from the target distribution
        n_epochs: training epochs
        lr: learning rate
        batch_size: batch size
        verbose: print progress

    Returns:
        list of losses
    """
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    dataset = torch.utils.data.TensorDataset(target_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            loss = -flow.log_prob(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, NLL: {avg_loss:.4f}")

    return losses


def verify_flow(flow, n_test=1000):
    """
    Verify flow invertibility and sample statistics.
    Matches NSPS's verify_flow pattern.

    Returns dict with diagnostics.
    """
    device = next(flow.parameters()).device

    with torch.no_grad():
        # Check invertibility: z -> eps -> z_rec
        z_test = torch.randn(n_test, flow.dim, device=device)
        eps, _ = flow.forward(z_test)
        z_rec = flow.inverse(eps)
        max_error = (z_test - z_rec).abs().max().item()

        # Check sample statistics
        eps_samples = torch.randn(10000, flow.dim, device=device)
        z_samples = flow.inverse(eps_samples)
        mean = z_samples.mean(dim=0)
        std = z_samples.std(dim=0)

    print(f"  Max invertibility error: {max_error:.2e}")
    print(f"  Sample mean: [{mean[0].item():.3f}, {mean[1].item():.3f}]")
    print(f"  Sample std:  [{std[0].item():.3f}, {std[1].item():.3f}]")

    ok = max_error < 1e-2
    print(f"  Flow verification {'PASSED' if ok else 'FAILED'}")

    return {"max_error": max_error, "mean": mean.cpu(), "std": std.cpu(), "ok": ok}
