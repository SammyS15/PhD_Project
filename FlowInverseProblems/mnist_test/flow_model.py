"""
RealNVP normalizing flow for arbitrary latent dimensions.

The 2D toy version hard-codes dim=2 in its coupling layers (alternating single
dimensions). This version uses half-dimensional masks so it works for any dim —
including LATENT_DIM=32 for the MNIST VAE latent space.

Convention (same as new_toy_2d/flow_model.py):
    forward(z)   → (eps, log_det)   data  → noise  (training / density eval)
    inverse(eps) → z                noise → data   (MCMC gradient chain)
    log_prob(z)  → scalar           exact log-density (training objective)
"""

import torch
import torch.nn as nn
import numpy as np


# ── Coupling layer ─────────────────────────────────────────────────────────────

class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for d-dimensional inputs.

    Splits the d dimensions into two halves:
        conditioning half  → passes through unchanged
        transformed half   → y = x * exp(s(x_cond)) + t(x_cond)

    Alternating which half is conditioned gives expressivity across all dims.
    """

    def __init__(self, dim: int, hidden_dim: int = 256, mask_first_half: bool = True):
        super().__init__()
        d1 = dim // 2
        d2 = dim - d1

        # Boolean mask: True = conditioning dim, False = transformed dim
        if mask_first_half:
            mask = torch.cat([torch.ones(d1), torch.zeros(d2)]).bool()
            self.cond_dim  = d1
            self.trans_dim = d2
        else:
            mask = torch.cat([torch.zeros(d1), torch.ones(d2)]).bool()
            self.cond_dim  = d2
            self.trans_dim = d1

        self.register_buffer('mask', mask)

        # Network: cond_dim → (s, t) for the trans_dim transformed coordinates
        self.net = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.trans_dim * 2),  # first half=s, second=t
        )
        # Initialise near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x_cond):
        out = self.net(x_cond)
        s = torch.tanh(out[:, :self.trans_dim]) * 2.0   # bounded scale
        t = out[:, self.trans_dim:]
        return s, t

    def forward(self, x):
        """x → y, log_det.  x: (B, dim)"""
        x_cond  = x[:,  self.mask]
        x_trans = x[:, ~self.mask]
        s, t = self._st(x_cond)

        y = x.clone()
        y[:, ~self.mask] = x_trans * torch.exp(s) + t
        return y, s.sum(dim=-1)    # log_det = sum of log-scale

    def inverse(self, y):
        """y → x.  y: (B, dim)"""
        y_cond  = y[:,  self.mask]
        y_trans = y[:, ~self.mask]
        s, t = self._st(y_cond)

        x = y.clone()
        x[:, ~self.mask] = (y_trans - t) * torch.exp(-s)
        return x


# ── RealNVP ────────────────────────────────────────────────────────────────────

class RealNVP(nn.Module):
    """
    RealNVP flow for arbitrary dim.

    Alternates mask_first_half=True/False across layers so every dimension
    gets transformed at least once every 2 layers.
    """

    def __init__(self, dim: int = 32, hidden_dim: int = 256, n_layers: int = 12):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, hidden_dim, mask_first_half=(i % 2 == 0))
            for i in range(n_layers)
        ])

    def forward(self, z):
        """z → (eps, log_det).  z: (B, dim)"""
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in self.layers:
            x, ld = layer.forward(x)
            log_det += ld
        return x, log_det

    def inverse(self, eps):
        """eps → z.  eps: (B, dim)  — the key function for MCMC."""
        x = eps
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, z):
        """log p(z) via change of variables.  z: (B, dim) → (B,)"""
        eps, log_det = self.forward(z)
        log_base = -0.5 * (self.dim * np.log(2 * np.pi) + (eps ** 2).sum(dim=-1))
        return log_base + log_det

    def sample(self, n: int):
        eps = torch.randn(n, self.dim, device=next(self.parameters()).device)
        return self.inverse(eps)


# ── Training ───────────────────────────────────────────────────────────────────

def train_flow(flow, target_samples, n_epochs: int = 400, lr: float = 5e-4,
               batch_size: int = 512, verbose: bool = True):
    """
    Train by maximum likelihood on target_samples (the VAE latent codes).
    Same signature as new_toy_2d/flow_model.py for drop-in compatibility.
    """
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    dataset = torch.utils.data.TensorDataset(target_samples)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(1, n_epochs + 1):
        epoch_loss, n_batches = 0.0, 0
        for (batch,) in loader:
            loss = -flow.log_prob(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        avg = epoch_loss / n_batches
        losses.append(avg)
        if verbose and epoch % 50 == 0:
            print(f'  Flow Epoch {epoch:3d}/{n_epochs}  NLL={avg:.4f}')
    return losses


def verify_flow(flow, n_test: int = 1000):
    """Invertibility check — same interface as new_toy_2d/flow_model.py."""
    device = next(flow.parameters()).device
    with torch.no_grad():
        z = torch.randn(n_test, flow.dim, device=device)
        eps, _ = flow.forward(z)
        z_rec  = flow.inverse(eps)
        max_err = (z - z_rec).abs().max().item()
        samples = flow.sample(5000)
    print(f'  Max invertibility error : {max_err:.2e}')
    print(f'  Sample mean (first 4)   : {samples.mean(0)[:4].cpu().numpy().round(3)}')
    print(f'  Sample std  (first 4)   : {samples.std(0)[:4].cpu().numpy().round(3)}')
    ok = max_err < 1e-2
    print(f'  Flow verification {"PASSED" if ok else "FAILED"}')
    return {'max_error': max_err, 'ok': ok}
