"""
Forward measurement operators and target distributions.

All operators are pure functions (no hidden state) for clean composition
with the flow inverse and decoder in the MCMC gradient chain.

Matches NSPS's operators.py pattern but in PyTorch.
"""

import torch
import numpy as np


# ============================================================
# Target Distributions (to train the flow on)
# ============================================================

def sample_gaussian_mixture(n_samples, n_components=4, std=0.3, radius=2.0,
                            device="cpu"):
    """Sample from a 2D Gaussian mixture in a ring."""
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    means = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    assignments = np.random.randint(0, n_components, n_samples)
    samples = means[assignments] + np.random.randn(n_samples, 2) * std
    return torch.tensor(samples, dtype=torch.float32, device=device)


def sample_two_moons(n_samples, noise=0.1, device="cpu"):
    """Sample from Two Moons distribution."""
    n = n_samples // 2
    theta1 = np.random.uniform(0, np.pi, n)
    x1, y1 = np.cos(theta1), np.sin(theta1)
    theta2 = np.random.uniform(0, np.pi, n_samples - n)
    x2, y2 = 1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5
    x = np.concatenate([
        np.stack([x1, y1], axis=1),
        np.stack([x2, y2], axis=1),
    ], axis=0)
    x += np.random.randn(*x.shape) * noise
    return torch.tensor(x, dtype=torch.float32, device=device)


def log_prob_gaussian_mixture(z, n_components=4, std=0.3, radius=2.0):
    """Analytical log p(z) for verification."""
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    means = torch.tensor(
        np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1),
        dtype=z.dtype, device=z.device,
    )
    diff = z.unsqueeze(1) - means.unsqueeze(0)  # (N, K, 2)
    log_probs = -0.5 * (diff ** 2).sum(dim=-1) / (std ** 2) - np.log(2 * np.pi * std ** 2)
    return torch.logsumexp(log_probs, dim=-1) - np.log(n_components)


# ============================================================
# Forward Measurement Operators
# ============================================================

def identity(x):
    """Identity operator (denoising problem): A(x) = x."""
    return x


def make_inpainting_mask(mask):
    """Create inpainting operator from binary mask.

    Args:
        mask: Binary tensor, same shape as x. 1 = observed, 0 = missing.

    Returns:
        Callable: x -> x * mask
    """
    def operator(x):
        return x * mask
    return operator


def make_partial_observation(A):
    """Create linear forward model y = A @ z.

    Args:
        A: (m, d) observation matrix.

    Returns:
        Callable: z -> A @ z
    """
    def operator(z):
        if z.dim() == 1:
            return A @ z
        return z @ A.T
    return operator


def make_noisy_observation(forward_op, z_true, sigma_n, seed=None):
    """Generate a noisy observation y = A(z_true) + noise.

    Args:
        forward_op: forward operator A
        z_true: ground truth sample
        sigma_n: noise standard deviation
        seed: optional random seed

    Returns:
        y: noisy observation
    """
    if seed is not None:
        torch.manual_seed(seed)
    y_clean = forward_op(z_true)
    return y_clean + sigma_n * torch.randn_like(y_clean)


# ============================================================
# Decoder (latent -> "pixel" space)
# ============================================================

def identity_decoder(z):
    """Identity decoder: z -> x = z. For 2D experiments."""
    return z


def nonlinear_decoder(z):
    """Simple nonlinear decoder for more interesting experiments."""
    return z + 0.1 * torch.sin(3 * z)
