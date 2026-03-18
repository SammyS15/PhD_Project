"""
Forward operators for MNIST inverse problems.

Each operator returns:
    forward_op : Callable x (784,) → y (obs_dim,)   [differentiable]
    obs_dim    : int — dimension of the observation
    mask_flat  : (784,) bool tensor — True = observed pixel (for visualisation)

The operator must be differentiable w.r.t. x because gradients flow
through it during MCMC:
    ∇_eps log p(eps|y) = ∇_eps [ A(D(G(eps))) ]^T · (y - A(D(G(eps)))) / σ²
"""

import torch


# ── Inpainting ─────────────────────────────────────────────────────────────────

def make_inpainting_op(mask_type: str = 'right_half', device: str = 'cpu'):
    """
    Returns (forward_op, obs_dim, mask_flat).

    mask_type options:
        'right_half'  — observe columns 14:28 (392 pixels)
        'left_half'   — observe columns 0:14
        'top_half'    — observe rows 0:14
        'bottom_half' — observe rows 14:28
        'random_50'   — random 50% of pixels (seed=0)
    """
    mask_img = torch.zeros(28, 28, dtype=torch.bool)

    if mask_type == 'right_half':
        mask_img[:, 14:] = True
    elif mask_type == 'left_half':
        mask_img[:, :14] = True
    elif mask_type == 'top_half':
        mask_img[:14, :] = True
    elif mask_type == 'bottom_half':
        mask_img[14:, :] = True
    elif mask_type == 'random_50':
        torch.manual_seed(0)
        idx = torch.randperm(784)[:392]
        mask_img.view(-1)[idx] = True
    else:
        raise ValueError(f'Unknown mask_type: {mask_type!r}')

    mask_flat = mask_img.view(-1).to(device)   # (784,) bool
    obs_dim   = int(mask_flat.sum().item())

    def forward_op(x):
        """x: (..., 784) → (..., obs_dim).  Differentiable."""
        return x[..., mask_flat]

    return forward_op, obs_dim, mask_flat


# ── Compressed sensing ─────────────────────────────────────────────────────────

def make_cs_op(obs_dim: int = 200, n: int = 784, seed: int = 0, device: str = 'cpu'):
    """
    Random Gaussian measurement matrix A ∈ R^{obs_dim × 784}.
    y = A x  where A is normalised so ||Ax||² ≈ ||x||² in expectation.
    """
    torch.manual_seed(seed)
    A = torch.randn(obs_dim, n, device=device) / (obs_dim ** 0.5)

    def forward_op(x):
        """x: (..., 784) → (..., obs_dim).  Differentiable."""
        return x @ A.T

    mask_flat = torch.ones(784, dtype=torch.bool, device=device)   # all pixels "observed"
    return forward_op, obs_dim, mask_flat


# ── Observation helper ─────────────────────────────────────────────────────────

def make_observation(forward_op, x_true, sigma_n: float, seed: int = 42):
    """y = A(x_true) + N(0, sigma_n² I).  Returns y tensor."""
    torch.manual_seed(seed)
    with torch.no_grad():
        Ax = forward_op(x_true)
        y  = Ax + sigma_n * torch.randn_like(Ax)
    return y


# ── Visualisation helper ────────────────────────────────────────────────────────

def overlay_observation(x_flat, mask_flat, fill: float = 0.5):
    """
    Returns a (28, 28) image where:
        observed pixels  → their actual value
        masked pixels    → fill value (default 0.5 = grey)
    Useful for visualising what is given to the sampler.
    """
    img = x_flat.clone().detach().cpu()
    img[~mask_flat.cpu()] = fill
    return img.view(28, 28).numpy()
