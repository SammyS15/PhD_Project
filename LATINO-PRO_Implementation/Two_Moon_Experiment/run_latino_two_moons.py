"""
run_latino_two_moons.py
=======================
LATINO algorithm applied to a 2D Two Moons inverse problem.

Goal: Show that LATINO's iterative encode-noise-denoise-proximal loop
does not produce well-calibrated posteriors, even on a simple 2D problem.

Setup:
  - Prior: Two Moons distribution (learned via score-based diffusion model)
  - Forward operator: A = I (identity) -- pure denoising problem
  - Observation: y = x_true + sigma_y * eps
  - Encoder/Decoder: Identity (we work directly in 2D, no latent space)
  - "Consistency model": Tweedie single-step denoising using the learned score:
      x_0_hat = x_t + sigma_t^2 * score(t, x_t)

LATINO Algorithm (adapted for VE-SDE in 2D):
  Input: noisy observation y, trained score model, noise level sigma_y
  Initialize: x^(0) = y

  For k = 1 to N:
    1. Add noise:   x_t = x^{k-1} + sigma(t_k) * eps,  eps ~ N(0, I)
    2. Tweedie:     x_0_hat = x_t + sigma(t_k)^2 * score(t_k, x_t)
    3. Proximal:    x^{k} = (x_0_hat + gamma_k * y) / (1 + gamma_k)
                    where gamma_k = delta_k / sigma_y^2

  Return x^(N)

Requires: trained score model checkpoint in models/ directory.
  Run train_diffusion_model.py first.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.interpolate import interpn
from matplotlib.colors import Normalize
from pathlib import Path
from tqdm import tqdm

import torch.distributions as tfd
from score_models import ScoreModel

plt.style.use("dark_background")
params = {
    'legend.fontsize': 14,
    'figure.figsize': (5, 5),
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 20,
}
pylab.rcParams.update(params)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Two Moons Distribution (same as train_diffusion_model.py)
# ============================================================

def two_moons(modes=1024, width=0.1, size=1, device=DEVICE):
    outer_circ_x = torch.cos(torch.linspace(0, np.pi, modes)) - 0.5
    outer_circ_y = torch.sin(torch.linspace(0, np.pi, modes)) - 0.25
    inner_circ_x = -torch.cos(torch.linspace(0, np.pi, modes)) + 0.5
    inner_circ_y = -torch.sin(torch.linspace(0, np.pi, modes)) + 0.25
    x = torch.concat([outer_circ_x, inner_circ_x])
    y = torch.concat([outer_circ_y, inner_circ_y])
    coords = size * torch.stack([x, y], dim=1).to(device)
    mixture = tfd.Categorical(probs=torch.ones(2 * modes).to(device), validate_args=False)
    component = tfd.Independent(
        tfd.Normal(loc=coords, scale=width, validate_args=False), 1
    )
    return tfd.MixtureSameFamily(mixture, component, validate_args=False)


def density_scatter(points, fig=None, ax=None, sort=True, bins=40,
                    cmap="magma", norm=None, **kwargs):
    x = points[:, 0]
    y = points[:, 1]
    if ax is None:
        fig, ax = plt.subplots()
    data, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)
    x_bins = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_bins = 0.5 * (y_edges[1:] + y_edges[:-1])
    ax.hist2d(x, y, bins=bins, density=True, cmap=cmap, norm=norm, **kwargs)
    # z = interpn(
    #     (x_bins, y_bins), data, np.vstack([x, y]).T,
    #     method="splinef2d", bounds_error=False,
    # )
    # z[np.where(np.isnan(z))] = 0.0
    # if sort:
    #     idx = z.argsort()
    #     x, y, z = x[idx], y[idx], z[idx]
    # if norm is None:
    #     norm = Normalize(vmin=z.min(), vmax=z.max())
    # ax.scatter(x, y, c=z, cmap=cmap, norm=norm, **kwargs)
    return ax

# ============================================================
# LATINO for 2D (VE-SDE formulation)
# ============================================================

def get_timesteps(N, T=1.0, epsilon=0.0):
    """Evenly spaced timesteps from T (noisy) down to near epsilon (clean)."""
    return torch.linspace(T, epsilon + (T - epsilon) / N, N)


def get_delta(t_k, df, sigma_y):
    """
    Adaptive delta_k. Simplified schedule for the 2D case.
    Mirrors the spirit of the image-space LATINO reference:
    higher delta at high noise levels, lower at low noise.
    """
    if t_k > 0.3:
        return 3.0 * df / 10.0
    else:
        return 2.0 * df / 10.0

@torch.no_grad()
def latino_2d(score_model, y, sigma_y, N=8, device=DEVICE):
    """
    LATINO algorithm for 2D data with identity forward operator.

    Args:
        score_model: Trained ScoreModel instance
        y:           Noisy observations [B, 2] in data space
        sigma_y:     Observation noise std-dev
        N:           Number of LATINO iterations
        device:      Torch device

    Returns:
        x: Restored points [B, 2]
    """
    B = y.shape[0]
    timesteps = get_timesteps(N, T=score_model.sde.T, epsilon=score_model.sde.epsilon)

    # # Initialize: x^(0) = y (start from noisy observation)
    # x = y.clone()

    # Initalize x^(0) = N(0, 1) samples (pure noise) to show that LATINO does not converge to the correct posterior
    x = torch.randn_like(y)

    for k, t_k in enumerate(timesteps):
        t_val = t_k.item()
        sigma_t = score_model.sde.sigma(t_k).item()

        # 1) Add noise: x_t = x + sigma(t_k) * eps
        eps = torch.randn_like(x)
        x_t = x + sigma_t * eps

        # 2) Tweedie single-step denoise:
        #    x_0_hat = x_t + sigma_t^2 * score(t_k, x_t)
        t_batch = torch.full((B,), t_val, device=device)
        score = score_model.score(t_batch, x_t)
        x_0_hat = x_t + sigma_t**2 * score

        # 3) Proximal step (identity operator, L2 data fidelity):
        #    x^{k} = (x_0_hat + gamma * y) / (1 + gamma)
        df = torch.norm(x_0_hat - y, dim=1).mean().item()
        delta_k = get_delta(t_val, df, sigma_y)
        gamma_k = delta_k / (sigma_y**2)

        x = (x_0_hat + gamma_k * y) / (1.0 + gamma_k)

    return x

# ============================================================
# Baseline: Single Tweedie Denoise (no LATINO loop)
# ============================================================

@torch.no_grad()
def tweedie_denoise(score_model, y, sigma_y, device=DEVICE):
    """
    Simple baseline: one-shot Tweedie denoising at the noise level
    matching sigma_y. No iterative loop, no proximal step.
    """
    B = y.shape[0]

    # Find t such that sigma(t) ~ sigma_y
    # sigma(t) = sigma_min * (sigma_max/sigma_min)^(t/T)
    # t = T * log(sigma_y/sigma_min) / log(sigma_max/sigma_min)
    sde = score_model.sde
    t_val = sde.T * np.log(sigma_y / sde.sigma_min) / np.log(sde.sigma_max / sde.sigma_min)
    t_val = np.clip(t_val, sde.epsilon, sde.T)

    t_batch = torch.full((B,), t_val, device=device)
    score = score_model.score(t_batch, y)
    sigma_t = sde.sigma(t_batch[0]).item()
    x_0_hat = y + sigma_t**2 * score

    return x_0_hat

# ============================================================
# Main Experiment
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LATINO on Two Moons")
    # parser.add_argument("--checkpoint-dir", type=str, default="models",
    #                     help="Path to trained score model checkpoint directory")
    parser.add_argument("--sigma-y", type=float, default=0.3,
                        help="Observation noise std-dev")
    parser.add_argument("--N", type=int, default=8,
                        help="Number of LATINO iterations")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load trained score model ----
    checkpoint_path = './models/'
    print(f"Loading score model from {checkpoint_path}...")
    score_model = ScoreModel(checkpoints_directory=checkpoint_path).to(DEVICE)
    score_model.eval()
    print(f"  SDE: VE-SDE, sigma_min={score_model.sde.sigma_min}, "
          f"sigma_max={score_model.sde.sigma_max}")

    # ---- Sample ground truth ----
    sigma_x0 = 0.1
    dist = two_moons(width=sigma_x0)
    x_true = dist.sample([args.n_samples]).to(DEVICE)
    print(f"Ground truth: {x_true.shape}")

    # ---- Create noisy observations ----
    noise = args.sigma_y * torch.randn_like(x_true)
    y = x_true + noise
    print(f"Observations (sigma_y={args.sigma_y}): {y.shape}")

    # ---- Run LATINO ----
    print(f"\nRunning LATINO (N={args.N}, sigma_y={args.sigma_y})...")
    x_latino = latino_2d(
        score_model, y,
        sigma_y=args.sigma_y,
        N=args.N,
        device=DEVICE,
    )

    # ---- Run baseline (single Tweedie) ----
    print("Running baseline (single Tweedie denoise)...")
    x_tweedie = tweedie_denoise(score_model, y, sigma_y=args.sigma_y, device=DEVICE)

    # ---- Compute metrics ----
    mse_noisy = torch.mean((y - x_true) ** 2).item()
    mse_latino = torch.mean((x_latino - x_true) ** 2).item()
    mse_tweedie = torch.mean((x_tweedie - x_true) ** 2).item()

    print(f"\nMean Squared Error:")
    print(f"  Noisy observation:  {mse_noisy:.4f}")
    print(f"  Tweedie (1-step):   {mse_tweedie:.4f}")
    print(f"  LATINO (N={args.N}):      {mse_latino:.4f}")

    # ---- Visualization ----
    x_true_np = x_true.cpu().numpy()
    y_np = y.cpu().numpy()
    x_latino_np = x_latino.cpu().numpy()
    x_tweedie_np = x_tweedie.cpu().numpy()

    xmax, ymax = 2.0, 1.5

    # Plot 1: Four-panel comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"LATINO on Two Moons  (N={args.N}, $\\sigma_y$={args.sigma_y})",
        fontsize=20,
    )

    density_scatter(x_true_np, ax=axes[0, 0], s=1)
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].set_xlim(-xmax, xmax)
    axes[0, 0].set_ylim(-ymax, ymax)
    axes[0, 0].set_ylabel(r"$x_2$")

    density_scatter(y_np, ax=axes[0, 1], s=1)
    axes[0, 1].set_title(f"Noisy Observations ($\\sigma_y$={args.sigma_y})")
    axes[0, 1].set_xlim(-xmax, xmax)
    axes[0, 1].set_ylim(-ymax, ymax)

    density_scatter(x_tweedie_np, ax=axes[1, 0], s=1)
    axes[1, 0].set_title(f"Tweedie Denoise (MSE={mse_tweedie:.4f})")
    axes[1, 0].set_xlim(-xmax, xmax)
    axes[1, 0].set_ylim(-ymax, ymax)
    axes[1, 0].set_xlabel(r"$x_1$")
    axes[1, 0].set_ylabel(r"$x_2$")

    density_scatter(x_latino_np, ax=axes[1, 1], s=1)
    axes[1, 1].set_title(f"LATINO N={args.N} (MSE={mse_latino:.4f})")
    axes[1, 1].set_xlim(-xmax, xmax)
    axes[1, 1].set_ylim(-ymax, ymax)
    axes[1, 1].set_xlabel(r"$x_1$")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"latino_two_moons_comparison_sigma_y_{args.sigma_y}.png",
                bbox_inches="tight", dpi=300)
    print(f"\nSaved comparison to {OUTPUT_DIR}/latino_two_moons_comparison_sigma_y_{args.sigma_y}.png")

    # Plot 2: Overlay -- LATINO vs Ground Truth
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_true_np[:, 0], x_true_np[:, 1], s=1, alpha=0.3, c="cyan", label="Ground Truth")
    ax.scatter(x_latino_np[:, 0], x_latino_np[:, 1], s=1, alpha=0.3, c="magenta", label="LATINO")
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(f"LATINO vs Ground Truth (N={args.N}, $\\sigma_y$={args.sigma_y})")
    ax.legend(markerscale=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"latino_vs_gt_overlay_sigma_y_{args.sigma_y}.png",
                bbox_inches="tight", dpi=300)
    print(f"Saved overlay to {OUTPUT_DIR}/latino_vs_gt_overlay_sigma_y_{args.sigma_y}.png")

    # Plot 3: Diffusion model prior samples (unconditional) for reference
    print("\nGenerating unconditional prior samples for reference...")
    x_prior = score_model.sample([args.n_samples, 2], steps=500)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Distribution Comparison", fontsize=18)

    density_scatter(x_true_np, ax=axes[0], s=1)
    axes[0].set_title("True Two Moons")
    axes[0].set_xlim(-xmax, xmax)
    axes[0].set_ylim(-ymax, ymax)

    density_scatter(x_prior.cpu().numpy(), ax=axes[1], s=1)
    axes[1].set_title("Diffusion Prior Samples")
    axes[1].set_xlim(-xmax, xmax)
    axes[1].set_ylim(-ymax, ymax)

    density_scatter(x_latino_np, ax=axes[2], s=1)
    axes[2].set_title(f"LATINO Restored (N={args.N})")
    axes[2].set_xlim(-xmax, xmax)
    axes[2].set_ylim(-ymax, ymax)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"latino_prior_vs_posterior_sigma_y_{args.sigma_y}.png",
                bbox_inches="tight", dpi=300)
    print(f"Saved prior comparison to {OUTPUT_DIR}/latino_prior_vs_posterior_sigma_y_{args.sigma_y}.png")

    print("\nDone.")
