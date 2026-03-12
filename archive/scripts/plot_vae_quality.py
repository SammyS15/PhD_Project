"""Generate figure showing VAE autoencoding quality and prior samples.

Row 1: Original MNIST digits and their reconstructions (encode → decode).
Row 2: Samples from the Gaussian prior z ~ N(0,I) decoded through the VAE.

Produces figures/vae_quality_d{dim}.png for d=2 and d=20.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from lip.problems import MNISTVAE
from train_vae import load_mnist


def load_test_digits(n=8, seed=0):
    """Load a few MNIST digits for display."""
    images = load_mnist()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(images), n, replace=False)
    return images[idx]


def make_figure(latent_dim, n_show=8):
    problem = MNISTVAE(latent_dim=latent_dim)
    key = jax.random.PRNGKey(42)

    # --- Row 1: autoencoding quality ---
    originals = load_test_digits(n_show, seed=0)
    originals_jax = jnp.array(originals)
    z_enc = problem.encoder(originals_jax)
    reconstructions = np.array(problem.decoder(z_enc))

    # --- Row 2: prior samples z ~ N(0, I) ---
    z_prior = jax.random.normal(key, (n_show, latent_dim))
    prior_samples = np.array(problem.decoder(z_prior))

    # --- Plot ---
    fig, axes = plt.subplots(3, n_show, figsize=(1.6 * n_show, 1.6 * 3))

    for i in range(n_show):
        # Original
        axes[0, i].imshow(originals[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")

        # Reconstruction
        axes[1, i].imshow(reconstructions[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")

        # Prior sample
        axes[2, i].imshow(prior_samples[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=11, rotation=0, labelpad=70, va="center")
    axes[1, 0].set_ylabel("Reconstructed", fontsize=11, rotation=0, labelpad=70, va="center")
    axes[2, 0].set_ylabel("Prior sample", fontsize=11, rotation=0, labelpad=70, va="center")

    fig.suptitle(f"MNIST VAE (latent dim = {latent_dim})", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0.08, 0, 1, 0.95])
    return fig


if __name__ == "__main__":
    for d in [2, 20]:
        fig = make_figure(d)
        out = f"figures/vae_quality_d{d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")
