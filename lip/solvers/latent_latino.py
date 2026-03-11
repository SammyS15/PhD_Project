"""Latent LATINO -- LAtent consisTency INverse sOlver.

Follows Algorithm 1 of Spagnoletti et al. (arXiv:2503.12615).
The iterate x lives in PIXEL space. Each step:
  1. Encode x to latent z, add noise:  z_noisy = E(x) + σ_k · ε
  2. Denoise in latent space (PF-ODE):  z_clean = denoise(z_noisy, σ_k)
  3. Decode back to pixel space:        u = D(z_clean)
  4. Proximal step in pixel space:      x = (δ·y + σ_n²·u) / (δ + σ_n²)

Returns z = E(x_final) for comparison against the latent posterior.
"""

import jax
import jax.numpy as jnp


def latent_latino(problem, y, key, *, N=64, sigma_max=2.0, sigma_min=0.01):
    sigma_schedule = jnp.geomspace(sigma_max, sigma_min, N)

    # Initialize in pixel space (observation itself, as in the paper)
    x = y

    for k in range(N):
        sigma_k = sigma_schedule[k]
        key, subkey = jax.random.split(key)

        # Encode to latent + noise (VE-SDE equivalent of √α_t E(x) + √(1-α_t) ε)
        z = problem.encoder(x)
        z_noisy = z + sigma_k * jax.random.normal(subkey, z.shape)

        # Denoise in latent space (deterministic PF-ODE)
        z_clean = problem.denoise(z_noisy, sigma_k)

        # Decode back to pixel space
        u = problem.decoder(z_clean)

        # Proximal step in pixel space: prox_{δ g_y}(u)
        delta_k = float(sigma_k) ** 2
        x = (delta_k * y + problem.sigma_n**2 * u) / (delta_k + problem.sigma_n**2)

    # Return latent encoding of final pixel-space iterate
    return problem.encoder(x)
