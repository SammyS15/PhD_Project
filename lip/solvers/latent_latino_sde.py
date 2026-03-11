"""Latent LATINO+SDE -- LATINO with stochastic denoiser in latent space.

Same as Latent LATINO but uses the stochastic (SDE) denoiser instead
of the deterministic PF-ODE. In pixel space this nearly calibrates
(z_std=0.979), but in latent space the encode-decode round-trip
breaks the variance correction, making it worse than deterministic LATINO.
"""

import jax
import jax.numpy as jnp


def latent_latino_sde(problem, y, key, *, N=64, sigma_max=2.0, sigma_min=0.01):
    """Latent LATINO with stochastic (SDE) denoiser."""
    sigma_schedule = jnp.geomspace(sigma_max, sigma_min, N)
    x = y

    for k in range(N):
        sigma_k = sigma_schedule[k]
        key, k1, k2 = jax.random.split(key, 3)

        z = problem.encoder(x)
        z_noisy = z + sigma_k * jax.random.normal(k1, z.shape)

        # Stochastic denoiser (key=k2 enables SDE mode)
        z_clean = problem.denoise(z_noisy, sigma_k, key=k2)

        u = problem.decoder(z_clean)
        delta_k = float(sigma_k) ** 2
        x = (delta_k * y + problem.sigma_n**2 * u) / (delta_k + problem.sigma_n**2)

    return problem.encoder(x)
