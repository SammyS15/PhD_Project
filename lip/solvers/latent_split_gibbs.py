"""Latent Split Gibbs -- alternating denoising prior + Langevin likelihood.

Inspired by PnP-DM (Wu et al., NeurIPS 2024). Alternates:
  (a) Denoising diffusion step: z → z_noisy → denoise(z_noisy) (prior)
  (b) Langevin step on likelihood: z += η·∇_z log p(y|D(z))

No Tweedie approximation needed. Asymptotically exact as K→∞.
In practice, limited iterations cause mode-trapping on bimodal posteriors.
"""

import jax
import jax.numpy as jnp


def latent_split_gibbs(problem, y, key, *, K=200, sigma_denoise=0.5,
                        n_langevin=5, eta=0.01):
    """Split Gibbs: alternate denoising prior + Langevin likelihood."""
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (*y.shape[:-1], problem.d_latent))

    for k in range(K):
        key, k1, k2 = jax.random.split(key, 3)
        # Anneal noise level
        t = 1.0 - k / K
        sigma_k = sigma_denoise * t + 0.01 * (1 - t)

        # (a) Denoising diffusion prior step
        z_noisy = z + sigma_k * jax.random.normal(k1, z.shape)
        z = problem.denoise(z_noisy, sigma_k, key=k2)

        # (b) Langevin steps on likelihood
        for l in range(n_langevin):
            key, subkey = jax.random.split(key)
            residual = y - problem.decoder(z)
            J = problem.decoder_jacobian(z)
            grad_ll = jnp.einsum('...pi,...p->...i', J, residual) / problem.sigma_n**2
            z = z + eta * grad_ll
            z = z + jnp.sqrt(2 * eta) * jax.random.normal(subkey, z.shape)

    return z
