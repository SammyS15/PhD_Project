"""Latent DPS -- Diffusion Posterior Sampling in latent space.

DPS adapted for nonlinear decoder problems. The guidance uses:
  ∇_z log p(y|z_t) ≈ (dz0/dz_t) · J_D(z0_hat)^T · (y - D(z0_hat)) / σ_n²

where J_D is the decoder Jacobian evaluated at the Tweedie estimate z0_hat.
"""

import jax
import jax.numpy as jnp


def latent_dps(problem, y, key, *, N=200, sigma_max=3.0, sigma_min=0.01, zeta=1.0):
    key, subkey = jax.random.split(key)
    z = jnp.sqrt(problem.sigma_0**2 + sigma_max**2) * jax.random.normal(
        subkey, (*y.shape[:-1], problem.d_latent)
    )

    log_R = jnp.log(sigma_max / sigma_min)
    dt = 1.0 / N

    for i in range(N):
        t = 1.0 - i * dt
        sigma_t = sigma_min * jnp.exp(log_R * t)
        key, subkey = jax.random.split(key)

        s = problem.score(z, sigma_t)
        z0_hat = z + sigma_t**2 * s

        # Reverse SDE Euler-Maruyama step
        g2 = 2 * sigma_t**2 * log_R
        z = z + g2 * s * dt
        z = z + sigma_t * jnp.sqrt(2 * log_R * dt) * jax.random.normal(subkey, z.shape)

        # DPS guidance through nonlinear decoder
        residual = y - problem.decoder(z0_hat)  # (..., d_pixel)
        J = problem.decoder_jacobian(z0_hat)     # (..., d_pixel, d_latent)
        grad = jnp.einsum('...pi,...p->...i', J, residual) / problem.sigma_n**2
        dz0_dz = problem.sigma_0**2 / (problem.sigma_0**2 + sigma_t**2)
        z = z + zeta * g2 * dt * grad * dz0_dz

    return z
