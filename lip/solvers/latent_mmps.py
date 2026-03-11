"""Latent MMPS -- Moment-Matching Posterior Sampling in latent space.

Extension of MMPS (Rozet et al., 2024) to latent space. The likelihood
approximation propagates Tweedie covariance through the decoder Jacobian:

  p(y|z_t) ≈ N(y | D(ẑ₀), σ_n²I + J_D · V[z|z_t] · J_Dᵀ)

This reduces to standard DPS when V_t is dropped (= 0).
"""

import jax
import jax.numpy as jnp


def latent_mmps(problem, y, key, *, N=200, sigma_max=3.0, sigma_min=0.01, zeta=1.1):
    """Latent MMPS: DPS + Tweedie covariance through decoder Jacobian."""
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

        # Tweedie posterior covariance (scalar, isotropic)
        V_t = problem.tweedie_cov(z, sigma_t)

        # Reverse SDE Euler-Maruyama step
        g2 = 2 * sigma_t**2 * log_R
        z = z + g2 * s * dt
        z = z + sigma_t * jnp.sqrt(2 * log_R * dt) * jax.random.normal(subkey, z.shape)

        # MMPS guidance through decoder Jacobian
        # Σ_y = σ_n²I + V_t · J_D · J_D^T
        residual = y - problem.decoder(z0_hat)
        J = problem.decoder_jacobian(z0_hat)
        JJT = jnp.einsum('...pi,...qi->...pq', J, J)
        Sigma_y = problem.sigma_n**2 * jnp.eye(problem.d_pixel) + V_t * JJT

        # grad = J^T · Σ_y^{-1} · residual
        Sigma_y_inv_r = jnp.linalg.solve(Sigma_y, residual[..., None])[..., 0]
        grad = jnp.einsum('...pi,...p->...i', J, Sigma_y_inv_r)

        dz0_dz = problem.sigma_0**2 / (problem.sigma_0**2 + sigma_t**2)
        z = z + zeta * g2 * dt * grad * dz0_dz

    return z
