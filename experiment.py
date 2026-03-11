"""Experiment: H1 — Latent MMPS (Moment-Matching Posterior Sampling in latent space).

Idea: Port MMPS to latent space by propagating Tweedie covariance through
the decoder Jacobian. The likelihood becomes:
  p(y|z_t) ≈ N(y | D(z0_hat), σ_n²I + J_D · V_t · J_D^T)

where V_t = V[z0|z_t] is the (scalar, isotropic) Tweedie covariance.
"""

import jax
import jax.numpy as jnp
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test, latent_posterior_test


def latent_mmps(problem, y, key, *, N=200, sigma_max=3.0, sigma_min=0.01, zeta=1.0):
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
        # Σ_y = σ_n²I + V_t · J_D · J_D^T  (in pixel space)
        # grad_z = J_D^T · Σ_y^{-1} · (y - D(z0_hat))
        residual = y - problem.decoder(z0_hat)     # (..., d_pixel)
        J = problem.decoder_jacobian(z0_hat)        # (..., d_pixel, d_latent)

        # Σ_y = σ_n²I + V_t · J·J^T  shape (..., d_pixel, d_pixel)
        JJT = jnp.einsum('...pi,...qi->...pq', J, J)  # (..., d_pixel, d_pixel)
        Sigma_y = problem.sigma_n**2 * jnp.eye(problem.d_pixel) + V_t * JJT

        # Solve Σ_y^{-1} · residual
        Sigma_y_inv_r = jnp.linalg.solve(Sigma_y, residual[..., None])[..., 0]

        # grad = J^T · Σ_y^{-1} · residual
        grad = jnp.einsum('...pi,...p->...i', J, Sigma_y_inv_r)

        dz0_dz = problem.sigma_0**2 / (problem.sigma_0**2 + sigma_t**2)
        z = z + zeta * g2 * dt * grad * dz0_dz

    return z


if __name__ == "__main__":
    print("=" * 70)
    print("H1: Latent MMPS — NonlinearDecoder2D (alpha=0.5)")
    print("=" * 70)
    problem = NonlinearDecoder2D(alpha=0.5)
    result = latent_calibration_test(problem, latent_mmps, jax.random.PRNGKey(0), n=200)
    print(f"HPD mean: {result['hpd_mean']:.3f} (target: 0.500)")
    print(f"HPD KS:   {result['hpd_ks']:.3f} (target: → 0)")

    print()
    print("=" * 70)
    print("H1: Latent MMPS — FoldedDecoder2D (alpha=1.0)")
    print("=" * 70)
    problem2 = FoldedDecoder2D(alpha=1.0)
    result2 = latent_calibration_test(problem2, latent_mmps, jax.random.PRNGKey(0), n=200)
    print(f"HPD mean: {result2['hpd_mean']:.3f} (target: 0.500)")
    print(f"HPD KS:   {result2['hpd_ks']:.3f} (target: → 0)")
