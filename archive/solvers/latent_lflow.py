"""Latent LFlow -- Flow matching in latent space with Jacobian-aware guidance.

Extension of LFlow (Askari et al., 2025) to latent space. Uses the same
Jacobian-aware covariance as Latent MMPS but with ODE (flow matching)
instead of SDE (diffusion):

  v_guided = v(z_t) - (V_t/t) · (σ_n²I + V_t·J^T·J)^{-1} · J^T · (y - D(ẑ₀))

Deterministic given initial noise. Works well on unimodal posteriors but
lacks SDE noise for mode exploration on bimodal problems.
"""

import jax
import jax.numpy as jnp


def latent_lflow(problem, y, key, *, N=200, t_max=0.999, t_min=0.001, zeta=1.0):
    """Latent LFlow: flow matching with Jacobian-aware guidance."""
    key, subkey = jax.random.split(key)
    var_t = (1 - t_max)**2 * problem.sigma_0**2 + t_max**2
    z = jnp.sqrt(var_t) * jax.random.normal(subkey, (*y.shape[:-1], problem.d_latent))

    dt = (t_max - t_min) / N

    for i in range(N):
        t = t_max - i * dt
        sigma_t2 = (1 - t)**2 * problem.sigma_0**2 + t**2

        # Tweedie denoised estimate in latent space
        z0_hat = (1 - t) * problem.sigma_0**2 / sigma_t2 * z

        # Velocity field in latent space
        z1_hat = t / sigma_t2 * z
        v = z1_hat - z0_hat

        # Tweedie covariance for flow interpolant
        V_t = problem.sigma_0**2 * t**2 / sigma_t2

        # Jacobian-aware guidance (push-through identity)
        residual = y - problem.decoder(z0_hat)
        J = problem.decoder_jacobian(z0_hat)
        JTr = jnp.einsum('...pi,...p->...i', J, residual)
        JTJ = jnp.einsum('...pi,...pj->...ij', J, J)
        M = problem.sigma_n**2 * jnp.eye(problem.d_latent) + V_t * JTJ
        grad = jnp.linalg.solve(M, JTr[..., None])[..., 0]

        guidance = V_t / t * grad
        v_guided = v - zeta * guidance

        z = z - v_guided * dt

    return z
