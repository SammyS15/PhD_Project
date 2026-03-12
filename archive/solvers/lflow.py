"""LFlow -- Latent Refinement via Flow Matching.

Askari et al., "Latent Refinement via Flow Matching for Training-free
Linear Inverse Problem Solving", arXiv:2511.06138, NeurIPS 2025.

Uses OT interpolant x_t = (1-t)x_0 + t*z_1 with posterior velocity:
  v_t^y(x) = v_t(x) - (t/(1-t)) * nabla_{x_t} log p(y|x_t)

Theoretically exact for Gaussians, limited by Euler discretization.
"""

import jax
import jax.numpy as jnp


def lflow(problem, y, key, *, N=200, t_max=0.999, t_min=0.001, zeta=1.0):
    key, subkey = jax.random.split(key)
    # Initialize from marginal at t ~ 1
    mean_t = (1 - t_max) * problem.mu_0
    var_t = (1 - t_max) ** 2 * problem.sigma_0**2 + t_max**2
    x = mean_t + jnp.sqrt(var_t) * jax.random.normal(subkey, y.shape)

    dt = (t_max - t_min) / N

    for i in range(N):
        t = t_max - i * dt
        sigma_t2 = (1 - t) ** 2 * problem.sigma_0**2 + t**2

        # Tweedie denoised estimate: E[x0 | x_t]
        x0_hat = problem.mu_0 + (1 - t) * problem.sigma_0**2 / sigma_t2 * (
            x - (1 - t) * problem.mu_0
        )

        # Velocity field: v(x_t, t) = E[z1|x_t] - E[x0|x_t]
        z1_hat = t / sigma_t2 * (x - (1 - t) * problem.mu_0)
        v = z1_hat - x0_hat

        # Posterior covariance (exact Tweedie for Gaussian)
        V_t = problem.sigma_0**2 * t**2 / sigma_t2

        # Guidance
        guidance = V_t / t * (y - x0_hat) / (problem.sigma_n**2 + V_t)
        v_guided = v - zeta * guidance

        # Euler step backward in t
        x = x - v_guided * dt

    return x
