"""DPS -- Diffusion Posterior Sampling.

Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse
Problems", arXiv:2209.14687, 2023.

Reverse VE-SDE with likelihood guidance using Tweedie mean only:
  p(y|x_t) ~ N(y | x0_hat, sigma_n^2)

Ignoring the posterior covariance makes guidance too strong, producing
under-dispersed posteriors closer to MAP than true posterior.
"""

import jax
import jax.numpy as jnp


def dps(problem, y, key, *, N=200, sigma_max=3.0, sigma_min=0.01, zeta=1.0):
    key, subkey = jax.random.split(key)
    x = problem.mu_0 + jnp.sqrt(problem.sigma_0**2 + sigma_max**2) * jax.random.normal(
        subkey, y.shape
    )

    log_R = jnp.log(sigma_max / sigma_min)
    dt = 1.0 / N

    for i in range(N):
        t = 1.0 - i * dt
        sigma_t = sigma_min * jnp.exp(log_R * t)
        key, subkey = jax.random.split(key)

        s = problem.score(x, sigma_t)
        x0_hat = x + sigma_t**2 * s

        # Reverse SDE Euler-Maruyama step
        g2 = 2 * sigma_t**2 * log_R
        x = x + g2 * s * dt
        x = x + sigma_t * jnp.sqrt(2 * log_R * dt) * jax.random.normal(subkey, x.shape)

        # DPS guidance: p(y|x_t) ~ N(y | x0_hat, sigma_n^2)
        dx0_dx = problem.sigma_0**2 / (problem.sigma_0**2 + sigma_t**2)
        x = x + zeta * g2 * dt * (y - x0_hat) / problem.sigma_n**2 * dx0_dx

    return x
