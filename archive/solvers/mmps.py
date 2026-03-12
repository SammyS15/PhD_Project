"""MMPS -- Moment-Matching Posterior Sampling.

Rozet et al., "Learning Diffusion Priors from Observations by Expectation
Maximization", arXiv:2405.13712, 2024.

Like DPS but uses the full Tweedie posterior moments in the likelihood:
  p(y|x_t) ~ N(y | x0_hat, sigma_n^2 + V[x0|x_t])

Exactly calibrated for Gaussian priors (the moment-matched likelihood
is the true marginal likelihood).
"""

import jax
import jax.numpy as jnp


def mmps(problem, y, key, *, N=200, sigma_max=3.0, sigma_min=0.01, zeta=1.0):
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

        # Tweedie posterior covariance
        V_t = problem.tweedie_cov(x, sigma_t)

        # Reverse SDE Euler-Maruyama step
        g2 = 2 * sigma_t**2 * log_R
        x = x + g2 * s * dt
        x = x + sigma_t * jnp.sqrt(2 * log_R * dt) * jax.random.normal(subkey, x.shape)

        # MMPS guidance: sigma_n^2 + V_t in denominator (vs sigma_n^2 for DPS)
        dx0_dx = problem.sigma_0**2 / (problem.sigma_0**2 + sigma_t**2)
        x = x + zeta * g2 * dt * (y - x0_hat) / (problem.sigma_n**2 + V_t) * dx0_dx

    return x
