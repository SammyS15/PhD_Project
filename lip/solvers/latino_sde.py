"""LATINO+SDE -- LATINO with stochastic reverse SDE denoiser.

Same structure as vanilla LATINO, but replaces the deterministic PF-ODE
denoiser with the stochastic reverse SDE. The SDE injects additional noise
during denoising, which restores most of the missing variance.
Nearly calibrated.
"""

import jax
import jax.numpy as jnp


def latino_sde(problem, y, key, *, N=64, sigma_max=2.0, sigma_min=0.01):
    sigma_schedule = jnp.geomspace(sigma_max, sigma_min, N)
    x = y
    for k in range(N):
        sigma_k = sigma_schedule[k]
        key, k1, k2 = jax.random.split(key, 3)
        x_noisy = x + sigma_k * jax.random.normal(k1, x.shape)
        u = problem.denoise(x_noisy, sigma_k, key=k2)  # stochastic denoiser
        delta_k = float(sigma_k) ** 2
        x = (delta_k * y + problem.sigma_n**2 * u) / (delta_k + problem.sigma_n**2)
    return x
