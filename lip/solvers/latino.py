"""LATINO -- LAtent consisTency INverse sOlver (deterministic PF-ODE).

Spagnoletti et al., "LATINO-PRO: LAtent consisTency INverse sOlver with
PRompt Optimization", arXiv:2503.12615, 2025.

Each iteration: add noise sigma_k -> denoise via PF-ODE -> proximal step.
Provably under-dispersed: the proximal contraction cannot be compensated
by the deterministic denoiser.
"""

import jax
import jax.numpy as jnp


def latino(problem, y, key, *, N=64, sigma_max=2.0, sigma_min=0.01):
    sigma_schedule = jnp.geomspace(sigma_max, sigma_min, N)
    x = y
    for k in range(N):
        sigma_k = sigma_schedule[k]
        key, subkey = jax.random.split(key)
        x_noisy = x + sigma_k * jax.random.normal(subkey, x.shape)
        u = problem.denoise(x_noisy, sigma_k)
        delta_k = float(sigma_k) ** 2
        x = (delta_k * y + problem.sigma_n**2 * u) / (delta_k + problem.sigma_n**2)
    return x
