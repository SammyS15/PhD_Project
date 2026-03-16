"""DMAP -- Diffusion MAP in latent space.

Paper: Xu et al. "Rethinking Diffusion Posterior Sampling: From Conditional
Score Estimator to Maximizing a Posterior" (ICLR 2025).

Algorithm 2: at each reverse step t (T -> 1):
  1. Sample z_{t-1} ~ p_theta(z_{t-1} | z_t) via reverse SDE;
     compute mean mu_{t-1} = E[z_{t-1} | z_t].
  2. Run K inner iterations:
     a. Gradient step:  z_{t-1} -= zeta_t * grad_z ||f(E[z0|z_{t-1}]) - y||^2
     b. Projection:     z_{t-1}  = mu_{t-1} + r * (z_{t-1}-mu_{t-1})
                                             / ||z_{t-1}-mu_{t-1}||
        where r = sqrt(d) * sigma_noise is the typical Gaussian distance.
"""

import jax
import jax.numpy as jnp


def _dmap_single(problem, y, key, *, N=500, K=5, sigma_max=1.0, sigma_min=0.01,
                 zeta=0.1):
    """Single-sample DMAP: reverse VE-SDE + spherical-projection inner loop."""
    d = problem.d_latent
    s02 = problem.sigma_0 ** 2
    sn2 = problem.sigma_n ** 2

    # Noise schedule
    sigmas = jnp.geomspace(sigma_max, sigma_min, N + 1)

    # Initialize from the noised prior
    key, k_init = jax.random.split(key)
    z = jnp.sqrt(s02 + sigmas[0] ** 2) * jax.random.normal(k_init, (d,))

    # Likelihood loss
    def _loss(z_t, sigma):
        z0_hat = z_t * s02 / (s02 + sigma ** 2)
        x0_hat = problem.decoder(z0_hat)
        return jnp.sum((y - x0_hat) ** 2) / (2 * sn2)

    grad_loss = jax.grad(_loss)

    def step(carry, k):
        z, key = carry
        sigma_k = sigmas[k]
        sigma_next = sigmas[k + 1]
        key, k_noise = jax.random.split(key)

        # reverse VE-SDE step
        score = -z / (s02 + sigma_k ** 2)
        ds2 = sigma_k ** 2 - sigma_next ** 2
        noise = jax.random.normal(k_noise, (d,))
        mu = z + ds2 * score                         
        z_prev = mu + jnp.sqrt(ds2) * noise          

        # Sphere radius
        r = jnp.sqrt(jnp.array(d, dtype=jnp.float32) * ds2)

        def inner_step(z_inner, _):
            # K steps of gradient + spherical projection
            g = grad_loss(z_inner, sigma_next)
            z_inner = z_inner - zeta * g
            diff = z_inner - mu
            z_inner = mu + r * diff / (jnp.linalg.norm(diff) + 1e-8)
            return z_inner, None

        z_prev, _ = jax.lax.scan(inner_step, z_prev, None, length=K)

        return (z_prev, key), None

    (z, _), _ = jax.lax.scan(step, (z, key), jnp.arange(N))
    return z


# JIT cache: single-sample and vmapped batch versions
_jit_single = None
_jit_batch = None
_jit_config = None


def dmap(problem, y, key, *, N=500, K=5, sigma_max=1.0, sigma_min=0.01,
         zeta=0.1, batch_size=0, **kwargs):
    """DMAP: reverse VE-SDE with spherical-projection inner optimization.

    Args:
        problem: MNISTVAE instance.
        y: Observation(s), shape (d_pixel,) or (batch, d_pixel).
        key: JAX PRNG key.
        N: Number of reverse SDE steps.
        K: Number of inner gradient+projection iterations per step.
        sigma_max: Starting noise level.
        sigma_min: Final noise level.
        zeta: Gradient step size for inner optimization.
        batch_size: Max samples to vmap at once (0 = all). Reduce if OOM.

    Returns:
        z: Latent sample(s), shape (d_latent,) or (batch, d_latent).
    """
    global _jit_single, _jit_batch, _jit_config

    config = (id(problem), N, K, sigma_max, sigma_min, zeta)
    if _jit_single is None or _jit_config != config:
        _fn = lambda y, key: _dmap_single(
            problem, y, key, N=N, K=K, sigma_max=sigma_max,
            sigma_min=sigma_min, zeta=zeta,
        )
        _jit_single = jax.jit(_fn)
        _jit_batch = jax.jit(jax.vmap(_fn))
        _jit_config = config

    if y.ndim == 1:
        return _jit_single(y, key)

    keys = jax.random.split(key, y.shape[0])
    n = y.shape[0]

    if batch_size <= 0 or n <= batch_size:
        return _jit_batch(y, keys)

    chunks = []
    for i in range(0, n, batch_size):
        chunks.append(_jit_batch(y[i:i+batch_size], keys[i:i+batch_size]))
    return jnp.concatenate(chunks)
