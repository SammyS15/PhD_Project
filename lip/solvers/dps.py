"""DPS -- Diffusion Posterior Sampling in latent space.

Paper: Chung et al. "Diffusion Posterior Sampling for General Noisy
Inverse Problems" (arXiv:2209.14687, ICLR 2023).

Reverse VE-SDE with Tweedie-mean likelihood guidance.  At each reverse
step k (from high noise σ_k to low noise σ_{k+1}):

  1. Prior score:   s = -z / (σ₀² + σ_k²)
  2. Tweedie mean:  ẑ₀ = z · σ₀² / (σ₀² + σ_k²)
  3. Decode:        x̂₀ = D(ẑ₀)
  4. Guidance:      g = ∇_{z_t} [ -‖y - x̂₀‖² / (2σ_n²) ]
  5. Reverse SDE:   z ← z + Δσ² · (s + ζ·g) + √Δσ² · ε

Known limitation: ignores the Tweedie covariance V[z₀|z_t], making
guidance too strong at high noise.  For MNISTVAE the decoder Jacobian
norm is large (~10-50), so ζ must be dampened well below 1.0.
"""

import jax
import jax.numpy as jnp


def _dps_single(problem, y, key, *, N=500, sigma_max=1.0, sigma_min=0.01,
                zeta=0.003):
    """Single-sample DPS: reverse VE-SDE with Tweedie mean guidance."""
    d = problem.d_latent
    s02 = problem.sigma_0 ** 2
    sn2 = problem.sigma_n ** 2

    # Noise schedule
    sigmas = jnp.geomspace(sigma_max, sigma_min, N + 1)

    # Initialize from noised prior
    key, k_init = jax.random.split(key)
    z = jnp.sqrt(s02 + sigmas[0] ** 2) * jax.random.normal(k_init, (d,))

    def _guidance_loglik(z_t, sigma):
        """Approximate log p(y|z_t) using only the Tweedie mean."""
        z0_hat = z_t * s02 / (s02 + sigma ** 2)
        x0_hat = problem.decoder(z0_hat)
        return -jnp.sum((y - x0_hat) ** 2) / (2 * sn2)

    grad_guidance = jax.grad(_guidance_loglik)

    def step(carry, k):
        z, key = carry
        sigma_k = sigmas[k]
        sigma_next = sigmas[k + 1]
        key, k_noise = jax.random.split(key)

        # Prior score
        score = -z / (s02 + sigma_k ** 2)

        # DPS guidance
        g = grad_guidance(z, sigma_k)

        
        z0_hat = z * s02 / (s02 + sigma_k ** 2)
        x0_hat = problem.decoder(z0_hat)
        # Residual norm
        residual_norm = jnp.sqrt(jnp.sum((y - x0_hat) ** 2)) + 1e-8
 
        ds2 = sigma_k ** 2 - sigma_next ** 2
        z = z + ds2 * score \
            + jnp.sqrt(ds2) * jax.random.normal(k_noise, (d,)) \
            + zeta * g / residual_norm

        return (z, key), None

    (z, _), _ = jax.lax.scan(step, (z, key), jnp.arange(N))
    return z


# JIT cache: single-sample and vmapped batch versions
_jit_single = None
_jit_batch = None
_jit_config = None


def dps(problem, y, key, *, N=500, sigma_max=1.0, sigma_min=0.01,
        zeta=0.003, batch_size=0, **kwargs):
    """DPS: reverse VE-SDE with Tweedie mean guidance.

    Args:
        problem: MNISTVAE instance.
        y: Observation(s), shape (d_pixel,) or (batch, d_pixel).
        key: JAX PRNG key.
        N: Number of reverse steps.
        sigma_max: Starting noise level.
        sigma_min: Final noise level.
        zeta: Guidance strength (0.01-0.5 for neural decoders).
        batch_size: Max samples to vmap at once (0 = all). Reduce if OOM.

    Returns:
        z: Latent sample(s), shape (d_latent,) or (batch, d_latent).
    """
    global _jit_single, _jit_batch, _jit_config

    config = (id(problem), N, sigma_max, sigma_min, zeta)
    if _jit_single is None or _jit_config != config:
        _fn = lambda y, key: _dps_single(
            problem, y, key, N=N, sigma_max=sigma_max,
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

    # Process in chunks to limit memory on laptops
    chunks = []
    for i in range(0, n, batch_size):
        chunks.append(_jit_batch(y[i:i+batch_size], keys[i:i+batch_size]))
    return jnp.concatenate(chunks)
