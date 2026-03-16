"""MMPS -- Moment Matching Posterior Sampling in latent space.

Paper: Rozet et al. "Learning Diffusion Priors from Observations
by Expectation Maximization" (NeurIPS 2024).

Improves upon DPS by incorporating the Tweedie covariance V[z₀|z_t]
into the likelihood approximation via moment matching (Eq. 19-20),
solved efficiently with the conjugate gradient method (Algorithm 4).

At each reverse step k:
  1. Tweedie mean:  ẑ₀ = z · σ₀² / (σ₀² + σ_k²)
  2. Decode:        x̂₀ = D(ẑ₀)
  3. Solve CG:      u = (Σ_y + A·V[z₀|z_t]·Aᵀ)⁻¹ (y - x̂₀)
     where the matrix-vector product uses VJPs through the denoiser
  4. Likelihood score: ∇_{z_t} log q(y|z_t) via VJP with Aᵀu
  5. Combined score → reverse SDE step
"""

import jax
import jax.numpy as jnp
from functools import partial


def _conjugate_gradient(matvec, b, x0, n_iters=3):
    """    
    Args:
        matvec: Function v -> Mv (matrix-vector product operator)
        b: Right-hand side vector
        x0: Initial guess
        n_iters: Number of CG iterations (paper uses 1-3)
    
    Returns:
        Approximate solution v
    """
    r = b - matvec(x0)
    p = r
    x = x0

    def cg_step(carry, _):
        x, r, p = carry
        Ap = matvec(p)
        r_dot_r = jnp.dot(r, r)
        alpha = r_dot_r / (jnp.dot(p, Ap) + 1e-10)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        beta = jnp.dot(r_new, r_new) / (r_dot_r + 1e-10)
        p_new = r_new + beta * p
        return (x_new, r_new, p_new), None

    (x, _, _), _ = jax.lax.scan(cg_step, (x, r, p), None, length=n_iters)
    return x


def _mmps_single(problem, y, key, *, N=500, sigma_max=1.0, sigma_min=0.01,
                 zeta=5.0, cg_iters=2):
    """Single-sample MMPS: reverse VE-SDE with moment-matched guidance.
    
    Args:
        problem: MNISTVAE instance with .decoder(), .sigma_0, .sigma_n, .d_latent
        y: Observation vector, shape (d_pixel,)
        key: JAX PRNG key
        N: Number of reverse steps
        sigma_max: Starting noise level
        sigma_min: Final noise level
        zeta: Guidance strength (can be ~1.0 since MMPS is more stable than DPS)
        cg_iters: Number of conjugate gradient iterations (1-3 recommended)
    
    Returns:
        z: Latent sample, shape (d_latent,)
    """
    d = problem.d_latent
    s02 = problem.sigma_0 ** 2
    sn2 = problem.sigma_n ** 2

    # Noise schedule
    sigmas = jnp.geomspace(sigma_max, sigma_min, N + 1)

    # Initialize from noised prior
    key, k_init = jax.random.split(key)
    z = jnp.sqrt(s02 + sigmas[0] ** 2) * jax.random.normal(k_init, (d,))

    def step(carry, k):
        z, key = carry
        sigma_k = sigmas[k]
        sigma_next = sigmas[k + 1]
        key, k_noise = jax.random.split(key)

        sk2 = sigma_k ** 2  # σ_t² in the paper's notation

        # Prior score (same as DPS)
        score_prior = -z / (s02 + sk2)

        # Tweedie mean: E[z₀|z_t] = z_t · σ₀² / (σ₀² + σ_k²)
        z0_hat = z * s02 / (s02 + sk2)

        # Decode to observation space
        x0_hat = problem.decoder(z0_hat)

        # === Residual ===
        residual = y - x0_hat  # shape (d_pixel,)

        # The Tweedie covariance factor (scalar for Gaussian prior):
        tweedie_cov_scale = sk2 * s02 / (s02 + sk2)

        def decode_from_z0(z0):
            return problem.decoder(z0)

        def matvec(v):
            """Compute (Σ_y + A·V[z₀|z_t]·Aᵀ) · v using VJP/JVP."""
            # Aᵀv = J_Dᵀ · v (vector-Jacobian product)
            _, vjp_fn = jax.vjp(decode_from_z0, z0_hat)
            JTv = vjp_fn(v)[0]  # shape (d_latent,)

            # A · (V · Aᵀv) = J_D · (tweedie_cov_scale · J_Dᵀ · v)
            _, JJTv = jax.jvp(decode_from_z0, (z0_hat,),
                              (tweedie_cov_scale * JTv,))

            return sn2 * v + JJTv

        u0 = residual / sn2  # initial guess
        u = _conjugate_gradient(matvec, residual, u0, n_iters=cg_iters)

        denoiser_scale = s02 / (s02 + sk2)
        _, vjp_fn = jax.vjp(decode_from_z0, z0_hat)
        # Likelihood score
        score_likelihood = denoiser_scale * vjp_fn(u)[0]
        
        ds2 = sk2 - sigma_next ** 2
        z = z + ds2 * (score_prior + zeta * score_likelihood) \
            + jnp.sqrt(ds2) * jax.random.normal(k_noise, (d,))

        return (z, key), None

    (z, _), _ = jax.lax.scan(step, (z, key), jnp.arange(N))
    return z


# JIT cache: single-sample and vmapped batch versions
_jit_single = None
_jit_batch = None
_jit_config = None


def mmps(problem, y, key, *, N=500, sigma_max=1.0, sigma_min=0.01,
         zeta=1.0, cg_iters=3, batch_size=0, **kwargs):
    """MMPS: Moment Matching Posterior Sampling.

    Args:
        problem: MNISTVAE instance.
        y: Observation(s), shape (d_pixel,) or (batch, d_pixel).
        key: JAX PRNG key.
        N: Number of reverse steps.
        sigma_max: Starting noise level.
        sigma_min: Final noise level.
        zeta: Guidance strength. MMPS is much more stable than DPS,
              so zeta ~ 1.0 often works (vs 0.003 for DPS).
        cg_iters: Conjugate gradient iterations (1-3 recommended).
        batch_size: Max samples to vmap at once (0 = all). Reduce if OOM.

    Returns:
        z: Latent sample(s), shape (d_latent,) or (batch, d_latent).
    """
    global _jit_single, _jit_batch, _jit_config

    config = (id(problem), N, sigma_max, sigma_min, zeta, cg_iters)
    if _jit_single is None or _jit_config != config:
        _fn = lambda y, key: _mmps_single(
            problem, y, key, N=N, sigma_max=sigma_max,
            sigma_min=sigma_min, zeta=zeta, cg_iters=cg_iters,
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