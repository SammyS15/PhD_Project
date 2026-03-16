"""DPS-CSE: Oracle Conditional Score Estimator variant of DPS.

"CSE" is the Xu et al. 2025 (ICLR) name for the class of methods (DPS,
MCGDiff, etc.) that approximate the conditional score
    nabla_z log p_t(z | y).
Standard DPS approximates it as:
    s_uncond(z, sigma) + zeta * nabla_z log p(y | z0_hat(z))

The Oracle CSE decomposes the conditional score as:

    nabla_{z_t} log p_t(z_t | y)
    = nabla_{z_t} log p_t(z_t)          [exact prior score]
    + nabla_{z_t} log p(y | z_t)        [likelihood term, approximated]

The prior score is exact for the Gaussian VE-SDE:
    nabla_{z_t} log p_t(z_t) = -z_t / (s0^2 + sigma^2)

The likelihood term is approximated via the Tweedie chain rule:
    nabla_{z_t} log p(y | z_t)
    ≈ [s0^2 / (s0^2 + sigma^2)] * nabla_{z0} log p(y | z0_hat)

where z0_hat = z_t * s0^2/(s0^2+sigma^2) is the Tweedie denoised estimate.

Note: Using log_posterior (prior + likelihood) for the guidance gradient and
letting it replace the prior score gives a prior contribution of
    -z_t * s0^2 / (s0^2 + sigma^2)^2
which is weaker than the correct -z_t / (s0^2 + sigma^2) by a factor of
s0^2/(s0^2+sigma^2). The fix separates the exact prior score from the
likelihood guidance, ensuring the SDE has the correct centering force.

Key: for MNISTVAE we have exact access to log p(y|z), so the oracle
likelihood gradient is computable. Answers: "if DPS had a perfect likelihood
gradient, how well would it do?"
"""

import jax
import jax.numpy as jnp


def _oracle_cse_single(problem, y, key, *, N=500, sigma_max=1.0, sigma_min=0.01):
    """Single-sample Oracle CSE: reverse VE-SDE with true posterior score."""
    d = problem.d_latent
    s02 = problem.sigma_0 ** 2

    sigmas = jnp.geomspace(sigma_max, sigma_min, N + 1)

    key, k_init = jax.random.split(key)
    z = jnp.sqrt(s02 + sigmas[0] ** 2) * jax.random.normal(k_init, (d,))

    # Oracle conditional score:
    #   exact prior score + likelihood guidance via Tweedie chain rule
    #
    #   nabla_{z_t} log p_t(z_t | y)
    #   = -z_t / (s02 + sigma^2)                           [exact prior score]
    #   + (s02/(s02+sigma^2)) * nabla_{z0} log p(y|z0_hat) [likelihood term]
    def _oracle_score(z_t, sigma):
        z0_hat = z_t * s02 / (s02 + sigma ** 2)
        scale = s02 / (s02 + sigma ** 2)
        prior_score = -z_t / (s02 + sigma ** 2)
        likelihood_grad = jax.grad(lambda z: problem.log_likelihood(z, y))(z0_hat)
        return prior_score + scale * likelihood_grad

    def step(carry, k):
        z, key = carry
        sigma_k = sigmas[k]
        sigma_next = sigmas[k + 1]
        key, k_noise = jax.random.split(key)

        score = _oracle_score(z, sigma_k)
        ds2 = sigma_k ** 2 - sigma_next ** 2
        z = (z + ds2 * score
             + jnp.sqrt(ds2) * jax.random.normal(k_noise, (d,)))

        return (z, key), None

    (z, _), _ = jax.lax.scan(step, (z, key), jnp.arange(N))
    return z


# JIT cache
_jit_single = None
_jit_batch = None
_jit_config = None


def dps_cse(problem, y, key, *, N=500, sigma_max=1.0, sigma_min=0.01,
            batch_size=0, **kwargs):
    """Oracle CSE: reverse VE-SDE using the true posterior score via Tweedie.

    This is an oracle method (requires exact log p(z|y)) that answers:
    "if DPS had a perfect conditional score estimator, how well would it do?"

    Args:
        problem: MNISTVAE instance.
        y: Observation(s), shape (d_pixel,) or (batch, d_pixel).
        key: JAX PRNG key.
        N: Number of reverse steps.
        sigma_max: Starting noise level.
        sigma_min: Final noise level.
        batch_size: Max samples to vmap at once (0 = all).

    Returns:
        z: Latent sample(s), shape (d_latent,) or (batch, d_latent).
    """
    global _jit_single, _jit_batch, _jit_config

    config = (id(problem), N, sigma_max, sigma_min)
    if _jit_single is None or _jit_config != config:
        _fn = lambda y, key: _oracle_cse_single(
            problem, y, key, N=N, sigma_max=sigma_max, sigma_min=sigma_min,
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
