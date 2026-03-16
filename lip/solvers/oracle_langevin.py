"""Oracle Langevin -- ULA on exact log-posterior, initialized from encoder.

Uses the exact log-posterior gradient: ∇_z log p(z|y) = -(z/σ₀²) + J^T(y-D(z))/σ_n²
This is available for any VAE latent inverse problem since both the decoder
and prior are known.

For MNISTVAE with sigma_n=0.2, the posterior is extremely concentrated
(std ~0.015), requiring very small step sizes (lr ~5e-7).
"""

import jax
import jax.numpy as jnp


def _oracle_langevin_single(problem, y, key, *, N=3000, lr):
    """Single-sample ULA on exact log-posterior with lax.scan."""
    d = problem.d_latent
    z = problem.encoder(y)
    grad_fn = jax.grad(lambda z: problem.log_posterior(z, y))

    def step(carry, _):
        z, key = carry
        key, k1 = jax.random.split(key)
        g = grad_fn(z)
        z = z + lr * g + jnp.sqrt(2 * lr) * jax.random.normal(k1, (d,))
        return (z, key), None

    (z, _), _ = jax.lax.scan(step, (z, key), None, length=N)
    return z


# JIT cache: single-sample and vmapped batch versions
_jit_single = None
_jit_batch = None
_jit_config = None


def oracle_langevin(problem, y, key, *, N=3000, lr=None, batch_size=0,
                    **kwargs):
    """ULA on exact log-posterior, initialized from encoder.

    lr defaults to 5e-7 * (sigma_n / 0.2)^2, scaling with posterior width.
    """
    global _jit_single, _jit_batch, _jit_config

    if lr is None:
        lr = 5e-7 * (problem.sigma_n / 0.2) ** 2

    config = (id(problem), N, lr)
    if _jit_single is None or _jit_config != config:
        _fn = lambda y, key: _oracle_langevin_single(
            problem, y, key, N=N, lr=lr,
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
