"""Oracle Langevin -- ULA on exact log-posterior, initialized from encoder.

This is an oracle method that requires access to the exact log-posterior
gradient. It validates that calibrated sampling is achievable and serves
as a reference for practical methods.

For MNISTVAE with sigma_n=0.2, the posterior is extremely concentrated
(std ~0.015), requiring very small step sizes (lr ~1e-6).
"""

import jax
import jax.numpy as jnp


def _oracle_langevin_single(problem, y, key, *, N=3000, lr=1e-6):
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


# JIT-compiled version for speed
_jit_solve = None
_jit_config = None


def oracle_langevin(problem, y, key, *, N=3000, lr=1e-6, **kwargs):
    """ULA on exact log-posterior, initialized from encoder."""
    global _jit_solve, _jit_config

    config = (id(problem), N, lr)
    if _jit_solve is None or _jit_config != config:
        _jit_solve = jax.jit(
            lambda y, key: _oracle_langevin_single(problem, y, key, N=N, lr=lr)
        )
        _jit_config = config

    if y.ndim == 1:
        return _jit_solve(y, key)

    keys = jax.random.split(key, y.shape[0])
    results = []
    for i in range(y.shape[0]):
        results.append(_jit_solve(y[i], keys[i]))
    return jnp.stack(results)
