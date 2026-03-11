"""MAP-Laplace -- Newton MAP + Laplace approximation sampler.

Finds the true MAP via Newton iteration on the exact log-posterior,
then samples from the Laplace (Gaussian) approximation at the MAP.

For nearly-Gaussian posteriors (tight sigma_n), this is fast and accurate.
No diffusion model needed — uses only the decoder Jacobian and Hessian.
"""

import jax
import jax.numpy as jnp


def _map_laplace_single(problem, y, key, *, n_newton=10):
    """Single-sample: Newton MAP + Laplace sample."""
    d = problem.d_latent
    z = problem.encoder(y)

    # Newton iteration to find exact MAP
    for _ in range(n_newton):
        g = jax.grad(lambda z: problem.log_posterior(z, y))(z)
        H = jax.hessian(lambda z: problem.log_posterior(z, y))(z)
        dz = -jnp.linalg.solve(H, g)
        z = z + dz

    # Laplace sample at MAP
    H = jax.hessian(lambda z: problem.log_posterior(z, y))(z)
    cov = jnp.linalg.inv(-H)
    L = jnp.linalg.cholesky(cov)
    z_sample = z + L @ jax.random.normal(key, (d,))
    return z_sample


def map_laplace(problem, y, key, *, n_newton=10, **kwargs):
    """Newton MAP + Laplace approximation sampler."""
    if y.ndim == 1:
        return _map_laplace_single(problem, y, key, n_newton=n_newton)
    keys = jax.random.split(key, y.shape[0])
    results = []
    for i in range(y.shape[0]):
        results.append(_map_laplace_single(problem, y[i], keys[i],
                                            n_newton=n_newton))
    return jnp.stack(results)
