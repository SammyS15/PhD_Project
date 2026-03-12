"""Iteration 20: Check grid sampler KS floor + try higher grid resolution.

If the grid posterior sampler itself has KS > 0, that's a floor for all methods.
Also try increasing grid resolution from 200 to 400 to reduce this floor.
"""
import time
import jax
import jax.numpy as jnp
from lip import MNISTVAE
from lip.metrics import latent_calibration_test

problem = MNISTVAE(latent_dim=2, sigma_n=0.2)


# Grid posterior sampler — samples directly from the grid posterior
def _grid_sample_single(problem, y, key, grid_size=200):
    """Sample from grid-discretized posterior (gold standard)."""
    z1, z2, Z1, Z2, p, dz = problem.posterior_grid(y, grid_size=grid_size)
    # Sample from flattened probability
    log_p_flat = jnp.log(p + 1e-30).reshape(-1)
    log_p_flat = log_p_flat - jax.scipy.special.logsumexp(log_p_flat)
    idx = jax.random.categorical(key, log_p_flat)
    i, j = jnp.divmod(idx, z2.shape[0])
    return jnp.array([z1[j], z2[i]])


def grid_sampler(problem, y, key, **kwargs):
    if y.ndim == 1:
        return _grid_sample_single(problem, y, key)
    keys = jax.random.split(key, y.shape[0])
    return jnp.stack([_grid_sample_single(problem, y[i], keys[i]) for i in range(y.shape[0])])


# Grid sampler with higher resolution
def _grid_sample_fine_single(problem, y, key):
    """Sample from grid-discretized posterior with 400x400 grid."""
    return _grid_sample_single(problem, y, key, grid_size=400)


def grid_sampler_fine(problem, y, key, **kwargs):
    if y.ndim == 1:
        return _grid_sample_fine_single(problem, y, key)
    keys = jax.random.split(key, y.shape[0])
    return jnp.stack([_grid_sample_fine_single(problem, y[i], keys[i]) for i in range(y.shape[0])])


if __name__ == "__main__":
    # Grid sampler floor at n=1000
    print("=== Grid sampler KS floor ===")
    key = jax.random.PRNGKey(0)
    t0 = time.time()
    r = latent_calibration_test(problem, grid_sampler, key, n=1000)
    t1 = time.time()
    print(f"Grid 200x200, n=1000: hpd={r['hpd_mean']:.3f}, std={r['hpd_std']:.3f}, KS={r['hpd_ks']:.3f} ({t1-t0:.0f}s)")

    # Fine grid sampler
    print("\n=== Fine grid (400x400) sampler ===")
    key = jax.random.PRNGKey(0)
    t0 = time.time()
    r = latent_calibration_test(problem, grid_sampler_fine, key, n=1000)
    t1 = time.time()
    print(f"Grid 400x400, n=1000: hpd={r['hpd_mean']:.3f}, std={r['hpd_std']:.3f}, KS={r['hpd_ks']:.3f} ({t1-t0:.0f}s)")

    # Oracle Langevin at n=1000 for comparison
    print("\n=== Oracle Langevin (N=3000, lr=5e-7) ===")
    from lip.solvers import oracle_langevin
    key = jax.random.PRNGKey(0)
    t0 = time.time()
    r = latent_calibration_test(problem, oracle_langevin, key, n=1000)
    t1 = time.time()
    print(f"n=1000: hpd={r['hpd_mean']:.3f}, std={r['hpd_std']:.3f}, KS={r['hpd_ks']:.3f} ({t1-t0:.0f}s)")
