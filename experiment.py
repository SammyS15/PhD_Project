"""H1h: Oracle ULA verification with n=500 + record baseline.

Verify step=0.3, N=5000 achieves calibration with larger sample.
Then commit as the Oracle baseline.
"""
import jax
import jax.numpy as jnp
from lip import MNISTVAE
from lip.metrics import latent_calibration_test

problem = MNISTVAE(latent_dim=2, sigma_n=0.2)


def _precond_ula_single(problem, y, key, *, N=5000, step_size=0.3, burnin=1000):
    d = problem.d_latent
    z_map = problem.encoder(y)
    H = jax.hessian(lambda z: problem.log_posterior(z, y))(z_map)
    neg_H = -H
    L = jnp.linalg.cholesky(neg_H)
    L_inv = jnp.linalg.inv(L)
    cov = L_inv.T @ L_inv
    grad_log_p = jax.grad(lambda z: problem.log_posterior(z, y))

    def step(carry, _):
        z, key = carry
        key, k1 = jax.random.split(key)
        g = grad_log_p(z)
        z = z + step_size * (cov @ g) + jnp.sqrt(2 * step_size) * (L_inv.T @ jax.random.normal(k1, (d,)))
        return (z, key), None

    (z_final, _), _ = jax.lax.scan(step, (z_map, key), None, length=N + burnin)
    return z_final


def oracle_precond_ula(problem, y, key, *, N=5000, step_size=0.3, burnin=1000, **kwargs):
    if y.ndim == 1:
        return _precond_ula_single(problem, y, key, N=N, step_size=step_size, burnin=burnin)
    keys = jax.random.split(key, y.shape[0])
    return jax.vmap(
        lambda yi, ki: _precond_ula_single(problem, yi, ki, N=N, step_size=step_size, burnin=burnin)
    )(y, keys)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # Verification with n=500
    print("=== Oracle Preconditioned ULA (step=0.3, N=5000, n=500) ===")
    result = latent_calibration_test(
        problem, oracle_precond_ula, key, n=500,
        step_size=0.3, N=5000, burnin=1000
    )
    print(f"HPD mean: {result['hpd_mean']:.3f} (target: 0.500)")
    print(f"HPD std:  {result['hpd_std']:.3f} (target: 0.289)")
    print(f"HPD KS:   {result['hpd_ks']:.3f} (target: < 0.1)")

    # Also re-evaluate existing solvers with fixed grid
    print("\n=== All solvers with adaptive fine grid (n=200) ===")
    from lip.solvers import LATENT_ALL
    for name, solver in LATENT_ALL.items():
        k, key = jax.random.split(key)
        r = latent_calibration_test(problem, solver, k, n=200)
        ok = "✓" if 0.45 <= r['hpd_mean'] <= 0.55 and r['hpd_ks'] < 0.10 else " "
        print(f"  {name:<25s}: hpd={r['hpd_mean']:.3f} KS={r['hpd_ks']:.3f} {ok}")
