"""Experiment: H10 — Annealed ULA on exact log-posterior (oracle).

Simpler than MALA (no MH step), faster. Use as oracle reference.
"""

import jax
import jax.numpy as jnp
from functools import partial
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test


def annealed_ula(problem, y, key, *, n_steps=1000, eta=0.005,
                  n_anneal=300, T_max=3.0):
    """Annealed ULA on exact log-posterior (no MH correction)."""
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (*y.shape[:-1], problem.d_latent))

    grad_fn = jax.grad(lambda z, y: problem.log_posterior(z, y))

    for i in range(n_steps):
        if i < n_anneal:
            T = T_max * (1 - i / n_anneal) + 1.0 * (i / n_anneal)
        else:
            T = 1.0

        key, subkey = jax.random.split(key)

        if z.ndim == 1:
            g = grad_fn(z, y)
        else:
            g = jax.vmap(grad_fn)(z, y)

        z = z + eta * g / T + jnp.sqrt(2 * eta / T) * jax.random.normal(subkey, z.shape)

    return z


if __name__ == "__main__":
    p1 = NonlinearDecoder2D(alpha=0.5)
    p2 = FoldedDecoder2D(alpha=1.0)

    print("H10: Annealed ULA oracle (n_steps=1000)")
    for name, p in [("NonlinearDecoder2D", p1), ("FoldedDecoder2D", p2)]:
        r = latent_calibration_test(p, annealed_ula, jax.random.PRNGKey(0), n=100)
        ok = "✓" if 0.45 <= r['hpd_mean'] <= 0.55 and r['hpd_ks'] < 0.10 else " "
        print(f"  {name}: hpd={r['hpd_mean']:.3f} KS={r['hpd_ks']:.3f} {ok}")

    # Compare
    from lip.solvers.latent_mmps import latent_mmps
    solver = partial(latent_mmps, zeta=1.1)
    print("\nLatent MMPS (zeta=1.1):")
    for name, p in [("NonlinearDecoder2D", p1), ("FoldedDecoder2D", p2)]:
        r = latent_calibration_test(p, solver, jax.random.PRNGKey(0), n=100)
        ok = "✓" if 0.45 <= r['hpd_mean'] <= 0.55 and r['hpd_ks'] < 0.10 else " "
        print(f"  {name}: hpd={r['hpd_mean']:.3f} KS={r['hpd_ks']:.3f} {ok}")
