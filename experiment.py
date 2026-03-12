"""Scratch pad for prototyping new solvers on MNISTVAE.

Quick test pattern:
    python experiment.py
"""
import time
import jax
import jax.numpy as jnp
from lip import MNISTVAE
from lip.metrics import latent_calibration_test

problem = MNISTVAE(latent_dim=2, sigma_n=0.2)


if __name__ == "__main__":
    from lip.solvers import oracle_langevin, latent_latino

    key = jax.random.PRNGKey(0)

    print("=== Oracle Langevin (N=3000, lr=5e-7) ===")
    t0 = time.time()
    r = latent_calibration_test(problem, oracle_langevin, key, n=200)
    t1 = time.time()
    print(f"hpd_mean={r['hpd_mean']:.3f}, KS={r['hpd_ks']:.3f} ({t1-t0:.0f}s)")

    print("\n=== Latent LATINO (N=64) ===")
    t0 = time.time()
    r = latent_calibration_test(problem, latent_latino, key, n=200)
    t1 = time.time()
    print(f"hpd_mean={r['hpd_mean']:.3f}, KS={r['hpd_ks']:.3f} ({t1-t0:.0f}s)")
