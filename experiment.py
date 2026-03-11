"""Experiment: H1b verification — larger sample, timing, and zeta=1.1 vs 1.15."""

import jax
import jax.numpy as jnp
import time
from functools import partial
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test
from lip.solvers.latent_mmps import latent_mmps
from lip.solvers.latent_dps import latent_dps


if __name__ == "__main__":
    problem = NonlinearDecoder2D(alpha=0.5)
    problem2 = FoldedDecoder2D(alpha=1.0)

    # Larger sample verification with zeta=1.1
    print("=" * 70)
    print("Verification with n=500, zeta=1.1")
    print("=" * 70)
    solver = partial(latent_mmps, zeta=1.1)

    t0 = time.time()
    r1 = latent_calibration_test(problem, solver, jax.random.PRNGKey(42), n=500)
    t_mmps_nl = time.time() - t0
    print(f"NonlinearDecoder2D: hpd_mean={r1['hpd_mean']:.3f}, KS={r1['hpd_ks']:.3f} (time={t_mmps_nl:.1f}s)")

    t0 = time.time()
    r2 = latent_calibration_test(problem2, solver, jax.random.PRNGKey(42), n=500)
    t_mmps_f = time.time() - t0
    print(f"FoldedDecoder2D:    hpd_mean={r2['hpd_mean']:.3f}, KS={r2['hpd_ks']:.3f} (time={t_mmps_f:.1f}s)")

    # DPS timing for comparison
    t0 = time.time()
    r3 = latent_calibration_test(problem, latent_dps, jax.random.PRNGKey(42), n=500)
    t_dps = time.time() - t0
    print(f"\nDPS reference:      hpd_mean={r3['hpd_mean']:.3f}, KS={r3['hpd_ks']:.3f} (time={t_dps:.1f}s)")
    print(f"Cost ratio: Latent MMPS / Latent DPS = {t_mmps_nl/t_dps:.1f}x")

    # Also test zeta=1.15
    print("\n" + "=" * 70)
    print("Verification with n=500, zeta=1.15")
    print("=" * 70)
    solver2 = partial(latent_mmps, zeta=1.15)
    r4 = latent_calibration_test(problem, solver2, jax.random.PRNGKey(42), n=500)
    r5 = latent_calibration_test(problem2, solver2, jax.random.PRNGKey(42), n=500)
    print(f"NonlinearDecoder2D: hpd_mean={r4['hpd_mean']:.3f}, KS={r4['hpd_ks']:.3f}")
    print(f"FoldedDecoder2D:    hpd_mean={r5['hpd_mean']:.3f}, KS={r5['hpd_ks']:.3f}")
