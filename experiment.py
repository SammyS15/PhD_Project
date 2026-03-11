"""Experiment: H12 — Adaptive zeta: find optimal zeta for each alpha.

Higher alpha needs higher zeta to compensate for decoder nonlinearity.
Can we find a simple zeta(alpha) rule?
"""

import jax
import jax.numpy as jnp
from functools import partial
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test
from lip.solvers.latent_mmps import latent_mmps


if __name__ == "__main__":
    print("Optimal zeta search per alpha — NonlinearDecoder2D")
    print(f"{'alpha':>6} | {'best_zeta':>9} | {'hpd_mean':>8} | {'hpd_ks':>6}")
    print("-" * 45)

    best_results = {}
    for alpha in [0.0, 0.2, 0.5, 0.7, 1.0, 1.5]:
        problem = NonlinearDecoder2D(alpha=alpha)
        best_zeta, best_dist = 1.0, 999
        best_r = None
        for zeta in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
            solver = partial(latent_mmps, zeta=zeta)
            r = latent_calibration_test(problem, solver, jax.random.PRNGKey(0), n=200)
            dist = abs(r['hpd_mean'] - 0.5)
            if dist < best_dist:
                best_dist = dist
                best_zeta = zeta
                best_r = r
        ok = "✓" if 0.45 <= best_r['hpd_mean'] <= 0.55 and best_r['hpd_ks'] < 0.10 else " "
        print(f"{alpha:6.1f} | {best_zeta:9.1f} | {best_r['hpd_mean']:8.3f} | {best_r['hpd_ks']:6.3f} {ok}")
        best_results[alpha] = best_zeta

    # Check if linear fit zeta = 1.0 + c*alpha works
    alphas = list(best_results.keys())
    zetas = list(best_results.values())
    print(f"\nOptimal zetas: {dict(zip(alphas, zetas))}")

    # Test the best fixed zeta on FoldedDecoder2D
    print("\nFoldedDecoder2D check with various zeta:")
    problem2 = FoldedDecoder2D(alpha=1.0)
    for zeta in [1.0, 1.1, 1.2, 1.3]:
        solver = partial(latent_mmps, zeta=zeta)
        r = latent_calibration_test(problem2, solver, jax.random.PRNGKey(0), n=200)
        ok = "✓" if 0.45 <= r['hpd_mean'] <= 0.55 and r['hpd_ks'] < 0.10 else " "
        print(f"  zeta={zeta:.1f}: hpd_mean={r['hpd_mean']:.3f}, KS={r['hpd_ks']:.3f} {ok}")
