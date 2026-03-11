"""Experiment: H11 — Fewer diffusion steps for speed.

Test N=50, 100, 200 to see if Latent MMPS can be faster.
"""

import jax
import time
from functools import partial
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test
from lip.solvers.latent_mmps import latent_mmps


if __name__ == "__main__":
    p1 = NonlinearDecoder2D(alpha=0.5)
    p2 = FoldedDecoder2D(alpha=1.0)

    print("H11: Step count sweep (zeta=1.1)")
    print(f"{'N':>5} | {'NL hpd':>7} {'NL ks':>6} | {'F hpd':>7} {'F ks':>6} | {'time':>6}")
    print("-" * 55)

    for N in [50, 100, 150, 200, 300]:
        solver = partial(latent_mmps, zeta=1.1, N=N)
        t0 = time.time()
        r1 = latent_calibration_test(p1, solver, jax.random.PRNGKey(0), n=200)
        r2 = latent_calibration_test(p2, solver, jax.random.PRNGKey(0), n=200)
        elapsed = time.time() - t0

        ok1 = "✓" if 0.45 <= r1['hpd_mean'] <= 0.55 and r1['hpd_ks'] < 0.10 else " "
        ok2 = "✓" if 0.45 <= r2['hpd_mean'] <= 0.55 and r2['hpd_ks'] < 0.10 else " "
        print(f"{N:5d} | {r1['hpd_mean']:7.3f} {r1['hpd_ks']:6.3f} {ok1}| {r2['hpd_mean']:7.3f} {r2['hpd_ks']:6.3f} {ok2}| {elapsed:6.1f}s")
