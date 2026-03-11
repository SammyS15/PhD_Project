"""Experiment: H8 — Phase diagram: sweep alpha on NonlinearDecoder2D.

At what nonlinearity does Latent MMPS break? Also compare DPS and LATINO.
"""

import jax
import jax.numpy as jnp
from functools import partial
from lip import NonlinearDecoder2D
from lip.metrics import latent_calibration_test
from lip.solvers.latent_mmps import latent_mmps
from lip.solvers.latent_dps import latent_dps
from lip.solvers.latent_latino import latent_latino


if __name__ == "__main__":
    print("Phase diagram: NonlinearDecoder2D, alpha sweep")
    print(f"{'alpha':>6} | {'MMPS hpd':>8} {'MMPS ks':>7} | {'DPS hpd':>8} {'DPS ks':>7} | {'LAT hpd':>8} {'LAT ks':>7}")
    print("-" * 75)

    solver_mmps = partial(latent_mmps, zeta=1.1)

    for alpha in [0.0, 0.2, 0.5, 0.7, 1.0, 1.5]:
        problem = NonlinearDecoder2D(alpha=alpha)
        r_mmps = latent_calibration_test(problem, solver_mmps, jax.random.PRNGKey(0), n=200)
        r_dps = latent_calibration_test(problem, latent_dps, jax.random.PRNGKey(0), n=200)
        r_lat = latent_calibration_test(problem, latent_latino, jax.random.PRNGKey(0), n=200)

        m_ok = "✓" if 0.45 <= r_mmps['hpd_mean'] <= 0.55 and r_mmps['hpd_ks'] < 0.10 else " "
        print(f"{alpha:6.1f} | {r_mmps['hpd_mean']:8.3f} {r_mmps['hpd_ks']:7.3f} {m_ok}| {r_dps['hpd_mean']:8.3f} {r_dps['hpd_ks']:7.3f} | {r_lat['hpd_mean']:8.3f} {r_lat['hpd_ks']:7.3f}")
