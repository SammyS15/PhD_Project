"""Experiment: H15 — Latent MMPS with sigma_n sensitivity test.

Before trying more methods, test how Latent MMPS behaves at different
noise levels. This affects practical applicability: high SNR (small σ_n)
is the hardest regime for posterior sampling (posterior is tight).
"""

import jax
import jax.numpy as jnp
from functools import partial
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test
from lip.solvers.latent_mmps import latent_mmps
from lip.solvers.latent_dps import latent_dps


if __name__ == "__main__":
    print("H15: Noise level sensitivity — NonlinearDecoder2D (alpha=0.5)")
    print(f"{'sigma_n':>8} | {'MMPS hpd':>8} {'MMPS ks':>7} | {'DPS hpd':>8} {'DPS ks':>7}")
    print("-" * 55)

    solver_mmps = partial(latent_mmps, zeta=1.1)
    for sigma_n in [0.1, 0.2, 0.3, 0.5, 1.0]:
        p = NonlinearDecoder2D(alpha=0.5, sigma_n=sigma_n)
        r_mmps = latent_calibration_test(p, solver_mmps, jax.random.PRNGKey(0), n=200)
        r_dps = latent_calibration_test(p, latent_dps, jax.random.PRNGKey(0), n=200)
        m_ok = "✓" if 0.45 <= r_mmps['hpd_mean'] <= 0.55 and r_mmps['hpd_ks'] < 0.10 else " "
        print(f"{sigma_n:8.1f} | {r_mmps['hpd_mean']:8.3f} {r_mmps['hpd_ks']:7.3f} {m_ok}| {r_dps['hpd_mean']:8.3f} {r_dps['hpd_ks']:7.3f}")

    print()
    print("H15: Noise level sensitivity — FoldedDecoder2D (alpha=1.0)")
    print(f"{'sigma_n':>8} | {'MMPS hpd':>8} {'MMPS ks':>7} | {'DPS hpd':>8} {'DPS ks':>7}")
    print("-" * 55)

    for sigma_n in [0.1, 0.2, 0.3, 0.5, 1.0]:
        p = FoldedDecoder2D(alpha=1.0, sigma_n=sigma_n)
        r_mmps = latent_calibration_test(p, solver_mmps, jax.random.PRNGKey(0), n=200)
        r_dps = latent_calibration_test(p, latent_dps, jax.random.PRNGKey(0), n=200)
        m_ok = "✓" if 0.45 <= r_mmps['hpd_mean'] <= 0.55 and r_mmps['hpd_ks'] < 0.10 else " "
        print(f"{sigma_n:8.1f} | {r_mmps['hpd_mean']:8.3f} {r_mmps['hpd_ks']:7.3f} {m_ok}| {r_dps['hpd_mean']:8.3f} {r_dps['hpd_ks']:7.3f}")
