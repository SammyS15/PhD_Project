"""Experiment: H3 — Latent LATINO + SDE denoiser.

In pixel space, switching from PF-ODE to SDE denoiser nearly calibrated
LATINO (z_std 0.765 → 0.979). Test the same trick in latent space.
"""

import jax
import jax.numpy as jnp
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test


def latent_latino_sde(problem, y, key, *, N=64, sigma_max=2.0, sigma_min=0.01):
    """Latent LATINO with stochastic (SDE) denoiser."""
    sigma_schedule = jnp.geomspace(sigma_max, sigma_min, N)
    x = y

    for k in range(N):
        sigma_k = sigma_schedule[k]
        key, k1, k2 = jax.random.split(key, 3)

        z = problem.encoder(x)
        z_noisy = z + sigma_k * jax.random.normal(k1, z.shape)

        # Stochastic denoiser (key=k2 enables SDE mode)
        z_clean = problem.denoise(z_noisy, sigma_k, key=k2)

        u = problem.decoder(z_clean)
        delta_k = float(sigma_k) ** 2
        x = (delta_k * y + problem.sigma_n**2 * u) / (delta_k + problem.sigma_n**2)

    return problem.encoder(x)


if __name__ == "__main__":
    from lip.solvers.latent_latino import latent_latino

    p1 = NonlinearDecoder2D(alpha=0.5)
    p2 = FoldedDecoder2D(alpha=1.0)

    print("H3: Latent LATINO + SDE vs deterministic LATINO")
    print()
    for name, solver in [("LATINO (det)", latent_latino), ("LATINO+SDE", latent_latino_sde)]:
        r1 = latent_calibration_test(p1, solver, jax.random.PRNGKey(0), n=200)
        r2 = latent_calibration_test(p2, solver, jax.random.PRNGKey(0), n=200)
        ok1 = "✓" if 0.45 <= r1['hpd_mean'] <= 0.55 and r1['hpd_ks'] < 0.10 else " "
        ok2 = "✓" if 0.45 <= r2['hpd_mean'] <= 0.55 and r2['hpd_ks'] < 0.10 else " "
        print(f"{name:15s}  NL: hpd={r1['hpd_mean']:.3f} KS={r1['hpd_ks']:.3f} {ok1}  F: hpd={r2['hpd_mean']:.3f} KS={r2['hpd_ks']:.3f} {ok2}")
