"""Experiment: H14 — Latent LFlow: flow matching in latent space.

Port LFlow to latent space with Jacobian-aware covariance, analogous
to how Latent MMPS extends DPS. The guided velocity becomes:

  v_guided = v(z_t) - (V_t/t) * J_D^T * Σ_y^{-1} * (y - D(z0_hat))

where Σ_y = σ_n²I + V_t * J_D * J_D^T (same as Latent MMPS).

This is purely ODE-based (deterministic given initial noise).
"""

import jax
import jax.numpy as jnp
from functools import partial
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test
from lip.solvers.latent_mmps import latent_mmps


def latent_lflow(problem, y, key, *, N=200, t_max=0.999, t_min=0.001, zeta=1.0):
    """Latent LFlow: flow matching with Jacobian-aware guidance."""
    key, subkey = jax.random.split(key)
    # Initialize from marginal at t ~ 1: z_t ≈ noise
    var_t = (1 - t_max)**2 * problem.sigma_0**2 + t_max**2
    z = jnp.sqrt(var_t) * jax.random.normal(subkey, (*y.shape[:-1], problem.d_latent))

    dt = (t_max - t_min) / N

    for i in range(N):
        t = t_max - i * dt
        sigma_t2 = (1 - t)**2 * problem.sigma_0**2 + t**2

        # Tweedie denoised estimate in latent space
        z0_hat = (1 - t) * problem.sigma_0**2 / sigma_t2 * z

        # Velocity field in latent space
        z1_hat = t / sigma_t2 * z
        v = z1_hat - z0_hat

        # Tweedie covariance for flow interpolant
        V_t = problem.sigma_0**2 * t**2 / sigma_t2

        # Jacobian-aware guidance (same structure as Latent MMPS)
        residual = y - problem.decoder(z0_hat)
        J = problem.decoder_jacobian(z0_hat)
        JJT = jnp.einsum('...pi,...qi->...pq', J, J)
        Sigma_y = problem.sigma_n**2 * jnp.eye(problem.d_pixel) + V_t * JJT
        Sigma_y_inv_r = jnp.linalg.solve(Sigma_y, residual[..., None])[..., 0]
        grad = jnp.einsum('...pi,...p->...i', J, Sigma_y_inv_r)

        guidance = V_t / t * grad
        v_guided = v - zeta * guidance

        z = z - v_guided * dt

    return z


if __name__ == "__main__":
    p1 = NonlinearDecoder2D(alpha=0.5)
    p2 = FoldedDecoder2D(alpha=1.0)

    print("H14: Latent LFlow")
    print()

    # Test various zeta
    for zeta in [0.8, 1.0, 1.1, 1.2, 1.5]:
        solver = partial(latent_lflow, zeta=zeta)
        r1 = latent_calibration_test(p1, solver, jax.random.PRNGKey(0), n=200)
        r2 = latent_calibration_test(p2, solver, jax.random.PRNGKey(0), n=200)
        ok1 = "✓" if 0.45 <= r1['hpd_mean'] <= 0.55 and r1['hpd_ks'] < 0.10 else " "
        ok2 = "✓" if 0.45 <= r2['hpd_mean'] <= 0.55 and r2['hpd_ks'] < 0.10 else " "
        print(f"  zeta={zeta:.1f}: NL hpd={r1['hpd_mean']:.3f} KS={r1['hpd_ks']:.3f} {ok1} F hpd={r2['hpd_mean']:.3f} KS={r2['hpd_ks']:.3f} {ok2}")

    # Compare with MMPS
    print("\nLatent MMPS (zeta=1.1) reference:")
    solver_mmps = partial(latent_mmps, zeta=1.1)
    r1 = latent_calibration_test(p1, solver_mmps, jax.random.PRNGKey(0), n=200)
    r2 = latent_calibration_test(p2, solver_mmps, jax.random.PRNGKey(0), n=200)
    print(f"  NL hpd={r1['hpd_mean']:.3f} KS={r1['hpd_ks']:.3f}  F hpd={r2['hpd_mean']:.3f} KS={r2['hpd_ks']:.3f}")
