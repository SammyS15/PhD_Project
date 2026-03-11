"""Experiment: H13 — Auto-tune zeta based on decoder Jacobian.

Hypothesis: optimal zeta correlates with decoder nonlinearity.
Measure nonlinearity via Hessian-to-Jacobian ratio at the Tweedie estimate,
and use it to set zeta adaptively per step.

Simpler approach: measure E[||H_D||/||J_D||] at z~N(0,I) and set zeta = 1 + c*ratio.
"""

import jax
import jax.numpy as jnp
from functools import partial
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test
from lip.solvers.latent_mmps import latent_mmps


def measure_nonlinearity(problem, key, n_samples=50):
    """Estimate decoder nonlinearity as mean ||D(z) - J(0)·z|| / ||D(z)||."""
    z = jax.random.normal(key, (n_samples, problem.d_latent))
    Dz = problem.decoder(z)
    # Linear approximation at origin
    J0 = problem.decoder_jacobian(jnp.zeros(problem.d_latent))
    Dz_lin = z @ J0.T  # (..., d_pixel)
    nonlin = jnp.mean(jnp.linalg.norm(Dz - Dz_lin, axis=-1) / (jnp.linalg.norm(Dz, axis=-1) + 1e-8))
    return float(nonlin)


if __name__ == "__main__":
    print("H13: Nonlinearity measurement and zeta auto-tuning")
    print()

    # Measure nonlinearity for each problem
    problems = {
        "NL(α=0.0)": NonlinearDecoder2D(alpha=0.0),
        "NL(α=0.2)": NonlinearDecoder2D(alpha=0.2),
        "NL(α=0.5)": NonlinearDecoder2D(alpha=0.5),
        "NL(α=0.7)": NonlinearDecoder2D(alpha=0.7),
        "NL(α=1.0)": NonlinearDecoder2D(alpha=1.0),
        "NL(α=1.5)": NonlinearDecoder2D(alpha=1.5),
        "Folded(α=1.0)": FoldedDecoder2D(alpha=1.0),
    }

    print(f"{'Problem':>16} | {'nonlin':>7} | {'best_zeta':>9} | {'proposed':>8}")
    print("-" * 55)

    # Known optimal zetas from H12 sweep
    known_best = {
        "NL(α=0.0)": 1.1, "NL(α=0.2)": 1.1, "NL(α=0.5)": 1.2,
        "NL(α=0.7)": 1.2, "NL(α=1.0)": 1.3, "NL(α=1.5)": 1.4,
        "Folded(α=1.0)": 1.1,
    }

    nonlins = {}
    for name, p in problems.items():
        nl = measure_nonlinearity(p, jax.random.PRNGKey(0))
        nonlins[name] = nl
        best = known_best.get(name, "?")
        # Proposed: zeta = 1.0 + 0.5 * nonlinearity
        proposed = 1.0 + 0.5 * nl
        print(f"{name:>16} | {nl:7.3f} | {best:>9} | {proposed:8.2f}")

    # Test the auto-tuned zeta
    print("\nVerification: auto-tuned zeta = 1.0 + 0.5 * nonlinearity")
    print(f"{'Problem':>16} | {'auto_zeta':>9} | {'hpd':>6} | {'ks':>6}")
    print("-" * 45)
    for name, p in problems.items():
        nl = nonlins[name]
        auto_zeta = 1.0 + 0.5 * nl
        solver = partial(latent_mmps, zeta=auto_zeta)
        r = latent_calibration_test(p, solver, jax.random.PRNGKey(0), n=200)
        ok = "✓" if 0.45 <= r['hpd_mean'] <= 0.55 and r['hpd_ks'] < 0.10 else " "
        print(f"{name:>16} | {auto_zeta:9.2f} | {r['hpd_mean']:6.3f} | {r['hpd_ks']:6.3f} {ok}")
