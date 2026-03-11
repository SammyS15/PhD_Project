"""Experiment: H16 — Woodbury-efficient Latent MMPS.

When d_pixel >> d_latent, the d_pixel×d_pixel solve in Σ_y is expensive.
Use the Woodbury identity to reduce to a d_latent×d_latent solve:

  Σ_y^{-1} = σ_n^{-2}(I - V_t·J·(J^T·J + σ_n²/V_t·I)^{-1}·J^T / σ_n²)

Then: J^T·Σ_y^{-1}·r = (1/σ_n²)(J^T·r - V_t·J^T·J·(J^T·J + σ_n²/V_t·I)^{-1}·J^T·r / σ_n²)

Simplification: let A = J^T·J (d_latent×d_latent), b = J^T·r (d_latent)
  grad = (1/σ_n²)(b - V_t·A·(A + σ_n²/V_t·I)^{-1}·b / σ_n²)
       = (A + σ_n²/V_t·I)^{-1}·b / V_t  ... let me derive carefully

Actually: Σ_y^{-1}·r = r/σ_n² - V_t·J·(V_t·J^T·J + σ_n²·I)^{-1}·J^T·r / σ_n⁴
So: J^T·Σ_y^{-1}·r = J^T·r/σ_n² - V_t·J^T·J·(V_t·J^T·J + σ_n²·I)^{-1}·J^T·r / σ_n⁴

Let A = J^T·J, b = J^T·r:
  grad = b/σ_n² - V_t·A·(V_t·A + σ_n²·I)^{-1}·b / σ_n⁴
       = (V_t·A + σ_n²·I)^{-1}·b·(V_t·A + σ_n²·I)/(σ_n²) - V_t·A·(V_t·A+σ_n²·I)^{-1}·b/σ_n⁴
Hmm, let me just use the formula: (V_t·A + σ_n²·I)^{-1}·b / σ_n²
which I can verify numerically.

Actually the clean Woodbury result for J^T·Σ_y^{-1}·r when Σ_y = σ²I + V·J·J^T is:
  J^T·(σ²I + V·J·J^T)^{-1}·r = (σ²I + V·J^T·J)^{-1}·J^T·r / σ²... no.

Let me use the push-through identity:
  J^T·(σ²I + V·J·J^T)^{-1} = (σ²I + V·J^T·J)^{-1}·J^T

So: grad = (σ_n²·I + V_t·J^T·J)^{-1}·J^T·r

This is a d_latent×d_latent solve! Much better.
"""

import jax
import jax.numpy as jnp
from functools import partial
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test


def latent_mmps_woodbury(problem, y, key, *, N=200, sigma_max=3.0, sigma_min=0.01, zeta=1.1):
    """Latent MMPS with Woodbury-efficient gradient (d_latent×d_latent solve)."""
    key, subkey = jax.random.split(key)
    z = jnp.sqrt(problem.sigma_0**2 + sigma_max**2) * jax.random.normal(
        subkey, (*y.shape[:-1], problem.d_latent)
    )

    log_R = jnp.log(sigma_max / sigma_min)
    dt = 1.0 / N

    for i in range(N):
        t = 1.0 - i * dt
        sigma_t = sigma_min * jnp.exp(log_R * t)
        key, subkey = jax.random.split(key)

        s = problem.score(z, sigma_t)
        z0_hat = z + sigma_t**2 * s
        V_t = problem.tweedie_cov(z, sigma_t)

        g2 = 2 * sigma_t**2 * log_R
        z = z + g2 * s * dt
        z = z + sigma_t * jnp.sqrt(2 * log_R * dt) * jax.random.normal(subkey, z.shape)

        # Woodbury-efficient guidance:
        # grad = (σ_n²·I + V_t·J^T·J)^{-1} · J^T · residual
        # This is a d_latent×d_latent solve instead of d_pixel×d_pixel
        residual = y - problem.decoder(z0_hat)
        J = problem.decoder_jacobian(z0_hat)  # (..., d_pixel, d_latent)
        JTr = jnp.einsum('...pi,...p->...i', J, residual)  # (..., d_latent)
        JTJ = jnp.einsum('...pi,...pj->...ij', J, J)  # (..., d_latent, d_latent)
        M = problem.sigma_n**2 * jnp.eye(problem.d_latent) + V_t * JTJ
        grad = jnp.linalg.solve(M, JTr[..., None])[..., 0]

        dz0_dz = problem.sigma_0**2 / (problem.sigma_0**2 + sigma_t**2)
        z = z + zeta * g2 * dt * grad * dz0_dz

    return z


if __name__ == "__main__":
    from lip.solvers.latent_mmps import latent_mmps

    p1 = NonlinearDecoder2D(alpha=0.5)
    p2 = FoldedDecoder2D(alpha=1.0)

    print("H16: Woodbury-efficient Latent MMPS (d_latent×d_latent solve)")
    print()

    # Verify equivalence with original
    for name, solver in [
        ("Original (d_pixel solve)", partial(latent_mmps, zeta=1.1)),
        ("Woodbury (d_latent solve)", partial(latent_mmps_woodbury, zeta=1.1)),
    ]:
        r1 = latent_calibration_test(p1, solver, jax.random.PRNGKey(0), n=200)
        r2 = latent_calibration_test(p2, solver, jax.random.PRNGKey(0), n=200)
        ok1 = "✓" if 0.45 <= r1['hpd_mean'] <= 0.55 and r1['hpd_ks'] < 0.10 else " "
        ok2 = "✓" if 0.45 <= r2['hpd_mean'] <= 0.55 and r2['hpd_ks'] < 0.10 else " "
        print(f"  {name:30s}: NL hpd={r1['hpd_mean']:.3f} KS={r1['hpd_ks']:.3f} {ok1} F hpd={r2['hpd_mean']:.3f} KS={r2['hpd_ks']:.3f} {ok2}")

    # Time comparison with high d_pixel
    import time
    print("\n  Timing with d_pixel=8 (RandomMLP):")

    import sys
    sys.path.insert(0, '.')

    # Use inline RandomMLP for timing
    class QuickMLP:
        sigma_0 = 1.0; sigma_n = 0.3; d_latent = 2; d_pixel = 16
        def __init__(self):
            key = jax.random.PRNGKey(99)
            k1, k2 = jax.random.split(key)
            self.W = jax.random.normal(k1, (2, 16)) * 0.5
        def decoder(self, z): return jnp.tanh(z @ self.W)
        def decoder_jacobian(self, z):
            if z.ndim == 1: return jax.jacobian(self.decoder)(z)
            return jax.vmap(jax.jacobian(self.decoder))(z)
        def score(self, z, sigma): return -z / (1.0 + sigma**2)
        def tweedie_cov(self, z, sigma): return sigma**2 / (1.0 + sigma**2)
        def log_prior(self, z): return -0.5 * jnp.sum(z**2, axis=-1)
        def log_likelihood(self, z, y):
            r = y - self.decoder(z)
            return -0.5 * jnp.sum(r**2, axis=-1) / 0.09
        def log_posterior(self, z, y): return self.log_prior(z) + self.log_likelihood(z, y)
        def sample_joint(self, key, n):
            k1, k2 = jax.random.split(key)
            z = jax.random.normal(k1, (n, 2))
            y = self.decoder(z) + 0.3 * jax.random.normal(k2, (n, 16))
            return z, y

    qp = QuickMLP()
    k = jax.random.PRNGKey(0)
    _, y_batch = qp.sample_joint(k, 50)

    # Warm up
    _ = latent_mmps(qp, y_batch[0], k)
    _ = latent_mmps_woodbury(qp, y_batch[0], k)

    t0 = time.time()
    for y in y_batch: latent_mmps(qp, y, k)
    t_orig = time.time() - t0

    t0 = time.time()
    for y in y_batch: latent_mmps_woodbury(qp, y, k)
    t_wood = time.time() - t0

    print(f"    Original (16×16 solve): {t_orig:.2f}s")
    print(f"    Woodbury  (2×2 solve):  {t_wood:.2f}s")
    print(f"    Speedup: {t_orig/t_wood:.1f}×")
