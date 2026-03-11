"""Experiment: H4 — Split Gibbs sampler in latent space.

Inspired by PnP-DM (Wu et al., NeurIPS 2024). Alternate:
  (a) Denoising diffusion step: z → z_noisy → denoise(z_noisy) (prior)
  (b) Langevin step on likelihood: z += η·∇_z log p(y|D(z))

No Tweedie approximation needed. Asymptotically exact as K→∞.
"""

import jax
import jax.numpy as jnp
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test


def latent_split_gibbs(problem, y, key, *, K=200, sigma_denoise=0.5,
                        n_langevin=5, eta=0.01):
    """Split Gibbs: alternate denoising prior + Langevin likelihood."""
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (*y.shape[:-1], problem.d_latent))

    for k in range(K):
        # (a) Denoising diffusion prior step
        key, k1, k2 = jax.random.split(key, 3)
        # Anneal noise level
        t = 1.0 - k / K
        sigma_k = sigma_denoise * t + 0.01 * (1 - t)

        z_noisy = z + sigma_k * jax.random.normal(k1, z.shape)
        z = problem.denoise(z_noisy, sigma_k, key=k2)

        # (b) Langevin steps on likelihood
        for l in range(n_langevin):
            key, subkey = jax.random.split(key)
            residual = y - problem.decoder(z)
            J = problem.decoder_jacobian(z)
            grad_ll = jnp.einsum('...pi,...p->...i', J, residual) / problem.sigma_n**2
            z = z + eta * grad_ll
            z = z + jnp.sqrt(2 * eta) * jax.random.normal(subkey, z.shape)

    return z


if __name__ == "__main__":
    p1 = NonlinearDecoder2D(alpha=0.5)
    p2 = FoldedDecoder2D(alpha=1.0)

    # Test various configs
    configs = [
        dict(K=200, sigma_denoise=0.5, n_langevin=5, eta=0.01),
        dict(K=200, sigma_denoise=0.3, n_langevin=5, eta=0.01),
        dict(K=200, sigma_denoise=0.5, n_langevin=10, eta=0.005),
        dict(K=100, sigma_denoise=1.0, n_langevin=5, eta=0.01),
    ]

    from functools import partial
    print("H4: Split Gibbs in latent space")
    for cfg in configs:
        solver = partial(latent_split_gibbs, **cfg)
        r1 = latent_calibration_test(p1, solver, jax.random.PRNGKey(0), n=200)
        r2 = latent_calibration_test(p2, solver, jax.random.PRNGKey(0), n=200)
        ok1 = "✓" if 0.45 <= r1['hpd_mean'] <= 0.55 and r1['hpd_ks'] < 0.10 else " "
        ok2 = "✓" if 0.45 <= r2['hpd_mean'] <= 0.55 and r2['hpd_ks'] < 0.10 else " "
        print(f"  {cfg}")
        print(f"    NL: hpd={r1['hpd_mean']:.3f} KS={r1['hpd_ks']:.3f} {ok1}  F: hpd={r2['hpd_mean']:.3f} KS={r2['hpd_ks']:.3f} {ok2}")
