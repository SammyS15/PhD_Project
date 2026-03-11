"""Experiment: H5 — Few-particle SMC on top of Latent MMPS.

Run N_particles parallel MMPS trajectories per observation,
weight by p(y|D(z)), select one. Corrects Tweedie approximation error.
"""

import jax
import jax.numpy as jnp
from functools import partial
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test
from lip.solvers.latent_mmps import latent_mmps


def latent_mmps_smc(problem, y, key, *, N_particles=5, **mmps_kw):
    """Run N_particles MMPS trajectories, select one by importance weight.

    y has shape (..., d_pixel). We run N_particles independent trajectories
    and select one per observation.
    """
    keys = jax.random.split(key, N_particles + 1)
    key_select = keys[0]

    # Run N_particles independent MMPS trajectories
    # Each returns shape (..., d_latent)
    z_particles = jnp.stack([
        latent_mmps(problem, y, keys[i + 1], **mmps_kw)
        for i in range(N_particles)
    ], axis=0)  # (N_particles, ..., d_latent)

    # Log-weights: log p(y|D(z)) + log p(z)
    log_w = jnp.stack([
        problem.log_likelihood(z_particles[i], y) + problem.log_prior(z_particles[i])
        for i in range(N_particles)
    ], axis=0)  # (N_particles, ...)

    # Normalize and select per observation
    log_w = log_w - jax.scipy.special.logsumexp(log_w, axis=0, keepdims=True)

    # For each observation in batch, select one particle
    # idx shape: (...)
    batch_shape = y.shape[:-1]
    if len(batch_shape) == 0:
        idx = jax.random.categorical(key_select, log_w[:, None])
        return z_particles[idx[0]]
    else:
        # Categorical per batch element
        idx = jax.random.categorical(key_select, log_w.T)  # (batch,)
        # Gather: z_particles[idx[b], b, :]
        return z_particles[idx, jnp.arange(batch_shape[0])]


if __name__ == "__main__":
    p1 = NonlinearDecoder2D(alpha=0.5)
    p2 = FoldedDecoder2D(alpha=1.0)

    print("H5: Few-particle SMC with Latent MMPS proposals")

    for np_ in [1, 2, 5, 10]:
        if np_ == 1:
            solver = partial(latent_mmps, zeta=1.1)
        else:
            solver = partial(latent_mmps_smc, N_particles=np_, zeta=1.1)
        r1 = latent_calibration_test(p1, solver, jax.random.PRNGKey(0), n=200)
        r2 = latent_calibration_test(p2, solver, jax.random.PRNGKey(0), n=200)
        ok1 = "✓" if 0.45 <= r1['hpd_mean'] <= 0.55 and r1['hpd_ks'] < 0.10 else " "
        ok2 = "✓" if 0.45 <= r2['hpd_mean'] <= 0.55 and r2['hpd_ks'] < 0.10 else " "
        print(f"  {np_:2d} particles: NL hpd={r1['hpd_mean']:.3f} KS={r1['hpd_ks']:.3f} {ok1} F hpd={r2['hpd_mean']:.3f} KS={r2['hpd_ks']:.3f} {ok2}")
