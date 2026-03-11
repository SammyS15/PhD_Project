"""Experiment: H9 — Random MLP decoder.

Add a RandomMLPDecoder2D problem: frozen random MLP, R²→R⁸.
Tests whether Latent MMPS generalizes beyond hand-crafted analytic decoders.
Uses JAX autodiff for Jacobian (no analytic formula).
"""

import jax
import jax.numpy as jnp
from functools import partial
from lip.metrics import latent_calibration_test
from lip.solvers.latent_mmps import latent_mmps
from lip.solvers.latent_dps import latent_dps


class RandomMLPDecoder2D:
    """2D Gaussian prior with frozen random MLP decoder to R^d_pixel."""

    def __init__(self, d_pixel=8, hidden=32, sigma_n=0.3, seed=42):
        self.sigma_0 = 1.0
        self.sigma_n = sigma_n
        self.d_latent = 2
        self.d_pixel = d_pixel

        # Frozen random MLP weights: 2 → hidden → hidden → d_pixel
        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)
        scale = 1.0 / jnp.sqrt(jnp.array([2.0, hidden, hidden]))
        self.W1 = scale[0] * jax.random.normal(k1, (2, hidden))
        self.b1 = jnp.zeros(hidden)
        self.W2 = scale[1] * jax.random.normal(k2, (hidden, hidden))
        self.b2 = jnp.zeros(hidden)
        self.W3 = scale[2] * jax.random.normal(k3, (hidden, d_pixel))
        self.b3 = jnp.zeros(d_pixel)

    def decoder(self, z):
        """MLP decoder: 2 → hidden → hidden → d_pixel with tanh activations."""
        h = jnp.tanh(z @ self.W1 + self.b1)
        h = jnp.tanh(h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3

    def decoder_jacobian(self, z):
        """Autodiff Jacobian."""
        if z.ndim == 1:
            return jax.jacobian(self.decoder)(z)
        return jax.vmap(jax.jacobian(self.decoder))(z)

    def encoder(self, x, *, n_iter=20):
        """Gauss-Newton least-squares inverse."""
        z0 = jnp.zeros((*x.shape[:-1], self.d_latent))

        def step(z, _):
            residual = x - self.decoder(z)
            J = self.decoder_jacobian(z)
            JTJ = jnp.einsum('...pi,...pj->...ij', J, J)
            JTr = jnp.einsum('...pi,...p->...i', J, residual)
            JTJ = JTJ + 1e-6 * jnp.eye(self.d_latent)
            dz = jnp.linalg.solve(JTJ, JTr[..., None])[..., 0]
            return z + dz, None

        z_final, _ = jax.lax.scan(step, z0, None, length=n_iter)
        return z_final

    def score(self, z, sigma):
        return -z / (self.sigma_0**2 + sigma**2)

    def denoise(self, z, sigma, key=None):
        sigma_end = 1e-3
        s02 = self.sigma_0**2
        if key is None:
            return z * jnp.sqrt((s02 + sigma_end**2) / (s02 + sigma**2))
        else:
            Phi = (s02 + sigma_end**2) / (s02 + sigma**2)
            V = (s02 + sigma_end**2) * (sigma**2 - sigma_end**2) / (s02 + sigma**2)
            return Phi * z + jnp.sqrt(V) * jax.random.normal(key, z.shape)

    def tweedie_cov(self, z, sigma):
        return sigma**2 * self.sigma_0**2 / (self.sigma_0**2 + sigma**2)

    def log_prior(self, z):
        return -0.5 * jnp.sum(z**2, axis=-1)

    def log_likelihood(self, z, y):
        residual = y - self.decoder(z)
        return -0.5 * jnp.sum(residual**2, axis=-1) / self.sigma_n**2

    def log_posterior(self, z, y):
        return self.log_prior(z) + self.log_likelihood(z, y)

    def sample_joint(self, key, n):
        k1, k2 = jax.random.split(key)
        z = jax.random.normal(k1, (n, self.d_latent))
        y = self.decoder(z) + self.sigma_n * jax.random.normal(k2, (n, self.d_pixel))
        return z, y

    def posterior_grid(self, y, *, grid_range=4.0, grid_size=200):
        z1 = jnp.linspace(-grid_range, grid_range, grid_size)
        z2 = jnp.linspace(-grid_range, grid_range, grid_size)
        Z1, Z2 = jnp.meshgrid(z1, z2)
        z_grid = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)
        log_p = self.log_posterior(z_grid, y)
        log_p = log_p - jnp.max(log_p)
        p = jnp.exp(log_p).reshape(grid_size, grid_size)
        dz = float((z1[1] - z1[0]) * (z2[1] - z2[0]))
        p = p / (jnp.sum(p) * dz)
        return z1, z2, Z1, Z2, p, dz

    def hpd_level(self, z, y, *, grid_range=4.0, grid_size=200):
        def _hpd_single(z_i, y_i):
            _, _, _, _, p_grid, dz = self.posterior_grid(
                y_i, grid_range=grid_range, grid_size=grid_size
            )
            log_p_z = self.log_posterior(z_i, y_i)
            z1 = jnp.linspace(-grid_range, grid_range, grid_size)
            z2 = jnp.linspace(-grid_range, grid_range, grid_size)
            Z1, Z2 = jnp.meshgrid(z1, z2)
            z_grid = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)
            log_p_grid = self.log_posterior(z_grid, y_i)
            log_norm = jax.scipy.special.logsumexp(log_p_grid) + jnp.log(dz)
            p_at_z = jnp.exp(log_p_z - log_norm)
            alpha = jnp.sum(jnp.where(p_grid >= p_at_z, p_grid, 0.0)) * dz
            return alpha

        if z.ndim == 1:
            return _hpd_single(z, y)
        return jax.vmap(_hpd_single)(z, y)


if __name__ == "__main__":
    problem = RandomMLPDecoder2D(d_pixel=8, hidden=32, seed=42)

    print("H9: Random MLP Decoder (R²→R⁸)")
    print()

    solver_mmps = partial(latent_mmps, zeta=1.1)
    r_mmps = latent_calibration_test(problem, solver_mmps, jax.random.PRNGKey(0), n=100)
    ok = "✓" if 0.45 <= r_mmps['hpd_mean'] <= 0.55 and r_mmps['hpd_ks'] < 0.10 else " "
    print(f"  MMPS (z=1.1): hpd={r_mmps['hpd_mean']:.3f} KS={r_mmps['hpd_ks']:.3f} {ok}")

    r_dps = latent_calibration_test(problem, latent_dps, jax.random.PRNGKey(0), n=100)
    print(f"  DPS:          hpd={r_dps['hpd_mean']:.3f} KS={r_dps['hpd_ks']:.3f}")

    # Try different zeta if needed
    if not (0.45 <= r_mmps['hpd_mean'] <= 0.55):
        print("\n  Zeta sweep:")
        for zeta in [0.8, 1.0, 1.2, 1.3, 1.5]:
            solver = partial(latent_mmps, zeta=zeta)
            r = latent_calibration_test(problem, solver, jax.random.PRNGKey(0), n=100)
            ok = "✓" if 0.45 <= r['hpd_mean'] <= 0.55 and r['hpd_ks'] < 0.10 else " "
            print(f"    zeta={zeta:.1f}: hpd={r['hpd_mean']:.3f} KS={r['hpd_ks']:.3f} {ok}")
