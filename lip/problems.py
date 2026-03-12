"""MNISTVAE problem definition for posterior sampling benchmarks."""

from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp


_DATA_DIR = Path(__file__).parent / "data"


@dataclass
class MNISTVAE:
    """MNIST inverse problem using a pretrained VAE decoder.

    Prior: z ~ N(0, I_{d_latent})
    Decoder: D(z) = neural_net(z) in [0,1]^784  (28x28 MNIST image)
    Forward model: y = D(z) + n, n ~ N(0, sigma_n^2 I)

    The decoder is a trained MLP VAE. Weights are loaded from
    lip/data/vae_mnist_d{latent_dim}.npz.
    Train with: python scripts/train_vae.py --latent-dim <dim>
    """

    sigma_n: float = 0.2
    sigma_0: float = 1.0
    latent_dim: int = 2
    weights_path: str = None
    _params: dict = field(default=None, repr=False, init=False)

    def __post_init__(self):
        from . import vae
        if self.weights_path is None:
            self.weights_path = str(_DATA_DIR / f"vae_mnist_d{self.latent_dim}.npz")
        self._params, loaded_dim = vae.load_params(self.weights_path)
        if loaded_dim != self.latent_dim:
            raise ValueError(
                f"Weight file has latent_dim={loaded_dim}, expected {self.latent_dim}"
            )
        # Pre-compile the single-sample Jacobian function
        self._jac_fn = jax.jacfwd(lambda z: vae.decode_single(self._params, z))

    @property
    def d_latent(self):
        return self.latent_dim

    @property
    def d_pixel(self):
        return 784

    def decoder(self, z):
        """Neural network decoder D: R^d_latent -> [0,1]^784.

        z has shape (..., d_latent). Returns shape (..., 784).
        """
        from . import vae
        shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.d_latent)
        x_flat = jax.vmap(lambda zi: vae.decode_single(self._params, zi))(z_flat)
        return x_flat.reshape(*shape, 784)

    def decoder_jacobian(self, z):
        """Jacobian J_D(z), shape (..., 784, d_latent).

        Computed via forward-mode AD (efficient when d_latent << d_pixel).
        """
        shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.d_latent)
        J_flat = jax.vmap(self._jac_fn)(z_flat)
        return J_flat.reshape(*shape, 784, self.d_latent)

    def encoder(self, x, *, n_iter=10):
        """VAE encoder: returns posterior mean mu(x).

        x has shape (..., 784). Returns z with shape (..., d_latent).
        Uses the VAE encoder network (not Gauss-Newton).
        """
        from . import vae
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, 784)
        mu_flat = jax.vmap(
            lambda xi: vae.encode_single(self._params, xi)[0]
        )(x_flat)
        return mu_flat.reshape(*shape, self.d_latent)

    def score(self, z, sigma):
        """Score of the noised prior N(0, sigma_0^2 I): nabla_z log p_sigma(z)."""
        return -z / (self.sigma_0**2 + sigma**2)

    def denoise(self, z, sigma, key=None):
        """Tweedie denoiser for isotropic Gaussian prior."""
        sigma_end = 1e-3
        s02 = self.sigma_0**2
        if key is None:
            return z * jnp.sqrt((s02 + sigma_end**2) / (s02 + sigma**2))
        else:
            Phi = (s02 + sigma_end**2) / (s02 + sigma**2)
            V = (s02 + sigma_end**2) * (sigma**2 - sigma_end**2) / (s02 + sigma**2)
            return Phi * z + jnp.sqrt(V) * jax.random.normal(key, z.shape)

    def tweedie_cov(self, z, sigma):
        """Tweedie posterior covariance V[z0|z_t] (scalar, isotropic)."""
        return sigma**2 * self.sigma_0**2 / (self.sigma_0**2 + sigma**2)

    def log_prior(self, z):
        """Log prior: z ~ N(0, sigma_0^2 I)."""
        return -0.5 * jnp.sum(z**2, axis=-1) / self.sigma_0**2

    def log_likelihood(self, z, y):
        """Log likelihood: y ~ N(D(z), sigma_n^2 I)."""
        residual = y - self.decoder(z)
        return -0.5 * jnp.sum(residual**2, axis=-1) / self.sigma_n**2

    def log_posterior(self, z, y):
        """Unnormalized log posterior."""
        return self.log_prior(z) + self.log_likelihood(z, y)

    def sample_joint(self, key, n):
        """Sample n (z_true, y_obs) pairs."""
        k1, k2 = jax.random.split(key)
        z = self.sigma_0 * jax.random.normal(k1, (n, self.d_latent))
        x = self.decoder(z)
        y = x + self.sigma_n * jax.random.normal(k2, x.shape)
        return z, y

    # -- Grid-based calibration (only for d_latent=2) --

    def posterior_grid(self, y, *, grid_range=None, grid_size=200):
        """Evaluate posterior on a 2D grid (only for latent_dim=2).

        Uses an adaptive fine grid centered on the encoder MAP estimate
        because the MNISTVAE posterior is very concentrated (std ~0.015).
        """
        if self.d_latent != 2:
            raise NotImplementedError(
                f"posterior_grid requires d_latent=2, got {self.d_latent}. "
                "Use MCMC-based calibration for higher dimensions."
            )
        if grid_range is None:
            # Adaptive: center on encoder MAP, +/-0.2 covers ~13 stds
            z_map = self.encoder(y)
            fine_range = 0.2
            z1 = jnp.linspace(z_map[0] - fine_range, z_map[0] + fine_range, grid_size)
            z2 = jnp.linspace(z_map[1] - fine_range, z_map[1] + fine_range, grid_size)
        else:
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

    def posterior_mean_cov(self, y, **grid_kw):
        """Posterior mean and covariance via grid integration (d_latent=2 only)."""
        _, _, Z1, Z2, p, dz = self.posterior_grid(y, **grid_kw)
        mu1 = jnp.sum(Z1 * p) * dz
        mu2 = jnp.sum(Z2 * p) * dz
        var1 = jnp.sum((Z1 - mu1) ** 2 * p) * dz
        var2 = jnp.sum((Z2 - mu2) ** 2 * p) * dz
        cov12 = jnp.sum((Z1 - mu1) * (Z2 - mu2) * p) * dz
        return jnp.array([mu1, mu2]), jnp.array([[var1, cov12], [cov12, var2]])

    def posterior_mean_cov_batch(self, y_batch, *, grid_range=4.0, grid_size=200):
        """Posterior mean (n,2) and covariance (n,2,2) for a batch of y."""
        return jax.vmap(
            lambda y: self.posterior_mean_cov(
                y, grid_range=grid_range, grid_size=grid_size
            )
        )(y_batch)

    def hpd_level(self, z, y, *, grid_range=4.0, grid_size=200):
        """HPD credibility level (d_latent=2 only).

        For MNISTVAE, uses an adaptive fine grid centered on the MAP
        (encoder estimate) because the posterior is very concentrated
        (std ~0.015) and a coarse grid on [-4,4] cannot resolve it.
        """
        if self.d_latent != 2:
            raise NotImplementedError("hpd_level requires d_latent=2")

        def _hpd_single(z_i, y_i):
            # Find MAP via encoder
            z_map = self.encoder(y_i)
            # Use fine grid centered on MAP: +/-0.2 covers ~13 stds
            fine_range = 0.2
            fine_size = grid_size
            z1 = jnp.linspace(z_map[0] - fine_range, z_map[0] + fine_range, fine_size)
            z2 = jnp.linspace(z_map[1] - fine_range, z_map[1] + fine_range, fine_size)
            Z1, Z2 = jnp.meshgrid(z1, z2)
            z_grid = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)
            log_p_grid = self.log_posterior(z_grid, y_i)
            log_p_max = jnp.max(log_p_grid)
            p_grid = jnp.exp(log_p_grid - log_p_max).reshape(fine_size, fine_size)
            dz = float((z1[1] - z1[0]) * (z2[1] - z2[0]))
            p_grid = p_grid / (jnp.sum(p_grid) * dz)

            # HPD level: fraction of posterior mass at density >= p(z_i|y_i)
            log_p_z = self.log_posterior(z_i, y_i)
            p_at_z = jnp.exp(log_p_z - log_p_max) / (jnp.sum(jnp.exp(log_p_grid - log_p_max)) * dz)
            alpha = jnp.sum(jnp.where(p_grid >= p_at_z, p_grid, 0.0)) * dz
            return alpha

        if z.ndim == 1:
            return _hpd_single(z, y)
        # Sequential loop to avoid OOM with high-dimensional decoder
        levels = [_hpd_single(z[i], y[i]) for i in range(z.shape[0])]
        return jnp.array(levels)

    def plot(self, solver_samples, y_star, title, ax=None, _grid_cache=None):
        """Plot solver samples vs posterior."""
        import matplotlib.pyplot as plt
        import numpy as np

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        if self.d_latent == 2:
            if _grid_cache is not None:
                z1, z2, p = _grid_cache
            else:
                z1, z2, _, _, p, _ = self.posterior_grid(y_star)
                z1, z2, p = np.array(z1), np.array(z2), np.array(p)
            ax.contourf(z1, z2, p, levels=30, cmap="Blues", alpha=0.7)
            ax.contour(z1, z2, p, levels=8, colors="steelblue", linewidths=0.5)
            samples = np.array(solver_samples)
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c="red", alpha=0.3,
                       label="Solver")
            ax.set_xlabel("z1")
            ax.set_ylabel("z2")
            ax.set_aspect("equal")
        else:
            samples = np.array(solver_samples)
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c="red", alpha=0.3,
                       label="Solver")
            ax.set_xlabel("z1")
            ax.set_ylabel("z2")

        ax.set_title(title)
        ax.legend(fontsize=8)
        return ax

    def plot_reconstruction(self, z, y=None, ax=None):
        """Plot decoded image from latent z, optionally alongside observation y."""
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.array(self.decoder(z)).reshape(28, 28)
        n_panels = 1 if y is None else 2
        if ax is None:
            _, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))
            if n_panels == 1:
                axes = [axes]
        else:
            axes = [ax]

        axes[0].imshow(x, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Decoder(z)")
        axes[0].axis("off")

        if y is not None and len(axes) > 1:
            y_img = np.array(y).reshape(28, 28)
            axes[1].imshow(y_img, cmap="gray")
            axes[1].set_title("Observation y")
            axes[1].axis("off")

        return axes
