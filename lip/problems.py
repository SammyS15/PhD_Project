"""MNISTVAE problem definition for posterior sampling benchmarks."""

from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


_DATA_DIR = Path(__file__).parent / "data"

# Grid defaults: [-3, 3]^2 at 1600x1600.
# Posterior std ~ 0.015 -> spacing 0.00375 -> ~4 pts per posterior std.
# Covers the full N(0,1) prior so no modes are missed.
# Chunked evaluation avoids OOM (~36s per grid).
GRID_RANGE = 3.0
GRID_SIZE = 1600
CHUNK_SIZE = 5000


@dataclass
class MNISTVAE:
    """MNIST inverse problem using a pretrained VAE decoder.

    Prior: z ~ N(0, I_2)
    Decoder: D(z) = neural_net(z) in [0,1]^784  (28x28 MNIST image)
    Forward model: y = D(z) + n, n ~ N(0, sigma_n^2 I)

    The decoder is a trained MLP VAE. Weights are loaded from
    lip/data/vae_mnist_d2.npz.
    Train with: python scripts/train_vae.py
    """

    sigma_n: float = 0.2
    sigma_0: float = 1.0
    weights_path: str = None
    _params: dict = field(default=None, repr=False, init=False)

    def __post_init__(self):
        from . import vae
        if self.weights_path is None:
            self.weights_path = str(_DATA_DIR / "vae_mnist_d2.npz")
        self._params, loaded_dim = vae.load_params(self.weights_path)
        if loaded_dim != 2:
            raise ValueError(
                f"Weight file has latent_dim={loaded_dim}, expected 2"
            )
        self._jac_fn = jax.jacfwd(lambda z: vae.decode_single(self._params, z))

    @property
    def d_latent(self):
        return 2

    @property
    def d_pixel(self):
        return 784

    def decoder(self, z):
        """Neural network decoder D: R^2 -> [0,1]^784."""
        from . import vae
        shape = z.shape[:-1]
        z_flat = z.reshape(-1, 2)
        x_flat = jax.vmap(lambda zi: vae.decode_single(self._params, zi))(z_flat)
        return x_flat.reshape(*shape, 784)

    def decoder_jacobian(self, z):
        """Jacobian J_D(z), shape (..., 784, 2)."""
        shape = z.shape[:-1]
        z_flat = z.reshape(-1, 2)
        J_flat = jax.vmap(self._jac_fn)(z_flat)
        return J_flat.reshape(*shape, 784, 2)

    def encoder(self, x, *, n_iter=10):
        """VAE encoder: returns posterior mean mu(x)."""
        from . import vae
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, 784)
        mu_flat = jax.vmap(
            lambda xi: vae.encode_single(self._params, xi)[0]
        )(x_flat)
        return mu_flat.reshape(*shape, 2)

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
        z = self.sigma_0 * jax.random.normal(k1, (n, 2))
        x = self.decoder(z)
        y = x + self.sigma_n * jax.random.normal(k2, x.shape)
        return z, y

    # -- Grid-based posterior evaluation --

    def posterior_grid(self, y, *, grid_range=GRID_RANGE, grid_size=GRID_SIZE,
                       chunk_size=CHUNK_SIZE, center=None):
        """Evaluate posterior on a 2D grid.

        Default: [-3, 3]^2 at 1600x1600 covering the full prior.
        With center: grid_range is a half-width around center (for fast HPD).
        Chunked evaluation avoids OOM.

        Returns (z1, z2, p, dz, log_p_max) where:
          z1, z2: 1D coordinate arrays (grid_size,)
          p: (grid_size, grid_size) normalized posterior density
          dz: grid cell area
          log_p_max: log-posterior shift constant (for HPD normalization)
        """
        if center is not None:
            c1, c2 = float(center[0]), float(center[1])
        else:
            c1, c2 = 0.0, 0.0
        z1 = jnp.linspace(c1 - grid_range, c1 + grid_range, grid_size)
        z2 = jnp.linspace(c2 - grid_range, c2 + grid_range, grid_size)
        Z1, Z2 = jnp.meshgrid(z1, z2)
        z_grid = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)

        n = z_grid.shape[0]
        log_p_chunks = []
        for i in range(0, n, chunk_size):
            log_p_chunks.append(self.log_posterior(z_grid[i:i + chunk_size], y))
        log_p = jnp.concatenate(log_p_chunks)

        log_p_max = jnp.max(log_p)
        log_p = log_p - log_p_max
        p = jnp.exp(log_p).reshape(grid_size, grid_size)
        dz = float((z1[1] - z1[0]) * (z2[1] - z2[0]))
        p = p / (jnp.sum(p) * dz)
        return z1, z2, p, dz, log_p_max

    def save_posterior_grid(self, y, path, **grid_kw):
        """Compute and save posterior grid + observation to .npz for caching."""
        z1, z2, p, dz, _ = self.posterior_grid(y, **grid_kw)
        np.savez(str(path),
                 z1=np.array(z1), z2=np.array(z2),
                 p=np.array(p), dz=np.array(dz),
                 y=np.array(y), sigma_n=np.array(self.sigma_n))
        return z1, z2, p, dz

    @staticmethod
    def load_posterior_grid(path):
        """Load cached posterior grid. Returns (z1, z2, p, dz, y)."""
        data = np.load(str(path))
        return (jnp.array(data["z1"]), jnp.array(data["z2"]),
                jnp.array(data["p"]), float(data["dz"]),
                jnp.array(data["y"]))

    def posterior_mean_cov(self, y, **grid_kw):
        """Posterior mean and covariance via grid integration."""
        z1, z2, p, dz, _ = self.posterior_grid(y, **grid_kw)
        Z1, Z2 = jnp.meshgrid(z1, z2)
        mu1 = jnp.sum(Z1 * p) * dz
        mu2 = jnp.sum(Z2 * p) * dz
        var1 = jnp.sum((Z1 - mu1) ** 2 * p) * dz
        var2 = jnp.sum((Z2 - mu2) ** 2 * p) * dz
        cov12 = jnp.sum((Z1 - mu1) * (Z2 - mu2) * p) * dz
        return jnp.array([mu1, mu2]), jnp.array([[var1, cov12], [cov12, var2]])

    def hpd_level(self, z, y, *, hpd_range=0.3, grid_size=200):
        """HPD credibility level using adaptive grid centered on z.

        Centers a small grid (±hpd_range) on each sample point z_i. The
        posterior mode is always close to z_i (within a few posterior stds
        ~0.015), so hpd_range=0.3 safely captures the full posterior mass.
        grid_size=200 gives spacing 0.003, ~5 pts per posterior std.

        Returns alpha in [0, 1]: fraction of posterior mass at density >= p(z|y).
        Calibrated samples have alpha ~ Uniform(0, 1).
        """
        def _hpd_single(z_i, y_i):
            z1, z2, p_grid, dz, log_p_max = self.posterior_grid(
                y_i, grid_range=hpd_range, grid_size=grid_size,
                center=z_i,
            )
            log_p_z = self.log_posterior(z_i, y_i)
            p_at_z = jnp.max(p_grid) * jnp.exp(log_p_z - log_p_max)
            alpha = jnp.sum(jnp.where(p_grid >= p_at_z, p_grid, 0.0)) * dz
            return alpha

        if z.ndim == 1:
            return _hpd_single(z, y)
        levels = [_hpd_single(z[i], y[i]) for i in range(z.shape[0])]
        return jnp.array(levels)

    def plot(self, solver_samples, y_star, title, ax=None, _grid_cache=None):
        """Plot solver samples vs posterior contours."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        if _grid_cache is not None:
            z1, z2, p = _grid_cache
        else:
            z1, z2, p, _, _ = self.posterior_grid(y_star)
            z1, z2, p = np.array(z1), np.array(z2), np.array(p)
        ax.contourf(z1, z2, p, levels=30, cmap="Blues", alpha=0.7)
        ax.contour(z1, z2, p, levels=8, colors="steelblue", linewidths=0.5)
        samples = np.array(solver_samples)
        ax.scatter(samples[:, 0], samples[:, 1], s=1, c="red", alpha=0.3,
                   label="Solver")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_aspect("equal")

        ax.set_title(title)
        ax.legend(fontsize=8)
        return ax

    def plot_reconstruction(self, z, y=None, ax=None):
        """Plot decoded image from latent z, optionally alongside observation y."""
        import matplotlib.pyplot as plt

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
