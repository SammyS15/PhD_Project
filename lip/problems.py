"""Problem definitions for posterior sampling benchmarks."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class Gaussian1D:
    """1D Gaussian prior with identity forward model.

    Prior: x ~ N(mu_0, sigma_0^2)
    Forward model: y = x + n, n ~ N(0, sigma_n^2)
    Posterior: p(x|y) = N(posterior_mean(y), posterior_std^2)
    """

    mu_0: float = 0.0
    sigma_0: float = 1.0
    sigma_n: float = 0.5

    def score(self, x, sigma):
        """Score of the noised prior: nabla_x log p_sigma(x)."""
        return -(x - self.mu_0) / (self.sigma_0**2 + sigma**2)

    def denoise(self, x, sigma, key=None):
        """Denoise from noise level sigma.

        key=None -> deterministic (PF-ODE).
        key provided -> stochastic (reverse SDE).
        """
        if key is None:
            sigma_end = 1e-3
            return self.mu_0 + (x - self.mu_0) * jnp.sqrt(
                (self.sigma_0**2 + sigma_end**2) / (self.sigma_0**2 + sigma**2)
            )
        else:
            sigma_end = 1e-3
            Phi = (self.sigma_0**2 + sigma_end**2) / (self.sigma_0**2 + sigma**2)
            V = (
                (self.sigma_0**2 + sigma_end**2)
                * (sigma**2 - sigma_end**2)
                / (self.sigma_0**2 + sigma**2)
            )
            return self.mu_0 + Phi * (x - self.mu_0) + jnp.sqrt(V) * jax.random.normal(
                key, x.shape
            )

    def tweedie_cov(self, x, sigma):
        """Tweedie posterior covariance V[x0 | x_t]."""
        return sigma**2 * self.sigma_0**2 / (self.sigma_0**2 + sigma**2)

    def sample_joint(self, key, n):
        """Sample n (x_true, y_obs) pairs."""
        k1, k2 = jax.random.split(key)
        x = self.mu_0 + self.sigma_0 * jax.random.normal(k1, (n,))
        y = x + self.sigma_n * jax.random.normal(k2, (n,))
        return x, y

    def posterior_mean(self, y):
        s2 = self.posterior_std**2
        return s2 * (y / self.sigma_n**2 + self.mu_0 / self.sigma_0**2)

    @property
    def posterior_std(self):
        return jnp.sqrt(
            self.sigma_0**2 * self.sigma_n**2 / (self.sigma_0**2 + self.sigma_n**2)
        )

    def plot(self, posterior_result, calibration_result, title, axes=None):
        """Diagnostic plot: posterior histogram (left) + QQ calibration (right)."""
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        import numpy as np

        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(12, 4))

        samples = np.array(posterior_result["samples"])
        mu_p = posterior_result["target_mean"]
        sigma_p = posterior_result["target_std"]

        # Left: posterior histogram vs analytic
        bins = np.linspace(mu_p - 4 * sigma_p, mu_p + 4 * sigma_p, 80)
        r = np.linspace(bins[0] - 0.5, bins[-1] + 0.5, 300)
        axes[0].hist(samples, bins=bins, density=True, alpha=0.5, label=title)
        axes[0].plot(r, norm.pdf(r, mu_p, sigma_p), "r-", lw=2, label="Analytic posterior")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("Density")
        axes[0].set_title(f"{title} (y = {posterior_result.get('y_star', '?')})")
        axes[0].text(
            0.02, 0.95,
            f"mu={posterior_result['mean']:.3f}, sigma={posterior_result['std']:.3f}\n"
            f"target: mu={mu_p:.3f}, sigma={sigma_p:.3f}",
            transform=axes[0].transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axes[0].legend(fontsize=8)

        # Right: QQ plot of z-scores
        z = np.sort(np.array(calibration_result["z_scores"]))
        n = len(z)
        theoretical = norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n))
        axes[1].plot(theoretical, z, ".", ms=1, alpha=0.3)
        axes[1].plot([-4, 4], [-4, 4], "r-", lw=2)
        axes[1].set_xlabel("Theoretical N(0,1)")
        axes[1].set_ylabel(f"{title} z-scores")
        axes[1].set_title(
            f"QQ Plot (z-mean={calibration_result['z_mean']:.3f}, "
            f"z-std={calibration_result['z_std']:.3f})"
        )
        axes[1].set_aspect("equal")
        axes[1].set_xlim(-4, 4)
        axes[1].set_ylim(-4, 4)

        return axes


@dataclass
class NonlinearDecoder2D:
    """2D Gaussian latent prior with nonlinear decoder to ℝ³.

    Prior: z ~ N(0, I_2)
    Decoder: D(z) = [z1 + α·z2², z2 + α·sin(z1), β·z1·z2]
    Forward model: y = D(z) + n, n ~ N(0, σ_n² I)

    The nonlinearity parameter α controls how far from linear the decoder is.
    At α=0, β=0 the decoder is [z1, z2, 0] and everything is Gaussian.
    As α grows, Jacobian distortion breaks Tweedie-based methods.
    """

    alpha: float = 0.5
    beta: float = 0.5
    sigma_n: float = 0.3
    sigma_0: float = 1.0

    @property
    def d_latent(self):
        return 2

    @property
    def d_pixel(self):
        return 3

    def decoder(self, z):
        """Nonlinear decoder D: ℝ² → ℝ³.  z has shape (..., 2)."""
        z1, z2 = z[..., 0], z[..., 1]
        return jnp.stack([
            z1 + self.alpha * z2**2,
            z2 + self.alpha * jnp.sin(z1),
            self.beta * z1 * z2,
        ], axis=-1)

    def decoder_jacobian(self, z):
        """Analytic Jacobian J_D(z), shape (..., 3, 2)."""
        z1, z2 = z[..., 0], z[..., 1]
        ones = jnp.ones_like(z1)
        return jnp.stack([
            jnp.stack([ones, 2 * self.alpha * z2], axis=-1),
            jnp.stack([self.alpha * jnp.cos(z1), ones], axis=-1),
            jnp.stack([self.beta * z2, self.beta * z1], axis=-1),
        ], axis=-2)

    def encoder(self, x, *, n_iter=10):
        """Least-squares encoder E: ℝ³ → ℝ² via Gauss-Newton.

        Solves  E(x) = argmin_z ||D(z) - x||²  iteratively.
        At α=0, β=0 the decoder is [z1, z2, 0] so the initial guess z=x[...,:2]
        is exact; for nonzero α the iterations refine it.

        x has shape (..., 3), returns z with shape (..., 2).
        """
        # Initial guess: first two pixel components (exact when α=β=0)
        z0 = x[..., :2]

        def step(z, _):
            residual = x - self.decoder(z)          # (..., 3)
            J = self.decoder_jacobian(z)             # (..., 3, 2)
            JTJ = jnp.einsum('...pi,...pj->...ij', J, J)  # (..., 2, 2)
            JTr = jnp.einsum('...pi,...p->...i', J, residual)  # (..., 2)
            dz = jnp.linalg.solve(JTJ, JTr[..., None])[..., 0]
            return z + dz, None

        z_final, _ = jax.lax.scan(step, z0, None, length=n_iter)
        return z_final

    def score(self, z, sigma):
        """Score of the noised prior N(0, I): ∇_z log p_σ(z)."""
        return -z / (self.sigma_0**2 + sigma**2)

    def denoise(self, z, sigma, key=None):
        """Tweedie denoiser for N(0, I) prior."""
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
        """Log prior: z ~ N(0, I). z has shape (..., 2)."""
        return -0.5 * jnp.sum(z**2, axis=-1)

    def log_likelihood(self, z, y):
        """Log likelihood: y ~ N(D(z), σ_n² I). Shapes (..., 2) and (..., 3)."""
        residual = y - self.decoder(z)
        return -0.5 * jnp.sum(residual**2, axis=-1) / self.sigma_n**2

    def log_posterior(self, z, y):
        """Unnormalized log posterior."""
        return self.log_prior(z) + self.log_likelihood(z, y)

    def sample_joint(self, key, n):
        """Sample n (z_true, y_obs) pairs."""
        k1, k2 = jax.random.split(key)
        z = jax.random.normal(k1, (n, self.d_latent))
        y = self.decoder(z) + self.sigma_n * jax.random.normal(k2, (n, self.d_pixel))
        return z, y

    def posterior_grid(self, y, *, grid_range=4.0, grid_size=200):
        """Evaluate posterior on a 2D grid for a single observation y (shape (3,)).

        Returns (z1, z2, Z1, Z2, p, dz) where Z1/Z2 are meshgrid arrays.
        """
        z1 = jnp.linspace(-grid_range, grid_range, grid_size)
        z2 = jnp.linspace(-grid_range, grid_range, grid_size)
        Z1, Z2 = jnp.meshgrid(z1, z2)
        z_grid = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)  # (G, 2)
        log_p = self.log_posterior(z_grid, y)
        log_p = log_p - jnp.max(log_p)
        p = jnp.exp(log_p).reshape(grid_size, grid_size)
        dz = float((z1[1] - z1[0]) * (z2[1] - z2[0]))
        p = p / (jnp.sum(p) * dz)
        return z1, z2, Z1, Z2, p, dz

    def posterior_mean_cov(self, y, **grid_kw):
        """Posterior mean and covariance via grid integration for a single y."""
        _, _, Z1, Z2, p, dz = self.posterior_grid(y, **grid_kw)
        mu1 = jnp.sum(Z1 * p) * dz
        mu2 = jnp.sum(Z2 * p) * dz
        var1 = jnp.sum((Z1 - mu1) ** 2 * p) * dz
        var2 = jnp.sum((Z2 - mu2) ** 2 * p) * dz
        cov12 = jnp.sum((Z1 - mu1) * (Z2 - mu2) * p) * dz
        return jnp.array([mu1, mu2]), jnp.array([[var1, cov12], [cov12, var2]])

    def posterior_mean_cov_batch(self, y_batch, *, grid_range=4.0, grid_size=200):
        """Posterior mean (n,2) and covariance (n,2,2) for a batch of y values."""
        return jax.vmap(
            lambda y: self.posterior_mean_cov(y, grid_range=grid_range, grid_size=grid_size)
        )(y_batch)

    def plot(self, solver_samples, y_star, title, ax=None, _grid_cache=None):
        """2D scatter of solver samples overlaid on posterior contours.

        Pass _grid_cache=(z1, z2, p) to avoid recomputing the posterior grid.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        if _grid_cache is not None:
            z1, z2, p = _grid_cache
        else:
            z1, z2, _, _, p, _ = self.posterior_grid(y_star)
            z1, z2, p = np.array(z1), np.array(z2), np.array(p)

        ax.contourf(z1, z2, p, levels=30, cmap="Blues", alpha=0.7)
        ax.contour(z1, z2, p, levels=8, colors="steelblue", linewidths=0.5)

        samples = np.array(solver_samples)
        ax.scatter(samples[:, 0], samples[:, 1], s=1, c="red", alpha=0.3, label="Solver")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

        return ax
