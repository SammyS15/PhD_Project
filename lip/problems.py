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
