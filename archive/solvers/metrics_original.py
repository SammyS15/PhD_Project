"""Calibration and posterior diagnostics for posterior sampling benchmarks."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp


def calibration_test(problem, solver, key, *, n=10_000, **solver_kw):
    """One sample per random (x_true, y) pair. Returns z-score statistics."""
    k1, k2 = jax.random.split(key)
    x_true, y_obs = problem.sample_joint(k1, n)
    x_samples = solver(problem, y_obs, k2, **solver_kw)
    z = (x_samples - problem.posterior_mean(y_obs)) / problem.posterior_std
    return {"z_mean": float(jnp.mean(z)), "z_std": float(jnp.std(z)), "z_scores": z}


def posterior_test(problem, solver, y, key, *, n=50_000, **solver_kw):
    """Many samples for a fixed y. Returns mean/std vs analytic target."""
    y_batch = jnp.full(n, y)
    x_samples = solver(problem, y_batch, key, **solver_kw)
    return {
        "mean": float(jnp.mean(x_samples)),
        "std": float(jnp.std(x_samples)),
        "target_mean": float(problem.posterior_mean(y)),
        "target_std": float(problem.posterior_std),
        "y_star": float(y),
        "samples": x_samples,
    }


def benchmark(problem, solvers=None, key=None, *, y_star=1.5, n_samples=50_000,
              n_cal=10_000, output_dir=None):
    """Run all solvers on a problem. Optionally save plots and JSON to output_dir."""
    if solvers is None:
        from .solvers import ALL
        solvers = ALL
    if key is None:
        key = jax.random.PRNGKey(0)

    results = {}
    for i, (name, solver) in enumerate(solvers.items()):
        k1, k2 = jax.random.split(jax.random.PRNGKey(i))
        post = posterior_test(problem, solver, y_star, k1, n=n_samples)
        cal = calibration_test(problem, solver, k2, n=n_cal)
        results[name] = {**post, **cal}

    print_table(problem, results, y_star)

    if output_dir is not None:
        _save_results(problem, results, y_star, Path(output_dir))

    return results


def print_table(problem, results, y_star=1.5):
    """Print summary table of benchmark results."""
    target_mean = float(problem.posterior_mean(y_star))
    target_std = float(problem.posterior_std)

    print(f"{'Method':<18} {'mu':>7} {'sigma':>7} {'z-mean':>7} {'z-std':>7}")
    print(f"{'Target':<18} {target_mean:7.3f} {target_std:7.3f} {'0.000':>7} {'1.000':>7}")
    print("\u2500" * 50)
    for name, r in results.items():
        print(f"{name:<18} {r['mean']:7.3f} {r['std']:7.3f} {r['z_mean']:7.3f} {r['z_std']:7.3f}")


def _save_results(problem, results, y_star, output_dir):
    """Save per-solver plots and a summary JSON to output_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-solver diagnostic plots
    for name, r in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        post = {k: r[k] for k in ("mean", "std", "target_mean", "target_std", "y_star", "samples")}
        cal = {k: r[k] for k in ("z_mean", "z_std", "z_scores")}
        problem.plot(post, cal, name, axes=axes)
        plt.tight_layout()
        problem_name = type(problem).__name__.lower()
        solver_name = name.lower().replace('+', '_')
        fig.savefig(output_dir / f"{problem_name}_{solver_name}.png", dpi=150)
        plt.close(fig)

    # JSON with quantitative results (no arrays)
    json_results = {
        "problem": type(problem).__name__,
        "y_star": y_star,
        "target_mean": float(problem.posterior_mean(y_star)),
        "target_std": float(problem.posterior_std),
        "solvers": {
            name: {
                "mean": r["mean"],
                "std": r["std"],
                "z_mean": r["z_mean"],
                "z_std": r["z_std"],
            }
            for name, r in results.items()
        },
    }
    problem_name = type(problem).__name__.lower()
    with open(output_dir / f"{problem_name}_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


# --- 2D latent-space metrics ---

def latent_calibration_test(problem, solver, key, *, n=500, grid_size=200,
                            grid_range=4.0, **solver_kw):
    """HPD calibration test for 2D latent problems.

    For each (z_true, y) pair, runs the solver and computes the HPD level
    of the solver sample under the true grid posterior.

    Under perfect calibration: HPD levels ~ Uniform(0,1), mean ≈ 0.5.
    Under-dispersed: mean < 0.5 (samples near mode).
    Over-dispersed: mean > 0.5 (samples in tails).
    """
    k1, k2 = jax.random.split(key)
    z_true, y_obs = problem.sample_joint(k1, n)
    z_samples = solver(problem, y_obs, k2, **solver_kw)

    hpd_levels = problem.hpd_level(
        z_samples, y_obs, grid_range=grid_range, grid_size=grid_size
    )

    # KS test against Uniform(0,1)
    sorted_levels = jnp.sort(hpd_levels)
    n_samples = len(sorted_levels)
    uniform_cdf = jnp.linspace(0.5 / n_samples, 1 - 0.5 / n_samples, n_samples)
    ks_stat = float(jnp.max(jnp.abs(sorted_levels - uniform_cdf)))

    return {
        "hpd_mean": float(jnp.mean(hpd_levels)),
        "hpd_std": float(jnp.std(hpd_levels)),
        "hpd_ks": ks_stat,
        "hpd_levels": hpd_levels,
    }


def latent_posterior_test(problem, solver, y, key, *, n=5000, **solver_kw):
    """Many samples for a fixed y. Returns solver sample stats vs grid posterior."""
    y_batch = jnp.broadcast_to(y, (n, *y.shape))
    z_samples = solver(problem, y_batch, key, **solver_kw)

    mu_grid, cov_grid = problem.posterior_mean_cov(y)
    return {
        "mean": jnp.asarray(z_samples.mean(axis=0)),
        "cov": jnp.asarray(jnp.cov(z_samples.T)),
        "target_mean": jnp.asarray(mu_grid),
        "target_cov": jnp.asarray(cov_grid),
        "y_star": jnp.asarray(y),
        "samples": z_samples,
    }


def latent_benchmark(problem, solvers=None, key=None, *, y_star=None,
                     n_samples=5000, n_cal=500, output_dir=None):
    """Run all latent solvers on a nonlinear decoder problem."""
    if solvers is None:
        from .solvers import LATENT_ALL
        solvers = LATENT_ALL
    if key is None:
        key = jax.random.PRNGKey(0)
    z_star = None
    if y_star is None:
        # Generate a typical observation
        k, key = jax.random.split(key)
        z_star = jnp.array([0.8, -0.5])
        y_star = problem.decoder(z_star) + problem.sigma_n * jax.random.normal(k, (problem.d_pixel,))

    results = {}
    for i, (name, solver) in enumerate(solvers.items()):
        k1, k2 = jax.random.split(jax.random.PRNGKey(i + 100))
        post = latent_posterior_test(problem, solver, y_star, k1, n=n_samples)
        cal = latent_calibration_test(problem, solver, k2, n=n_cal)
        results[name] = {**post, **cal}

    _print_latent_table(problem, results)

    if output_dir is not None:
        _save_latent_results(problem, results, y_star, Path(output_dir),
                             z_star=z_star)

    return results


def _print_latent_table(problem, results):
    """Print summary table for latent benchmark."""
    print(f"\n{'Method':<20} {'HPD mean':>9} {'HPD std':>9} {'KS stat':>9}")
    print(f"{'(calibrated)':<20} {'0.500':>9} {'0.289':>9} {'→ 0':>9}")
    print("─" * 50)

    for name, r in results.items():
        print(f"{name:<20} {r['hpd_mean']:9.3f} {r['hpd_std']:9.3f}"
              f" {r['hpd_ks']:9.3f}")


def _save_latent_results(problem, results, y_star, output_dir, z_star=None):
    """Save per-solver contour plots and summary JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute posterior grid once for all solver plots
    z1, z2, _, _, p, _ = problem.posterior_grid(y_star)
    grid_cache = (np.array(z1), np.array(z2), np.array(p))

    # Check if the problem has a pixel-space decoder (e.g. MNISTVAE)
    has_images = hasattr(problem, 'd_pixel') and problem.d_pixel > 10

    problem_name = type(problem).__name__.lower()
    for name, r in results.items():
        if has_images:
            n_sample_imgs = 4
            n_panels = 2 + n_sample_imgs  # ground truth + obs + samples
            if z_star is None:
                n_panels -= 1
            fig = plt.figure(figsize=(14, 9))
            gs = fig.add_gridspec(2, n_panels, height_ratios=[1.2, 0.5],
                                  hspace=0.35, wspace=0.3)
            # Top row: contour and HPD span half each
            half = n_panels // 2
            ax_contour = fig.add_subplot(gs[0, :half])
            ax_hpd = fig.add_subplot(gs[0, half:])
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ax_contour = axes[0]
            ax_hpd = axes[1]

        # Left: posterior contours + solver samples
        problem.plot(r["samples"], y_star, name, ax=ax_contour,
                     _grid_cache=grid_cache)

        # Right: HPD level histogram
        hpd = np.array(r["hpd_levels"])
        ax_hpd.hist(hpd, bins=20, range=(0, 1), density=True, alpha=0.7,
                     color="steelblue", edgecolor="white")
        ax_hpd.axhline(1.0, color="red", ls="--", lw=1.5, label="Uniform(0,1)")
        ax_hpd.set_xlabel("HPD level")
        ax_hpd.set_ylabel("Density")
        ax_hpd.set_title(
            f"HPD calibration (mean={r['hpd_mean']:.3f}, KS={r['hpd_ks']:.3f})")
        ax_hpd.set_xlim(0, 1)
        ax_hpd.legend(fontsize=9)

        # Bottom row: decoded images (ground truth, observation, posterior samples)
        if has_images:
            panels = []
            labels = []
            if z_star is not None:
                panels.append(np.array(problem.decoder(z_star)).reshape(28, 28))
                labels.append("Ground truth D(z*)")
            panels.append(np.clip(np.array(y_star).reshape(28, 28), 0, 1))
            labels.append("Observation y")
            # Pick a few evenly-spaced posterior samples
            samples = np.array(r["samples"])
            n_s = len(samples)
            idxs = np.linspace(0, n_s - 1, n_sample_imgs, dtype=int)
            for j, idx in enumerate(idxs):
                z_j = jnp.array(samples[idx])
                panels.append(np.array(problem.decoder(z_j)).reshape(28, 28))
                labels.append(f"Sample {j+1}")

            for i, (img, label) in enumerate(zip(panels, labels)):
                ax = fig.add_subplot(gs[1, i])
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(label, fontsize=9)
                ax.axis("off")

        plt.suptitle(name, fontsize=14, y=0.98)
        if not has_images:
            plt.tight_layout()
        solver_name = name.lower().replace('+', '_').replace(' ', '_')
        fig.savefig(output_dir / f"{problem_name}_{solver_name}.png", dpi=150)
        plt.close(fig)

    # JSON summary — convert arrays to lists for serialization
    def _to_list(v):
        return v.tolist() if hasattr(v, 'tolist') else v

    json_results = {
        "problem": type(problem).__name__,
        "y_star": _to_list(y_star),
        "solvers": {
            name: {
                "mean": _to_list(r["mean"]),
                "target_mean": _to_list(r["target_mean"]),
                "hpd_mean": r["hpd_mean"],
                "hpd_std": r["hpd_std"],
                "hpd_ks": r["hpd_ks"],
            }
            for name, r in results.items()
        },
    }
    with open(output_dir / f"{problem_name}_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
