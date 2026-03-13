"""Calibration and posterior diagnostics for MNISTVAE posterior sampling."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp


def latent_calibration_test(problem, solver, key, *, n=500, **solver_kw):
    """HPD calibration test.

    For each (z_true, y) pair, runs the solver and computes the HPD level
    of the solver sample under the true grid posterior.

    Under perfect calibration: HPD levels ~ Uniform(0,1), mean ~ 0.5.
    Under-dispersed: mean < 0.5 (samples near mode).
    Over-dispersed: mean > 0.5 (samples in tails).
    """
    k1, k2 = jax.random.split(key)
    z_true, y_obs = problem.sample_joint(k1, n)
    z_samples = solver(problem, y_obs, k2, **solver_kw)

    hpd_levels = problem.hpd_level(z_samples, y_obs)

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
    """Run all solvers on MNISTVAE and report calibration."""
    if solvers is None:
        from .solvers import SOLVERS
        solvers = SOLVERS
    if key is None:
        key = jax.random.PRNGKey(0)
    z_star = None
    if y_star is None:
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
    print(f"\n{'Method':<20} {'HPD mean':>9} {'HPD std':>9} {'KS stat':>9}")
    print(f"{'(calibrated)':<20} {'0.500':>9} {'0.289':>9} {'-> 0':>9}")
    print("-" * 50)
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

    # Compute (or load cached) wide posterior grid for plotting
    grid_path = output_dir / "posterior_grid.npz"
    if grid_path.exists():
        print("Loading cached posterior grid...")
        z1, z2, p, dz, _y = problem.load_posterior_grid(grid_path)
    else:
        print("Computing posterior grid (one-time, ~36s)...")
        z1, z2, p, dz = problem.save_posterior_grid(y_star, grid_path)
    grid_cache = (np.array(z1), np.array(z2), np.array(p))

    problem_name = type(problem).__name__.lower()
    for name, r in results.items():
        n_sample_imgs = 4
        n_panels = 2 + n_sample_imgs
        if z_star is None:
            n_panels -= 1
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, n_panels, height_ratios=[1.2, 0.5],
                              hspace=0.35, wspace=0.3)
        half = n_panels // 2
        ax_contour = fig.add_subplot(gs[0, :half])
        ax_hpd = fig.add_subplot(gs[0, half:])

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

        # Bottom row: decoded images
        panels = []
        labels = []
        if z_star is not None:
            panels.append(np.array(problem.decoder(z_star)).reshape(28, 28))
            labels.append("Ground truth D(z*)")
        panels.append(np.clip(np.array(y_star).reshape(28, 28), 0, 1))
        labels.append("Observation y")
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
        solver_name = name.lower().replace('+', '_').replace(' ', '_')
        fig.savefig(output_dir / f"{problem_name}_{solver_name}.png", dpi=150)
        plt.close(fig)

    # JSON summary
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
