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

def latent_calibration_test(problem, solver, key, *, n=500, grid_size=100,
                            grid_range=4.0, **solver_kw):
    """Calibration test for 2D latent problems using Mahalanobis distance.

    For each (z_true, y) pair, computes the grid-based posterior mean/cov
    and the Mahalanobis distance of the solver sample.
    Under calibration: d² ~ χ²(d_latent), so mean ≈ d_latent.
    """
    k1, k2 = jax.random.split(key)
    z_true, y_obs = problem.sample_joint(k1, n)
    z_samples = solver(problem, y_obs, k2, **solver_kw)

    mu, cov = problem.posterior_mean_cov_batch(
        y_obs, grid_range=grid_range, grid_size=grid_size
    )

    # Mahalanobis distance of solver samples
    diff = z_samples - mu  # (n, 2)
    cov_inv = jnp.linalg.inv(cov)  # (n, 2, 2)
    maha2 = jnp.einsum('ni,nij,nj->n', diff, cov_inv, diff)

    # Under χ²(d): mean = d, std = sqrt(2d)
    d = problem.d_latent
    return {
        "maha2_mean": float(jnp.mean(maha2)),
        "maha2_std": float(jnp.std(maha2)),
        "maha2_target_mean": float(d),
        "maha2_target_std": float(jnp.sqrt(2 * d)),
        "maha2": maha2,
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
        _save_latent_results(problem, results, y_star, Path(output_dir))

    return results


def _print_latent_table(problem, results):
    """Print summary table for latent benchmark."""
    d = problem.d_latent
    print(f"\n{'Method':<20} {'μ_z1':>7} {'μ_z2':>7} {'σ²_z1':>7} {'σ²_z2':>7}"
          f" {'d²_mean':>8} {'d²_std':>8}")
    print("─" * 70)

    for name, r in results.items():
        mu = jnp.asarray(r["mean"])
        cov = jnp.asarray(r["cov"])
        print(f"{name:<20} {float(mu[0]):7.3f} {float(mu[1]):7.3f}"
              f" {float(cov[0, 0]):7.3f} {float(cov[1, 1]):7.3f}"
              f" {r['maha2_mean']:8.3f} {r['maha2_std']:8.3f}")

    print(f"\n  Target d² ~ χ²({d}): mean = {d:.1f}, std = {jnp.sqrt(2*d):.3f}")


def _save_latent_results(problem, results, y_star, output_dir):
    """Save per-solver contour plots and summary JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute posterior grid once for all solver plots
    z1, z2, _, _, p, _ = problem.posterior_grid(y_star)
    grid_cache = (np.array(z1), np.array(z2), np.array(p))

    problem_name = type(problem).__name__.lower()
    for name, r in results.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        problem.plot(r["samples"], y_star, name, ax=ax, _grid_cache=grid_cache)
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
                "maha2_mean": r["maha2_mean"],
                "maha2_std": r["maha2_std"],
            }
            for name, r in results.items()
        },
    }
    with open(output_dir / f"{problem_name}_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
