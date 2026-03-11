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
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
