"""Benchmark latent solvers on MNISTVAE (pretrained VAE decoder on MNIST).

Usage:
    python scripts/run_mnist_vae.py [--latent-dim 2] [--n-cal 200] [--n-samples 2000]

This runs all latent solvers on the MNIST VAE problem and reports
HPD calibration (for d_latent=2) or reconstruction quality (for higher dims).
"""

import argparse
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp

import lip
from lip.solvers import LATENT_ALL


def main():
    parser = argparse.ArgumentParser(description="MNIST VAE benchmark")
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--sigma-n", type=float, default=5.0)
    parser.add_argument("--n-cal", type=int, default=200,
                        help="Number of calibration samples (grid eval is expensive)")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of posterior samples per solver")
    parser.add_argument("--solvers", nargs="*", default=None,
                        help="Subset of solvers to run (default: all)")
    args = parser.parse_args()

    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    output_dir = Path("results") / git_hash

    print("=" * 70)
    print(f"MNISTVAE (latent_dim={args.latent_dim}, sigma_n={args.sigma_n})")
    print("=" * 70)

    problem = lip.MNISTVAE(latent_dim=args.latent_dim, sigma_n=args.sigma_n)

    # Select solvers
    if args.solvers:
        solvers = {k: v for k, v in LATENT_ALL.items() if k in args.solvers}
    else:
        solvers = LATENT_ALL

    if args.latent_dim == 2:
        # Use full latent benchmark with HPD calibration
        lip.latent_benchmark(
            problem, solvers=solvers,
            n_samples=args.n_samples, n_cal=args.n_cal,
            output_dir=output_dir,
        )
    else:
        # For higher dims, run posterior test only (no grid-based calibration)
        print(f"\nNote: d_latent={args.latent_dim} > 2, skipping grid-based "
              "HPD calibration. Running posterior reconstruction test only.\n")
        _run_reconstruction_benchmark(problem, solvers, args, output_dir)


def _run_reconstruction_benchmark(problem, solvers, args, output_dir):
    """Benchmark for d_latent > 2: measure reconstruction quality."""
    key = jax.random.PRNGKey(0)

    # Generate test observation
    k1, key = jax.random.split(key)
    z_true = jax.random.normal(k1, (problem.d_latent,))
    k2, key = jax.random.split(key)
    y = problem.decoder(z_true) + problem.sigma_n * jax.random.normal(
        k2, (problem.d_pixel,)
    )

    print(f"{'Method':<20} {'||z-z*||':>9} {'||D(z)-D(z*)||':>15} {'log p(z|y)':>11}")
    print("─" * 58)

    for name, solver in solvers.items():
        k, key = jax.random.split(key)
        try:
            z_hat = solver(problem, y, k, N=100)
            z_err = float(jnp.linalg.norm(z_hat - z_true))
            x_hat = problem.decoder(z_hat)
            x_true = problem.decoder(z_true)
            x_err = float(jnp.linalg.norm(x_hat - x_true))
            lp = float(problem.log_posterior(z_hat, y))
            print(f"{name:<20} {z_err:9.3f} {x_err:15.3f} {lp:11.1f}")
        except Exception as e:
            print(f"{name:<20} FAILED: {e}")


if __name__ == "__main__":
    main()
