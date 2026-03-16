"""Sweep MMPS hyperparameters on MNISTVAE and save each config to MMPS_RESULT/.

Usage:
    python scripts/run_mmps_sweep.py
    python scripts/run_mmps_sweep.py --n-cal 50 --n-samples 100
    python scripts/run_mmps_sweep.py --configs 1 5 8 9 10
"""

import argparse
from functools import partial
from pathlib import Path

import lip
from lip.solvers.mmps import mmps

# ---- MMPS configurations to sweep ----
# MMPS self-calibrates via Tweedie covariance, so zeta ~ 1.0 is expected.
# Key axes: zeta (guidance strength), sigma_max, N, cg_iters.
CONFIGS = [
    (0,  {"zeta": 5.0, "sigma_max": 1.0, "N": 500, "cg_iters": 2}),
    (1,  {"zeta": 10.0, "sigma_max": 1.0, "N": 500, "cg_iters": 2}),
    (2,  {"zeta": 15.0, "sigma_max": 1.0, "N": 500, "cg_iters": 2}),
    (3,  {"zeta": 20.0, "sigma_max": 1.0, "N": 500, "cg_iters": 2}),
    (4,  {"zeta": 25.0, "sigma_max": 1.0, "N": 500, "cg_iters": 2}),
    (5,  {"zeta": 30.0, "sigma_max": 1.0, "N": 500, "cg_iters": 2}),
    (6,  {"zeta": 35.0, "sigma_max": 1.0, "N": 500, "cg_iters": 2}),
    (7,  {"zeta": 40.0, "sigma_max": 1.0, "N": 500, "cg_iters": 2}),
    (8,  {"zeta": 5.0, "sigma_max": 1.0, "N": 500, "cg_iters": 1}),
    (9,  {"zeta": 10.0, "sigma_max": 1.0, "N": 500, "cg_iters": 1}),
    (10,  {"zeta": 15.0, "sigma_max": 1.0, "N": 500, "cg_iters": 1}),
    (11,  {"zeta": 20.0, "sigma_max": 1.0, "N": 500, "cg_iters": 1}),
    (12,  {"zeta": 25.0, "sigma_max": 1.0, "N": 500, "cg_iters": 1}),
    (13,  {"zeta": 30.0, "sigma_max": 1.0, "N": 500, "cg_iters": 1}),
    (14,  {"zeta": 35.0, "sigma_max": 1.0, "N": 500, "cg_iters": 1}),
    (15,  {"zeta": 40.0, "sigma_max": 1.0, "N": 500, "cg_iters": 1}),
    (16,  {"zeta": 5.0, "sigma_max": 1.0, "N": 500, "cg_iters": 3}),
    (17,  {"zeta": 10.0, "sigma_max": 1.0, "N": 500, "cg_iters": 3}),
    (18,  {"zeta": 15.0, "sigma_max": 1.0, "N": 500, "cg_iters": 3}),
    (19,  {"zeta": 20.0, "sigma_max": 1.0, "N": 500, "cg_iters": 3}),
    (20,  {"zeta": 25.0, "sigma_max": 1.0, "N": 500, "cg_iters": 3}),
    (21,  {"zeta": 30.0, "sigma_max": 1.0, "N": 500, "cg_iters": 3}),
    (22,  {"zeta": 35.0, "sigma_max": 1.0, "N": 500, "cg_iters": 3}),
    (23,  {"zeta": 40.0, "sigma_max": 1.0, "N": 500, "cg_iters": 3}),
]


def _config_name(params):
    return (f"zeta: {params['zeta']} sigma_max: {params['sigma_max']} "
            f"N: {params['N']} cg: {params['cg_iters']}")


def main():
    parser = argparse.ArgumentParser(description="MMPS hyperparameter sweep")
    parser.add_argument("--sigma-n", type=float, default=0.2)
    parser.add_argument("--n-cal", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-tarp-sims", type=int, default=100)
    parser.add_argument("--n-tarp-samples", type=int, default=50)
    parser.add_argument("--configs", nargs="*", type=int, default=None,
                        help="Config indices to run (default: all 0-7)")
    args = parser.parse_args()

    output_root = Path("MMPS_RESULT")
    problem = lip.MNISTVAE(sigma_n=args.sigma_n)

    # Select which configs to run
    if args.configs is not None:
        configs = [(i, p) for i, p in CONFIGS if i in args.configs]
    else:
        configs = CONFIGS

    for idx, params in configs:
        name = _config_name(params)
        output_dir = output_root / name

        print("\n" + "=" * 70)
        print(f"Config {idx}: {name}")
        print(f"  zeta={params['zeta']}, sigma_max={params['sigma_max']}, "
              f"N={params['N']}, cg_iters={params['cg_iters']}")
        print("=" * 70)

        # Create a solver with these specific params
        solver_fn = partial(mmps, **params)
        solvers = {f"MMPS ({name})": solver_fn}

        lip.latent_benchmark(
            problem, solvers=solvers,
            n_samples=args.n_samples,
            n_cal=args.n_cal,
            n_tarp_sims=args.n_tarp_sims,
            n_tarp_samples=args.n_tarp_samples,
            output_dir=output_dir,
        )

    # Print final summary across all configs
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE — results in MMPS_RESULT/")
    print("=" * 70)
    for idx, params in configs:
        name = _config_name(params)
        json_path = output_root / name / "mnistvae_results.json"
        if json_path.exists():
            import json
            with open(json_path) as f:
                data = json.load(f)
            for sname, metrics in data["solvers"].items():
                print(f"  {name:<50} HPD={metrics['hpd_mean']:.3f}  "
                      f"KS={metrics['hpd_ks']:.3f}")


if __name__ == "__main__":
    main()
