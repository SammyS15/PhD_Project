"""Sweep DPS hyperparameters on MNISTVAE and save each config to DPS_RESULT/.

Usage:
    python scripts/run_dps_sweep.py
    python scripts/run_dps_sweep.py --n-cal 100 --n-samples 1000
    python scripts/run_dps_sweep.py --configs 0 3 5   # run only selected configs
"""

import argparse
from functools import partial
from pathlib import Path

import lip
from lip.solvers.dps import dps

# ---- 8 DPS configurations to sweep ----
CONFIGS = [
    # idx  zeta    sigma_max  N
    (0, {"zeta": 0.005, "sigma_max": 2.0, "N": 500}),
    (1, {"zeta": 0.005, "sigma_max": 1.0, "N": 500}),
    (2, {"zeta": 0.005, "sigma_max": 2.0, "N": 1000}),
    (3, {"zeta": 0.005, "sigma_max": 1.0, "N": 1000}),
    (4, {"zeta": 0.001, "sigma_max": 2.0, "N": 500}),
    (5, {"zeta": 0.001, "sigma_max": 1.0, "N": 500}),
    (6, {"zeta": 0.001, "sigma_max": 2.0, "N": 1000}),
    (7, {"zeta": 0.001, "sigma_max": 1.0, "N": 1000}),
    (8, {"zeta": 0.004, "sigma_max": 1.0, "N": 500}),
    (9, {"zeta": 0.003, "sigma_max": 1.0, "N": 500}),
    (10, {"zeta": 0.002, "sigma_max": 1.0, "N": 500}),
]


def _config_name(params):
    return f"zeta: {params['zeta']} sigma_max: {params['sigma_max']} N: {params['N']}"


def main():
    parser = argparse.ArgumentParser(description="DPS hyperparameter sweep")
    parser.add_argument("--sigma-n", type=float, default=0.2)
    parser.add_argument("--n-cal", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-tarp-sims", type=int, default=100)
    parser.add_argument("--n-tarp-samples", type=int, default=50)
    parser.add_argument("--configs", nargs="*", type=int, default=None,
                        help="Config indices to run (default: all 0-7)")
    args = parser.parse_args()

    output_root = Path("DPS_RESULT")
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
              f"N={params['N']}")
        print("=" * 70)

        # Create a solver with these specific params
        solver_fn = partial(dps, **params)
        solvers = {f"DPS ({name})": solver_fn}

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
    print("SWEEP COMPLETE — results in DPS_RESULT/")
    print("=" * 70)
    for idx, params in configs:
        name = _config_name(params)
        json_path = output_root / name / "mnistvae_results.json"
        if json_path.exists():
            import json
            with open(json_path) as f:
                data = json.load(f)
            for sname, metrics in data["solvers"].items():
                print(f"  {name:<35} HPD={metrics['hpd_mean']:.3f}  "
                      f"KS={metrics['hpd_ks']:.3f}")


if __name__ == "__main__":
    main()
