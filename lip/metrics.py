"""Calibration and posterior diagnostics for MNISTVAE posterior sampling."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tarp import get_tarp_coverage
from tqdm import tqdm

try:
    import torch
    from mira_score import get_device as _mira_get_device, mira as _mira_fn
    _MIRA_AVAILABLE = True
except ImportError:
    _MIRA_AVAILABLE = False


# ---------------------------------------------------------------------------
# HPD calibration test (existing)
# ---------------------------------------------------------------------------

def latent_calibration_test(problem, solver, key, *, n=500,
                            z_true=None, y_obs=None, **solver_kw):
    """HPD calibration test.

    For each (z_true, y) pair, runs the solver and computes the HPD level
    of the solver sample under the true grid posterior.

    Under perfect calibration: HPD levels ~ Uniform(0,1), mean ~ 0.5.
    Under-dispersed: mean < 0.5 (samples near mode).
    Over-dispersed: mean > 0.5 (samples in tails).

    Args:
        z_true: pre-generated ground truth (shared across solvers).
        y_obs:  pre-generated observations (shared across solvers).
    """
    if z_true is None:
        k1, key = jax.random.split(key)
        z_true, y_obs = problem.sample_joint(k1, n)

    z_samples = solver(problem, y_obs, key, **solver_kw)

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

# ---------------------------------------------------------------------------
# TARP coverage test  (Lemos, Coogan et al 2023, arXiv:2302.03026)
# ---------------------------------------------------------------------------

def tarp_calibration_test(problem, solver, key, *, n_sims=500,
                          n_samples=100, z_true=None, y_obs=None,
                          references=None, **solver_kw):
    """Run TARP coverage test for a solver.

    Generates *n_sims* (z_true, y) pairs and for each draws *n_samples*
    posterior samples from the solver, then computes TARP ECP vs alpha.

    Uses norm=False with reference points at the encoder MAP estimate for
    each observation.  This places references at the posterior center so
    that all distances are on the posterior scale (~0.015), giving TARP
    full power to distinguish calibrated vs over/under-dispersed posteriors.
    With prior-based or U[0,1] references the reference-to-cluster distance
    (~1-2) dwarfs the within-posterior scale, destroying sensitivity.

    Args:
        problem:   MNISTVAE instance.
        solver:    solver function  ``(problem, y, key, **kw) -> z``.
        key:       JAX PRNG key.
        n_sims:    number of (z_true, y) simulation pairs.
        n_samples: posterior samples per simulation.
        z_true:    pre-generated ground truth (shared across solvers).
        y_obs:     pre-generated observations (shared across solvers).
        references: pre-generated TARP reference points (shared across solvers).

    Returns:
        dict with tarp_ecp, tarp_alpha, tarp_ecp_bootstrap, tarp_alpha_bootstrap.
    """
    if z_true is None:
        k1, key = jax.random.split(key)
        z_true, y_obs = problem.sample_joint(k1, n_sims)

    n_sims = z_true.shape[0]

    # Draw n_samples posterior samples per simulation.
    # Loop over n_samples with batch size n_sims rather than one mega-batch of
    # n_sims*n_samples:
    #   - reuses the JIT-compiled vmap from the HPD test (same batch shape)
    #   - peak memory: n_sims * d_pixel instead of n_sims * n_samples * d_pixel
    #   - each iteration converts to numpy immediately, freeing JAX buffers
    samples = []
    for _ in range(n_samples):
        key, k_s = jax.random.split(key)
        z_s = np.array(solver(problem, y_obs, k_s, **solver_kw))  # (n_sims, d)
        samples.append(z_s)

    # z_stack: (n_sims, n_samples, d)
    z_stack = np.stack(samples, axis=1)

    # Reference points at the encoder MAP for each observation.
    # This places the reference at the posterior center so distances to
    # samples and theta are on the posterior scale (~0.015), not the
    # prior scale (~1.0).  TARP validity is preserved because the
    # reference depends on the data y, not on theta or the samples.
    # Used as centers for both MIRA and TARP.
    if references is None:
        references = np.array(problem.encoder(y_obs))

    # ---- MIRA score (computed here while z_stack is still (n_sims, n_samples, d)) ----
    mira_mean, mira_std = None, None
    if _MIRA_AVAILABLE:
        device = _mira_get_device()
        # MIRA expects: truth (T, q), posterior (M, T, S, q)
        truth_t = torch.from_numpy(np.array(z_true)).float().to(device)
        post_t = torch.from_numpy(z_stack).float().unsqueeze(0).to(device)
        # Use encoder MAP as reference_choice (not center_choice):
        #   - centers remain random U[0,1]^q per MC run, preserving MIRA validity
        #   - radius = ||random_center - encoder_MAP|| anchors ball size to the
        #     posterior scale rather than the prior scale (~1-2), giving MIRA
        #     power to distinguish calibrated vs miscalibrated samplers
        ref_t = torch.from_numpy(references).float().to(device)
        m_mean, m_std = _mira_fn(
            truth_t,
            post_t,
            reference_choice=ref_t,
            disable_tqdm=True)
        mira_mean = float(m_mean[0].cpu())
        mira_std = float(m_std[0].cpu())

    # Transpose to (n_samples, n_sims, d) for TARP
    z_stack_tarp = z_stack.transpose(1, 0, 2)

    ecp, alpha = get_tarp_coverage(
        z_stack_tarp, z_true, references=references,
        metric='euclidean', norm=False,
    )
    ecp_bootstrap, alpha_bootstrap = get_tarp_coverage(
        z_stack_tarp, z_true, references=references,
        metric='euclidean', norm=False, bootstrap=True,
    )

    return {
        "tarp_ecp": ecp,
        "tarp_alpha": alpha,
        "tarp_ecp_bootstrap": ecp_bootstrap,
        "alpha_bootstrap": alpha_bootstrap,
        "mira_mean": mira_mean,
        "mira_std": mira_std,
    }

# ---------------------------------------------------------------------------
# Posterior sampling test (for a fixed y)
# ---------------------------------------------------------------------------

def latent_posterior_test(problem, solver, y, key, *, n=5000, **solver_kw):
    """Run solver for a fixed y, returning samples for plotting."""
    y_batch = jnp.broadcast_to(y, (n, *y.shape))
    z_samples = solver(problem, y_batch, key, **solver_kw)
    return {"samples": z_samples}


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def latent_benchmark(problem, solvers=None, key=None, *, y_star=None,
                     n_samples=5000, n_cal=500, n_tarp_sims=None,
                     n_tarp_samples=100, output_dir=None):
    """Run all solvers on MNISTVAE and report calibration.

    Args:
        problem:        MNISTVAE instance.
        solvers:        dict of {name: solver_fn} (default: all registered).
        key:            JAX PRNG key.
        y_star:         fixed observation for posterior plots.
        n_samples:      posterior samples per solver for contour plots.
        n_cal:          ground truths for HPD, TARP, and MIRA (all three share
                        this count so comparisons are apples-to-apples).
        n_tarp_sims:    override n_cal for TARP/MIRA only (rarely needed).
        n_tarp_samples: posterior samples per simulation for TARP/MIRA.
        output_dir:     save plots and JSON here (None = skip).
    """
    if n_tarp_sims is None:
        n_tarp_sims = n_cal
    if solvers is None:
        from .solvers import SOLVERS
        solvers = SOLVERS
    if key is None:
        key = jax.random.PRNGKey(0)
    z_star = None
    if y_star is None:
        k, key = jax.random.split(key)
        z_star = jnp.array([0.8, -0.5])
        y_star = (problem.decoder(z_star)
                  + problem.sigma_n * jax.random.normal(k, (problem.d_pixel,)))

    # Pre-generate shared test data so all solvers are evaluated on the
    # exact same (z_true, y_obs) pairs -- apples-to-apples comparison.
    # HPD, TARP, and MIRA all use n_cal ground truths by default.
    k_hpd, k_tarp, key = jax.random.split(key, 3)
    z_true_hpd, y_obs_hpd = problem.sample_joint(k_hpd, n_cal)
    z_true_tarp, y_obs_tarp = problem.sample_joint(k_tarp, n_tarp_sims)
    # Encoder MAP references for TARP (shared so bootstrap is consistent).
    tarp_references = np.array(problem.encoder(y_obs_tarp))

    print(f"Shared test data: {n_cal} ground truths "
          f"(HPD: {n_cal}×1, TARP/MIRA: {n_tarp_sims}×{n_tarp_samples})")

    results = {}
    for i, (name, solver) in enumerate(solvers.items()):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(i + 100), 3)
        print(f"\n--- {name} ---")

        print(f"  Posterior samples (n={n_samples})...")
        post = latent_posterior_test(problem, solver, y_star, k1, n=n_samples)

        print(f"  HPD calibration   (n={n_cal})...")
        cal = latent_calibration_test(problem, solver, k2, n=n_cal,
                                      z_true=z_true_hpd, y_obs=y_obs_hpd)

        print(f"  TARP coverage     (sims={n_tarp_sims}, "
              f"samples={n_tarp_samples})...")
        tarp = tarp_calibration_test(problem, solver, k3,
                                     n_sims=n_tarp_sims,
                                     n_samples=n_tarp_samples,
                                     z_true=z_true_tarp,
                                     y_obs=y_obs_tarp,
                                     references=tarp_references)

        results[name] = {**post, **cal, **tarp}

    _print_latent_table(problem, results, n_tarp_sims=n_tarp_sims)

    if output_dir is not None:
        _save_latent_results(problem, results, y_star, Path(output_dir),
                             z_star=z_star, n_tarp_samples=n_tarp_samples,
                             n_tarp_sims=n_tarp_sims)

    return results


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_latent_table(problem, results, n_tarp_sims=500):
    has_mira = any(r.get("mira_mean") is not None for r in results.values())
    mira_std_target = np.sqrt(1.0 / (18.0 * n_tarp_sims))
    header = (f"{'Method':<20} {'HPD mean':>9} {'HPD std':>9} {'HPD KS':>9}"
              + (f" {'MIRA':>9} {'MIRA std':>9}" if has_mira else ""))
    target = (f"{'(calibrated)':<20} {'0.500':>9} {'0.289':>9} {'-> 0':>9}"
              + (f" {2/3:9.3f} {mira_std_target:9.4f}" if has_mira else ""))
    print(f"\n{header}")
    print(target)
    print("-" * len(header))
    for name, r in results.items():
        row = (f"{name:<20} {r['hpd_mean']:9.3f} {r['hpd_std']:9.3f}"
               f" {r['hpd_ks']:9.3f}")
        if has_mira:
            m, ms = r.get("mira_mean"), r.get("mira_std")
            row += f" {m:9.3f} {ms:9.3f}" if m is not None else f" {'N/A':>9} {'N/A':>9}"
        print(row)


# ---------------------------------------------------------------------------
# Plot / save
# ---------------------------------------------------------------------------

def _save_latent_results(problem, results, y_star, output_dir, z_star=None,
                         n_tarp_samples=100, n_tarp_sims=500):
    """Save two comparison figures, TARP/MIRA overlays, and JSON.

    Figure 1 (_diagnostics.png):  n_methods × 3
      Posterior Samples | HPD Level | TARP Coverage

    Figure 2 (_reconstructions.png):  n_methods × 5
      Ground Truth | Observation | Posterior Mean | Posterior Std | Residual
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute (or load cached) wide posterior grid for plotting
    grid_path = output_dir / "posterior_grid.npz"
    if grid_path.exists():
        print("Loading cached posterior grid...")
        z1, z2, p, _, _y = problem.load_posterior_grid(grid_path)
    else:
        print("Computing posterior grid (one-time, ~36s)...")
        z1, z2, p, _ = problem.save_posterior_grid(y_star, grid_path)
    grid_cache = (np.array(z1), np.array(z2), np.array(p))

    problem_name = type(problem).__name__.lower()
    obs_img = np.clip(np.array(y_star).reshape(28, 28), 0, 1)
    gt_img = (np.array(problem.decoder(z_star)).reshape(28, 28)
              if z_star is not None else None)

    n_methods = len(results)

    # -----------------------------------------------------------------------
    # Plot 1: Diagnostics — n_methods × 3  (Samples | HPD | TARP)
    # -----------------------------------------------------------------------
    _CELL_W, _CELL_H = 3.4, 3.0   # inches per cell
    fig1, axes1 = plt.subplots(
        n_methods, 3,
        figsize=(3 * _CELL_W, n_methods * _CELL_H),
        constrained_layout=True,
    )
    if n_methods == 1:
        axes1 = axes1[np.newaxis, :]

    for col, title in enumerate(["Posterior Samples", "HPD Level", "TARP Coverage"]):
        axes1[0, col].set_title(title, fontsize=11, fontweight="bold", pad=6)

    for row, (name, r) in enumerate(results.items()):
        ax_c, ax_h, ax_t = axes1[row, 0], axes1[row, 1], axes1[row, 2]

        # -- Posterior samples (contour) --
        problem.plot(r["samples"], y_star, name, ax=ax_c, _grid_cache=grid_cache)
        ax_c.set_ylabel(name, fontsize=9, labelpad=5)
        ax_c.tick_params(labelsize=7)

        # -- HPD histogram --
        hpd = np.array(r["hpd_levels"])
        ax_h.hist(hpd, bins=20, range=(0, 1), density=True,
                  color="steelblue", edgecolor="white", linewidth=0.4, alpha=0.85)
        ax_h.axhline(1.0, color="crimson", ls="--", lw=1.5)
        ax_h.set_xlim(0, 1)
        ax_h.set_xlabel("HPD level", fontsize=8)
        ax_h.set_ylabel("Density", fontsize=8)
        ax_h.tick_params(labelsize=7)
        ax_h.text(
            0.97, 0.97,
            f"mean = {r['hpd_mean']:.3f}\nKS = {r['hpd_ks']:.3f}",
            transform=ax_h.transAxes, fontsize=7.5,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.9),
        )

        # -- TARP coverage --
        if "tarp_ecp" in r:
            alpha    = np.array(r["tarp_alpha"])
            ecp_boot = np.array(r["tarp_ecp_bootstrap"])
            mean_ecp = ecp_boot.mean(axis=0)
            std_ecp  = ecp_boot.std(axis=0)
            ax_t.plot([0, 1], [0, 1], ls="--", color="k", lw=1.2)
            ax_t.plot(alpha, mean_ecp, lw=1.8, color="steelblue")
            ax_t.fill_between(alpha, mean_ecp - std_ecp, mean_ecp + std_ecp,
                              alpha=0.25, color="steelblue")
            ax_t.set_xlim(0, 1)
            ax_t.set_ylim(0, 1)
            ax_t.set_xlabel("Credibility level", fontsize=8)
            ax_t.set_ylabel("Expected coverage", fontsize=8)
            dev = r.get("tarp_max_dev")
            if dev is not None:
                ax_t.text(
                    0.03, 0.97, f"max dev = {dev:.3f}",
                    transform=ax_t.transAxes, fontsize=7.5,
                    ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.9),
                )
        else:
            ax_t.text(0.5, 0.5, "N/A", ha="center", va="center",
                      transform=ax_t.transAxes, fontsize=11, color="gray")
            ax_t.set_xlim(0, 1)
            ax_t.set_ylim(0, 1)
        ax_t.tick_params(labelsize=7)

    fig1.suptitle(f"{problem_name}  —  Diagnostics", fontsize=13, fontweight="bold")
    fig1.savefig(output_dir / f"{problem_name}_diagnostics.png",
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # -----------------------------------------------------------------------
    # Plot 2: Reconstructions — n_methods × 5  (GT | Obs | Mean | Std | Res)
    # Pre-compute decoded images and shared colour scales for Std / Residual.
    # -----------------------------------------------------------------------
    img_cols = []
    if gt_img is not None:
        img_cols.append(("Ground Truth",    "gray", "fixed"))
    img_cols += [
        ("Observation",    "gray", "fixed"),
        ("Posterior Mean", "gray", "fixed"),
        ("Posterior Std",  "hot",  "shared_std"),
        ("Residual |y−μ|", "hot",  "shared_res"),
    ]
    n_img_cols = len(img_cols)

    # Decode once per method; accumulate shared vmaxes
    _decoded_cache = {}
    _std_vmax, _res_vmax = 0.0, 0.0
    for name, r in results.items():
        samples_z = np.array(r["samples"])
        decoded   = np.array(problem.decoder(jnp.array(samples_z)))
        pm  = decoded.mean(axis=0).reshape(28, 28)
        ps  = decoded.std(axis=0).reshape(28, 28)
        res = np.abs(obs_img - pm)
        _std_vmax = max(_std_vmax, float(ps.max()))
        _res_vmax = max(_res_vmax, float(res.max()))
        _decoded_cache[name] = (pm, ps, res)
    _std_vmax = max(_std_vmax, 1e-6)
    _res_vmax = max(_res_vmax, 1e-6)

    _CELL_IMG = 2.2   # square inches per image cell
    fig2, axes2 = plt.subplots(
        n_methods, n_img_cols,
        figsize=(n_img_cols * _CELL_IMG + 0.8, n_methods * _CELL_IMG + 0.4),
        constrained_layout=True,
    )
    if n_methods == 1:
        axes2 = axes2[np.newaxis, :]
    if n_img_cols == 1:
        axes2 = axes2[:, np.newaxis]

    # Column headers
    for col, (title, _, _) in enumerate(img_cols):
        axes2[0, col].set_title(title, fontsize=11, fontweight="bold", pad=6)

    for row, (name, r) in enumerate(results.items()):
        pm, ps, res = _decoded_cache[name]
        img_data = []
        if gt_img is not None:
            img_data.append(gt_img)
        img_data += [obs_img, pm, ps, res]

        for col, (img, (lbl, cmap, scale)) in enumerate(zip(img_data, img_cols)):
            ax = axes2[row, col]
            if scale == "shared_std":
                vmax = _std_vmax
            elif scale == "shared_res":
                vmax = _res_vmax
            else:
                vmax = max(float(img.max()), 1e-6)
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax,
                           interpolation="nearest", aspect="equal")
            # Hide ticks/spines but keep ylabel alive for row labels
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(name, fontsize=9, labelpad=5)

    # Shared colourbars for Std and Residual columns
    _std_col = next(i for i, (_, _, s) in enumerate(img_cols) if s == "shared_std")
    _res_col = next(i for i, (_, _, s) in enumerate(img_cols) if s == "shared_res")
    _sm_std = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, _std_vmax))
    _sm_res = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, _res_vmax))
    fig2.colorbar(_sm_std, ax=list(axes2[:, _std_col]),
                  location="right", shrink=0.85, pad=0.03, label="σ")
    fig2.colorbar(_sm_res, ax=list(axes2[:, _res_col]),
                  location="right", shrink=0.85, pad=0.03, label="|residual|")

    fig2.suptitle(f"{problem_name}  —  Reconstructions", fontsize=13, fontweight="bold")
    fig2.savefig(output_dir / f"{problem_name}_reconstructions.png",
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ---- Combined TARP overlay plot ----
    has_tarp = any("tarp_ecp" in r for r in results.values())
    if has_tarp:
        fig_tarp, ax = plt.subplots(figsize=(6, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for (name, r), color in zip(results.items(), colors):
            if "tarp_ecp" not in r:
                continue
            k_sigma = [1]
            ecp = np.array(r["tarp_ecp"])
            alpha = np.array(r["tarp_alpha"])
            ecp_bootstrap = np.array(r["tarp_ecp_bootstrap"])
            ax.plot(alpha, ecp_bootstrap.mean(axis=0), label=f'{name}')
            # ax.plot(alpha, ecp, "o-", color=color, ms=3, lw=1.5, label=name)
            for k in k_sigma:
                ax.fill_between(alpha, ecp_bootstrap.mean(axis=0) - k * ecp_bootstrap.std(axis=0), ecp_bootstrap.mean(axis=0) + k * ecp_bootstrap.std(axis=0), alpha = 0.2)
        ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
        ax.set_xlabel("Credibility Level")
        ax.set_ylabel("Expected Coverage")
        ax.set_title("TARP")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        fig_tarp.tight_layout()
        fig_tarp.savefig(output_dir / f"{problem_name}_tarp_all.png", dpi=150)
        plt.close(fig_tarp)

    # ---- Combined MIRA plot ----
    has_mira = any(r.get("mira_mean") is not None for r in results.values())
    if has_mira:
        mira_names = [n for n, r in results.items() if r.get("mira_mean") is not None]
        mira_means = np.array([results[n]["mira_mean"] for n in mira_names])
        mira_stds  = np.array([results[n]["mira_std"]  for n in mira_names])
        x = np.arange(len(mira_names))

        mira_var_band = np.sqrt((1.0 / 18.0) / n_tarp_sims)

        fig_mira, ax_m = plt.subplots(figsize=(max(5, len(mira_names) * 1.2 + 2), 5))
        ax_m.errorbar(x, mira_means, yerr=mira_stds, fmt="o", capsize=5,
                      color="steelblue", zorder=3)
        ax_m.axhline(2 / 3, color="gray", lw=1.5, ls="--", zorder=2)
        ax_m.axhspan(2 / 3 - mira_var_band, 2 / 3 + mira_var_band,
                     color="gray", alpha=0.15, zorder=1)
        ax_m.axhline(0.5, color="black", lw=1.5, ls=":", zorder=2)
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(mira_names, rotation=20, ha="right")
        ax_m.set_ylabel("MIRA score")
        ax_m.set_title("MIRA")
        fig_mira.tight_layout()
        fig_mira.savefig(output_dir / f"{problem_name}_mira_all.png", dpi=150)
        plt.close(fig_mira)

    # ---- JSON summary ----
    json_results = {
        "problem": type(problem).__name__,
        "solvers": {
            name: {
                "hpd_mean": r["hpd_mean"],
                "hpd_std": r["hpd_std"],
                "hpd_ks": r["hpd_ks"],
                "tarp_max_dev": r.get("tarp_max_dev"),
                "mira_mean": r.get("mira_mean"),
                "mira_std": r.get("mira_std"),
            }
            for name, r in results.items()
        },
    }
    with open(output_dir / f"{problem_name}_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
