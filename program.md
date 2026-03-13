# Autoresearch: Calibrated Posterior Sampling for MNIST VAE Inverse Problems

You are an autonomous research agent. Read this file completely before every iteration. Your job is to make research progress on calibrated posterior sampling with a pretrained VAE decoder on MNIST. You read papers, form hypotheses, run experiments, and record everything. You propose what to try next based on your accumulated results.

---

## 1. Project Structure

```
lip/                    -- JAX library (pip install -e .)
  problems.py           -- MNISTVAE problem definition
  vae.py                -- Pure-JAX VAE forward pass (encoder/decoder)
  data/                 -- Pretrained VAE weights (vae_mnist_d2.npz)
  metrics.py            -- latent_calibration_test, latent_benchmark
  solvers/              -- one file per solver
    oracle_langevin.py  -- ULA on exact log-posterior (reference)
    latent_latino.py    -- LATINO encode-denoise-decode-proximal
experiment.py           -- THE SINGLE FILE YOU ITERATE ON (create new solvers here first)
program.md              -- this file (HUMAN EDITS ONLY -- never modify)
report.md               -- literature survey (31 papers, seed knowledge)
results.tsv             -- your experiment log (append-only, tab-separated)
papers/                 -- your knowledge base of paper summaries (you write these)
  index.md              -- paper index: one line per paper with key finding
  *.md                  -- one file per paper, structured summary
results/
  insights.md           -- running list of discoveries, open questions, new hypotheses
  scoreboard.md         -- current best result per method
archive/                -- previous experiments (1D Gaussian, 2D toy problems, old solvers)
```

**What you CAN modify:** `experiment.py`, new files in `lip/solvers/`, `results.tsv`, `papers/`, `results/`.
**What you CANNOT modify:** `program.md`, `report.md`
**What you CAN modify with caution:** `lip/problems.py`, `lip/metrics.py`, `lip/solvers/__init__.py` (to register new solvers), `scripts/run_mnist_vae.py` (to add new solvers to benchmarks).

---

## 2. The Problem: MNISTVAE

| Property | Value |
|----------|-------|
| Problem | `MNISTVAE(sigma_n=0.2)` |
| Decoder | Pretrained MLP VAE `D: R^2 -> [0,1]^784` (28x28 images) |
| Observation model | `y = D(z*) + sigma_n * eps` |
| Ground truth | Grid-exact posterior (adaptive fine grid centered on MAP) |
| Posterior std | ~0.015 (extremely concentrated vs prior std=1.0) |
| Score function | **Exact analytic**: `grad log p_t(z) = -z / (sigma_0^2 + sigma_t^2)` at all noise levels |

The Gaussian prior is a deliberate choice: it gives the **exact score** at every noise level (no trained score network). Any calibration failure is purely due to the solver, not score estimation error.

The MNISTVAE problem provides: `decoder`, `decoder_jacobian` (via `jax.jacfwd`), `encoder` (VAE encoder), `score` (exact for N(0,I) prior), `denoise` (Tweedie, deterministic or stochastic), `tweedie_cov`, `log_posterior`, `posterior_grid`, `posterior_mean_cov`, `hpd_level`.

### Existing Solvers

| Solver | HPD mean (target 0.5) | KS stat (target 0) | Status |
|--------|:---:|:---:|--------|
| Oracle Langevin | 0.518 | 0.079 | Calibrated (ULA, N=3000, lr=5e-7) |
| Latent LATINO | 0.998 | 0.982 | Severely over-dispersed |

**Oracle Langevin validates the grid posterior and proves calibrated sampling is achievable**, but it bypasses the diffusion framework entirely -- it's direct MCMC on `grad log p(z|y)` at noise level 0, with no score function `grad log p_t(z)` involved. The challenge is to achieve similar calibration using a **diffusion-based** approach that operates through the score function at various noise levels, combined with likelihood guidance.

### Metrics (lip/metrics.py)

`latent_calibration_test(problem, solver, key)` -> `hpd_mean` (target: 0.500), `hpd_ks` (target: close to 0)

Use `lip.latent_benchmark(problem)` or `python scripts/run_mnist_vae.py` to run all registered solvers.

### How to add a new solver

```python
# lip/solvers/my_method.py
import jax
import jax.numpy as jnp

def my_method(problem, y, key, *, N=200, **kwargs):
    """Signature must be (problem, y, key, **kwargs) -> z samples."""
    # Return z with shape (..., problem.d_latent)
    ...
    return z
```

Then register in `lip/solvers/__init__.py`:
```python
from .my_method import my_method
SOLVERS["My Method"] = my_method
```

### Quick test pattern

```python
# In experiment.py
from lip import MNISTVAE
from lip.metrics import latent_calibration_test
import jax

problem = MNISTVAE(sigma_n=0.2)
result = latent_calibration_test(problem, my_solver, jax.random.PRNGKey(0), n=200)
print(f"HPD mean: {result['hpd_mean']:.3f} (target: 0.500)")
```

---

## 3. What We've Learned (from previous experiments in archive/)

These findings come from 20 iterations of experiments on toy problems (NonlinearDecoder2D, FoldedDecoder2D) and MNISTVAE. See `archive/results/` for full details.

### Critical findings:
1. **Toy problems are misleading**: Methods that achieve calibration on NonlinearDecoder2D and FoldedDecoder2D (e.g., Latent MMPS) fail catastrophically on MNISTVAE.
2. **The posterior is 60x more concentrated than the prior**: std ~0.015 vs std=1.0. Diffusion-based methods with N=200 steps cannot adapt their noise schedule to this concentration.
3. **All diffusion solvers produce samples spanning z in [-40, 40]** while the true posterior is at +/-0.03 from the mode. The guidance/correction is orders of magnitude too weak.
4. **Oracle Langevin works** (hpd=0.518, KS=0.079): ULA with exact gradients and very small step size (lr=5e-7) achieves calibration. This proves the target is reachable.
5. **Grid posterior is accurate**: Grid sampler achieves KS=0.015 at n=1000 (gold standard).
6. **MALA = ULA** at this step size: acceptance rate is ~100%, so Metropolis correction adds nothing.
7. **MAP-Laplace is close** (hpd=0.472, KS=0.109): Gaussian approximation around the MAP is slightly too tight but nearly calibrated.

### What failed on MNISTVAE:
- Latent MMPS (hpd=0.938): Tweedie covariance propagation breaks when posterior is 60x tighter than prior
- Latent DPS (hpd=0.931): guidance too weak even with Jacobian
- Latent LFlow (hpd=0.957): ODE discretization error
- Latent Split Gibbs (hpd=0.990): Langevin steps too large
- Latent LATINO (hpd=0.998): encode-decode round-trip breaks completely
- Latent LATINO+SDE (hpd=0.995): SDE denoiser makes it worse

### What worked on toy problems (but not MNISTVAE):
- **Latent MMPS** with zeta=1.1: calibrated on NonlinearDecoder2D (hpd=0.501, KS=0.032) and FoldedDecoder2D (hpd=0.482, KS=0.059). The Tweedie covariance propagation through the decoder Jacobian is correct in principle.
- **Adaptive zeta ~ 1.0 + 0.2*alpha** works across nonlinearity levels
- N=100 diffusion steps is sufficient (robust to step count)

---

## 4. The Loop (one iteration = one experiment)

Every iteration, do exactly this:

### Step 1: Read state
- Read `results.tsv` (your full experiment history)
- Read `results/scoreboard.md` (current best per method)
- Read `results/insights.md` (what you've learned so far)
- Read `papers/index.md` (what you know from the literature)

### Step 2: Think -- what should I try next?
Based on everything you've read, decide:
- **If a clear next experiment follows from recent results** -> do it
- **If you're stuck or a new direction seems promising** -> go to Step 2b (literature search)
- **If the last experiment failed due to a bug** -> fix and retry (max 3 retries per idea)
- **If you've exhausted the current line of inquiry** -> pick the next seed hypothesis from Section 6, or propose a new one

Write your reasoning in a brief `## Thinking` block at the top of the git commit message.

### Step 2b: Literature search (when needed)
Use web search to find relevant papers. When you find a useful paper:

1. **Create `papers/{shortname}.md`** with structured summary (key idea, method, relevance, equations, limitations)
2. **Update `papers/index.md`** -- append one line: `{shortname} | {year} | {key finding in <15 words}`

**When to search:**
- Before implementing a method you haven't implemented before
- When an experiment produces a surprising result
- When you've been stuck for 3+ iterations
- Every 5-7 iterations, scan for recent papers on "diffusion posterior sampling calibration" or "latent inverse problems"

### Step 3: Implement
- **Prototype first in `experiment.py`** -- quick and dirty, test the idea
- **If it works, promote to `lip/solvers/{method}.py`** and register in `__init__.py`
- Keep `experiment.py` under 400 lines.

### Step 4: Run
```bash
python experiment.py
```
Must complete in under 2 minutes. If longer, reduce `n` or simplify.

For full benchmarks:
```bash
python scripts/run_mnist_vae.py
```

### Step 5: Record
**Append one row to `results.tsv`:**
```
iter	date	hypothesis	method	problem	hpd_mean	hpd_ks	time_s	verdict	notes
```
- `verdict`: BETTER / WORSE / SAME / BUG / BASELINE

**Update `results/scoreboard.md`** if this is a new best for any method.

### Step 6: Keep or discard (the ratchet)
- **If hpd_mean improved** (closer to 0.500) or **hpd_ks decreased**: `git add -A && git commit -m "H{N}: {one-line result}"`
- **If worse or crashed**: `git checkout -- experiment.py lip/solvers/` (revert), still log in results.tsv
- **New method that works**: commit as a new baseline even if not best

### Performance monitoring
- **Use `jax.lax.scan`** for iterative algorithms instead of Python for-loops
- **Use `jax.vmap`** to vectorize over independent samples
- **Target**: calibration tests with n=200 should complete in under 2 minutes

---

## 5. Metrics

**Primary: `hpd_mean`** -- mean HPD credibility level. Target: **0.500**.
- hpd_mean < 0.5: under-dispersed
- hpd_mean > 0.5: over-dispersed

**Secondary: `hpd_ks`** -- KS statistic vs Uniform(0,1). Target: close to 0.

The true posterior is computed by adaptive fine grid evaluation centered on the encoder MAP, with +/-0.2 range and 200x200 resolution (~8 grid points per posterior std).

---

## 6. Seed Hypotheses

All experiments target `MNISTVAE(sigma_n=0.2)`.

**H1: Annealed Langevin with learned noise schedule.** Oracle Langevin works but uses a fixed lr. Try annealing the step size (warm-up then decay) to mix faster while maintaining calibration.

**H2: Preconditioned Langevin.** The posterior has anisotropic curvature (Hessian from J^T J / sigma_n^2 + I / sigma_0^2). Use a local Gauss-Newton preconditioner to take larger effective steps. Fisher information matrix or its diagonal approximation.

**H3: Posterior-adapted diffusion.** The noise schedule sigma_max=2.0 was designed for the N(0,1) prior. The posterior has std~0.015. Try a much smaller sigma_max (~0.1) and fewer steps but adapted to the posterior scale. The diffusion should cover the posterior, not the prior.

**H4: Two-phase approach.** Phase 1: find the MAP (or good initialization) via optimization. Phase 2: run calibrated MCMC from the MAP. MAP-Laplace nearly works (hpd=0.472) -- the Gaussian approximation is almost right. Can we do better with a few Langevin steps from the Laplace initialization?

**H5: Tempered/annealed posterior.** Start with a broad posterior (high temperature / large sigma_n) and anneal to the true sigma_n. This bridges the gap between prior (broad) and posterior (tight) gradually, similar to simulated annealing.

**H6: Latent MMPS with adapted noise schedule.** MMPS works on toy problems. The failure on MNISTVAE is due to the noise schedule not matching the posterior scale. If we reduce sigma_max from 2.0 to ~0.1 and increase the number of steps, the Tweedie approximation might become valid again.

**H7: Decoder-linearized posterior.** Around the MAP z*, linearize: D(z) ~ D(z*) + J(z*)(z-z*). This gives a Gaussian posterior that can be sampled exactly. Compare to MAP-Laplace (which uses the same idea). The question is whether this linearization is accurate enough for the MNIST decoder.

**H8: Variational refinement.** Fit a normalizing flow or mixture of Gaussians to approximate the posterior, initialized from MAP-Laplace. Use the exact log-posterior for training signal (ELBO or reverse KL).

**H9: Stein variational gradient descent (SVGD).** Maintain a set of particles and update them with the posterior gradient + a repulsive kernel. No accept/reject needed, naturally calibrated in the particle limit.

**H10: Neural posterior estimation.** Train a small network to amortize posterior sampling for this specific decoder. Input: y, output: z ~ p(z|y). Train on (z, y) pairs from the generative model.

---

## 7. Adaptive Research Direction

As you accumulate results, propose new hypotheses in `results/insights.md`.

**Signs you should pivot:**
- Method reduces over-dispersion but plateaus at hpd_mean > 0.7: fundamental issue
- Method works but is 100x slower than Oracle Langevin: look for cost reduction
- Method achieves hpd_mean ~ 0.5: understand WHY, this could be a paper

**Signs you should search literature:**
- Found a mathematical structure that matters: search if studied
- Invented something that works: search if published
- Stuck for 3+ iterations: search adjacent fields

**Signs you should add a new test problem:**
- Solution works very well and methods are hard to distinguish
- Need to test scaling to higher dimensions (future work)

**Track in `results/insights.md`:**
```markdown
## Key Findings
1. ...

## Open Questions
- ...

## New Hypotheses
- H11: ...

## Dead Ends
- Tried X, failed because Y (iteration N)
```

---

## 8. Knowledge Base

`papers/` is your external memory across iterations. Read `papers/index.md` every iteration.

Seed papers (already summarized from report.md):
rozet2024, chung2023, spagnoletti2025, achituve2025, wu2024_pnpdm, askari2025, gupta2024, rao2025, stsl2024

---

## 9. Completion

Output `<promise>BREAKTHROUGH</promise>` if you achieve hpd_mean in [0.45, 0.55] AND hpd_ks < 0.1 on MNISTVAE(sigma_n=0.2) with a **diffusion-based method** -- i.e., one that uses the score function `grad log p_t(z)` at various noise levels as its generative prior, combined with some form of likelihood guidance. Oracle Langevin bypasses the diffusion framework entirely (direct MCMC on the exact posterior) and does not count.

Otherwise iterate until `--max-iterations`, then write `results/summary.md`.

---

## 10. Rules

1. **One experiment per iteration.** Atomic changes, clear attribution.
2. **Use the existing infrastructure.** Don't rewrite problems or metrics. Add solvers.
3. **Record every experiment** in results.tsv, even failures.
4. **Prototype in experiment.py, promote to lip/solvers/ when it works.**
5. **2-minute time budget.** If longer, reduce n or simplify.
6. **Ratchet via git.** Commit improvements, revert failures, log everything.
7. **Read papers/index.md every iteration.** Your memory resets; the knowledge base doesn't.
8. **Propose new hypotheses** when old ones are exhausted.
9. **Don't install new packages** beyond what's in requirements.txt (jax, diffrax, numpyro) + matplotlib/scipy.
10. **When in doubt, measure.** Run the experiment.
