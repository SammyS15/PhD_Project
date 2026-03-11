# Autoresearch: Calibrated Posterior Sampling for MNIST VAE Inverse Problems

You are an autonomous research agent. Read this file completely before every iteration. Your job is to make research progress on calibrated posterior sampling with a pretrained VAE decoder on MNIST. You read papers, form hypotheses, run experiments, and record everything. You propose what to try next based on your accumulated results.

---

## 1. Project Structure

```
lip/                    — JAX library (pip install -e .)
  problems.py           — Gaussian1D, NonlinearDecoder2D, FoldedDecoder2D, MNISTVAE
  vae.py                — Pure-JAX VAE forward pass (encoder/decoder) with weight loading
  data/                 — Pretrained VAE weights (vae_mnist_d2.npz, vae_mnist_d20.npz)
  metrics.py            — calibration_test, latent_calibration_test, benchmark, latent_benchmark
  solvers/              — one file per solver
    latino.py, dps.py, mmps.py, latino_sde.py, lflow.py          (pixel-space)
    latent_latino.py, latent_dps.py, _latent_proximal.py          (latent-space)
scripts/
  run_mnist_vae.py      — MNISTVAE benchmark (all latent solvers)
  run_gaussian.py       — 1D Gaussian benchmark (reference only)
  run_nonlinear.py      — 2D toy latent benchmarks (reference only)
experiment.py           — THE SINGLE FILE YOU ITERATE ON (create new solvers here first)
program.md              — this file (HUMAN EDITS ONLY — never modify)
report.md               — literature survey (31 papers, seed knowledge)
results.tsv             — your experiment log (append-only, tab-separated)
papers/                 — your knowledge base of paper summaries (you write these)
  index.md              — paper index: one line per paper with key finding
  *.md                  — one file per paper, structured summary
results/
  insights.md           — running list of discoveries, open questions, new hypotheses
  scoreboard.md         — current best result per method
figures/                — saved plots
```

**What you CAN modify:** `experiment.py`, new files in `lip/solvers/`, `results.tsv`, `papers/`, `results/`, `figures/`.
**What you CANNOT modify:** `program.md`, `report.md`
**What you CAN modify with caution:** `lip/problems.py`, `lip/metrics.py`, `lip/solvers/__init__.py` (to register new solvers), `scripts/run_mnist_vae.py` (to add new solvers to benchmarks).

---

## 2. The Existing Codebase

Before writing any code, understand what's already built:

### Target Problem: MNISTVAE

| Problem | Decoder | Posterior | Difficulty |
|---------|---------|-----------|------------|
| `MNISTVAE(latent_dim=2, sigma_n=0.2)` | Pretrained MLP VAE `D: ℝ²→[0,1]^784` (28×28 images) | Grid-exact (d_latent=2) | Realistic neural network decoder, large Jacobian norms |

**Observation model:** `y = D(z*) + σ_n · ε`, with `σ_n=0.2` (per-pixel SNR ≈ 1, digits visible but noisy).

**Ground truth latent:** `z* = [0.8, -0.5]` (produces a digit "8").

The MNISTVAE problem provides: `decoder`, `decoder_jacobian` (via `jax.jacfwd`), `encoder` (Gauss-Newton), `score` (exact for N(0,I) prior), `denoise` (Tweedie, deterministic or stochastic), `tweedie_cov`, `log_posterior`, `posterior_grid`, `posterior_mean_cov`, `hpd_level`.

**Reference problems (already solved, for context only):**
- `Gaussian1D` — Pixel-space baseline, analytic posterior. MMPS achieves z_std=1.002 (calibrated).
- `NonlinearDecoder2D` / `FoldedDecoder2D` — Toy 2D latent problems with analytic decoders.

### Existing Solvers on MNISTVAE (lip/solvers/)

All latent solvers are benchmarked on `MNISTVAE(latent_dim=2, sigma_n=0.2)`:

| Solver | HPD mean (target 0.5) | KS stat (target → 0) | Key issue |
|--------|----------------------|----------------------|-----------|
| Latent LATINO | 1.000 | 0.997 | Severely over-dispersed, proximal + encoder round-trip breaks |
| Latent DPS | 0.966 | 0.937 | Over-dispersed, guidance scaling insufficient |
| Latent MMPS | 0.926 | 0.843 | Over-dispersed, Tweedie linearization breaks on neural decoder |
| Latent LFlow | 0.977 | 0.920 | Over-dispersed, ODE discretization error |
| Latent LATINO+SDE | 0.998 | 0.987 | Severely over-dispersed |
| Latent Split Gibbs | 1.000 | 0.997 | Severely over-dispersed |

**No method achieves calibrated posteriors.** All are heavily over-dispersed (HPD mean >> 0.5), meaning solver samples land far from the true posterior mass. The diagnostic plots show samples scattered across latent space (z range ±40) while the true posterior is tightly concentrated.

### Metrics (lip/metrics.py)

**Latent-space:** `latent_calibration_test(problem, solver, key)` → `hpd_mean` (target: 0.500), `hpd_ks` (target: → 0)

Use `lip.latent_benchmark(problem)` or `python scripts/run_mnist_vae.py` to run all registered solvers. Output includes per-solver diagnostic plots with decoded image panels (ground truth, observation, posterior samples).

### How to add a new solver

```python
# lip/solvers/my_method.py
import jax
import jax.numpy as jnp

def my_method(problem, y, key, *, N=200, **kwargs):
    """Signature must be (problem, y, key, **kwargs) -> samples."""
    # Return z with shape (..., problem.d_latent)
    ...
    return z
```

Then register in `lip/solvers/__init__.py`:
```python
from .my_method import my_method
LATENT_ALL["My Method"] = my_method
```

### Quick test pattern

```python
# In experiment.py
from lip import MNISTVAE
from lip.metrics import latent_calibration_test, latent_posterior_test
import jax

problem = MNISTVAE(latent_dim=2, sigma_n=0.2)
result = latent_calibration_test(problem, my_solver, jax.random.PRNGKey(0), n=200)
print(f"HPD mean: {result['hpd_mean']:.3f} (target: 0.500)")
```

**Note on MNISTVAE:** The decoder Jacobian has large norms (~10-50). DPS-style guidance scaling (zeta) typically needs significant reduction (~0.01-0.5 vs 1.0 on toy problems). The decoder output is in [0,1]^784 (sigmoid activation).

---

## 3. The Loop (one iteration = one experiment)

Every iteration, do exactly this:

### Step 1: Read state
- Read `results.tsv` (your full experiment history)
- Read `results/scoreboard.md` (current best per method)
- Read `results/insights.md` (what you've learned so far)
- Read `papers/index.md` (what you know from the literature)

### Step 2: Think — what should I try next?
Based on everything you've read, decide:
- **If a clear next experiment follows from recent results** → do it
- **If you're stuck or a new direction seems promising** → go to Step 2b (literature search)
- **If the last experiment failed due to a bug** → fix and retry (max 3 retries per idea)
- **If you've exhausted the current line of inquiry** → pick the next seed hypothesis from Section 6, or propose a new one based on what you've learned

Write your reasoning in a brief `## Thinking` block at the top of the git commit message.

### Step 2b: Literature search (when needed)
Use web search or the ADS API to find relevant papers. When you find a useful paper:

1. **Create `papers/{shortname}.md`** with this exact structure:
```markdown
# {Title}
**Authors:** {authors}
**Year:** {year} | **Venue:** {venue}
**Link:** {arxiv url}

## Key idea (1-2 sentences)

## Method summary (≤10 lines)
{Focus on the math/algorithm, not the narrative.}

## Relevance to our problem
{Why does this matter for calibrated latent posterior sampling?}

## Key equations
{The 2-3 most important equations, in LaTeX}

## Limitations noted by authors

## Experimental takeaway
{Key numbers if available.}
```

2. **Update `papers/index.md`** — append one line:
```
{shortname} | {year} | {key finding in <15 words}
```

**When to search:**
- Before implementing a method you haven't implemented before
- When an experiment produces a surprising result (good or bad)
- When you've been stuck for 3+ iterations on the same idea
- When you want to check if someone has already tried your idea
- Every 5-7 iterations, do a broad scan for recent papers (2025-2026) on "diffusion posterior sampling calibration" or "latent inverse problems"

**When NOT to search:**
- When you have a clear next experiment from the last result
- When you're debugging a crash
- Don't read more than 2 papers per iteration — this is a research loop, not a literature review

### Step 3: Implement
- **Prototype first in `experiment.py`** — quick and dirty, test the idea
- **If it works, promote to `lip/solvers/{method}.py`** and register in `__init__.py`
- Keep `experiment.py` under 400 lines. It's a scratch pad, not an archive.

### Step 4: Run
```bash
python experiment.py
```
Must complete in under 2 minutes. If longer, reduce `n` or simplify.

For full benchmarks against all existing solvers:
```bash
python scripts/run_mnist_vae.py
```

### Step 5: Record
**Append one row to `results.tsv`:**
```
iter	date	hypothesis	method	problem	alpha	hpd_mean	hpd_ks	time_s	verdict	notes
```
- `verdict`: BETTER / WORSE / SAME / BUG / BASELINE
- For pixel-space experiments, use `z_std` instead of `hpd_mean`

**Update `results/scoreboard.md`** if this is a new best for any (method, problem) pair.

### Step 6: Keep or discard (the ratchet)
- **If hpd_mean improved** (closer to 0.500) or **hpd_ks decreased**: `git add -A && git commit -m "H{N}: {one-line result}"`
- **If worse or crashed**: `git checkout -- experiment.py lip/solvers/` (revert), still log in results.tsv
- **New method that works at all**: commit as a new baseline even if not best
- **After committing a new solver or improvement**: run `python scripts/run_mnist_vae.py` to benchmark against all existing solvers and update the scoreboard

### Performance monitoring
- **Check GPU utilization** with `nvidia-smi` when running experiments. If GPU usage is low (<30%), something is wrong:
  - Python for-loops over samples instead of vectorized JAX operations (use `jax.vmap` or `jax.lax.scan`)
  - Frequent Python-level iteration instead of compiled XLA kernels
  - Small batch sizes that don't saturate the GPU
- **Use `jax.lax.scan`** for iterative algorithms (Langevin, diffusion steps) instead of Python for-loops
- **Use `jax.vmap`** to vectorize over independent samples when possible
- **Target**: calibration tests with n=200 should complete in under 2 minutes

---

## 4. Metrics

**Latent-space primary metric: `hpd_mean`** — mean HPD credibility level of solver samples under the true grid posterior. Target: **0.500** (= Uniform(0,1)).
- hpd_mean < 0.5 → samples cluster near mode (under-dispersed)
- hpd_mean > 0.5 → samples in tails (over-dispersed)

**Secondary: `hpd_ks`** — KS statistic of HPD levels vs Uniform(0,1). Target: → 0.

**Reference (pixel-space) metric: `z_std`** — standard deviation of z-scores. Target: **1.000**. (Only relevant for Gaussian1D reference problem.)

The true posterior is computed by grid evaluation in `problem.posterior_grid()` over a 200×200 grid on [-4, 4]². This is exact for d_latent=2. If Oracle MCMC disagrees with the grid, debug the grid first.

---

## 5. Starting Point

The baselines are already implemented and benchmarked on MNISTVAE. You do NOT need to reimplement them. Your job starts at improving upon them.

Current state of affairs:
- **All existing methods fail on MNISTVAE**: no method achieves hpd_mean ≈ 0.500 (all are > 0.9, severely over-dispersed)
- **Best so far**: Latent MMPS (hpd_mean=0.926, KS=0.843) — still far from calibrated
- **The gap**: The neural network decoder's large Jacobian norms and nonlinearity break all existing approximations. Tweedie linearization (MMPS/DPS), encoder round-trips (LATINO), and Langevin steps (Split Gibbs) all produce samples that scatter far from the true posterior.
- **Key observation**: Solver samples span z ∈ [-40, 40] while the true posterior is concentrated near the origin. The methods need much tighter control of step sizes / guidance strength for neural decoders.

---

## 6. Seed Hypotheses

These are starting directions ordered by expected information gain. You will generate better hypotheses as you learn. All experiments target `MNISTVAE(latent_dim=2, sigma_n=0.2)`.

**H1: Oracle Langevin on exact log-posterior.** Not a diffusion method — just run annealed MALA/ULA on `problem.log_posterior(z, y)` with decreasing temperature. This establishes whether calibrated sampling is achievable at all on MNISTVAE, and validates the grid posterior. Start here.

**H2: Aggressive guidance scaling for DPS/MMPS.** Current solvers are over-dispersed with huge latent excursions (z ∈ [-40, 40]). Try much smaller guidance scale (zeta ≪ 1) and/or gradient clipping. The Jacobian norms of the neural decoder are ~10-50×, so guidance needs proportional dampening.

**H3: Latent MMPS + Jacobian-aware covariance scaling.** MMPS uses the Tweedie covariance which doesn't account for the decoder Jacobian conditioning. Scale the covariance correction by `(J^T J)^{-1}` or use a Gauss-Newton Hessian approximation for the likelihood term.

**H4: Few-particle SMC (LD-SMC style).** Run N=2,5,10 parallel reverse-SDE trajectories, weight by `p(y|D(ẑ₀))`, resample. Minimal change to Latent DPS — just add particles and weights. The resampling should concentrate particles near the true posterior.

**H5: Split Gibbs with tuned Langevin step size.** The current Split Gibbs is severely over-dispersed. The Langevin likelihood step may need much smaller step size for the neural decoder. Sweep step size and number of inner Langevin steps.

**H6: Latent LATINO with adaptive proximal strength.** LATINO's proximal step uses a fixed schedule. With a neural decoder, the proximal operator may need tighter coupling (larger δ_k) to prevent drift in pixel space.

**H7: Direct MCMC baseline (HMC/NUTS).** Use NumPyro to run HMC/NUTS on the exact `log_posterior`. This gives a gold-standard posterior to compare against and reveals whether the grid posterior is correct.

**H8: Noise schedule tuning.** The diffusion noise schedule (number of steps, min/max sigma) was tuned for toy problems. The neural decoder may need a different schedule — more steps, smaller sigma_max, or different spacing.

**H9: Encoder-free methods.** LATINO relies on a Gauss-Newton encoder which may be inaccurate for the neural decoder. Methods that avoid the encoder entirely (DPS, Split Gibbs, SMC) may have an advantage if properly tuned.

**H10: Sigma_n sensitivity sweep.** Run the best 2-3 methods across sigma_n ∈ [0.1, 0.2, 0.5, 1.0, 2.0] to map where calibration breaks down and understand the dependence on SNR.

---

## 7. Adaptive Research Direction

As you accumulate results, you SHOULD propose new hypotheses. Write them in `results/insights.md`.

**Signs you should pivot:**
- Method reduces over-dispersion but plateaus at hpd_mean > 0.7 → fundamental approximation issue, try a different family of methods
- Method works on toy NonlinearDecoder2D but fails on MNISTVAE → neural decoder Jacobian structure is the bottleneck
- Method is calibrated but 100× slower → look for cost reduction
- Unexpected method achieves hpd_mean≈0.5 → understand WHY, derive the math, this could be a paper
- Oracle MCMC (H1/H7) fails to calibrate → grid posterior may be wrong, debug metrics first

**Signs you should search literature:**
- You've found a specific mathematical structure matters → search if studied
- You've invented something that works → search if published
- Stuck for 3+ iterations → search adjacent fields

**Signs you should implement more methods from the litterature:**
- If your solution becomes very good, make sure to implement SOTA algorithms from the litterature to compare to

**Signs you should implement a new test**
- If your solution becomes very good and that it's difficult to see any difference between methods, you may consider extending the 
set of test problems. But keep in mind that test problems should be fast, and able to provide clear insights. Also compare SOTA on new problems.

**Track the evolving program in `results/insights.md`:**
```markdown
## Key Findings (numbered, one line each)
1. ...

## Open Questions
- ...

## New Hypotheses (agent-generated)
- H11: ...

## Dead Ends (don't retry)
- Tried X, failed because Y (iteration N)
```

---

## 8. Knowledge Base Management

`papers/` is your external memory across iterations.

**papers/index.md** — quick-reference table you read every iteration:
```markdown
| shortname | year | key finding |
|-----------|------|-------------|
| rozet2024 | 2024 | MMPS exact for Gaussian; Tweedie covariance in likelihood |
| chung2023 | 2023 | DPS is implicit MAP, not posterior sampling |
| ...       |      |             |
```

**Seed from report.md in iteration 0** (don't fetch — info is already there):
rozet2024, chung2023, spagnoletti2025, achituve2025, wu2024_pnpdm, askari2025, gupta2024, rao2025

---

## 9. Completion

Output `<promise>BREAKTHROUGH</promise>` if you achieve hpd_mean ∈ [0.45, 0.55] and hpd_ks < 0.1 on MNISTVAE(latent_dim=2, sigma_n=0.2) with cost ≤ 10× single-trajectory Latent DPS

Otherwise iterate until `--max-iterations`, then write `results/summary.md`:
```markdown
## Summary after N iterations
### What worked
### What didn't
### Best method found (with numbers)
### Most promising unexplored direction
### Recommended next steps for the human
```

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