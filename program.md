# Autoresearch: Calibrated Posterior Sampling for Latent Inverse Problems

You are an autonomous research agent. Read this file completely before every iteration. Your job is to make research progress on calibrated posterior sampling with latent diffusion models. You read papers, form hypotheses, run experiments, and record everything. You propose what to try next based on your accumulated results.

---

## 1. Project Structure

```
lip/                    — JAX library (pip install -e .)
  problems.py           — Gaussian1D, NonlinearDecoder2D, FoldedDecoder2D
  metrics.py            — calibration_test, latent_calibration_test, benchmark, latent_benchmark
  solvers/              — one file per solver
    latino.py, dps.py, mmps.py, latino_sde.py, lflow.py          (pixel-space)
    latent_latino.py, latent_dps.py, _latent_proximal.py          (latent-space)
scripts/
  run_gaussian.py       — 1D Gaussian benchmark (all pixel solvers)
  run_nonlinear.py      — 2D latent benchmarks (NonlinearDecoder2D + FoldedDecoder2D)
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
**What you CAN modify with caution:** , `lip/problems.py`, `lip/metrics.py`, `lip/solvers/__init__.py` (to register new solvers), `scripts/run_nonlinear.py` (to add new solvers to benchmarks).

---

## 2. The Existing Codebase

Before writing any code, understand what's already built:

### Problems (lip/problems.py — DO NOT MODIFY)

| Problem | Decoder | Posterior | Difficulty |
|---------|---------|-----------|------------|
| `Gaussian1D` | Identity (pixel-space) | Gaussian, analytic | Baseline |
| `NonlinearDecoder2D(alpha, beta)` | `[z1+αz2², z2+αsin(z1), βz1z2]` | Non-Gaussian, unimodal, grid-exact | Jacobian distortion |
| `FoldedDecoder2D(alpha)` | `[z1²-z2², 2z1z2, α(z1²+z2²)]` | Bimodal (D(z)=D(-z)), grid-exact | Representation error + multimodality |

All problems provide: `decoder`, `decoder_jacobian` (analytic), `encoder` (Gauss-Newton), `score` (exact for N(0,I) prior), `denoise` (Tweedie, deterministic or stochastic), `tweedie_cov`, `log_posterior`, `posterior_grid`, `posterior_mean_cov`.

### Existing Solvers (lip/solvers/)

**Pixel-space (1D Gaussian):**
| Solver | z-std | Status |
|--------|-------|--------|
| LATINO | 0.765 | Under-dispersed (proximal contraction) |
| DPS | 0.916 | Implicit MAP (Tweedie mean only) |
| **MMPS** | **1.002** | **Calibrated** (Tweedie mean + covariance) |
| LATINO+SDE | 0.979 | Nearly calibrated (stochastic denoiser) |
| LFlow | 0.986 | Nearly calibrated (ODE discretization limited) |

**Latent-space (2D problems):**
| Solver | HPD mean (target 0.5) | Key issue |
|--------|----------------------|-----------|
| Latent LATINO | ~0.5 (d²≈2.08) | Trapped in one mode on FoldedDecoder2D |
| Latent DPS | under-dispersed (d²≈1.25) | Finds both modes but each is too tight |

### Metrics (lip/metrics.py — DO NOT MODIFY)

**Pixel-space:** `calibration_test(problem, solver, key)` → `z_std` (target: 1.000)
**Latent-space:** `latent_calibration_test(problem, solver, key)` → `hpd_mean` (target: 0.500), `hpd_ks` (target: → 0)

Use `lip.benchmark(problem)` and `lip.latent_benchmark(problem)` to run all registered solvers.

### How to add a new solver

```python
# lip/solvers/my_method.py
import jax
import jax.numpy as jnp

def my_method(problem, y, key, *, N=200, **kwargs):
    """Signature must be (problem, y, key, **kwargs) -> samples."""
    # For pixel-space: return x with same shape as y
    # For latent-space: return z with shape (..., problem.d_latent)
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
from lip import NonlinearDecoder2D, FoldedDecoder2D
from lip.metrics import latent_calibration_test, latent_posterior_test
import jax

problem = NonlinearDecoder2D(alpha=0.5)
result = latent_calibration_test(problem, my_solver, jax.random.PRNGKey(0), n=200)
print(f"HPD mean: {result['hpd_mean']:.3f} (target: 0.500)")
```

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
python scripts/run_nonlinear.py
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

---

## 4. Metrics

**Latent-space primary metric: `hpd_mean`** — mean HPD credibility level of solver samples under the true grid posterior. Target: **0.500** (= Uniform(0,1)).
- hpd_mean < 0.5 → samples cluster near mode (under-dispersed)
- hpd_mean > 0.5 → samples in tails (over-dispersed)

**Secondary: `hpd_ks`** — KS statistic of HPD levels vs Uniform(0,1). Target: → 0.

**Pixel-space metric: `z_std`** — standard deviation of z-scores. Target: **1.000**.

The true posterior is computed by grid evaluation in `problem.posterior_grid()`. This is exact. If Oracle MCMC disagrees with the grid, debug the grid first.

---

## 5. Starting Point

The baselines are already implemented and benchmarked. You do NOT need to reimplement them. Your job starts at improving upon them.

Current state of affairs:
- **Pixel-space is solved**: MMPS achieves z_std=1.002 on Gaussian1D
- **Latent-space is not solved**: no method achieves hpd_mean≈0.500 on both NonlinearDecoder2D AND FoldedDecoder2D
- **The gap**: MMPS is exact for Gaussians but its Tweedie linearization breaks when the decoder is nonlinear. Latent LATINO avoids Tweedie but is structurally trapped in one mode.

---

## 6. Seed Hypotheses

These are starting directions ordered by expected information gain. You will generate better hypotheses as you learn.

**H2: Latent MMPS + second-order decoder correction.** The linearization D(z)≈D(ẑ₀)+J·(z-ẑ₀) drops the Hessian. Add bias:
```
E[D(z)|z_t] ≈ D(ẑ₀) + 0.5·Tr(H_D · V[z|z_t])
```
For our analytic decoder, H_D is known exactly. At what alpha does this matter?

**H3: Latent LATINO + SDE denoiser.** The pixel-space LATINO+SDE nearly calibrates (z_std=0.979). Does the same trick work in latent space? Just change `problem.denoise(z_noisy, sigma_k)` to `problem.denoise(z_noisy, sigma_k, key=subkey)` in latent_latino.py.

**H4: Split Gibbs in latent space.** Port PnP-DM to latent space: alternate (a) unconditional diffusion step on z (prior), (b) Langevin step on log p(y|D(z)) (likelihood). No Tweedie, no encoder. Should be asymptotically exact.

**H5: Few-particle SMC (LD-SMC style).** Run N=2,5,10 parallel reverse-SDE trajectories, weight by p(y|D(ẑ₀)), resample. Minimal change to Latent DPS — just add particles and weights.

**H6: Folding decoder stress test.** Run every new method on FoldedDecoder2D. Which ones find both modes? Get mode weights right? This is the hardest test.

**H7: Gauss-Newton proximal for LATINO.** Replace closed-form pixel-space proximal with iterative Gauss-Newton in latent space: linearize A·D(z) at current z, solve, iterate.

**H8: Phase diagrams.** For the best 2-3 methods, sweep alpha ∈ [0, 1.5] on NonlinearDecoder2D. Plot hpd_mean vs alpha → shows exactly where each method breaks.

**H9: Random MLP decoder.** Add a `RandomMLPDecoder2D` problem (frozen random MLP, R²→R¹⁶). Do conclusions generalize beyond the analytic decoder?

**H10: Annealed Langevin on exact log-posterior.** Not a diffusion method — just run annealed MALA/ULA on `problem.log_posterior(z, y)` with decreasing temperature. Use as a second oracle to validate grid posteriors and bound achievable calibration.

---

## 7. Adaptive Research Direction

As you accumulate results, you SHOULD propose new hypotheses. Write them in `results/insights.md`.

**Signs you should pivot:**
- Method works on NonlinearDecoder2D but breaks on FoldedDecoder2D → multimodality is the bottleneck
- Method works at alpha=0.3 but fails at alpha=0.7 → nonlinearity matters, investigate Jacobian spectrum
- Method is calibrated but 100× slower → look for cost reduction
- Unexpected method achieves hpd_mean≈0.5 → understand WHY, derive the math, this could be a paper

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

Output `<promise>BREAKTHROUGH</promise>` if you achieve perfect calibration with cost ≤ 10× single-trajectory Latent DPS

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