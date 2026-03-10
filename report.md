# Posterior Sampling for Inverse Problems with Diffusion/Flow Priors

This report surveys methods for solving inverse problems using diffusion and flow matching priors, with emphasis on posterior calibration and latent-space models.

**Test problem (this repo):** Prior $x \sim \mathcal{N}(\mu_0, \sigma_0^2)$, forward model $y = x + n$, $n \sim \mathcal{N}(0, \sigma_n^2)$.

---

<<<<<<< HEAD
## Part I — Implemented Methods

### 1. LATINO — LAtent consisTency INverse sOlver
=======
## Part I: Methods Implemented in This Repository

### 1. LATINO — LAtent consisTency INverse sOlver

**Paper:** Spagnoletti, Prost, Almansa, Papadakis, Pereyra. *"LATINO-PRO"* ([arXiv:2503.12615](https://arxiv.org/abs/2503.12615), ICCV 2025).
>>>>>>> c4a9935 (Expand report with SOTA calibration methods and failure mode analysis)

Iterative noise–denoise–proximal loop. At each step $k$:

1. **Noise:** $x_{\text{noisy}} = x + \sigma_k \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$
2. **Denoise:** $u = \text{PF-ODE}(x_{\text{noisy}}, \sigma_k \to 0)$ using the prior score
3. **Proximal step:** $x = \frac{\delta_k y + \sigma_n^2 u}{\delta_k + \sigma_n^2}$ for $A = I$

**Gaussian calibration:** Under-dispersed. The proximal step is a contraction; deterministic PF-ODE provides no mechanism to restore lost variance.

> Spagnoletti, Prost, Almansa, Papadakis, Pereyra. "LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization." ICCV 2025. [arXiv:2503.12615](https://arxiv.org/abs/2503.12615)

### 2. DPS — Diffusion Posterior Sampling

<<<<<<< HEAD
Reverse-time SDE with likelihood gradient guidance:
=======
---

### 2. DPS — Diffusion Posterior Sampling

**Paper:** Chung, Kim, Mccann, Klasky, Ye. *"Diffusion Posterior Sampling for General Noisy Inverse Problems"* ([arXiv:2209.14687](https://arxiv.org/abs/2209.14687), ICLR 2023).

**Core idea:** Run the reverse-time SDE from noise to data, adding a likelihood gradient at each step. Decomposes the posterior score via Bayes' rule:
>>>>>>> c4a9935 (Expand report with SOTA calibration methods and failure mode analysis)

$$\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y | x_t)$$

Likelihood approximated using only the Tweedie posterior **mean**:

$$p(y | x_t) \approx \mathcal{N}(y \mid \hat{x}_0(x_t),\; \sigma_n^2)$$

**Gaussian calibration:** Over-confident. Ignoring $V[x|x_t]$ makes guidance too strong at large $\sigma_t$, producing biased and under-dispersed posteriors.

> Chung, Kim, McCann, Klasky, Ye. "Diffusion Posterior Sampling for General Noisy Inverse Problems." ICLR 2023. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

<<<<<<< HEAD
### 3. MMPS — Moment-Matching Posterior Sampling

Improves DPS by incorporating the Tweedie posterior covariance:

$$p(y | x_t) \approx \mathcal{N}\big(y \mid \hat{x}_0,\; \sigma_n^2 + V[x | x_t]\big)$$

**Gaussian calibration:** **Exact.** For Gaussian priors, the moment-matched likelihood is the true marginal likelihood, so MMPS with $\zeta=1$ recovers the exact posterior.

> Rozet, Andry, Lanusse, Louppe. "Learning Diffusion Priors from Observations by Expectation Maximization." 2024. [arXiv:2405.13712](https://arxiv.org/abs/2405.13712)

### 4. LATINO + SDE

Variant replacing the deterministic PF-ODE denoiser with the stochastic reverse SDE, restoring variance lost by the proximal step.

**Gaussian calibration:** Nearly calibrated (z-std ≈ 0.98).

### 5. LFlow — Latent Refinement via Flow Matching

Uses flow matching with OT interpolant $x_t = (1-t)x_0 + tz_1$ instead of diffusion SDE. Posterior velocity derived via the continuity equation:

$$v_t^y(x) = v_t(x) - \frac{t}{1-t}\,\nabla_{x_t} \log p(y|x_t)$$

with MMPS-style covariance in the likelihood. Pure ODE (deterministic given initial noise).

**Gaussian calibration:** Theoretically exact (verified by solving the linear ODE analytically with scipy DOP853: output mean and variance match the posterior to machine precision). Euler discretization with N=200 gives z-std ≈ 0.986; converges slowly because the $t/(1-t)$ factor makes the ODE stiff.

> Askari, Luo, Sun, Roosta. "Latent Refinement via Flow Matching for Training-free Linear Inverse Problem Solving." NeurIPS 2025. [arXiv:2511.06138](https://arxiv.org/abs/2511.06138)

### Calibration Summary (1D Gaussian)

| Method | μ (target: 1.200) | σ (target: 0.447) | z-std (target: 1.000) |
|---|---|---|---|
| Vanilla LATINO | 1.337 | 0.327 | 0.765 |
| DPS | 1.452 | 0.356 | 0.916 |
| **MMPS** | **1.179** | **0.446** | **1.002** |
| LATINO + SDE | 1.209 | 0.436 | 0.979 |
| **LFlow** | **1.199** | **0.442** | **0.986** |

---

## Part II — Other Notable Methods (Not Yet Implemented)

### PSLD — Posterior Sampling with Latent Diffusion
=======
**Critical limitation:** [Kwon et al. (2025)](https://arxiv.org/abs/2501.18913) showed that DPS actually behaves as **implicit MAP estimation**, not posterior sampling — it produces high-quality but low-diversity outputs.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

### 3. MMPS — Moment-Matching Posterior Sampling

**Paper:** Rozet, Andry, Lanusse, Louppe. *"Learning Diffusion Priors from Observations by Expectation Maximization"* ([arXiv:2405.13712](https://arxiv.org/abs/2405.13712), 2024).
>>>>>>> c4a9935 (Expand report with SOTA calibration methods and failure mode analysis)

First framework extending DPS to latent diffusion models. Adds a "gluing objective" to keep latents in the encoder's range space, preventing decode-encode round-trip artifacts.

> Rout, Raoof, Daras, Caramanis, Dimakis, Shakkottai. "Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models." NeurIPS 2023. [arXiv:2307.00619](https://arxiv.org/abs/2307.00619)

### STSL — Beyond First-Order Tweedie

Derives a tractable second-order Tweedie approximation via a surrogate loss, reducing the quality-limiting bias of first-order DPS/PSLD in latent space. 4–8× fewer NFEs than PSLD.

> Rout, Chen, Kumar, Caramanis, Shakkottai, Chu. "Beyond First-Order Tweedie: Solving Inverse Problems using Latent Diffusion." CVPR 2024. [arXiv:2312.00852](https://arxiv.org/abs/2312.00852)

### ReSample — Hard Data Consistency

Replaces soft DPS gradient guidance with hard-constrained optimization at each step, then resamples to stay on the noisy data manifold. Avoids backpropagating through the decoder.

> Song, Kwon, Zhang, Hu, Qu, Shen. "Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency." ICLR 2024. [arXiv:2307.08123](https://arxiv.org/abs/2307.08123)

### DAPS — Decoupled Annealing Posterior Sampling

Decouples consecutive diffusion steps, allowing large jumps between iterates while ensuring time-marginals anneal to the true posterior. Much better exploration of multimodal posteriors than DPS.

> Zhang, Chu, Berner, Meng, Anandkumar, Song. "Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing." CVPR 2025 (Oral). [arXiv:2407.01521](https://arxiv.org/abs/2407.01521)

### SILO — Solving Inverse Problems with Latent Operators

Learns a degradation operator directly in latent space. Encoder and decoder each used only once (not during iterative sampling), avoiding decoder Jacobian issues entirely. 2.6–10× faster.

> Raphaeli, Man, Elad. "SILO: Solving Inverse Problems with Latent Operators." ICCV 2025. [arXiv:2501.11746](https://arxiv.org/abs/2501.11746)

### LD-SMC — Sequential Monte Carlo in Latent Space

SMC-based posterior sampling directly in latent space. **Asymptotically exact** as the number of particles → ∞. The only latent-space method with formal convergence guarantees.

> Achituve, Habi, Rosenfeld, Netzer, Diamant, Fetaya. "Inverse Problem Sampling in Latent Space Using Sequential Monte Carlo." ICML 2025. [arXiv:2502.05908](https://arxiv.org/abs/2502.05908)

### MGPS — Midpoint Guidance Posterior Sampling

Novel decomposition of diffusion transitions allowing a trade-off between guidance complexity and prior transition fidelity. Validated on both pixel-space and latent diffusion models.

> Moufad et al. "Variational Diffusion Posterior Sampling with Midpoint Guidance." ICLR 2025 (Oral). [arXiv:2410.09945](https://arxiv.org/abs/2410.09945)

### P2L — Prompt-Tuning Latent Diffusion

Jointly optimizes text embeddings during reverse diffusion, while projecting latents into the encoder's range space. Identifies that latent drift outside the encoder's range is a major artifact source.

> Chung, Ye, Milanfar, Delbracio. "Prompt-tuning Latent Diffusion Models for Inverse Problems." ICML 2024. [arXiv:2310.01110](https://arxiv.org/abs/2310.01110)

### C-DPS — Coupled Data and Measurement Space Dynamics

Introduces a parallel forward diffusion in measurement space $\{y_t\}$ alongside $\{x_t\}$, enabling a closed-form posterior $p(x_{t-1}|x_t, y_{t-1})$ without likelihood approximation.

> Mohajer Hamidi, Yang. "Coupled Data and Measurement Space Dynamics for Enhanced Diffusion Posterior Sampling." NeurIPS 2025. [arXiv:2510.09676](https://arxiv.org/abs/2510.09676)

### FlowDPS — Flow-Driven Posterior Sampling

Extends DPS-style guidance to flow matching models. Derives a flow-version of Tweedie's formula with adaptive guidance weighting (strong early, fading late). Validated with Stable Diffusion 3.0.

> Kim, Kim, Ye. "FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems." ICCV 2025. [arXiv:2503.08136](https://arxiv.org/abs/2503.08136)

### "Rethinking DPS"

Shows that DPS's conditional score approximation is actually closer to **MAP estimation** than posterior sampling. The estimated conditional score has mean significantly deviating from zero, making it an invalid score estimation. DPS "generates high-quality samples with significantly lower diversity."

> Xu, Cai, Zhang, Ge, He, Sun, Liu, Zhang, Li, Wang. "Rethinking Diffusion Posterior Sampling: From Conditional Score Estimator to Maximizing a Posterior." ICLR 2025. [arXiv:2501.18913](https://arxiv.org/abs/2501.18913)

---

<<<<<<< HEAD
## Part III — State of the Art for Calibrated Posteriors
=======
### 4. LATINO + SDE
>>>>>>> c4a9935 (Expand report with SOTA calibration methods and failure mode analysis)

### Which method should you use?

No method currently achieves all four desiderata simultaneously: (a) works in latent space, (b) handles nonlinear forward models, (c) provides calibrated posteriors, (d) is computationally tractable.

**Tier 1 — Asymptotically exact (SMC-based):**
- **Twisted Diffusion Sampler (TDS)** [Wu et al., NeurIPS 2023]: Uses SMC with "twisting" — incorporates heuristic likelihood approximations as proposals while correcting via importance weights. Asymptotically exact as #particles → ∞, improvements seen with as few as 2 particles. Pixel-space only. [arXiv:2306.17775](https://arxiv.org/abs/2306.17775)
- **Filtering Posterior Sampling (FPS)** [Dou & Song, ICLR 2024]: Establishes equivalence between diffusion posterior sampling and Bayesian filtering, then applies SMC. Global convergence guarantees. Pixel-space, linear problems. [OpenReview](https://openreview.net/forum?id=tplXNcHZs1)
- **MCGdiff** [Cardoso et al., 2023]: Constructs Feynman-Kac model from the SGM prior, solved by SMC. Consistent under mild assumptions. Pixel-space, linear problems. [arXiv:2308.07983](https://arxiv.org/abs/2308.07983)
- **LD-SMC** [Achituve et al., ICML 2025]: The only **latent-space** method with formal convergence guarantees. Converges to the true posterior as #particles → ∞. Decoder appears only in likelihood evaluations, not gradient flows. In practice, tested with only N ∈ {1,5,10} particles.

**Tier 1b — Split Gibbs / MCMC (non-asymptotic guarantees):**
- **PnP-DM** [Wu et al., NeurIPS 2024]: Split Gibbs sampler alternating between exact likelihood conditional and denoising diffusion prior. Non-asymptotic convergence in Fisher information divergence at rate $O(1/K)$. **Best empirical calibration found in the literature: 97.46% coverage within 3σ credible intervals** (vs 88.77% for DPS). Pixel-space only. [arXiv:2405.18782](https://arxiv.org/abs/2405.18782)
- **DPnP** [Xu & Chi, NeurIPS 2024]: First provably-robust posterior sampling for nonlinear inverse problems using unconditional diffusion priors. Alternates proximal consistency sampler and denoising diffusion sampler. Pixel-space. [arXiv:2403.17042](https://arxiv.org/abs/2403.17042)

**Tier 2 — Theoretically motivated, exact for Gaussians:**
- **MMPS** [Rozet et al., 2024]: Best calibration among single-trajectory methods. Exact for Gaussian, well-motivated for non-Gaussian.
- **LFlow** [Askari et al., 2025]: Same Tweedie covariance idea in the flow matching framework. Theoretically exact for Gaussian. ODE-based (no SDE noise), so more sensitive to discretization.
- **D-Flow SGLD** [2026]: Performs posterior inference in the source/noise space of flow matching models via preconditioned SGLD, then pushes forward through the learned flow. Principled but requires diminishing step sizes for exact convergence. [arXiv:2602.21469](https://arxiv.org/abs/2602.21469)

**Tier 3 — Practically robust, no formal guarantees:**
- **DAPS** [Zhang et al., 2025]: Best for hard nonlinear problems (phase retrieval). Decoupled annealing enables better mode exploration.
- **LATINO-PRO** [Spagnoletti et al., 2025]: Fastest (8 NFEs). Best for latent consistency models. Provably under-dispersed.

---

<<<<<<< HEAD
## Part IV — Open Failure Modes

### 1. The decoder Jacobian problem (latent-space specific)
=======
### 5. LFlow — Latent Refinement via Flow Matching

**Paper:** Askari, Luo, Sun, Roosta. *"Latent Refinement via Flow Matching for Training-free Linear Inverse Problem Solving"* ([arXiv:2511.06138](https://arxiv.org/abs/2511.06138), NeurIPS 2025).
>>>>>>> c4a9935 (Expand report with SOTA calibration methods and failure mode analysis)

All latent-space methods must deal with $y = A \cdot D(z) + n$, where $D$ is the nonlinear decoder. Computing $\nabla_z \log p(y|z_t)$ requires either:
- **Backpropagation through $D$**: expensive, noisy Jacobian, $D$ sees OOD latents during sampling [SILO, PSLD]
- **Avoiding $D$ entirely**: LATINO-PRO (proximal splitting), SILO (learned latent operator) — sidesteps the issue but sacrifices posterior fidelity

<<<<<<< HEAD
The decoder Jacobian is non-uniform across the latent space. Regions of high distortion see stronger memorization and different score behavior, meaning uniform guidance strategies are suboptimal [Rao et al., 2025: arXiv:2511.20592].

**No clean solution exists.** This is arguably the central open problem.
=======
**Velocity field:**
>>>>>>> c4a9935 (Expand report with SOTA calibration methods and failure mode analysis)

### 2. Tweedie approximation breakdown (multimodal posteriors)

<<<<<<< HEAD
The Tweedie denoiser $\hat{x}_0 = \mathbb{E}[x_0|x_t]$ is a posterior mean — it averages over modes. For multimodal posteriors:
- At large $\sigma_t$: $\hat{x}_0$ is a blurry average between modes, pointing guidance toward a non-existent "average mode"
- The covariance $V[x_0|x_t]$ captures inter-mode spread, but the Gaussian likelihood approximation $\mathcal{N}(y|A\hat{x}_0, \sigma_n^2 + AV_tA^T)$ remains fundamentally **unimodal**
- **All single-trajectory methods (DPS, MMPS, LFlow) share this failure mode.** A single reverse trajectory cannot fork into multiple modes

DAPS mitigates this via decoupled annealing (allows large jumps). LD-SMC uses multiple particles. Neither fully solves the problem in high dimensions.

### 3. Score estimation error amplification
=======
**Posterior guidance** (from continuity equation):
>>>>>>> c4a9935 (Expand report with SOTA calibration methods and failure mode analysis)

With *learned* scores (not the exact analytic scores used in our notebooks):
- Score estimation error is amplified by the guidance step and accumulated over hundreds of reverse steps
- In latent space, errors are further amplified by the decoder Jacobian
- **No existing method provides calibration guarantees with imperfect scores.** All theoretical results (PSLD, LD-SMC) assume access to the true score

### 4. Encode-decode cycle inconsistency

Methods alternating between latent and pixel space (ReSample, P2L, split approaches):
- $D(E(x)) \neq x$ in general — VAE reconstruction error compounds over iterations
- Worst for fine details and out-of-distribution content — precisely the regime where inverse problems push estimates
- P2L [Chung et al., 2024] identifies latent drift outside the encoder's range as a major artifact source

<<<<<<< HEAD
### 5. Nonlinear forward models

Even in pixel space, DPS/MMPS break down for nonlinear forward models (phase retrieval, non-Cartesian MRI):
- The Gaussian likelihood approximation does not hold
- The guidance gradient landscape becomes non-convex with spurious local minima
- DAPS handles this better via global exploration, but without guarantees
=======
**Posterior covariance** for a Gaussian prior with OT interpolant:
>>>>>>> c4a9935 (Expand report with SOTA calibration methods and failure mode analysis)

### 6. The calibration gap

<<<<<<< HEAD
The field overwhelmingly reports reconstruction quality (PSNR, SSIM, LPIPS, FID). **Almost no papers report posterior calibration metrics** (coverage probability, z-score distributions, QQ plots). Among those that do:
- Most methods are under-dispersed (overconfident)
- Good PSNR does not imply calibrated posteriors
- Methods can produce sharp reconstructions while systematically missing posterior tails

The most rigorous calibration validation in the literature is PnP-DM [Wu et al., 2024], which reports **97.46% coverage within 3σ credible intervals** vs 88.77% for DPS — a 10 percentage point gap despite similar PSNR. The BIPSDA benchmark [Scope Crafts & Villa, 2025: arXiv:2503.03007] and a UQ benchmark [Qiu et al., 2026: arXiv:2602.04189] are beginning to address the lack of standardized calibration protocols.

This repository's Gaussian calibration analysis (comparing empirical z-score distributions to $\mathcal{N}(0,1)$) is one of the few rigorous calibration diagnostics in the literature.

---

## Part V — Summary: What's Missing

The biggest gap is a method that simultaneously:
1. **Works in latent space** (to leverage pretrained LDMs like Stable Diffusion)
2. **Handles nonlinear forward models** (beyond $y = Ax + n$)
3. **Provides calibrated posteriors** (not just good point estimates)
4. **Is computationally tractable** (not requiring thousands of decoder evaluations)

**Nobody has all four.** The most promising research directions are:
- **SMC-based methods** (LD-SMC): only path to asymptotic exactness, needs computational breakthroughs for particle efficiency
- **MMPS/LFlow + decoder-aware corrections**: the Tweedie covariance needs to account for decoder nonlinearity ($V[D(z_0)|z_t] \neq D(V[z_0|z_t])$)
- **DAPS-style decoupling**: practically robust for hard problems, needs formal calibration analysis
- **Hybrid approaches**: use cheap latent-space prior steps with occasional expensive pixel-space likelihood corrections

---

## References

### Implemented in this repository

1. Chung, Kim, McCann, Klasky, Ye. "Diffusion Posterior Sampling for General Noisy Inverse Problems." ICLR 2023. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

2. Rozet, Andry, Lanusse, Louppe. "Learning Diffusion Priors from Observations by Expectation Maximization." 2024. [arXiv:2405.13712](https://arxiv.org/abs/2405.13712)

3. Spagnoletti, Prost, Almansa, Papadakis, Pereyra. "LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization." ICCV 2025. [arXiv:2503.12615](https://arxiv.org/abs/2503.12615)

4. Askari, Luo, Sun, Roosta. "Latent Refinement via Flow Matching for Training-free Linear Inverse Problem Solving." NeurIPS 2025. [arXiv:2511.06138](https://arxiv.org/abs/2511.06138)

### Asymptotically exact / SMC-based methods

5. Wu, Trippe, Naesseth, Blei, Cunningham. "Practical and Asymptotically Exact Conditional Sampling in Diffusion Models." NeurIPS 2023. [arXiv:2306.17775](https://arxiv.org/abs/2306.17775)

6. Cardoso, Janati El Idrissi, Le Corff, Moulines. "Monte Carlo Guided Diffusion for Bayesian Linear Inverse Problems." 2023. [arXiv:2308.07983](https://arxiv.org/abs/2308.07983)

7. Dou, Song. "Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective." ICLR 2024. [OpenReview](https://openreview.net/forum?id=tplXNcHZs1)

8. Achituve, Habi, Rosenfeld, Netzer, Diamant, Fetaya. "Inverse Problem Sampling in Latent Space Using Sequential Monte Carlo." ICML 2025. [arXiv:2502.05908](https://arxiv.org/abs/2502.05908)

9. Wu, Han, Naesseth, Cunningham. "Reverse Diffusion Sequential Monte Carlo Samplers." 2025. [arXiv:2508.05926](https://arxiv.org/abs/2508.05926)

### Split Gibbs / MCMC methods

10. Wu, Sun, Chen, Zhang, Yue, Bouman. "Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors." NeurIPS 2024. [arXiv:2405.18782](https://arxiv.org/abs/2405.18782)

11. Xu, Chi. "Provably Robust Score-Based Diffusion Posterior Sampling for Plug-and-Play Image Reconstruction." NeurIPS 2024. [arXiv:2403.17042](https://arxiv.org/abs/2403.17042)

### Latent-space methods

12. Rout, Raoof, Daras, Caramanis, Dimakis, Shakkottai. "Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models." NeurIPS 2023. [arXiv:2307.00619](https://arxiv.org/abs/2307.00619)

13. Rout, Chen, Kumar, Caramanis, Shakkottai, Chu. "Beyond First-Order Tweedie: Solving Inverse Problems using Latent Diffusion." CVPR 2024. [arXiv:2312.00852](https://arxiv.org/abs/2312.00852)

14. Song, Kwon, Zhang, Hu, Qu, Shen. "Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency." ICLR 2024. [arXiv:2307.08123](https://arxiv.org/abs/2307.08123)

15. Chung, Ye, Milanfar, Delbracio. "Prompt-tuning Latent Diffusion Models for Inverse Problems." ICML 2024. [arXiv:2310.01110](https://arxiv.org/abs/2310.01110)

16. Raphaeli, Man, Elad. "SILO: Solving Inverse Problems with Latent Operators." ICCV 2025. [arXiv:2501.11746](https://arxiv.org/abs/2501.11746)

17. Rao, Qu, Moyer. "Latent Diffusion Inversion Requires Understanding the Latent Space." 2025. [arXiv:2511.20592](https://arxiv.org/abs/2511.20592)

### Guidance and decoupling methods

18. Zhang, Chu, Berner, Meng, Anandkumar, Song. "Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing." CVPR 2025 (Oral). [arXiv:2407.01521](https://arxiv.org/abs/2407.01521)

19. Moufad et al. "Variational Diffusion Posterior Sampling with Midpoint Guidance." ICLR 2025 (Oral). [arXiv:2410.09945](https://arxiv.org/abs/2410.09945)

20. Mohajer Hamidi, Yang. "C-DPS: Coupled Data and Measurement Space Dynamics for Enhanced Diffusion Posterior Sampling." NeurIPS 2025. [arXiv:2510.09676](https://arxiv.org/abs/2510.09676)

21. Li, Pereira. "Solving Inverse Problems via Diffusion Optimal Control." NeurIPS 2024. [arXiv:2412.16748](https://arxiv.org/abs/2412.16748)

### Flow matching methods

22. Kim, Kim, Ye. "FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems." ICCV 2025. [arXiv:2503.08136](https://arxiv.org/abs/2503.08136)

23. "D-Flow SGLD: Source-Space Posterior Sampling for Scientific Inverse Problems with Flow Matching." 2026. [arXiv:2602.21469](https://arxiv.org/abs/2602.21469)

### Analysis and benchmarks

24. Xu, Cai, Zhang, Ge, He, Sun, Liu, Zhang, Li, Wang. "Rethinking Diffusion Posterior Sampling: From Conditional Score Estimator to Maximizing a Posterior." ICLR 2025. [arXiv:2501.18913](https://arxiv.org/abs/2501.18913)

25. Scope Crafts, Villa. "Benchmarking Diffusion Annealing-Based Bayesian Inverse Problem Solvers." 2025. [arXiv:2503.03007](https://arxiv.org/abs/2503.03007)

26. Qiu, Yang, Liu, Wang, Shen. "Benchmarking Uncertainty Quantification of Plug-and-Play Diffusion Priors for Inverse Problems Solving." 2026. [arXiv:2602.04189](https://arxiv.org/abs/2602.04189)

27. Chung, Kim, Ye. "Diffusion Models for Inverse Problems." Survey, 2025. [arXiv:2508.01975](https://arxiv.org/abs/2508.01975)
=======
**Gaussian calibration result:** Theoretically exact — the guided ODE recovers the exact posterior mean and variance (verified analytically with a high-precision ODE solver). Practically, Euler discretization introduces slow convergence (z-std=0.986 at N=200 vs MMPS's 1.002) because the ODE has stiff-like behavior from the $t/(1-t)$ factor and lacks the SDE's self-correcting noise.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

### Gaussian Calibration Summary

| Method | μ (target: 1.200) | σ (target: 0.447) | z-std (target: 1.000) |
|--------|-------------------|--------------------|-----------------------|
| Vanilla LATINO | 1.337 | 0.327 | 0.765 |
| DPS | 1.452 | 0.356 | 0.916 |
| **MMPS** | **1.179** | **0.446** | **1.002** |
| LATINO + SDE | 1.209 | 0.436 | 0.979 |
| **LFlow** | **1.199** | **0.442** | **0.986** |

---

## Part II: State of the Art for Calibrated Posteriors

### The fundamental hardness result

[Gupta et al. (ICML 2024)](https://arxiv.org/abs/2402.12727) proved that **the worst-case complexity of diffusion posterior sampling is super-polynomial**, even when unconditional sampling is fast. This means no algorithm can be simultaneously general, fast, and exact. All practical methods must trade off between these.

### Best current approaches for calibrated posteriors

#### PnP-DM — Plug-and-Play Diffusion Models (NeurIPS 2024)

**Paper:** Wu, Sun, Chen, Zhang, Yue, Bouman. ([arXiv:2405.18782](https://arxiv.org/abs/2405.18782)) | [Project](https://imaging.cms.caltech.edu/pnpdm/)

The strongest current method for calibrated posteriors. Uses a **split Gibbs sampler** (MCMC) that alternates:
1. **Likelihood step:** Sample $z \sim p(z | x, y)$ — a Gaussian proximal step
2. **Prior step:** Sample $x \sim p(x | z)$ — a Bayesian denoising problem (exactly what diffusion models do)

**Key advantage:** No Tweedie approximation. No guidance heuristics. The diffusion model is used as a denoising oracle, not for likelihood estimation. Non-asymptotic stationarity guarantees. Captures 97.5% of ground truth pixels in 3-sigma credible intervals. Demonstrated on black hole imaging with multimodal posteriors.

**Limitation:** MCMC — requires many iterations. Operates in **pixel space only**.

#### DPnP — Diffusion Plug-and-Play (NeurIPS 2024)

**Paper:** Xu & Chi. ([arXiv:2403.17042](https://arxiv.org/abs/2403.17042))

First provably robust posterior sampling method for **nonlinear** inverse problems. Both asymptotic and non-asymptotic guarantees, with graceful degradation under score estimation error.

#### G-DPS — Gibbs Posterior Sampler (Feb 2025)

**Paper:** Giovannelli. ([arXiv:2602.11059](https://arxiv.org/abs/2602.11059))

Augments the problem with the full diffusion chain as auxiliary variables. All conditionals are Gaussian — "remarkably simple." Convergence guaranteed, but linear forward models only.

#### DAPS — Decoupled Annealing Posterior Sampling (CVPR 2025 Oral)

**Paper:** Zhang, Chu, Berner, Meng, Anandkumar, Song. ([arXiv:2407.01521](https://arxiv.org/abs/2407.01521))

Decouples consecutive diffusion steps, allowing large jumps in sample space. Time-marginals provably anneal to the true posterior. Works in both pixel and **latent space** (demonstrated with Stable Diffusion). Very expensive.

---

## Part III: Failure Modes None of the Methods Fully Address

### 1. The Tweedie approximation is fundamentally unimodal

DPS, MMPS, LFlow all approximate $p(x_0|x_t)$ as Gaussian. This is exact for Gaussians but breaks for multimodal posteriors. At large noise, $\mathbb{E}[x_0|x_t]$ is a blurry average over modes — the Gaussian approximation is wrong. MMPS adds the covariance but remains unimodal.

**Affected:** DPS, MMPS, TMPD, LFlow, and all first/second-order Tweedie methods.
**Mitigation:** MCMC correction (PnP-DM), multi-particle methods with repulsion.

### 2. Latent space introduces three compounding errors

This is the **key unsolved problem** for latent models:

- **Decoder Jacobian distortion:** The Jacobian $J_D(z)$ has decaying singular values, creating anisotropic latent dimensions where some directions matter far more than others for data-space fidelity. ([arXiv:2511.20592](https://arxiv.org/pdf/2511.20592))
- **Representation error:** The encoder is many-to-one. Many latents decode to images consistent with measurements. [PSLD](https://arxiv.org/abs/2307.00619) showed vanilla DPS extensions to latent space simply don't work without a "gluing" penalty.
- **Nonlinearity of decode(encode(·)):** Even linear forward models $y = Ax + n$ become nonlinear in latent space: $y = A \cdot D(z) + n$, destroying closed-form proximal steps.

**Proposed solutions:** [ReSample](https://arxiv.org/abs/2307.08123) (hard data consistency via optimization), [SILO](https://openaccess.thecvf.com/content/ICCV2025/papers/Raphaeli_SILO_Solving_Inverse_Problems_with_Latent_Operators_ICCV_2025_paper.pdf) (learned latent operators), Jacobian-aware weighting. None fully resolve the issue.

### 3. ODE methods systematically under-disperse

[Analysis of deterministic ODE samplers](https://arxiv.org/abs/2508.16154) shows they concentrate samples due to score errors propagating coherently (no stochastic correction). SDE methods self-correct via noise injection.

**Affected:** LFlow, LATINO (PF-ODE), consistency models.
**Mitigation:** Use SDE samplers or hybrid approaches.

### 4. Calibration ≠ reconstruction quality

A [comprehensive UQ benchmark (Feb 2026)](https://arxiv.org/abs/2602.04189) found dramatic differences:

| Method | Reconstruction quality | Calibration |
|--------|----------------------|-------------|
| DPS, DiffPIR, DDNM | Good PSNR/SSIM | Substantially overconfident |
| REDDiff | Good PSNR/SSIM | Near-zero variance (point estimate) |
| PnP-DM, MCG-Diff | Good PSNR/SSIM | Reasonably calibrated |

Most papers report PSNR/SSIM/LPIPS but never validate calibration.

### 5. No posterior guarantees with learned scores

- Unconditional diffusion sampling requires only L2 score accuracy.
- Posterior sampling requires much stronger conditions (MGF bounds, log-concavity).
- [Annealed Langevin (Wu et al., 2025)](https://arxiv.org/abs/2510.26324) achieves polynomial convergence with L4 score error bounds under local log-concavity — the best known result, but still restrictive.
- Performance guarantees can **diverge with increasing dimension** ([arXiv:2505.18276](https://arxiv.org/pdf/2505.18276)).

### 6. Nonlinear forward models break guidance

All theory assumes linear $A$. Nonlinear models introduce expensive Jacobians, severe nonconvexity, and local minima. All gradient-guidance methods degrade. Proximal methods lose closed forms.

### 7. Manifold departure

Score functions are trained only on the noisy data manifold. Measurement-consistency projections can throw samples off-manifold where scores are unreliable. [MCG (Chung et al., NeurIPS 2022)](https://arxiv.org/abs/2206.00941) mitigates this with tangent-plane corrections.

---

## Part IV: Error Decomposition

Total error in diffusion posterior sampling decomposes into:

1. **Initialization/truncation error** — starting from finite rather than infinite noise
2. **Score approximation error** — learned score ≠ true score (**often dominant in practice**)
3. **Discretization error** — finite steps in ODE/SDE solver (mitigable with more steps)
4. **Likelihood approximation error** — Tweedie, guidance heuristics (**structural to the method class**)
5. **Latent-space error** — decoder nonlinearity, Jacobian distortion, representation gap (**fundamental to non-invertible architectures**)

For inverse problems, items 4 and 5 are the additional error sources that don't exist in unconditional sampling. This is the core reason why unconditional diffusion models work well but posterior sampling remains challenging.

---

## Part V: Summary — What to Use When

| Goal | Best approach | Trade-off |
|------|--------------|-----------|
| Fast reconstructions | LFlow / LATINO | Not calibrated; ~8 NFEs |
| Calibrated posteriors (pixel space) | **PnP-DM** (Split Gibbs) | ~100-1000× slower |
| Calibrated posteriors (latent space) | **Open problem** | Decoder Jacobian unsolved |
| Nonlinear forward models | DPnP / PnP-DM | Even more expensive |
| Multimodal posteriors | PnP-DM / DAPS | Must avoid Tweedie-based guidance |

**The uncomfortable truth:** No existing method provides calibrated posteriors with latent models efficiently. PnP-DM works but only in pixel space. Latent methods (LFlow, LATINO, PSLD) trade calibration for speed. The decoder Jacobian problem — bridging pixel-space measurements to latent-space priors without expensive or approximate Jacobian computation — remains the key open challenge.
>>>>>>> c4a9935 (Expand report with SOTA calibration methods and failure mode analysis)
