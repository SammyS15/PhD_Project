# Posterior Sampling for Inverse Problems with Diffusion/Flow Priors

This report surveys methods for solving inverse problems using diffusion and flow matching priors, with emphasis on posterior calibration and latent-space models.

**Test problem (this repo):** Prior $x \sim \mathcal{N}(\mu_0, \sigma_0^2)$, forward model $y = x + n$, $n \sim \mathcal{N}(0, \sigma_n^2)$.

---

## Part I — Implemented Methods

### 1. LATINO — LAtent consisTency INverse sOlver

Iterative noise–denoise–proximal loop. At each step $k$:

1. **Noise:** $x_{\text{noisy}} = x + \sigma_k \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$
2. **Denoise:** $u = \text{PF-ODE}(x_{\text{noisy}}, \sigma_k \to 0)$ using the prior score
3. **Proximal step:** $x = \frac{\delta_k y + \sigma_n^2 u}{\delta_k + \sigma_n^2}$ for $A = I$

**Gaussian calibration:** Under-dispersed. The proximal step is a contraction; deterministic PF-ODE provides no mechanism to restore lost variance.

> Spagnoletti, Prost, Almansa, Papadakis, Pereyra. "LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization." ICCV 2025. [arXiv:2503.12615](https://arxiv.org/abs/2503.12615)

### 2. DPS — Diffusion Posterior Sampling

Reverse-time SDE with likelihood gradient guidance:

$$\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y | x_t)$$

Likelihood approximated using only the Tweedie posterior **mean**:

$$p(y | x_t) \approx \mathcal{N}(y \mid \hat{x}_0(x_t),\; \sigma_n^2)$$

**Gaussian calibration:** Over-confident. Ignoring $V[x|x_t]$ makes guidance too strong at large $\sigma_t$, producing biased and under-dispersed posteriors.

> Chung, Kim, McCann, Klasky, Ye. "Diffusion Posterior Sampling for General Noisy Inverse Problems." ICLR 2023. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

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

## Part III — State of the Art for Calibrated Posteriors

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

## Part IV — Open Failure Modes

### 1. The decoder Jacobian problem (latent-space specific)

All latent-space methods must deal with $y = A \cdot D(z) + n$, where $D$ is the nonlinear decoder. Computing $\nabla_z \log p(y|z_t)$ requires either:
- **Backpropagation through $D$**: expensive, noisy Jacobian, $D$ sees OOD latents during sampling [SILO, PSLD]
- **Avoiding $D$ entirely**: LATINO-PRO (proximal splitting), SILO (learned latent operator) — sidesteps the issue but sacrifices posterior fidelity

The decoder Jacobian is non-uniform across the latent space. Regions of high distortion see stronger memorization and different score behavior, meaning uniform guidance strategies are suboptimal [Rao et al., 2025: arXiv:2511.20592].

**No clean solution exists.** This is arguably the central open problem.

### 2. Tweedie approximation breakdown (multimodal posteriors)

The Tweedie denoiser $\hat{x}_0 = \mathbb{E}[x_0|x_t]$ is a posterior mean — it averages over modes. For multimodal posteriors:
- At large $\sigma_t$: $\hat{x}_0$ is a blurry average between modes, pointing guidance toward a non-existent "average mode"
- The covariance $V[x_0|x_t]$ captures inter-mode spread, but the Gaussian likelihood approximation $\mathcal{N}(y|A\hat{x}_0, \sigma_n^2 + AV_tA^T)$ remains fundamentally **unimodal**
- **All single-trajectory methods (DPS, MMPS, LFlow) share this failure mode.** A single reverse trajectory cannot fork into multiple modes

DAPS mitigates this via decoupled annealing (allows large jumps). LD-SMC uses multiple particles. Neither fully solves the problem in high dimensions.

### 3. Score estimation error amplification

With *learned* scores (not the exact analytic scores used in our notebooks):
- Score estimation error is amplified by the guidance step and accumulated over hundreds of reverse steps
- In latent space, errors are further amplified by the decoder Jacobian
- **No existing method provides calibration guarantees with imperfect scores.** All theoretical results (PSLD, LD-SMC) assume access to the true score

### 4. Encode-decode cycle inconsistency

Methods alternating between latent and pixel space (ReSample, P2L, split approaches):
- $D(E(x)) \neq x$ in general — VAE reconstruction error compounds over iterations
- Worst for fine details and out-of-distribution content — precisely the regime where inverse problems push estimates
- P2L [Chung et al., 2024] identifies latent drift outside the encoder's range as a major artifact source

### 5. Nonlinear forward models

Even in pixel space, DPS/MMPS break down for nonlinear forward models (phase retrieval, non-Cartesian MRI):
- The Gaussian likelihood approximation does not hold
- The guidance gradient landscape becomes non-convex with spurious local minima
- DAPS handles this better via global exploration, but without guarantees

### 6. The calibration gap

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
