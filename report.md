# Methods for Solving Inverse Problems with Generative Priors

This report summarizes the diffusion/flow-based inverse problem solvers studied in this repository, tested on a 1D Gaussian calibration problem: prior $x \sim \mathcal{N}(\mu_0, \sigma_0^2)$, forward model $y = x + n$ with $n \sim \mathcal{N}(0, \sigma_n^2)$.

---

## 1. LATINO — LAtent consisTency INverse sOlver

**Paper:** Spagnoletti, Prost, Almansa, Papadakis, Pereyra. *"LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization"* (arXiv:2503.12615, 2025).

**Core idea:** Iterative noise-denoise-proximal loop. At each step $k$:

1. **Noise:** $x_{\text{noisy}} = x + \sigma_k \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$
2. **Denoise:** $u = \text{PF-ODE}(x_{\text{noisy}}, \sigma_k \to 0)$ using the prior score
3. **Proximal step:** $x = \text{prox}_{\delta_k \cdot g}(u) = \frac{\delta_k y + \sigma_n^2 u}{\delta_k + \sigma_n^2}$ for $A = I$

The sigma schedule is geometric: $\sigma_k \in [\sigma_{\max}, \sigma_{\min}]$. Delta schedule choices: vanishing ($\delta = \sigma_k^2$), constant ($\delta = \sigma_n^2$), or adaptive.

**Gaussian calibration result:** The PF-ODE denoiser is deterministic, so the only stochasticity comes from injected noise. The proximal step contracts the distribution, producing **under-dispersed** posteriors (variance too small). The analytic stationary distribution at fixed $\sigma$ does not match the true posterior for any delta schedule choice.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

## 2. DPS — Diffusion Posterior Sampling

**Paper:** Chung, Kim, Mccann, Klasky, Ye. *"Diffusion Posterior Sampling for General Noisy Inverse Problems"* (ICLR 2023).

**Core idea:** Run the reverse-time SDE from noise to data, adding a likelihood gradient at each step. Decomposes the posterior score via Bayes' rule:

$$\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y | x_t)$$

The likelihood is approximated using only the Tweedie posterior **mean**:

$$p(y | x_t) \approx \mathcal{N}(y \mid \hat{x}_0(x_t), \sigma_n^2)$$

where $\hat{x}_0 = x_t + \sigma_t^2 \nabla \log p_t(x_t)$ is the Tweedie denoised estimate.

**Gaussian calibration result:** This approximation ignores the posterior covariance $V[x|x_t]$, which causes the guidance to be **too strong** at large noise levels. The resulting posteriors have correct mean but slightly **under-dispersed** variance and shifted mean (biased toward the observation).

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

## 3. MMPS — Moment-Matching Posterior Sampling

**Paper:** Rozet, Andry, Lanusse, Louppe. *"Learning Diffusion Priors from Observations by Expectation Maximization"* (arXiv:2405.13712, 2024).

**Core idea:** Improves DPS by incorporating both Tweedie posterior moments (mean **and** covariance):

$$p(y | x_t) \approx \mathcal{N}\big(y \mid \hat{x}_0, \sigma_n^2 + V[x | x_t]\big)$$

where $V[x|x_t] = \sigma_t^2 \sigma_0^2 / (\sigma_0^2 + \sigma_t^2)$ is computed from Tweedie's second identity. The added covariance in the denominator tempers the guidance at large noise levels.

**Gaussian calibration result:** For the Gaussian case, this likelihood approximation is **exact** (not an approximation). With $\zeta = 1$, MMPS produces **perfectly calibrated** posteriors matching the analytic posterior in both mean and variance.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

## 4. LATINO + SDE

**Variant:** Replace the deterministic PF-ODE denoiser in LATINO with the stochastic reverse SDE. The SDE injects additional noise during denoising, which can restore variance lost in the proximal step.

**Gaussian calibration result:** Significantly improves over vanilla LATINO, producing **nearly calibrated** posteriors. The stochastic denoiser compensates for the variance contraction of the proximal step.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

## 5. LFlow — Latent Refinement via Flow Matching

**Paper:** Askari, Luo, Sun, Roosta. *"Latent Refinement via Flow Matching for Training-free Linear Inverse Problem Solving"* (NeurIPS 2025, arXiv:2511.06138).

**Core idea:** Uses flow matching (optimal transport interpolant) instead of diffusion for the generative prior. The OT path is $x_t = (1-t) x_0 + t z_1$ with $z_1 \sim \mathcal{N}(0, I)$, evolving from data ($t=0$) to noise ($t=1$).

### Velocity field

The marginal velocity field is:

$$v_t(x) = \mathbb{E}[z_1 - x_0 \mid x_t = x] = \frac{x}{t} - \frac{\hat{x}_0(x_t)}{t}$$

For sampling, integrate backward from $t \approx 1$ to $t \approx 0$.

### Posterior guidance

The posterior velocity field (conditioned on observation $y$) is derived via the continuity equation:

$$v_t^y(x) = v_t(x) - \frac{t}{1-t} \nabla_{x_t} \log p(y | x_t)$$

The likelihood uses the full Tweedie moments (like MMPS):

$$p(y | x_t) \approx \mathcal{N}\big(y \mid \hat{x}_0, \sigma_n^2 + V[x_0 | x_t]\big)$$

### Posterior covariance

The posterior covariance $V[x_0 | x_t]$ is derived using the Jacobian of the optimal vector field. For a Gaussian prior with OT interpolant:

$$V[x_0 | x_t] = \frac{\sigma_0^2 t^2}{(1-t)^2 \sigma_0^2 + t^2}$$

This coincides with the exact Tweedie posterior covariance for the Gaussian case, confirming that the LFlow covariance formula is exact in this setting.

### Key difference from diffusion methods

LFlow uses **ODE sampling** (no stochastic noise injection during reverse integration). All stochasticity comes from the initial noise sample at $t \approx 1$. The OT interpolant produces straight trajectories, enabling efficient sampling.

**Gaussian calibration result:** See notebook for results. For the Gaussian case with exact velocity field and exact posterior covariance, LFlow should produce well-calibrated posteriors.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

## Summary Table

| Method | Prior score | Guidance | Covariance | Stochastic | Calibrated (Gaussian) |
|--------|------------|----------|------------|------------|----------------------|
| Vanilla LATINO | VE-SDE PF-ODE | Proximal step | N/A | Noise injection only | No (under-dispersed) |
| DPS | VE-SDE reverse SDE | Score guidance | Ignored | Yes (SDE) | Nearly (biased) |
| MMPS | VE-SDE reverse SDE | Score guidance | Tweedie $V[x|x_t]$ | Yes (SDE) | Yes (exact) |
| LATINO + SDE | VE-SDE reverse SDE | Proximal step | N/A | Yes (SDE denoiser) | Nearly |
| LFlow | OT flow matching ODE | Velocity guidance | Tweedie $V[x_0|x_t]$ | No (ODE only) | See notebook |
