# Research Insights

## Key Findings
1. Pixel-space is solved: MMPS achieves z_std=1.002 on Gaussian1D
2. Latent LATINO: hpd_mean=0.435 on NonlinearDecoder2D (under-dispersed), 0.741 on FoldedDecoder2D (over-dispersed, trapped in one mode)
3. Latent DPS: hpd_mean=0.381 on NonlinearDecoder2D (under-dispersed), 0.256 on FoldedDecoder2D (very under-dispersed)
4. Both baseline methods are worse on FoldedDecoder2D — multimodality is a major challenge
5. **Latent MMPS achieves calibrated posteriors on BOTH problems** (iter 2):
   - NonlinearDecoder2D (α=0.5): hpd_mean=0.501, KS=0.032
   - FoldedDecoder2D: hpd_mean=0.482, KS=0.059
   - Cost: 4.7x single-trajectory Latent DPS
6. The key insight: propagating Tweedie covariance through the decoder Jacobian (Σ_y = σ_n²I + V_t·J·J^T) correctly accounts for decoder anisotropy
7. Optimal zeta=1.1 (slight over-guidance needed), not 1.0 — likely compensating for decoder nonlinearity not captured by first-order Jacobian
8. Latent MMPS handles multimodality (FoldedDecoder2D) despite Tweedie being unimodal — the SDE noise injection provides mode exploration

## Open Questions
- Why does zeta=1.1 work better than zeta=1.0? Likely compensating for higher-order decoder terms
- Can second-order decoder correction (H2) extend calibration to alpha>0.7?
- How sensitive is the result to N (number of diffusion steps)?
- Interesting: LATINO actually beats DPS at alpha=0.7 (0.497 vs 0.408) but diverges at alpha≥1.0

## Phase Diagram (H8+H12 results, iters 3-4)
- With adaptive zeta, Latent MMPS is calibrated at ALL alpha ∈ [0, 1.5]
- Optimal zeta ≈ 1.0 + 0.2·alpha (linear scaling with nonlinearity)
- Best universal default: zeta=1.1 (works on both problems up to alpha≈0.7)
- DPS consistently under-dispersed (~0.41-0.44) at all alpha
- LATINO surprisingly good at alpha=0.7 but catastrophically bad at alpha=1.5
- FoldedDecoder2D prefers lower zeta (1.0-1.1); higher zeta over-corrects

## Step Count Robustness (H11, iter 5)
- Latent MMPS is calibrated at ALL tested step counts: N=50, 100, 150, 200, 300
- N=100 is the sweet spot: calibrated + fast (1.4s for n=200 test cases)
- N=50 still passes criteria — method is very robust to discretization
- SDE noise provides self-correction that compensates for coarse steps

## Random MLP Decoder (H9, iter 11)
- RandomMLPDecoder2D (R²→R⁸, frozen random MLP with tanh): MMPS works with zeta=1.0
- DPS is competitive here (0.459 vs 0.463) because MLP is relatively linear
- Optimal zeta is LOWER for MLP (1.0) vs analytic decoders (1.1-1.4)
- Rule of thumb: more nonlinear decoder → higher zeta needed
- The zeta correction compensates for first-order Jacobian linearization error

## Zeta Auto-Tuning (H13, iter 12)
- Formula: zeta = 1.0 + 0.5 * nonlinearity, where nonlinearity = E[||D(z)-J(0)z|| / ||D(z)||]
- Works perfectly for NonlinearDecoder2D at ALL alpha (0.0-1.5): hpd ∈ [0.500, 0.515]
- Fails for FoldedDecoder2D: nonlinearity=1.0 due to global symmetry D(z)=D(-z), but optimal zeta is still 1.1
- The measure conflates global topology (folding) with local nonlinearity (Jacobian curvature)
- For injective decoders, the formula is reliable. For non-injective, use fixed zeta=1.1

## Latent LFlow (H14, iter 13)
- ODE-based flow matching with Jacobian-aware guidance works on NonlinearDecoder2D (zeta=1.0: hpd=0.504)
- Fails to simultaneously calibrate on FoldedDecoder2D — no single zeta works for both
- Root cause: ODE has no stochastic noise for mode exploration; Euler discretization error is coherent
- Confirms: SDE noise injection (as in Latent MMPS) is essential for bimodal posteriors
- SDE > ODE for posterior sampling in latent space

## Noise Level Sensitivity (H15, iter 14)
- MMPS calibrated at ALL sigma_n ∈ [0.1, 1.0] on NonlinearDecoder2D
- DPS catastrophically fails at low noise: hpd=0.065 at sigma_n=0.1 (guidance too strong)
- The Tweedie covariance V_t is crucial at low noise: without it (DPS), guidance overwhelms the prior
- FoldedDecoder2D borderline at sigma_n=0.1 (KS=0.108) — tight bimodal posterior is hardest case

## Literature Search (iter 16)
- Searched for "moment matching + decoder Jacobian + latent space + calibration"
- No existing paper combines MMPS-style Tweedie covariance with decoder Jacobian for calibrated latent posteriors
- Closest: STSL (2nd-order Tweedie, CVPR 2024) — reconstruction quality focus, not calibration
- Closest: C-DPS (coupled dynamics, NeurIPS 2025) — pixel+measurement space, not latent Jacobian-aware
- Closest: LD-SMC (Achituve, ICML 2025) — formal convergence but needs many particles
- **Latent MMPS is novel**: the push-through identity + Tweedie covariance propagation through J_D is not in the literature

## New Hypotheses (agent-generated)
- H13b: Local nonlinearity measure for zeta auto-tuning
- Write a paper on Latent MMPS

## Dead Ends
- H2 (second-order Hessian correction): buggy with batched inputs, and unnecessary since adaptive zeta achieves same result more simply
- H3 (LATINO+SDE): SDE denoiser makes LATINO WORSE in latent space (0.575/0.806 vs 0.447/0.724). The pixel-space proximal is exact but the encode-decode round-trip breaks the variance correction.
- H4 (Split Gibbs): OK on NonlinearDecoder2D (0.539) but fails on FoldedDecoder2D (0.258). Langevin gets trapped in one mode; denoising doesn't provide enough mode mixing with limited K.
- H5 (SMC): Simple SIR resampling on top of MMPS makes things WORSE (0.353/0.332). Importance resampling collapses diversity — selects high-likelihood particles, destroying calibration. Would need proper sequential resampling during diffusion (like LD-SMC), not post-hoc.

## FoldedDecoder2D Stress Test (H6, iter 10)
- Latent MMPS (zeta=1.1) is calibrated at ALL FoldedDecoder2D alpha ∈ [0.1, 3.0]
- α=3.0: MMPS hpd=0.516, KS=0.085 ✓ vs DPS 0.093 vs LATINO 0.854
- The method is remarkably robust to the strength of the folding nonlinearity
- DPS and LATINO degrade catastrophically at high α (opposite directions)
- Key: MMPS's Jacobian-aware covariance naturally adapts to the decoder geometry

## Oracle Comparison (H10, iter 9)
- Annealed ULA (1000 steps, exact log-posterior): NL=0.537, F=0.571
- Latent MMPS (200 steps, Tweedie): NL=0.492, F=0.482 (n=200)
- MMPS is competitive with or better than ULA oracle! The Tweedie approximation + SDE noise provides excellent calibration without needing exact posterior gradients over many steps. ULA struggles on FoldedDecoder2D bimodality.
