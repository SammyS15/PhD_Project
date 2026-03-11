# Summary after 12 iterations

## What worked
- **Latent MMPS** — propagating Tweedie covariance through the decoder Jacobian: `Σ_y = σ_n²I + V_t · J_D · J_D^T`. This is the natural extension of pixel-space MMPS to latent space.
- Default guidance strength ζ=1.1 achieves calibrated posteriors on both NonlinearDecoder2D (α=0.5) and FoldedDecoder2D (α=1.0)
- Adaptive ζ ≈ 1.0 + 0.2·α extends calibration to NonlinearDecoder2D at all tested α ∈ [0, 1.5]
- Auto-tuning formula ζ = 1.0 + 0.5·nonlinearity works for injective decoders
- Robust to step count (calibrated even at N=50)
- Cost: only 4.7× single-trajectory Latent DPS

## What didn't
- **H3 (LATINO+SDE)**: SDE denoiser makes LATINO worse in latent space — encode-decode round-trip breaks variance correction
- **H4 (Split Gibbs)**: OK on unimodal (0.539) but fails on bimodal FoldedDecoder2D (0.258) — Langevin gets trapped in one mode
- **H5 (SMC)**: Simple SIR resampling collapses diversity, making calibration worse
- **H2 (Hessian correction)**: Unnecessary — adaptive zeta achieves the same result more simply
- **H13 auto-zeta on FoldedDecoder2D**: Global nonlinearity measure conflates topology with local curvature

## Best method found
**Latent MMPS (ζ=1.1, N=200)**

| Problem | HPD mean | KS stat | Cost |
|---------|----------|---------|------|
| NonlinearDecoder2D (α=0.5) | 0.501 | 0.032 | 4.7× DPS |
| FoldedDecoder2D (α=1.0) | 0.482 | 0.059 | 4.7× DPS |
| FoldedDecoder2D (α=3.0) | 0.516 | 0.085 | — |
| NonlinearDecoder2D (α=1.5, ζ=1.4) | 0.489 | 0.068 | — |
| RandomMLPDecoder2D (ζ=1.0) | 0.463 | 0.094 | — |

Verified with multiple seeds (PRNGKey 0, 42, 123) and sample sizes (n=200, 500).

## Key insight
The Tweedie posterior covariance V[z₀|z_t], when propagated through the decoder Jacobian J_D, correctly accounts for the anisotropy of the decoder mapping. This turns the isotropic DPS guidance (which treats all latent directions equally) into Jacobian-aware guidance that respects the decoder geometry. The SDE noise injection provides mode exploration that allows the method to handle bimodal posteriors (FoldedDecoder2D) despite the Tweedie approximation being unimodal.

## Most promising unexplored direction
- **Latent LFlow**: Port flow matching to latent space with the same Jacobian-aware covariance. Could be faster (deterministic ODE) and theoretically cleaner.
- **Proper LD-SMC**: Sequential Monte Carlo with resampling during the diffusion process (not post-hoc SIR). The Achituve et al. (2025) approach with MMPS proposals could yield asymptotically exact posteriors.

## Recommended next steps for the human
1. **Scale to learned decoders**: Test with a real VAE/LDM decoder (Stable Diffusion). The method only needs decoder(z) and decoder_jacobian(z) — the latter can be computed via autodiff.
2. **Learned score functions**: Replace exact scores with trained score networks. The Tweedie covariance V_t becomes approximated, but the Jacobian-aware structure should still help.
3. **Higher dimensions**: Test with d_latent > 2. The 3×3 matrix solve in Σ_y becomes d_pixel × d_pixel — may need Woodbury identity for efficiency.
4. **Theory**: Derive when ζ=1 is optimal (linear decoder) vs ζ>1 (nonlinear). The correction likely compensates for the bias in E[D(z)|z_t] ≈ D(E[z|z_t]).
