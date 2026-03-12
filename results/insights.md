# Research Insights

## Key Findings
1. Oracle Langevin (ULA, N=3000, lr=5e-7) achieves calibrated posteriors on MNISTVAE: hpd=0.518, KS=0.079
2. All diffusion-based solvers fail catastrophically (hpd > 0.9) -- posterior is 60x tighter than prior
3. Grid posterior is accurate: grid sampler achieves KS=0.015 (gold standard)
4. MAP-Laplace is close (hpd=0.472, KS=0.109) -- Gaussian approximation slightly too tight
5. Latent MMPS worked on toy problems but fails on MNISTVAE due to concentration ratio

## Open Questions
- Can diffusion-based methods be adapted to work when posterior is much tighter than prior?
- Is there a noise schedule that bridges the 60x concentration gap?
- Can preconditioned Langevin match Oracle quality with fewer steps?
- Would a two-phase approach (MAP finding + local MCMC) be practical?

## New Hypotheses
(to be generated as research progresses)

## Dead Ends (from previous experiments)
- All standard diffusion solvers on MNISTVAE: LATINO, DPS, MMPS, LFlow, Split Gibbs (all hpd > 0.9)
- MALA at lr=5e-7: identical to ULA (acceptance rate ~100%)
- MAP-initialized MALA with Newton step: slightly under-dispersed (hpd~0.45)
