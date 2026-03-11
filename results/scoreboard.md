# Scoreboard — Best Results per (Method, Problem)

## NonlinearDecoder2D (alpha=0.5, beta=0.5, sigma_n=0.3)

| Method | HPD mean | KS stat | Iter | Notes |
|--------|----------|---------|------|-------|
| Latent LATINO | 0.435 | 0.139 | 0 | Baseline |
| Latent DPS | 0.381 | 0.191 | 0 | Baseline |
| **Latent MMPS (ζ=1.1)** | **0.501** | **0.032** | 2 | **CALIBRATED** (n=500) |

## FoldedDecoder2D (alpha=1.0, sigma_n=0.3)

| Method | HPD mean | KS stat | Iter | Notes |
|--------|----------|---------|------|-------|
| Latent LATINO | 0.741 | 0.431 | 0 | Over-dispersed, one mode |
| Latent DPS | 0.256 | 0.353 | 0 | Under-dispersed |
| **Latent MMPS (ζ=1.1)** | **0.482** | **0.059** | 2 | **CALIBRATED** (n=500, 4.7x DPS cost) |

## Target: HPD mean ∈ [0.45, 0.55], KS stat < 0.10

## SUCCESS CRITERIA MET ✓
- NonlinearDecoder2D (α=0.5): hpd_mean=0.501 ∈ [0.45,0.55], KS=0.032 < 0.10
- FoldedDecoder2D: hpd_mean=0.482 ∈ [0.45,0.55], KS=0.059 < 0.10
- Cost: 4.7x single-trajectory Latent DPS ≤ 10x
