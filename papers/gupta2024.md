# Diffusion Posterior Sampling is Computationally Intractable
**Authors:** Gupta, Chen, Chen
**Year:** 2024 | **Venue:** ICML 2024
**Link:** https://arxiv.org/abs/2402.12727

## Key idea (1-2 sentences)
Proved that worst-case complexity of diffusion posterior sampling is super-polynomial, even when unconditional sampling is efficient.

## Method summary (≤10 lines)
Theoretical result via reduction from planted clique. No algorithm can be simultaneously general, fast, and exact for diffusion posterior sampling.

## Relevance to our problem
Sets fundamental limits. We must trade off generality, speed, or exactness. Our toy problems with exact scores sidestep score error.

## Key equations
No algorithmic equations — hardness result.

## Limitations noted by authors
Worst-case result; practical problems may have exploitable structure.

## Experimental takeaway
Motivates problem-specific structure (Gaussian prior, analytic decoder) over general methods.
