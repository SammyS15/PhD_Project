# LATINO-PRO Implementation

Personal implementation of **LATINO-PRO** (LAtent consisTency INverse sOlver with PRompt Optimization)


## Repository Structure

### Python Scripts

| Script | Description | SR Scale | Key Difference |
|--------|-------------|----------|----------------|
| `run_latino.py` | Initial personal implementation | 4x | Bicubic upsample init; σ_y = 0.05; ablation studies included |
| `run_latino_16_scale_factor.py` | Extended personal implementation | 16x | Pseudoinverse init (`A^T(y)`); σ_y = 0.01; matches paper SR defaults |
| `updated_run_latino.py` | Most faithful to original paper | 16x | CLI arguments; conjugate gradient init; closest to paper results; **recommended** |

**Which to use:**
- For understanding the ablation studies (prompt importance, proximal importance). These versions use `sample.png`

  → `run_latino.py`

  → `run_latino_16_scale_factor.py`
- Closest version to LATINO-PRO implementation. This version uses the same sample as in the LATINO-PRO repo `60007.png`

  → `updated_run_latino.py`

## Degradation Types

Each script tests the algorithm against three image corruption models:

| Degradation | Forward Model `A` | Parameters Used |
|-------------|-------------------|-----------------|
| **Bicubic Super-Resolution** | Bicubic downsample | scale_factor = 4 (`run_latino.py`) or 16 (others) |
| **Gaussian Blur Deblurring** | Gaussian convolution | σ = 3.0 |
| **Inpainting** | Binary pixel mask | 50% random pixels (`run_latino*.py`) or 200×200 square mask (`updated_run_latino.py`) |


## Output Directories

### `Output_Results/` — from `run_latino.py` (scale_factor = 4)

Six subdirectories covering base experiments and ablation studies:

```
Output_Results/
  bicubic/                          # Bicubic SR, correct prompt
  gaussian_blur/                    # Gaussian deblurring, correct prompt
  inpainting/                       # Inpainting, correct prompt
  NonSense_Prompt/                  # All 3 degradations, semantically wrong prompt
  No_Proximal/                      # All 3 degradations, proximal operator disabled (δ_k ≈ 0)
  NonSense_Prompt_No_Proximal/      # All 3 degradations, wrong prompt AND proximal disabled
```

### `Output_Results_x16/` — from `run_latino_16_scale_factor.py` (scale_factor = 16)

Same six-subdirectory structure as above, but using 16x super-resolution and the paper's default noise level (σ_y = 0.01).

### `Updated_Output_Results/` — from `updated_run_latino.py`

Three subdirectories, one per degradation, with richer per-iteration output:

```
Updated_Output_Results/
  super_resolution_bicubic/
  deblurring_gaussian/
  inpainting_squared_mask/
```

Each degradation directory contains:

```
<degradation>/
  clean.png            # Ground truth: the original uncorrupted image
  degraded.png         # Observation: the corrupted input fed to the algorithm
  restored.png         # Output: the LATINO algorithm's final restored image
  LATINO_Results.png   # 2x3 comparison visualization (see below)
  iter/                # Per-iteration snapshots for debugging/visualization
    999_x.png          # Decoded image after denoising at timestep 999 (iteration 1)
    999_prox.png       # Same image after proximal step at timestep 999
    874_x.png          # Decoded image at timestep 874 (iteration 2)
    874_prox.png       # After proximal step at timestep 874
    ...                # Continues for timesteps: 749, 624, 499, 374, 249, 124
```

## How to Read `LATINO_Results.png`

Each result folder in `Updated_Output_Results/` contains a 2×3 grid visualization:

```
┌─────────────────────┬─────────────────────┬─────────────────────────────┐
│   Ground Truth      │   LATINO Result      │  |GT - LATINO|              │
│   (clean image)     │   (restored image)   │  (pixel-wise difference)    │
├─────────────────────┼─────────────────────┼─────────────────────────────┤
│   Degraded Obs      │   Re-degraded        │  |y - A(LATINO)|            │
│   (algorithm input) │   A(restored)        │  (observation residual)     │
└─────────────────────┴─────────────────────┴─────────────────────────────┘
```

**What each panel tells you:**
- **Top-left (Ground Truth):** The original clean image — the reconstruction target.
- **Top-middle (LATINO Result):** The algorithm's output after N iterations.
- **Top-right (|GT − LATINO|):** Pixel-wise absolute difference between truth and result. Brighter = larger error. Darker = more faithful reconstruction.
- **Bottom-left (Degraded Obs):** What the algorithm saw as input — the corrupted observation `y`.
- **Bottom-middle (Re-degraded):** Applying the degradation operator `A` to the restored image, i.e., `A(x^(N))`. This should closely match the observation `y` if the algorithm respects data fidelity.
- **Bottom-right (|y − A(LATINO)|):** Residual between observation and re-degraded result. Should be close to the noise level σ_y for a good solution.

## Interpreting PSNR

PSNR (Peak Signal-to-Noise Ratio) is the primary metric logged by all scripts:

```
PSNR = 20 * log10(MAX / RMSE)
```

where `MAX = 1.0` (images normalized to [0, 1]) and `RMSE` is the root-mean-squared pixel error.

Higher PSNR = better reconstruction. Rough reference ranges for these tasks (SDXL-scale images):
- **> 30 dB** — good reconstruction, visually close to ground truth
- **25–30 dB** — acceptable, some visible artifacts
- **< 25 dB** — poor reconstruction

The scripts report PSNR for the restored image against ground truth. The degraded observation's PSNR is also reported as a baseline to confirm the algorithm improves over the input.


## Understanding Proximal Operator & Prompt Importance (in `run_latino.py` and `run_latino_16_scale_factor.py`)

The `NonSense_Prompt`, `No_Proximal`, and `NonSense_Prompt_No_Proximal` folders test what happens when key components are disabled:

| Ablation | What changes | What it measures |
|----------|-------------|-----------------|
| **Nonsense Prompt** | Prompt replaced with semantically wrong text (e.g., "a photo of a red sports car") | Importance of text conditioning — does the algorithm degrade without meaningful guidance? |
| **No Proximal** | δ_k set to ≈ 0, disabling the proximal step | Importance of data fidelity — can the diffusion prior alone restore images without being anchored to `y`? |
| **Combined** | Both: wrong prompt + no proximal | Baseline of pure unconditional diffusion with no task-specific guidance whatsoever |

Comparing these against the base results shows which components contribute most to reconstruction quality.

