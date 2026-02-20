import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset

from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    LCMScheduler,
)
from huggingface_hub import hf_hub_download
import deepinv as dinv
from deepinv.physics.blur import gaussian_blur
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

device = torch.device("cpu")

print(f"PyTorch: {torch.__version__}")


# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
# print("CUDA available:", torch.cuda.is_available())
# print("GPU name:", torch.cuda.get_device_name(0))

# ---- Experiment parameters (edit these) ----
IMAGE_PATH   = "/home/sammys15/scratch/PhD_Project/LATINO-PRO_Implementation/sample.png"            # path to your ground-truth image
N            = 4                       # LATINO iterations (4 or 8)
SIGMA_Y      = 0.05                    # observation noise std-dev
PROMPT       = "a high quality photo"  # text conditioning
OUTPUT_DIR   = "/home/sammys15/scratch/PhD_Project/LATINO-PRO_Implementation/Output_Results"

# ---- Degradation type (change this to switch experiments) ----
# Options: "bicubic", "gaussian_blur", "inpainting"
DEGRADATION_TYPE = "inpainting"

# Bicubic super-resolution parameters
SCALE_FACTOR = 4                       # downsampling factor

# Gaussian blur parameters
BLUR_SIGMA   = 3.0                     # std-dev of the Gaussian kernel

# Inpainting parameters
MASK_RATIO   = 0.5                     # fraction of pixels to KEEP (0.5 = 50% observed)

# ---- Ablation test flags (toggle these to run specific tests) ----
# Test 1: Nonsense prompt -- use a prompt unrelated to the image content
#   to see how much the text conditioning matters.
#   Results go to: Output_Results/NonSense_Prompt/<DEGRADATION_TYPE>/
NONSENSE_PROMPT        = False 
NONSENSE_PROMPT_TEXT   = "a photo of a red sports car"   # the misleading prompt

# Test 2: Disable proximal operator -- set delta_k ~ 0 so the proximal
#   step has no effect. Tests what the diffusion prior alone can do.
#   Results go to: Output_Results/No_Proximal/<DEGRADATION_TYPE>/
DISABLE_PROXIMAL       = True

# Auto-select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Dimension must be divisible by scale_factor * 8
divisor = SCALE_FACTOR * 8
TARGET_SIZE = (1024 // divisor) * divisor
print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}  (divisible by {divisor})")

# Build the actual output path based on which test flags are active
# Normal run:       Output_Results/<DEGRADATION_TYPE>/
# Nonsense prompt:  Output_Results/NonSense_Prompt/<DEGRADATION_TYPE>/
# No proximal:      Output_Results/No_Proximal/<DEGRADATION_TYPE>/
# Both:             Output_Results/NonSense_Prompt_No_Proximal/<DEGRADATION_TYPE>/
_test_suffix = ""
if NONSENSE_PROMPT and DISABLE_PROXIMAL:
    _test_suffix = "NonSense_Prompt_No_Proximal"
elif NONSENSE_PROMPT:
    _test_suffix = "NonSense_Prompt"
elif DISABLE_PROXIMAL:
    _test_suffix = "No_Proximal"

if _test_suffix:
    RESULT_DIR = Path(OUTPUT_DIR) / _test_suffix / DEGRADATION_TYPE
else:
    RESULT_DIR = Path(OUTPUT_DIR) / DEGRADATION_TYPE
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Choose the actual prompt based on the nonsense flag
ACTIVE_PROMPT = NONSENSE_PROMPT_TEXT if NONSENSE_PROMPT else PROMPT

# ============================================================
# Model Loading
# ============================================================

def load_pipeline(device="cuda"):
    BASE_CACHE = "/home/sammys15/scratch/PhD_Project_Scratch/LATINO-PRO_Implementation/model_cache"

    vae = AutoencoderKL.from_pretrained(
        f"{BASE_CACHE}/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )

    base_model_path = f"{BASE_CACHE}/stable-diffusion-xl-base-1.0"
    unet_config = UNet2DConditionModel.load_config(base_model_path, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)

    unet.load_state_dict(
        torch.load(
            f"{BASE_CACHE}/DMD2/dmd2_sdxl_4step_unet_fp16.bin",
            map_location=device,
            weights_only=True,
        )
    )

    pipe = DiffusionPipeline.from_pretrained(
        base_model_path,
        unet=unet,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = pipe.vae.to(dtype=torch.float32)
    return pipe


# ============================================================
# VAE Encode / Decode
# ============================================================

def vae_encode(vae, x, scaling_factor):
    """Encode image to latent space.  x in [0, 1] -> z."""
    x_norm = (2.0 * x - 1.0).to(dtype=torch.float32)
    z = vae.encode(x_norm).latent_dist.mean * scaling_factor
    return z


def vae_decode(vae, z, scaling_factor):
    """Decode latent to image space.  z -> x in [0, 1]."""
    z_f32 = z.to(dtype=torch.float32)
    x = vae.decode(z_f32 / scaling_factor).sample
    x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


# ============================================================
# Diffusion Utilities
# ============================================================

def forward_diffusion(z, alpha_t):
    """Forward process: z_t = sqrt(alpha_t)*z + sqrt(1-alpha_t)*eps."""
    noise = torch.randn_like(z)
    return torch.sqrt(alpha_t) * z + torch.sqrt(1.0 - alpha_t) * noise


def consistency_denoise(pipe, z_t, timestep, prompt_embeds, pooled_prompt_embeds,
                        target_size=(1024, 1024)):
    """Single-step denoising via the consistency model (Tweedie formula)."""
    device = z_t.device
    dtype = z_t.dtype
    t = torch.tensor([timestep], device=device, dtype=torch.long)

    add_time_ids = torch.tensor(
        [[target_size[0], target_size[1], 0, 0, target_size[0], target_size[1]]],
        device=device, dtype=dtype,
    )
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds.to(dtype=dtype),
        "time_ids": add_time_ids,
    }

    noise_pred = pipe.unet(
        z_t, t,
        encoder_hidden_states=prompt_embeds.to(dtype=dtype),
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    alpha_t = pipe.scheduler.alphas_cumprod[timestep].to(device=device, dtype=dtype)

    pred_type = pipe.scheduler.config.prediction_type
    if pred_type == "epsilon":
        z_0 = (z_t - torch.sqrt(1.0 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
    elif pred_type == "v_prediction":
        z_0 = torch.sqrt(alpha_t) * z_t - torch.sqrt(1.0 - alpha_t) * noise_pred
    elif pred_type == "sample":
        z_0 = noise_pred
    else:
        raise ValueError(f"Unknown prediction type: {pred_type}")

    return z_0


# ============================================================
# Proximal Operator  (uses deepinv's built-in prox_l2)
# ============================================================

def get_delta(t_k, df, scale_factor=4):
    """Adaptive delta_k based on timestep and measurement error (from reference)."""
    if scale_factor <= 16:
        return 3.0 * df / 10.0 if t_k > 300 else 2.0 * df / 10.0
    else:
        return 1.5 * df / 10.0 if t_k > 300 else 3.0 * df / 10.0


def get_timesteps(N):
    """Evenly spaced timesteps from 999 down (matching reference implementation)."""
    step = 1000 // N
    return [999 - i * step for i in range(N)]


# ============================================================
# Metrics
# ============================================================

def compute_psnr(pred, target):
    """Peak Signal-to-Noise Ratio (assumes [0, 1] range)."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(1.0 / mse)).item()


# ============================================================
# Degradation Setup
# ============================================================

def build_physics(degradation_type, img_size, device,
                  scale_factor=4, blur_sigma=3.0, mask_ratio=0.5):
    """
    Factory function: create the deepinv physics operator for the chosen
    degradation experiment.

    Args:
        degradation_type: "bicubic", "gaussian_blur", or "inpainting"
        img_size:         (C, H, W) of the *ground-truth* image
        device:           torch device
        scale_factor:     downsampling factor      (bicubic only)
        blur_sigma:       Gaussian kernel std-dev   (gaussian_blur only)
        mask_ratio:       fraction of pixels kept   (inpainting only)

    Returns:
        physics:  deepinv physics operator with .A(), .A_adjoint(), .prox_l2()
    """
    if degradation_type == "bicubic":
        physics = dinv.physics.Downsampling(
            img_size=img_size,
            factor=scale_factor,
            filter="bicubic",
            device=device,
        )
    elif degradation_type == "gaussian_blur":
        kernel = gaussian_blur(sigma=(blur_sigma, blur_sigma), device=device)
        physics = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=kernel,
            device=device,
        )
    elif degradation_type == "inpainting":
        physics = dinv.physics.Inpainting(
            img_size=img_size,
            mask=mask_ratio,
            pixelwise=True,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown degradation_type='{degradation_type}'. "
            "Choose from: 'bicubic', 'gaussian_blur', 'inpainting'"
        )
    return physics


def build_initial_estimate(y, degradation_type, target_h, target_w):
    """
    Create the initial estimate x^(0) from the observation y.

    For bicubic SR the observation is low-res, so we upsample.
    For blur and inpainting the observation is already full-res,
    so we just use y directly.
    """
    if degradation_type == "bicubic":
        x0 = F.interpolate(y, size=(target_h, target_w),
                           mode="bicubic", align_corners=False)
        return x0.clamp(0.0, 1.0)
    else:
        # gaussian_blur and inpainting: y is already (B, C, H, W)
        return y.clone().clamp(0.0, 1.0)


# ============================================================
# LATINO  --  Algorithm 1
# ============================================================

@torch.no_grad()
def latino(pipe, y, physics, prompt="a high quality photo", N=4,
           sigma_y=0.05, scale_factor=4, degradation_type="bicubic",
           target_size_hw=None, disable_proximal=False,
           device="cuda", verbose=True):
    """
    LATINO: LAtent consisTency INverse sOlver.

    Args:
        pipe:             SDXL pipeline (DMD2 UNet + LCM scheduler)
        y:                Degraded observation  [B, C, H_y, W_y]  in [0, 1]
        physics:          deepinv forward operator
        prompt:           Text prompt for conditioning
        N:                Number of iterations (4 or 8)
        sigma_y:          Observation noise std-dev
        scale_factor:     Downsampling factor (used for adaptive delta)
        degradation_type: "bicubic", "gaussian_blur", or "inpainting"
        target_size_hw:   (H, W) of the full-res image (required for bicubic;
                          inferred from y for blur/inpainting)
        disable_proximal: If True, set delta_k=0 to disable the proximal
                          operator (ablation test)
        device:           Torch device
        verbose:          Print per-iteration progress

    Returns:
        x:  Reconstructed image  [B, C, H, W]  in [0, 1]
    """
    vae = pipe.vae
    s = vae.config.scaling_factor                       # 0.13025 for SDXL
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    timesteps = get_timesteps(N)
    print(f'Time Steps: {timesteps}')

    # Proximal step operates in [-1, 1] range (matching reference calibration)
    y_norm = (y * 2 - 1).float()

    # Encode text prompt  (no classifier-free guidance for DMD2)
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt, device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    # Determine target image size
    B, C, H_y, W_y = y.shape
    if degradation_type == "bicubic":
        H, W = H_y * scale_factor, W_y * scale_factor
    elif target_size_hw is not None:
        H, W = target_size_hw
    else:
        H, W = H_y, W_y   # blur / inpainting: same resolution

    # x^(0): build initial estimate depending on degradation type
    x = build_initial_estimate(y, degradation_type, H, W)

    pbar = tqdm(enumerate(timesteps), total=len(timesteps),
                desc="LATINO") if verbose else enumerate(timesteps)

    for k, t_k in pbar:
        # 1) Encode:  z = E(x^{k-1})
        z = vae_encode(vae, x, s)

        # 2) Forward diffusion:  z_{t_k} = sqrt(alpha)*z + sqrt(1-alpha)*eps
        alpha_t = alphas_cumprod[t_k].to(z.device, z.dtype)
        z_t = forward_diffusion(z, alpha_t)

        # Cast to UNet dtype (float16 on CUDA) for the denoising step
        z_t = z_t.to(dtype=pipe.unet.dtype)

        # 3) Denoise:  z_0 = G_theta(z_{t_k}, t_k, c)
        z_0 = consistency_denoise(
            pipe, z_t, t_k, prompt_embeds, pooled_prompt_embeds,
            target_size=(H, W),
        )

        # 4) Decode:  u^{k} = D(z_0)
        u = vae_decode(vae, z_0, s)

        # 5) Proximal step:  x^{k} = prox_{delta_k * g_y}(u^{k})
        #    Operate in [-1, 1] range; use delta_k directly as gamma
        u_norm = (u * 2 - 1).float()

        df = torch.norm(physics.A(u_norm) - y_norm).item()
        delta_k = get_delta(t_k, df, scale_factor=scale_factor)
        if disable_proximal:
            delta_k = 0.01          # ablation: diffusion prior only, no data fidelity
        prox_x_norm = physics.prox_l2(u_norm, y=y_norm, gamma=delta_k)
        x = ((prox_x_norm + 1) / 2).clamp(0.0, 1.0)

        if verbose:
            pbar.set_postfix(t=t_k, gamma=f"{delta_k:.4f}", df=f"{df:.4f}")

    return x

@torch.no_grad()
def latino_unconditional(pipe, y, physics, N=4,
                         sigma_y=0.05, scale_factor=4, degradation_type="bicubic",
                         target_size_hw=None, disable_proximal=False,
                         device="cuda", verbose=True):
    """
    LATINO solver without any text prompt conditioning.

    Args:
        pipe:             SDXL pipeline (DMD2 UNet + LCM scheduler)
        y:                Degraded observation  [B, C, H_y, W_y]  in [0, 1]
        N:                Number of iterations (e.g. 4)
        sigma_y:          Observation noise std-dev
        scale_factor:     Downsampling factor (used for adaptive delta)
        degradation_type: "bicubic", "gaussian_blur", or "inpainting"
        target_size_hw:   (H, W) of the full-res image (required for bicubic;
                          inferred from y for blur/inpainting)
        disable_proximal: If True, set delta_k=0 to disable the proximal
                          operator (ablation test)
        device:           Torch device
        verbose:          Print per-iteration progress

    Returns:
        x:  Reconstructed image  [B, C, H, W]  in [0, 1]
    """
    vae = pipe.vae
    s = vae.config.scaling_factor
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    timesteps = get_timesteps(N)
    print(f'Time Steps: {timesteps}')

    y_norm = (y * 2 - 1.0).float()

    B, C, H_y, W_y = y.shape
    if degradation_type == "bicubic":
        H, W = H_y * scale_factor, W_y * scale_factor
    elif target_size_hw is not None:
        H, W = target_size_hw
    else:
        H, W = H_y, W_y

    # x^(0): build initial estimate depending on degradation type
    x = build_initial_estimate(y, degradation_type, H, W)

    # Create unconditional (zero) embeddings
    dummy_text_embed = torch.zeros((B, 1, pipe.unet.config.cross_attention_dim), device=device)
    dummy_pooled_embed = torch.zeros((B, 1280), device=device)

    pbar = tqdm(enumerate(timesteps), total=len(timesteps),
                desc="LATINO Unconditional") if verbose else enumerate(timesteps)

    for k, t_k in pbar:
        # 1) Encode image to latent
        z = vae_encode(vae, x, s)

        # 2) Forward diffusion
        alpha_t = alphas_cumprod[t_k].to(z.device, z.dtype)
        z_t = forward_diffusion(z, alpha_t)
        z_t = z_t.to(dtype=pipe.unet.dtype)

        # 3) Denoise unconditionally
        z_0 = consistency_denoise(
            pipe, z_t, t_k, dummy_text_embed, dummy_pooled_embed,
            target_size=(H, W),
        )

        # 4) Decode latent
        u = vae_decode(vae, z_0, s)

        # 5) Proximal step
        u_norm = (u * 2 - 1).float()
        df = torch.norm(physics.A(u_norm) - y_norm).item()
        delta_k = get_delta(t_k, df, scale_factor=scale_factor)
        if disable_proximal:
            delta_k = 0.01          # ablation: diffusion prior only, no data fidelity
        prox_x_norm = physics.prox_l2(u_norm, y=y_norm, gamma=delta_k)
        x = ((prox_x_norm + 1) / 2).clamp(0.0, 1.0)

        if verbose:
            pbar.set_postfix(t=t_k, gamma=f"{delta_k:.4f}", df=f"{df:.4f}")

    return x

print("All components defined.")

# ============================================================
# Main Experiment
# ============================================================

print(f"\n{'='*60}")
print(f"  Degradation:      {DEGRADATION_TYPE}")
if DEGRADATION_TYPE == "bicubic":
    print(f"  Scale factor:     {SCALE_FACTOR}")
elif DEGRADATION_TYPE == "gaussian_blur":
    print(f"  Blur sigma:       {BLUR_SIGMA}")
elif DEGRADATION_TYPE == "inpainting":
    print(f"  Mask ratio:       {MASK_RATIO} (fraction kept)")
print(f"  Nonsense prompt:  {NONSENSE_PROMPT}  ('{ACTIVE_PROMPT}')")
print(f"  Disable proximal: {DISABLE_PROXIMAL}")
print(f"  Results dir:      {RESULT_DIR}")
print(f"{'='*60}\n")

# Build a short tag for plot titles so you can tell which test you're looking at
_mode_tag = ""
if NONSENSE_PROMPT:
    _mode_tag += " [nonsense prompt]"
if DISABLE_PROXIMAL:
    _mode_tag += " [no prox]"

image = Image.open(IMAGE_PATH).convert("RGB")
transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.CenterCrop((TARGET_SIZE, TARGET_SIZE)),
    transforms.ToTensor(),
])
gt = transform(image).unsqueeze(0).to(device)
print(f"Ground truth shape: {gt.shape}")

# Build the physics operator for the selected degradation
physics = build_physics(
    degradation_type=DEGRADATION_TYPE,
    img_size=(3, TARGET_SIZE, TARGET_SIZE),
    device=device,
    scale_factor=SCALE_FACTOR,
    blur_sigma=BLUR_SIGMA,
    mask_ratio=MASK_RATIO,
)

y = physics.A(gt)
if SIGMA_Y > 0:
    y = y + SIGMA_Y * torch.randn_like(y)
    y = y.clamp(0.0, 1.0)

print(f"Observation y shape: {y.shape}")

# Baseline: naive reconstruction (bicubic upsample for SR, raw y for others)
y_up = build_initial_estimate(y, DEGRADATION_TYPE, TARGET_SIZE, TARGET_SIZE)
baseline_psnr = compute_psnr(y_up, gt)
print(f"Baseline PSNR ({DEGRADATION_TYPE}): {baseline_psnr:.2f} dB")

# Visualise
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(gt[0].cpu().permute(1, 2, 0))
axes[0].set_title("Ground Truth")
axes[0].axis("off")
axes[1].imshow(y_up[0].cpu().permute(1, 2, 0))
axes[1].set_title(f"Degraded ({DEGRADATION_TYPE}) -- PSNR {baseline_psnr:.2f} dB")
axes[1].axis("off")
plt.tight_layout()
# plt.savefig(RESULT_DIR / "Degraded_Image.png", dpi=150, bbox_inches="tight")

pipe = load_pipeline(device=device)
print("Pipeline loaded.")
print(f"  Scheduler: {pipe.scheduler.__class__.__name__}")
print(f"  Prediction type: {pipe.scheduler.config.prediction_type}")
print(f"  VAE dtype: {pipe.vae.dtype}")
print(f"  UNet dtype: {pipe.unet.dtype}")

# ---- Conditional LATINO ----
N = 8
result = latino(
    pipe, y, physics,
    prompt=ACTIVE_PROMPT,
    N=N,
    sigma_y=SIGMA_Y,
    scale_factor=SCALE_FACTOR,
    degradation_type=DEGRADATION_TYPE,
    disable_proximal=DISABLE_PROXIMAL,
    device=device,
    verbose=True,
)

latino_psnr = compute_psnr(result, gt)
print(f"\nLATINO PSNR: {latino_psnr:.2f} dB  (baseline: {baseline_psnr:.2f} dB)")

# ---- Result validation plots (Conditional) ----
# Row 1: GT - LATINO Result  (should be small residual if LATINO works)
# Row 2: Degraded - A(LATINO Result)  (should be ~noise, proving data consistency)

gt_img  = gt[0].cpu().permute(1, 2, 0)
res_img = result[0].cpu().permute(1, 2, 0).clamp(0, 1)

# GT - LATINO result  (amplified for visibility)
amp = 1
diff_gt_latino = (gt_img - res_img).abs() * amp

# Degraded observation vs re-degraded LATINO result
#   y  = A(gt) + noise   and   A(result) should approximate y
#   so  y - A(result)  should look like pure noise
result_degraded = physics.A(result).cpu()
y_cpu = y.cpu()
# For bicubic, result_degraded and y are low-res; for blur/inpainting they are full-res
diff_degraded = (y_cpu - result_degraded).abs() * amp
diff_degraded_img = diff_degraded[0].permute(1, 2, 0)

# Also prepare displayable versions of y and A(result)
y_disp = build_initial_estimate(y_cpu, DEGRADATION_TYPE, TARGET_SIZE, TARGET_SIZE)
y_disp_img = y_disp[0].permute(1, 2, 0).clamp(0, 1)
res_deg_disp = build_initial_estimate(result_degraded, DEGRADATION_TYPE, TARGET_SIZE, TARGET_SIZE)
res_deg_disp_img = res_deg_disp[0].permute(1, 2, 0).clamp(0, 1)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f"Conditional LATINO  --  {DEGRADATION_TYPE}  (N={N}){_mode_tag}", fontsize=16)

# Row 1: GT vs LATINO result & their difference
axes[0, 0].imshow(gt_img)
axes[0, 0].set_title("Ground Truth")
axes[0, 0].axis("off")

axes[0, 1].imshow(res_img)
axes[0, 1].set_title(f"LATINO Result -- {latino_psnr:.2f} dB")
axes[0, 1].axis("off")

axes[0, 2].imshow(diff_gt_latino.clamp(0, 1))
axes[0, 2].set_title(f"|GT - LATINO| x{amp}")
axes[0, 2].axis("off")

# Row 2: Degraded obs vs re-degraded LATINO & their difference (should be noise)
axes[1, 0].imshow(y_disp_img)
axes[1, 0].set_title("Degraded (observation y)")
axes[1, 0].axis("off")

axes[1, 1].imshow(res_deg_disp_img)
axes[1, 1].set_title("A(LATINO Result)")
axes[1, 1].axis("off")

axes[1, 2].imshow(diff_degraded_img.clamp(0, 1))
axes[1, 2].set_title(f"|y - A(LATINO)| x{amp}  (should be noise)")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig(RESULT_DIR / "LATINO_Results_Prompt_Conditional.png", dpi=150, bbox_inches="tight")
print(f"Saved comparison to {RESULT_DIR}/LATINO_Results_Prompt_Conditional.png")

# save_image(gt, RESULT_DIR / "ground_truth.png")
# save_image(y_disp.clamp(0, 1), RESULT_DIR / f"degraded_{DEGRADATION_TYPE}.png")
# save_image(result.clamp(0, 1), RESULT_DIR / "latino_result.png")
# print(f"Images saved to {RESULT_DIR}/")

# ---- Unconditional LATINO ----
N = 8
result_unc = latino_unconditional(
    pipe, y, physics,
    N=N,
    sigma_y=SIGMA_Y,
    scale_factor=SCALE_FACTOR,
    degradation_type=DEGRADATION_TYPE,
    disable_proximal=DISABLE_PROXIMAL,
    device=device,
    verbose=True,
)

latino_unc_psnr = compute_psnr(result_unc, gt)
print(f"\nLATINO (unconditional) PSNR: {latino_unc_psnr:.2f} dB  (baseline: {baseline_psnr:.2f} dB)")

# ---- Result validation plots (Unconditional) ----
res_unc_img = result_unc[0].cpu().permute(1, 2, 0).clamp(0, 1)

diff_gt_latino_unc = (gt_img - res_unc_img).abs() * amp

result_unc_degraded = physics.A(result_unc).cpu()
diff_degraded_unc = (y_cpu - result_unc_degraded).abs() * amp
diff_degraded_unc_img = diff_degraded_unc[0].permute(1, 2, 0)

res_unc_deg_disp = build_initial_estimate(result_unc_degraded, DEGRADATION_TYPE, TARGET_SIZE, TARGET_SIZE)
res_unc_deg_disp_img = res_unc_deg_disp[0].permute(1, 2, 0).clamp(0, 1)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f"Unconditional LATINO  --  {DEGRADATION_TYPE}  (N={N}){_mode_tag}", fontsize=16)

# Row 1: GT vs LATINO result & their difference
axes[0, 0].imshow(gt_img)
axes[0, 0].set_title("Ground Truth")
axes[0, 0].axis("off")

axes[0, 1].imshow(res_unc_img)
axes[0, 1].set_title(f"LATINO Result -- {latino_unc_psnr:.2f} dB")
axes[0, 1].axis("off")

axes[0, 2].imshow(diff_gt_latino_unc.clamp(0, 1))
axes[0, 2].set_title(f"|GT - LATINO| x{amp}")
axes[0, 2].axis("off")

# Row 2: Degraded obs vs re-degraded LATINO & their difference (should be noise)
axes[1, 0].imshow(y_disp_img)
axes[1, 0].set_title("Degraded (observation y)")
axes[1, 0].axis("off")

axes[1, 1].imshow(res_unc_deg_disp_img)
axes[1, 1].set_title("A(LATINO Result)")
axes[1, 1].axis("off")

axes[1, 2].imshow(diff_degraded_unc_img.clamp(0, 1))
axes[1, 2].set_title(f"|y - A(LATINO)| x{amp}  (should be noise)")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig(RESULT_DIR / "LATINO_Results_Unconditional.png", dpi=150, bbox_inches="tight")
print(f"Saved comparison to {RESULT_DIR}/LATINO_Results_Unconditional.png")

# save_image(result_unc.clamp(0, 1), RESULT_DIR / "latino_result_unconditional.png")
# print(f"All images saved to {RESULT_DIR}/")