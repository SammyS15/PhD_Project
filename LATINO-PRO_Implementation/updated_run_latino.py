"""
Updated version of run_LATINO & run_LATINO_16_scale_factor with corrections based on main_LATINO.py and noise_schemes.py.
- Kept the correct scale factor for the super resolution case (16, not 4)
- Updated the proximial operator to match the reference implementation (using forward_model.prox_l2 with proper gamma)
    - Also fixed the delta schedules
- Used pipe.scheduler.step() with the modified noise prediction eps_cond, instead of manually iterating on x
- Had incorrect timesteps. Now it is [999, 874, 749, 624, 499, 374, 249, 124] for N=8 which matches the repo
- I was starting from random noise latents (with optional y_noise init) instead of starting from the noisy version of the initial estimate. 
    This is because the reference code's noise_pred_cond_y function is designed to work with the scheduler's noise addition, and it computes the modified noise prediction based on the current latents. 
    Starting from the noisy version of the initial estimate would not be compatible with how noise_pred_cond_y computes the modified noise prediction, which is based on the current latents and the scheduler's timesteps. 
    By starting from random noise latents, we ensure that the algorithm follows the intended flow of using the scheduler's noise addition and the modified noise prediction correctly.
- Defined normalization for sigma_y incorrectly, it's fixed now

SXDL --> Stability AI models
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    LCMScheduler,
)
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import deepinv as dinv
from deepinv.physics.blur import gaussian_blur
from torchvision.utils import save_image
from torchvision import transforms

import argparse


# Get argument passed from command line
parser = argparse.ArgumentParser(description="Run LATINO with specified parameters.")
parser.add_argument(
    "--degradation-type",
    type=str,
    default="super_resolution_bicubic",
    choices=["super_resolution_bicubic", "deblurring_gaussian", "inpainting_squared_mask"],
    help='Degradation type (default: "super_resolution_bicubic")'
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed (default: random)"
)
args = parser.parse_args()

# ============================================================
# Configuration
# ============================================================

IMAGE_PATH   = "/home/sammys15/scratch/LATINO-PRO/samples/60007.png"
N            = 8                        # LATINO iterations (reference default)
SIGMA_Y      = 0.01                     # observation noise std-dev (matching sr_x16.yaml)
PROMPT       = "a photo of a face"      # text conditioning (matching face.yaml)
OUTPUT_DIR   = "/home/sammys15/scratch/PhD_Project/LATINO-PRO_Implementation/Updated_Output_Results"
SEED         = args.seed if args.seed is not None else np.random.randint(0, 100000)  # random seed

# ---- Degradation type ----
DEGRADATION_TYPE = args.degradation_type  # Options: "super_resolution_bicubic", "deblurring_gaussian", "inpainting_squared_mask"

SCALE_FACTOR       = 16                 # downsampling factor (matching sr_x16.yaml)

# Gaussian blur parameters (unused for SR, but available)
BLUR_SIGMA         = 3.0

# Inpainting parameters (unused for SR, but available)
MASK_SIZE          = 200                # size parameter for squared mask (reference default)

# ---- Initialization strategy ----
# "y_noise": encode A^T y -> add noise at t=999 (reference default)
# "y":       encode A^T y -> use directly as starting latent
# "random":  pure random noise (pipe.prepare_latents)
INIT_STRATEGY = "y_noise"


# ============================================================
# Model Loading (matches main_LATINO.py LATINO path)
# ============================================================

def load_pipeline(device="cuda"):
    """Load SDXL + DMD2 pipeline exactly as in main_LATINO.py."""
    BASE = "/home/sammys15/scratch/PhD_Project_Scratch/LATINO-PRO_Implementation"

    SDXL_BASE_PATH = os.path.join(BASE, "sdxl-base")
    VAE_PATH       = os.path.join(BASE, "sdxl-vae-fp16-fix")
    UNET_PATH      = os.path.join(BASE, "DMD2")

    # Load VAE (fp16, matching reference line 105)
    vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.float16)

    # Load UNet on meta device, then load DMD2 weights (reference lines 117-127)
    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(
            SDXL_BASE_PATH, subfolder="unet"
        ).to(torch.float16)

    state_dict_path = os.path.join(UNET_PATH, "dmd2_sdxl_4step_unet_fp16.bin")
    unet.load_state_dict(torch.load(state_dict_path), assign=True)
    unet.to(device)

    # Load pipeline (reference lines 130-135, without variant="fp16")
    pipe = DiffusionPipeline.from_pretrained(
        SDXL_BASE_PATH,
        unet=unet,
        vae=vae,
        torch_dtype=torch.float16
    ).to(device)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    return pipe


# ============================================================
# Image Loading (matches reference utils.py)
# ============================================================

def load_image_tensor(path):
    """Load image as [0,1] tensor, shape (1, C, H, W). Matches reference."""
    img = Image.open(path).convert("RGB")
    return torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def crop_to_multiple(x, m=8):
    """Crop to nearest multiple of m. Matches reference."""
    H, W = x.shape[-2:]
    return x[:, :, :H - H % m, :W - W % m]


# ============================================================
# Forward Model (matches reference inverse_problems.py)
# ============================================================

def build_physics(degradation_type, x_clean, device, scale_factor=16,
                  blur_sigma=3.0, sigma_y=0.01, mask_size=200):
    """
    Build the deepinv physics operator.
    Returns (forward_model, transpose_operator).
    Matches reference inverse_problems.py.
    """
    noise_model = dinv.physics.GaussianNoise(sigma=sigma_y)

    if degradation_type == "super_resolution_bicubic":
        forward_model = dinv.physics.Downsampling(
            img_size=x_clean.shape[1:],
            factor=scale_factor,
            device=device,
            noise_model=noise_model,
            filter="bicubic",
            padding="reflect",
        )
        transpose_operator = forward_model.A_adjoint
    elif degradation_type == "deblurring_gaussian":
        kernel = gaussian_blur(sigma=(blur_sigma, blur_sigma))
        forward_model = dinv.physics.BlurFFT(
            img_size=x_clean.shape[1:],
            filter=kernel,
            device=device,
            noise_model=noise_model,
        )
        transpose_operator = forward_model.A_adjoint
    elif degradation_type == "inpainting_squared_mask":
        # Matches reference inverse_problems.py inpainting branch
        B, C, H, W = x_clean.shape
        mask = torch.ones((1, H, W), device=device)
        size = mask_size
        mask[:, H // 2 - size // 5 - 35 : H // 2 + size // 5 - 35,
             W // 2 - 4 * size // 5 - 2 : W // 2 + 4 * size // 5 + 2] = 0
        forward_model = dinv.physics.Inpainting(
            tensor_size=x_clean.shape,
            mask=mask,
            noise_model=noise_model,
        ).to(device)
        transpose_operator = forward_model.A_adjoint
    else:
        raise ValueError(
            f"Unsupported degradation: '{degradation_type}'. "
            "Choose from: 'super_resolution_bicubic', 'deblurring_gaussian', "
            "'inpainting_squared_mask'"
        )

    return forward_model, transpose_operator


# ============================================================
# Initial Estimate (matches reference utils.py _get_x_init)
# ============================================================

def get_x_init(y_norm, forward_model, transpose_operator):
    """
    Compute initial estimate for x using A^T(y).
    Matches reference _get_x_init for SR case.
    y_norm is in [-1, 1] range.
    Returns x_init in [-1, 1] range.
    """
    def AT(x):
        return transpose_operator(x)

    def A(x):
        return forward_model(x)

    def ATA(x):
        return AT(A(x))

    b = AT(y_norm.clone().detach())
    b.requires_grad = False

    u = dinv.optim.utils.conjugate_gradient(
        ATA,
        b,
        max_iter=100,
        tol=1e-5,
        eps=1e-8,
    )

    x_init = b.clip(-1, 1)
    return x_init


# ============================================================
# Delta Schedule (matches reference noise_schemes.py noise_pred_cond_y)
# ============================================================

def get_delta(t, df, degradation_type, scale_factor=16, sigma_kernel=None,
              sigma_y=0.01):
    """
    Compute the delta hyperparameter exactly as in the reference
    noise_pred_cond_y function in noise_schemes.py.
    """
    if degradation_type == "super_resolution_bicubic":
        if scale_factor == 16:
            if t > 300:
                delta = 3 * df / 1e1
            else:
                delta = 2 * df / 1e1
        elif scale_factor == 32:
            if t > 300:
                delta = 1.5 * df / 1e1
            else:
                delta = 3 * df / 1e1
        else:
            # Fallback for other factors (e.g., scale_factor=4)
            if t > 300:
                delta = 3 * df / 1e1
            else:
                delta = 2 * df / 1e1
    elif degradation_type == "deblurring_gaussian":
        if sigma_kernel is not None and sigma_kernel < 10:
            if t > 400:
                delta = 5 * df / 1e4
            else:
                delta = 2 * df / 1e4
        else:
            if t > 400:
                delta = 7 * df / 1e4
            else:
                delta = 3 * df / 1e4
    elif degradation_type == "inpainting_squared_mask":
        if t > 500:
            delta = 1
        else:
            delta = 0.5
    else:
        if t > 200:
            delta = 0.01
        else:
            delta = 1
    return delta


# ============================================================
# LATINO Algorithm (faithful to main_LATINO.py + noise_schemes.py)
# ============================================================

@torch.no_grad()
def latino(pipe, y, forward_model, transpose_operator, x_clean,
           prompt="a photo of a face", N=8, sigma_y=0.01,
           scale_factor=16, degradation_type="super_resolution_bicubic",
           init_strategy="y_noise", seed=42, device="cuda", logdir=None):
    """
    LATINO algorithm, faithfully following main_LATINO.py.

    Key steps per iteration (from noise_pred_cond_y in noise_schemes.py):
      1. UNet predicts noise_uncond from current latents
      2. Tweedie: z0_pred = sqrt(1/alpha_t) * (z_t - sqrt(1-alpha_t) * noise_uncond)
      3. Decode: x = D(z0_pred / scaling_factor).clip(-1, 1)
      4. Proximal: prox_x = prox_l2(x, y, gamma=delta * var_x_zt / sigma_y^2)
      5. Re-encode: z0_prox = E(prox_x).mean * scaling_factor
      6. Modified noise pred: eps_cond = sqrt(1/(1-alpha_t)) * z_t - sqrt(alpha_t/(1-alpha_t)) * z0_prox
      7. Scheduler step: z_{t-1} = scheduler.step(eps_cond, t, z_t)

    This matches the reference where:
      - main_LATINO.py lines 427-448 call noise_pred_cond_y
      - noise_schemes.py lines 6-94 implement the proximal + re-encode + noise reparameterization
      - main_LATINO.py line 815 calls pipe.scheduler.step(noise_pred, timestep, latents)
    """

    s = pipe.vae.config.scaling_factor  # 0.13025 for SDXL

    # ---- Normalize y to [-1, 1] (reference main_LATINO.py line 353) ----
    y_norm = y * 2 - 1
    sigma_y_norm = sigma_y * 2  # reference line 354

    # ---- Encode prompt (reference lines 164-176) ----
    # Conditional embeddings
    text_embeddings, _, pooled_text_embeds, _ = pipe.encode_prompt(
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    # Unconditional embeddings: encode empty string "" (reference line 171-176)
    # NOTE: The reference passes "" not zero tensors!
    uncond_embeddings, _, uncond_pooled_text_embeds, _ = pipe.encode_prompt(
        "",
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    # ---- Set up image dimensions (reference lines 182-184) ----
    image_height = 1024
    image_width = 1024

    # ---- Prepare initial latents (reference lines 186-194) ----
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.unet.config.in_channels,
        height=image_height,
        width=image_width,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )

    # ---- Prepare added_cond_kwargs (reference lines 196-214) ----
    time_ids = pipe._get_add_time_ids(
        original_size=(image_height, image_width),
        crops_coords_top_left=(0, 0),
        target_size=(image_height, image_width),
        dtype=torch.float16,
        text_encoder_projection_dim=1280,
    ).to(device)

    added_cond_kwargs = {
        "text_embeds": pooled_text_embeds,
        "time_ids": time_ids,
    }

    # ---- Set timesteps (reference lines 217-223) ----
    num_inference_steps = 8
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    custom_timesteps = torch.tensor(
        [999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long
    )
    pipe.scheduler.timesteps = custom_timesteps

    # ---- Apply init strategy (reference lines 384-395) ----
    if init_strategy in ("y_noise", "y"):
        x_init = get_x_init(y_norm, forward_model, transpose_operator)

        if logdir:
            save_image(x_init * 0.5 + 0.5, os.path.join(logdir, "x_init.png"))

        # Encode x_init to latent (reference line 392)
        qz = pipe.vae.encode(x_init.clip(-1, 1).half())
        mu_z = qz.latent_dist.mean * s

        if init_strategy == "y_noise":
            # Add noise at t=999 (reference line 395)
            noise = torch.randn_like(mu_z)
            latents = pipe.scheduler.add_noise(
                mu_z, noise=noise, timesteps=torch.tensor([999])
            )
        else:
            # Use encoded latent directly (reference line 402)
            latents = mu_z

    # ---- Main loop (reference lines 425-815) ----
    results_log = []
    pbar = tqdm(enumerate(pipe.scheduler.timesteps), total=len(pipe.scheduler.timesteps),
                desc="LATINO")

    for i, timestep in pbar:
        # Step 1: UNet forward pass (reference lines 429-435)
        # The reference uses text_embeddings (conditional) for the UNet call,
        # NOT uncond_embeddings. This is the conditional noise prediction.
        text_embeddings_iter = text_embeddings.detach().requires_grad_(True)

        noise_uncond = pipe.unet(
            latents,
            timestep,
            encoder_hidden_states=text_embeddings_iter,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Step 2: Call the equivalent of noise_pred_cond_y
        # (reference noise_schemes.py lines 17-94)

        # 2a. Tweedie formula: z0_pred (reference noise_schemes.py line 20)
        alpha_t = pipe.scheduler.alphas_cumprod[timestep]
        z0_pred = torch.sqrt(1 / alpha_t) * (
            latents - torch.sqrt(1 - alpha_t) * noise_uncond
        )

        # 2b. Decode to image space (reference noise_schemes.py line 23)
        x = pipe.vae.decode(z0_pred / s).sample.clip(-1, 1)

        # 2c. Compute data fidelity norm (reference noise_schemes.py line 25)
        df = torch.norm(forward_model(x.float()) - y_norm).item()

        # 2d. Compute delta (reference noise_schemes.py lines 28-72)
        delta = get_delta(
            timestep.item(), df, degradation_type, scale_factor=scale_factor
        )

        # 2e. Compute gamma (reference noise_schemes.py line 75)
        # gamma = delta * var_x_zt / sigma_y^2
        # where var_x_zt = 1 - alpha_t
        var_x_zt = 1 - alpha_t
        gamma = delta * var_x_zt / (sigma_y_norm ** 2)
        gamma = gamma.to(device=latents.device)

        # 2f. Proximal step (reference noise_schemes.py line 77)
        prox_x = forward_model.prox_l2(x.float(), y=y_norm, gamma=gamma)

        # 2g. Re-encode to latent (reference noise_schemes.py lines 80-81)
        qz = pipe.vae.encode(prox_x.clip(-1, 1).half())
        mu_z = qz.latent_dist.mean * s
        z0_pred_cond_y = mu_z

        # 2h. Modified noise prediction (reference noise_schemes.py line 85)
        noise_pred_cond_y = (
            torch.sqrt(1 / (1 - alpha_t)) * latents
            - torch.sqrt(alpha_t / (1 - alpha_t)) * z0_pred_cond_y
        )

        # Step 3: Scheduler step (reference main_LATINO.py line 815)
        latents = pipe.scheduler.step(
            noise_pred_cond_y, timestep, latents
        ).prev_sample

        # Log
        pbar.set_postfix(t=timestep.item(), delta=f"{delta:.4f}", df=f"{df:.4f}")

        if logdir:
            logdir_iter = os.path.join(logdir, "iter")
            os.makedirs(logdir_iter, exist_ok=True)
            save_image(
                torch.clamp(x * 0.5 + 0.5, 0, 1),
                os.path.join(logdir_iter, f"{timestep.item():3d}_x.png"),
            )
            save_image(
                torch.clamp(prox_x * 0.5 + 0.5, 0, 1),
                os.path.join(logdir_iter, f"{timestep.item():3d}_prox.png"),
            )

    # ---- Final decode (reference main_LATINO.py lines 840-844) ----
    decoded_image = pipe.vae.decode(latents / s).sample
    restored_x = (decoded_image / 2 + 0.5).clamp(0, 1)

    return restored_x


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
# Main
# ============================================================

def build_display_image(y, degradation_type, target_h, target_w):
    """
    Create a displayable version of the observation y.
    For SR the observation is low-res so we bicubic-upsample for display.
    For blur / inpainting y is already full-res.
    """
    if degradation_type == "super_resolution_bicubic":
        return F.interpolate(
            y, size=(target_h, target_w), mode="bicubic", align_corners=False
        ).clamp(0, 1)
    else:
        return y.clone().clamp(0, 1)


def plot_results(gt, result, y, forward_model, degradation_type, psnr_val,
                 target_h, target_w, save_path):
    """
    2x3 validation plot (matches the layout from run_latino.py).

    Row 1: Ground Truth | LATINO Result | |GT - Result| (amplified)
    Row 2: Degraded obs | A(Result)     | |y - A(Result)| (should be noise)
    """
    # Cast everything to float32 on CPU to avoid fp16 issues with
    # both deepinv operators and matplotlib's imshow
    gt_f32     = gt.float().cpu()
    result_f32 = result.float().cpu()
    y_f32      = y.float().cpu()

    gt_img  = gt_f32[0].permute(1, 2, 0).clamp(0, 1)
    res_img = result_f32[0].permute(1, 2, 0).clamp(0, 1)

    # |GT - LATINO|
    amp1 = 1
    diff_gt = (gt_img - res_img).abs() * amp1

    # Re-degrade the LATINO result (must be float32 for physics operators)
    result_degraded = forward_model.A(result.float()).float().cpu()

    # |y - A(result)| -- this should look like noise if data-consistent
    amp2 = 5
    diff_degraded = (y_f32 - result_degraded).abs() * amp2
    diff_degraded_img = diff_degraded[0].permute(1, 2, 0)

    # Displayable versions (upsample low-res for SR)
    y_disp = build_display_image(y_f32, degradation_type, target_h, target_w)
    y_disp_img = y_disp[0].permute(1, 2, 0).clamp(0, 1)

    res_deg_disp = build_display_image(result_degraded, degradation_type, target_h, target_w)
    res_deg_disp_img = res_deg_disp[0].permute(1, 2, 0).clamp(0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"LATINO  --  {degradation_type}  (N={N}, PSNR={psnr_val:.2f} dB)",
        fontsize=16,
    )

    # Row 1
    axes[0, 0].imshow(gt_img.numpy())
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(res_img.numpy())
    axes[0, 1].set_title(f"LATINO Result -- {psnr_val:.2f} dB")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(diff_gt.clamp(0, 1).numpy())
    axes[0, 2].set_title(f"|GT - LATINO| x{amp1}")
    axes[0, 2].axis("off")

    # Row 2
    axes[1, 0].imshow(y_disp_img.numpy())
    axes[1, 0].set_title("Degraded (observation y)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(res_deg_disp_img.numpy())
    axes[1, 1].set_title("A(LATINO Result)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(diff_degraded_img.clamp(0, 1).numpy())
    axes[1, 2].set_title(f"|y - A(LATINO)| x{amp2}  (should be noise)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # ---- Seed everything (reference lines 67-75) ----
    import random
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---- Load image (reference lines 314-347) ----
    xtemp = load_image_tensor(IMAGE_PATH)
    x_clean = crop_to_multiple(xtemp, m=8).to(device)

    # Normalize to [0, 1] (reference line 347)
    x_clean = (x_clean - x_clean.min()) / (x_clean.max() - x_clean.min())
    print(f"Ground truth shape: {x_clean.shape}")

    # ---- Build physics (reference lines 350-352) ----
    forward_model, transpose_operator = build_physics(
        degradation_type=DEGRADATION_TYPE,
        x_clean=x_clean,
        device=device,
        scale_factor=SCALE_FACTOR,
        blur_sigma=BLUR_SIGMA,
        sigma_y=SIGMA_Y,
        mask_size=MASK_SIZE,
    )

    y = forward_model(x_clean)
    print(f"Observation y shape: {y.shape}")

    # ---- Output directory ----
    RESULT_DIR = Path(OUTPUT_DIR) / DEGRADATION_TYPE
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logdir = str(RESULT_DIR)

    # ---- Load pipeline ----
    pipe = load_pipeline(device=device)
    print("Pipeline loaded.")
    print(f"  Scheduler: {pipe.scheduler.__class__.__name__}")
    print(f"  Prediction type: {pipe.scheduler.config.prediction_type}")
    print(f"  VAE dtype: {pipe.vae.dtype}")
    print(f"  UNet dtype: {pipe.unet.dtype}")

    # ---- Run LATINO ----
    result = latino(
        pipe=pipe,
        y=y,
        forward_model=forward_model,
        transpose_operator=transpose_operator,
        x_clean=x_clean,
        prompt=PROMPT,
        N=N,
        sigma_y=SIGMA_Y,
        scale_factor=SCALE_FACTOR,
        degradation_type=DEGRADATION_TYPE,
        init_strategy=INIT_STRATEGY,
        seed=SEED,
        device=device,
        logdir=logdir,
    )

    # ---- Metrics (reference lines 856-880) ----
    latino_psnr = compute_psnr(result, x_clean)
    print(f"\nLATINO PSNR: {latino_psnr:.2f} dB")

    # ---- Save results (reference lines 852-854) ----
    y_norm = y * 2 - 1
    save_image(result, os.path.join(logdir, "restored.png"))
    save_image(((y_norm + 1) / 2).clamp(0, 1).detach().cpu(), os.path.join(logdir, "degraded.png"))
    save_image(x_clean.detach().cpu(), os.path.join(logdir, "clean.png"))

    # ---- 2x3 Validation Plot ----
    H, W = x_clean.shape[-2:]
    plot_results(
        gt=x_clean,
        result=result,
        y=y,
        forward_model=forward_model,
        degradation_type=DEGRADATION_TYPE,
        psnr_val=latino_psnr,
        target_h=H,
        target_w=W,
        save_path=os.path.join(logdir, "LATINO_Results.png"),
    )

    print(f"All results saved to {logdir}")
