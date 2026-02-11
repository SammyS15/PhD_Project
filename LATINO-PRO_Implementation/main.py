'''
LATINO Algorithm 1 from LATINO-PRO: LAtent consisTency INverse sOlver

given
    - x^(0) = A^{†}y
    - text prompt c
    - number of step N = 4 or 8
    - latent consistency model Gθ
    - latent space decoder D
    - latent space encoder E
    - sequences {t_{k}, δ_{k}}^{N}_{k=1}.

for k = 1, . . . , N do
    ϵ ∼ N(0, Id)
    z_{t_{k}}^{(k)} ← √(α_{t_{k}}) E(x^{(k−1)}) + √(1 − α_{t_{k}}) ϵ   ▷ Encode
    u^{(k)} ← D(Gθ (z_{t_{k}}^{(k)}, t_{k}, c))                          ▷ Decode
    x(k) ← prox_{δ_{k} g_{y}} (u(k))                                      ▷ Proximal step
end for
return x^(N)
'''

import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    LCMScheduler,
)
from huggingface_hub import hf_hub_download
import deepinv as dinv
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image


# ============================================================
# Model Loading
# ============================================================

def load_pipeline(device="cuda"):
    """Load SDXL pipeline with DMD2 4-step distilled UNet and LCM scheduler."""
    # VAE with fp16-safe fix
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )

    # Load SDXL UNet architecture with DMD2 distilled weights
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    unet_config = UNet2DConditionModel.load_config(base_model_id, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
    unet.load_state_dict(
        torch.load(
            hf_hub_download("tianweiy/DMD2", "dmd2_sdxl_4step_unet_fp16.bin"),
            map_location=device,
            weights_only=True,
        )
    )

    # Build the full pipeline and swap in LCM scheduler
    pipe = DiffusionPipeline.from_pretrained(
        base_model_id,
        unet=unet,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Force VAE to float32 — SDXL VAE is numerically unstable in float16
    pipe.vae = pipe.vae.to(dtype=torch.float32)

    return pipe


# ============================================================
# VAE Encode / Decode
# ============================================================

def vae_encode(vae, x, scaling_factor):
    """Encode image to latent space.  x in [0, 1] → z."""
    x_norm = (2.0 * x - 1.0).to(dtype=torch.float32)
    z = vae.encode(x_norm).latent_dist.mean * scaling_factor
    return z


def vae_decode(vae, z, scaling_factor):
    """Decode latent to image space.  z → x in [0, 1]."""
    z_f32 = z.to(dtype=torch.float32)
    x = vae.decode(z_f32 / scaling_factor).sample
    x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


# ============================================================
# Diffusion Utilities
# ============================================================

def forward_diffusion(z, alpha_t):
    """Forward process: z_t = √α_t · z + √(1−α_t) · ε,  ε ∼ N(0, I)."""
    noise = torch.randn_like(z)
    return torch.sqrt(alpha_t) * z + torch.sqrt(1.0 - alpha_t) * noise


def consistency_denoise(pipe, z_t, timestep, prompt_embeds, pooled_prompt_embeds,
                        target_size=(1024, 1024)):
    """Single-step denoising via the consistency model (Tweedie formula)."""
    device = z_t.device
    dtype = z_t.dtype
    t = torch.tensor([timestep], device=device, dtype=torch.long)

    # SDXL requires additional conditioning: pooled text embeds + time ids
    add_time_ids = torch.tensor(
        [[target_size[0], target_size[1], 0, 0, target_size[0], target_size[1]]],
        device=device, dtype=dtype,
    )
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds.to(dtype=dtype),
        "time_ids": add_time_ids,
    }

    # UNet forward pass  →  noise prediction
    noise_pred = pipe.unet(
        z_t, t,
        encoder_hidden_states=prompt_embeds.to(dtype=dtype),
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # Apply Tweedie formula to get clean latent estimate z_0
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
    """Adaptive δ_k based on timestep and measurement error (from reference)."""
    if scale_factor <= 16:
        return 1.0 * df / 10.0 if t_k > 300 else 0.5 * df / 10.0
    else:
        return 1.5 * df / 10.0 if t_k > 300 else 3.0 * df / 10.0


def get_timesteps(N):
    """Evenly spaced timesteps from 999 down (matching reference implementation)."""
    step = 1000 // N
    return [999 - i * step for i in range(N)]


# ============================================================
# LATINO  —  Algorithm 1
# ============================================================

@torch.no_grad()
def latino(pipe, y, physics, prompt="a high quality photo", N=4,
           sigma_y=0.05, scale_factor=4, device="cuda", verbose=True):
    """
    LATINO: LAtent consisTency INverse sOlver.

    Args:
        pipe:         SDXL pipeline (DMD2 UNet + LCM scheduler)
        y:            Degraded observation  [B, C, H_low, W_low]  in [0, 1]
        physics:      deepinv forward operator  (e.g. Downsampling)
        prompt:       Text prompt for conditioning
        N:            Number of iterations (4 or 8)
        sigma_y:      Observation noise std-dev
        scale_factor: Downsampling factor (used for adaptive δ)
        device:       Torch device
        verbose:      Print per-iteration progress

    Returns:
        x:  Reconstructed image  [B, C, H, W]  in [0, 1]
    """
    vae = pipe.vae
    s = vae.config.scaling_factor                       # 0.13025 for SDXL
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    timesteps = get_timesteps(N)

    # Proximal step operates in [-1, 1] range (matching reference calibration)
    y_norm = (y * 2 - 1).float()
    sigma_y_norm = sigma_y * 2

    # Encode text prompt  (no classifier-free guidance for DMD2)
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt, device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    # Infer target image size from observation + scale factor
    B, C, H_low, W_low = y.shape
    H, W = H_low * scale_factor, W_low * scale_factor

    # ---- x^(0): initialise with bicubic upsampling of y ----
    x = F.interpolate(y, size=(H, W), mode="bicubic", align_corners=False)
    x = x.clamp(0.0, 1.0)

    pbar = tqdm(enumerate(timesteps), total=len(timesteps),
                desc="LATINO") if verbose else enumerate(timesteps)

    for k, t_k in pbar:
        # 1) Encode:  z = E(x^{k-1})
        z = vae_encode(vae, x, s)

        # 2) Forward diffusion:  z_{t_k} = √α_{t_k}·z + √(1−α_{t_k})·ε
        alpha_t = alphas_cumprod[t_k].to(z.device, z.dtype)
        z_t = forward_diffusion(z, alpha_t)

        # Cast to UNet dtype (float16 on CUDA) for the denoising step
        z_t = z_t.to(dtype=pipe.unet.dtype)

        # 3) Denoise:  z_0 = Gθ(z_{t_k}, t_k, c)   (single-step consistency model)
        z_0 = consistency_denoise(
            pipe, z_t, t_k, prompt_embeds, pooled_prompt_embeds,
            target_size=(H, W),
        )

        # 4) Decode:  u^{k} = D(z_0)
        u = vae_decode(vae, z_0, s)

        # 5) Proximal step:  x^{k} = prox_{δ_k · g_y}(u^{k})
        #    Operate in [-1, 1] range; use δ_k directly as gamma
        u_norm = (u * 2 - 1).float()

        df = torch.norm(physics.A(u_norm) - y_norm).item()
        delta_k = get_delta(t_k, df, scale_factor=scale_factor)

        prox_x_norm = physics.prox_l2(u_norm, y=y_norm, gamma=delta_k)
        x = ((prox_x_norm + 1) / 2).clamp(0.0, 1.0)

        if verbose:
            pbar.set_postfix(t=t_k, gamma=f"{delta_k:.4f}", df=f"{df:.4f}")

    return x


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

def main():
    parser = argparse.ArgumentParser(
        description="LATINO: LAtent consisTency INverse sOlver"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to ground-truth image")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--scale-factor", type=int, default=4,
                        help="Downsampling scale factor")
    parser.add_argument("--N", type=int, default=4, choices=[4, 8],
                        help="Number of LATINO iterations")
    parser.add_argument("--sigma-y", type=float, default=0.05,
                        help="Observation noise std-dev")
    parser.add_argument("--prompt", type=str, default="a high quality photo",
                        help="Text conditioning prompt")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ground-truth image ----
    # Snap target size to nearest valid: must be divisible by (scale_factor * 8)
    divisor = args.scale_factor * 8
    target_size = (1024 // divisor) * divisor
    print(f"Target image size: {target_size}x{target_size}  "
          f"(divisible by {divisor} = scale_factor×8)")

    image = Image.open(args.image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop((target_size, target_size)),
        transforms.ToTensor(),                     # → [0, 1]
    ])
    gt = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

    # ---- Forward model (bicubic downsampling) ----
    physics = dinv.physics.Downsampling(
        img_size=(3, target_size, target_size),
        factor=args.scale_factor,
        filter="bicubic",
        device=device,
    )

    # ---- Create degraded observation  y = A(x) + noise ----
    y = physics.A(gt)
    if args.sigma_y > 0:
        y = y + args.sigma_y * torch.randn_like(y)
        y = y.clamp(0.0, 1.0)

    # ---- Load SDXL + DMD2 pipeline ----
    print("Loading SDXL + DMD2 pipeline...")
    pipe = load_pipeline(device=device)

    # ---- Run LATINO ----
    print(f"Running LATINO  (N={args.N}, scale={args.scale_factor}x, "
          f"sigma_y={args.sigma_y})")
    result = latino(
        pipe, y, physics,
        prompt=args.prompt,
        N=args.N,
        sigma_y=args.sigma_y,
        scale_factor=args.scale_factor,
        device=device,
    )

    # ---- Save outputs ----
    save_image(gt, output_dir / "ground_truth.png")
    y_up = F.interpolate(y, size=(1024, 1024), mode="bicubic", align_corners=False)
    save_image(y_up.clamp(0, 1), output_dir / "degraded_bicubic.png")
    save_image(result, output_dir / "latino_result.png")

    # ---- Evaluate ----
    psnr = compute_psnr(result, gt)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
