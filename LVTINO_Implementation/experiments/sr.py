"""Super-resolution experiment using LATINO solver."""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

from models.lcm import LCMWrapper, get_device
from models.vae_utils import test_vae_roundtrip
from operators.downsample import DownsampleOperator, GaussianDownsampleOperator
from solver.latino import LATINOSolver
from metrics.psnr import psnr
from metrics.ssim import ssim


# Minimum recommended image size for SDXL
MIN_IMAGE_SIZE = 256


def load_image(path: str, size: tuple = None) -> torch.Tensor:
    """Load image as tensor.

    Args:
        path: Image file path
        size: Optional (H, W) to resize to

    Returns:
        Image tensor [1, 3, H, W] in range [0, 1]
    """
    img = Image.open(path).convert("RGB")

    if size is not None:
        img = img.resize((size[1], size[0]), Image.LANCZOS)

    # Convert to tensor
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    return img_tensor


def save_image(tensor: torch.Tensor, path: str):
    """Save tensor as image.

    Args:
        tensor: Image tensor [1, 3, H, W] or [3, H, W] in range [0, 1]
        path: Output path
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Clamp and convert
    tensor = torch.clamp(tensor, 0.0, 1.0)
    img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    Image.fromarray(img_np).save(path)


def run_super_resolution(
    image_path: str,
    output_dir: str,
    scale_factor: int = 4,
    num_iterations: int = 4,
    delta: float = 1.0,
    timesteps: list = None,
    noise_level: float = 0.0,
    use_gaussian_degradation: bool = False,
    prompt: str = "",
    device: str = "auto",
    dtype: torch.dtype = None,
):
    """Run LATINO super-resolution experiment.

    Args:
        image_path: Path to high-resolution ground truth image
        output_dir: Output directory for results
        scale_factor: Downsampling factor
        num_iterations: Number of LATINO iterations
        delta: Data fidelity weight
        timesteps: Diffusion timesteps
        noise_level: Gaussian noise standard deviation
        use_gaussian_degradation: Use Gaussian blur + subsample
        prompt: Optional text prompt
        device: Computation device ("auto", "cuda", "mps", "cpu")
        dtype: Model dtype (auto-selected if None)
    """
    # Auto-detect device
    device = get_device(device)
    print(f"Running super-resolution experiment")
    print(f"  Device: {device}")
    print(f"  Image: {image_path}")
    print(f"  Scale factor: {scale_factor}x")
    print(f"  Iterations: {num_iterations}")
    print(f"  Delta: {delta}")
    print(f"  Noise level: {noise_level}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load ground truth image
    print("\nLoading image...")
    gt_image = load_image(image_path)

    # Check image size
    _, _, H, W = gt_image.shape
    if H < MIN_IMAGE_SIZE or W < MIN_IMAGE_SIZE:
        print(f"\n  WARNING: Image size {H}x{W} is smaller than recommended {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}")
        print(f"  SDXL works best with larger images. Consider upscaling first or using a different model.")
        print(f"  The VAE downsamples by 8x, so {H}x{W} -> {H//8}x{W//8} latent")

    # Ensure dimensions are divisible by scale factor and 8 (for VAE)
    _, _, H, W = gt_image.shape
    H_new = (H // (scale_factor * 8)) * (scale_factor * 8)
    W_new = (W // (scale_factor * 8)) * (scale_factor * 8)

    if H_new != H or W_new != W:
        print(f"  Cropping image from {H}x{W} to {H_new}x{W_new}")
        gt_image = gt_image[:, :, :H_new, :W_new]

    gt_image = gt_image.to(device)

    # Create degradation operator
    print("\nCreating degradation operator...")
    if use_gaussian_degradation:
        operator = GaussianDownsampleOperator(scale_factor=scale_factor)
        print(f"  Using Gaussian blur + {scale_factor}x subsample")
    else:
        operator = DownsampleOperator(scale_factor=scale_factor)
        print(f"  Using average pooling {scale_factor}x")

    # Test adjoint property
    print("\nTesting operator adjoint property...")
    x_shape = gt_image.shape
    y_shape = operator.get_output_shape(x_shape)
    passed, rel_error = operator.test_adjoint(x_shape, y_shape, device=device)
    print(f"  Adjoint test: {'PASSED' if passed else 'FAILED'} (rel_error={rel_error:.6e})")

    if not passed:
        print("  WARNING: Adjoint property not satisfied!")

    # Create degraded observation
    print("\nCreating degraded observation...")
    y = operator.forward(gt_image)

    if noise_level > 0:
        noise = torch.randn_like(y) * noise_level
        y = y + noise
        y = torch.clamp(y, 0.0, 1.0)
        print(f"  Added Gaussian noise with sigma={noise_level}")

    print(f"  Ground truth shape: {list(gt_image.shape)}")
    print(f"  Observation shape: {list(y.shape)}")

    # Bicubic baseline (compute on CPU as MPS doesn't support bicubic)
    print("\nComputing bicubic baseline...")
    bicubic = torch.nn.functional.interpolate(
        y.cpu(), scale_factor=scale_factor, mode='bicubic', align_corners=False
    ).to(device)
    bicubic = torch.clamp(bicubic, 0.0, 1.0)
    bicubic_psnr = psnr(bicubic, gt_image)
    bicubic_ssim = ssim(bicubic, gt_image)
    print(f"  Bicubic PSNR: {bicubic_psnr:.2f} dB")
    print(f"  Bicubic SSIM: {bicubic_ssim:.4f}")

    # Load LCM model
    print("\nLoading LCM model...")
    lcm = LCMWrapper(device=device, dtype=dtype)

    # Test VAE roundtrip
    print("\nTesting VAE roundtrip...")
    gt_fp32 = gt_image.float()
    with torch.no_grad():
        vae_recon, vae_mse = test_vae_roundtrip(lcm.vae, gt_fp32.to(dtype))
    print(f"  VAE roundtrip MSE: {vae_mse:.6f}")

    # Set up timesteps
    if timesteps is None:
        timesteps = [800, 600, 400, 200]

    # Create LATINO solver
    print("\nCreating LATINO solver...")
    solver = LATINOSolver(
        lcm_model=lcm,
        operator=operator,
        delta=delta,
        timesteps=timesteps,
        device=device,
    )

    # Run LATINO
    print("\nRunning LATINO solver...")

    # Callback to save intermediate results
    intermediates = []
    def callback(k, x):
        intermediates.append(x.cpu().clone())
        iter_psnr = psnr(x, gt_image)
        iter_ssim = ssim(x, gt_image)
        print(f"  Iteration {k+1}: PSNR={iter_psnr:.2f} dB, SSIM={iter_ssim:.4f}")

    result = solver.solve(
        y=y,
        num_iterations=num_iterations,
        prompt=prompt,
        callback=callback,
        verbose=True,
    )

    # Final metrics
    final_psnr = psnr(result, gt_image)
    final_ssim = ssim(result, gt_image)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Bicubic:  PSNR={bicubic_psnr:.2f} dB, SSIM={bicubic_ssim:.4f}")
    print(f"LATINO:   PSNR={final_psnr:.2f} dB, SSIM={final_ssim:.4f}")
    print(f"Improvement: +{final_psnr - bicubic_psnr:.2f} dB PSNR")
    print("=" * 50)

    # Save results
    print("\nSaving results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_image(gt_image, os.path.join(output_dir, f"{timestamp}_gt.png"))
    save_image(y, os.path.join(output_dir, f"{timestamp}_degraded.png"))
    save_image(bicubic, os.path.join(output_dir, f"{timestamp}_bicubic.png"))
    save_image(result, os.path.join(output_dir, f"{timestamp}_latino.png"))

    # Save intermediates
    for k, x in enumerate(intermediates):
        save_image(x, os.path.join(output_dir, f"{timestamp}_iter{k+1}.png"))

    # Save metrics
    metrics_path = os.path.join(output_dir, f"{timestamp}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Scale factor: {scale_factor}x\n")
        f.write(f"Iterations: {num_iterations}\n")
        f.write(f"Delta: {delta}\n")
        f.write(f"Timesteps: {timesteps}\n")
        f.write(f"Noise level: {noise_level}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"\n")
        f.write(f"Bicubic PSNR: {bicubic_psnr:.4f} dB\n")
        f.write(f"Bicubic SSIM: {bicubic_ssim:.6f}\n")
        f.write(f"LATINO PSNR: {final_psnr:.4f} dB\n")
        f.write(f"LATINO SSIM: {final_ssim:.6f}\n")

    print(f"Results saved to: {output_dir}")

    return {
        'bicubic_psnr': bicubic_psnr,
        'bicubic_ssim': bicubic_ssim,
        'latino_psnr': final_psnr,
        'latino_ssim': final_ssim,
    }


def main():
    parser = argparse.ArgumentParser(description="LATINO Super-Resolution Experiment")

    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--scale", type=int, default=4,
                        help="Super-resolution scale factor")
    parser.add_argument("--iterations", type=int, default=4,
                        help="Number of LATINO iterations")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="Data fidelity weight")
    parser.add_argument("--timesteps", type=int, nargs="+", default=None,
                        help="Diffusion timesteps")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Noise level for degradation")
    parser.add_argument("--gaussian-blur", action="store_true",
                        help="Use Gaussian blur degradation")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for conditioning")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cuda/cpu/mps)")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 instead of FP16 (recommended for MPS/CPU)")

    args = parser.parse_args()

    # Auto dtype: fp32 if requested OR if not on CUDA
    device = get_device(args.device)
    if args.fp32 or device != "cuda":
        dtype = torch.float32
    else:
        dtype = torch.float16

    run_super_resolution(
        image_path=args.image,
        output_dir=args.output,
        scale_factor=args.scale,
        num_iterations=args.iterations,
        delta=args.delta,
        timesteps=args.timesteps,
        noise_level=args.noise,
        use_gaussian_degradation=args.gaussian_blur,
        prompt=args.prompt,
        device=args.device,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
