"""Structural Similarity Index (SSIM) metric."""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import math


def _create_gaussian_kernel(
    kernel_size: int = 11,
    sigma: float = 1.5,
    channels: int = 3,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create Gaussian kernel for SSIM computation.

    Args:
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of Gaussian
        channels: Number of image channels
        device: Tensor device
        dtype: Tensor dtype

    Returns:
        Gaussian kernel [channels, 1, kernel_size, kernel_size]
    """
    # Create 1D Gaussian
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()

    # Create 2D kernel via outer product
    kernel = torch.outer(g, g)

    # Expand for channels
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size).contiguous()

    return kernel


def _ssim_per_channel(
    img1: Tensor,
    img2: Tensor,
    kernel: Tensor,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Compute SSIM for each channel.

    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        kernel: Gaussian kernel
        k1: SSIM constant
        k2: SSIM constant
        data_range: Value range of images

    Returns:
        Tuple of (SSIM map, contrast-structure term)
    """
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    channels = img1.shape[1]
    kernel_size = kernel.shape[-1]
    padding = kernel_size // 2

    # Compute means
    mu1 = F.conv2d(img1, kernel, padding=padding, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=padding, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=padding, groups=channels) - mu1_mu2

    # Clamp variances to avoid numerical issues
    sigma1_sq = torch.clamp(sigma1_sq, min=0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0)

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    # Contrast-structure term (for MS-SSIM)
    cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    return ssim_map, cs


def ssim(
    img1: Tensor,
    img2: Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 1.0,
) -> float:
    """Compute SSIM between two images.

    Args:
        img1: First image [B, C, H, W] or [C, H, W]
        img2: Second image (same shape as img1)
        kernel_size: Gaussian kernel size
        sigma: Gaussian standard deviation
        k1: SSIM constant
        k2: SSIM constant
        data_range: Value range of images (1.0 for [0,1], 255 for [0,255])

    Returns:
        SSIM value (scalar)
    """
    # Ensure 4D tensors
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    channels = img1.shape[1]

    # Create kernel
    kernel = _create_gaussian_kernel(
        kernel_size, sigma, channels, img1.device, img1.dtype
    )

    # Compute SSIM
    ssim_map, _ = _ssim_per_channel(img1, img2, kernel, k1, k2, data_range)

    # Average over spatial dimensions and channels
    return ssim_map.mean().item()


def ssim_batch(
    imgs1: Tensor,
    imgs2: Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Compute SSIM for a batch of images.

    Args:
        imgs1: First batch [B, C, H, W]
        imgs2: Second batch [B, C, H, W]
        kernel_size: Gaussian kernel size
        sigma: Gaussian standard deviation
        k1: SSIM constant
        k2: SSIM constant
        data_range: Value range
        reduction: "mean", "sum", or "none"

    Returns:
        SSIM value(s)
    """
    channels = imgs1.shape[1]

    kernel = _create_gaussian_kernel(
        kernel_size, sigma, channels, imgs1.device, imgs1.dtype
    )

    ssim_map, _ = _ssim_per_channel(imgs1, imgs2, kernel, k1, k2, data_range)

    # Average over spatial dimensions and channels for each image
    ssim_vals = ssim_map.mean(dim=[1, 2, 3])

    if reduction == "mean":
        return ssim_vals.mean()
    elif reduction == "sum":
        return ssim_vals.sum()
    else:
        return ssim_vals


def ms_ssim(
    img1: Tensor,
    img2: Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 1.0,
    weights: Tensor = None,
) -> float:
    """Compute Multi-Scale SSIM (MS-SSIM).

    Args:
        img1: First image [B, C, H, W]
        img2: Second image
        kernel_size: Gaussian kernel size
        sigma: Gaussian standard deviation
        k1: SSIM constant
        k2: SSIM constant
        data_range: Value range
        weights: Scale weights (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    Returns:
        MS-SSIM value
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    if weights is None:
        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            device=img1.device, dtype=img1.dtype
        )

    levels = len(weights)
    channels = img1.shape[1]

    mcs = []
    for i in range(levels):
        kernel = _create_gaussian_kernel(
            kernel_size, sigma, channels, img1.device, img1.dtype
        )

        ssim_map, cs = _ssim_per_channel(img1, img2, kernel, k1, k2, data_range)

        if i < levels - 1:
            mcs.append(cs.mean())
            # Downsample
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
        else:
            mcs.append(ssim_map.mean())

    mcs = torch.stack(mcs)

    # Compute weighted product
    ms_ssim_val = torch.prod(mcs ** weights)

    return ms_ssim_val.item()
