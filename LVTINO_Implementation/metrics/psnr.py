"""Peak Signal-to-Noise Ratio (PSNR) metric."""

import torch
from torch import Tensor


def psnr(
    img1: Tensor,
    img2: Tensor,
    max_val: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """Compute PSNR between two images.

    PSNR = 10 * log10(max_val^2 / MSE)

    Args:
        img1: First image tensor
        img2: Second image tensor (same shape as img1)
        max_val: Maximum pixel value (1.0 for [0,1] range, 255 for [0,255])
        eps: Small value to avoid log(0)

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)

    if mse < eps:
        return float('inf')

    psnr_val = 10.0 * torch.log10(max_val ** 2 / mse)

    return psnr_val.item()


def psnr_batch(
    imgs1: Tensor,
    imgs2: Tensor,
    max_val: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Compute PSNR for a batch of images.

    Args:
        imgs1: First batch [B, C, H, W]
        imgs2: Second batch [B, C, H, W]
        max_val: Maximum pixel value
        reduction: "mean", "sum", or "none"

    Returns:
        PSNR value(s)
    """
    # Compute MSE per image
    mse = torch.mean((imgs1 - imgs2) ** 2, dim=[1, 2, 3])

    # Compute PSNR
    psnr_vals = 10.0 * torch.log10(max_val ** 2 / (mse + 1e-10))

    if reduction == "mean":
        return psnr_vals.mean()
    elif reduction == "sum":
        return psnr_vals.sum()
    else:
        return psnr_vals
