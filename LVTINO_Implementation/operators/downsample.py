"""Downsampling operator for super-resolution."""

import torch
import torch.nn.functional as F
from torch import Tensor
from .base import LinearOperator


class DownsampleOperator(LinearOperator):
    """Downsampling operator for super-resolution.

    Forward: Average pooling (downsampling)
    Adjoint: Bilinear upsampling (preserves adjoint property)

    The adjoint is scaled to satisfy <Ax, y> = <x, A^T y>.
    """

    def __init__(self, scale_factor: int = 4):
        """Initialize downsampling operator.

        Args:
            scale_factor: Downsampling factor (e.g., 4 for 4x SR)
        """
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        """Apply downsampling via average pooling.

        Args:
            x: High-resolution image [B, C, H, W]

        Returns:
            Low-resolution image [B, C, H//s, W//s]
        """
        return F.avg_pool2d(x, kernel_size=self.scale_factor, stride=self.scale_factor)

    def adjoint(self, y: Tensor) -> Tensor:
        """Apply adjoint (upsampling) operation.

        For average pooling with kernel size k, the adjoint is:
        - Upsample by factor k
        - Divide by k^2 (to preserve inner product)

        Args:
            y: Low-resolution image [B, C, H//s, W//s]

        Returns:
            Upsampled image [B, C, H, W]
        """
        # Upsample using bilinear interpolation
        x_up = F.interpolate(
            y,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False,
        )

        # Scale to preserve adjoint property
        # For average pooling: A^T = (1/k^2) * upsample
        # This ensures <Ax, y> = <x, A^T y>
        x_up = x_up / (self.scale_factor ** 2)

        return x_up

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Get output shape for given input shape.

        Args:
            input_shape: Input shape (B, C, H, W)

        Returns:
            Output shape (B, C, H//s, W//s)
        """
        B, C, H, W = input_shape
        return (B, C, H // self.scale_factor, W // self.scale_factor)

    def get_input_shape(self, output_shape: tuple) -> tuple:
        """Get input shape for given output shape.

        Args:
            output_shape: Output shape (B, C, H//s, W//s)

        Returns:
            Input shape (B, C, H, W)
        """
        B, C, H, W = output_shape
        return (B, C, H * self.scale_factor, W * self.scale_factor)


class GaussianDownsampleOperator(LinearOperator):
    """Downsampling with Gaussian blur followed by subsampling.

    This is a more realistic degradation model that applies:
    1. Gaussian blur
    2. Subsampling

    The adjoint applies the transpose operations.
    """

    def __init__(
        self,
        scale_factor: int = 4,
        kernel_size: int = 7,
        sigma: float = 1.6,
    ):
        """Initialize Gaussian downsampling operator.

        Args:
            scale_factor: Downsampling factor
            kernel_size: Gaussian kernel size
            sigma: Gaussian standard deviation
        """
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Create Gaussian kernel
        self.kernel = self._create_gaussian_kernel(kernel_size, sigma)

    def _create_gaussian_kernel(self, size: int, sigma: float) -> Tensor:
        """Create 2D Gaussian kernel.

        Args:
            size: Kernel size
            sigma: Standard deviation

        Returns:
            Gaussian kernel tensor [1, 1, size, size]
        """
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel = torch.outer(g, g)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def _get_kernel(self, x: Tensor) -> Tensor:
        """Get kernel on same device and expanded for channels."""
        kernel = self.kernel.to(x.device, x.dtype)
        # Expand for all channels
        C = x.shape[1]
        kernel = kernel.expand(C, 1, -1, -1)
        return kernel

    def forward(self, x: Tensor) -> Tensor:
        """Apply Gaussian blur + downsampling.

        Args:
            x: High-resolution image [B, C, H, W]

        Returns:
            Low-resolution image [B, C, H//s, W//s]
        """
        kernel = self._get_kernel(x)
        C = x.shape[1]

        # Apply Gaussian blur (groups=C for depthwise conv)
        pad = self.kernel_size // 2
        x_blur = F.conv2d(x, kernel, padding=pad, groups=C)

        # Subsample
        y = x_blur[:, :, ::self.scale_factor, ::self.scale_factor]

        return y

    def adjoint(self, y: Tensor) -> Tensor:
        """Apply adjoint: zero-insertion + Gaussian blur transpose.

        Args:
            y: Low-resolution image [B, C, H//s, W//s]

        Returns:
            Upsampled image [B, C, H, W]
        """
        B, C, H, W = y.shape
        kernel = self._get_kernel(y)

        # Zero-insertion upsampling
        x_up = torch.zeros(
            B, C, H * self.scale_factor, W * self.scale_factor,
            device=y.device, dtype=y.dtype
        )
        x_up[:, :, ::self.scale_factor, ::self.scale_factor] = y

        # Apply transposed Gaussian blur (same as blur for symmetric kernel)
        pad = self.kernel_size // 2
        x_adj = F.conv2d(x_up, kernel, padding=pad, groups=C)

        return x_adj
