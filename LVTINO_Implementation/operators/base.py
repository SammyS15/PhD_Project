"""Abstract base class for linear operators."""

from abc import ABC, abstractmethod
import torch
from torch import Tensor


class LinearOperator(ABC):
    """Abstract base class for linear operators.

    Linear operators must implement forward (A) and adjoint (A^T) operations.
    The adjoint must satisfy: <Ax, y> = <x, A^T y> for all x, y.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply forward operator A(x).

        Args:
            x: Input tensor

        Returns:
            A(x)
        """
        pass

    @abstractmethod
    def adjoint(self, y: Tensor) -> Tensor:
        """Apply adjoint operator A^T(y).

        Args:
            y: Input tensor

        Returns:
            A^T(y)
        """
        pass

    def __call__(self, x: Tensor) -> Tensor:
        """Alias for forward."""
        return self.forward(x)

    def T(self, y: Tensor) -> Tensor:
        """Alias for adjoint."""
        return self.adjoint(y)

    def normal(self, x: Tensor) -> Tensor:
        """Apply normal operator A^T A(x).

        Args:
            x: Input tensor

        Returns:
            A^T(A(x))
        """
        return self.adjoint(self.forward(x))

    @torch.no_grad()
    def test_adjoint(
        self,
        x_shape: tuple,
        y_shape: tuple,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ) -> tuple[bool, float]:
        """Test adjoint correctness: <Ax, y> should equal <x, A^T y>.

        Args:
            x_shape: Shape of input tensor x
            y_shape: Shape of output tensor y (result of forward)
            device: Device to run test on
            dtype: Data type for test tensors
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison

        Returns:
            Tuple of (passed: bool, relative_error: float)
        """
        # Generate random test vectors
        x = torch.randn(x_shape, device=device, dtype=dtype)
        y = torch.randn(y_shape, device=device, dtype=dtype)

        # Compute <Ax, y>
        Ax = self.forward(x)
        inner1 = torch.sum(Ax * y).item()

        # Compute <x, A^T y>
        ATy = self.adjoint(y)
        inner2 = torch.sum(x * ATy).item()

        # Compare
        abs_diff = abs(inner1 - inner2)
        max_val = max(abs(inner1), abs(inner2), 1e-8)
        rel_error = abs_diff / max_val

        passed = rel_error < rtol or abs_diff < atol

        return passed, rel_error
