"""Conjugate gradient solver for linear systems."""

import torch
from torch import Tensor
from typing import Callable, Optional


def conjugate_gradient(
    A_func: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Optional[Tensor] = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[Tensor, int, float]:
    """Conjugate gradient solver for Ax = b.

    Solves the linear system Ax = b where A is a symmetric positive definite
    operator provided as a function.

    For LATINO, we solve: (I + delta * A^T A) x = b
    where A_func computes (I + delta * A^T A)(x).

    Args:
        A_func: Function computing A(x) for the system matrix A
        b: Right-hand side vector
        x0: Initial guess (defaults to zeros)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance (relative residual)
        verbose: Print convergence info

    Returns:
        Tuple of (solution x, iterations used, final residual)
    """
    # Initialize
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    # Initial residual: r = b - Ax
    r = b - A_func(x)
    p = r.clone()

    # Initial residual norm
    r_norm_sq = torch.sum(r * r)
    r0_norm = torch.sqrt(r_norm_sq)

    if r0_norm < 1e-12:
        return x, 0, 0.0

    for k in range(max_iter):
        # Compute Ap
        Ap = A_func(p)

        # Step size: alpha = (r^T r) / (p^T Ap)
        pAp = torch.sum(p * Ap)

        if torch.abs(pAp) < 1e-12:
            # A is nearly singular in direction p
            break

        alpha = r_norm_sq / pAp

        # Update solution: x = x + alpha * p
        x = x + alpha * p

        # Update residual: r = r - alpha * Ap
        r = r - alpha * Ap

        # Check convergence
        r_norm_sq_new = torch.sum(r * r)
        rel_residual = torch.sqrt(r_norm_sq_new) / r0_norm

        if verbose:
            print(f"CG iter {k+1}: rel_residual = {rel_residual.item():.6e}")

        if rel_residual < tol:
            return x, k + 1, rel_residual.item()

        # Update search direction: p = r + beta * p
        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p

        r_norm_sq = r_norm_sq_new

    final_residual = torch.sqrt(r_norm_sq) / r0_norm
    return x, max_iter, final_residual.item()


def solve_proximal(
    operator,
    x_prior: Tensor,
    y: Tensor,
    delta: float,
    max_iter: int = 50,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tensor:
    """Solve the proximal operator for least squares data term.

    Solves: (I + delta * A^T A) x = x_prior + delta * A^T y

    This is equivalent to the proximal operator:
    prox_{delta * f}(x_prior) where f(x) = (1/2)||Ax - y||^2

    Args:
        operator: Linear operator with forward() and adjoint() methods
        x_prior: Prior estimate (from diffusion step)
        y: Observed measurement
        delta: Regularization parameter
        max_iter: Maximum CG iterations
        tol: Convergence tolerance
        verbose: Print convergence info

    Returns:
        Solution x
    """
    # Define A_func for CG: computes (I + delta * A^T A)(x)
    def A_func(x: Tensor) -> Tensor:
        return x + delta * operator.normal(x)

    # Compute right-hand side: b = x_prior + delta * A^T y
    b = x_prior + delta * operator.adjoint(y)

    # Solve using CG
    x, iterations, residual = conjugate_gradient(
        A_func, b, x0=x_prior, max_iter=max_iter, tol=tol, verbose=verbose
    )

    return x
