# PPAI — Polygonal Projection for Auditable Inference
# Copyright (C) 2026 Eldhose Mani. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# You may freely use, modify, and distribute this software, provided that:
#   - All derivative works are also licensed under AGPL-3.0
#   - You provide attribution to the original author
#   - If you offer this software (or derivatives) as a network service,
#     you must make the complete source code available to its users
# See the LICENSE file or <https://www.gnu.org/licenses/agpl-3.0.html>
# for the full license terms.
#
# Commercial License:
#   For use in proprietary software, SaaS platforms, or any context
#   where AGPL-3.0 obligations cannot be met, a separate commercial
#   license is required. Contact: eldhose.mani@hotmail.co.uk
#
# Unless you have obtained a commercial license, this file is governed
# by the AGPL-3.0 terms above.

"""Find optimal PPT projection parameters (α*, ψ*) for a weight matrix.

Uses SVD-guided initialization + local grid refinement to find the
angular span and rotation that minimize projection reconstruction error.
"""

import numpy as np
from .projection import build_ppt_projection, ProjectionSpec


def projection_error(W: np.ndarray, P: np.ndarray) -> float:
    """Compute Frobenius norm of projection reconstruction error.

    error = ||W - W @ P^T @ P||_F

    This measures how much information the projection loses.
    """
    W_reconstructed = W @ P.T @ P
    return float(np.linalg.norm(W - W_reconstructed, "fro"))


def relative_projection_error(W: np.ndarray, P: np.ndarray) -> float:
    """Compute relative projection error: ||W - W P^T P||_F / ||W||_F."""
    W_norm = np.linalg.norm(W, "fro")
    if W_norm < 1e-12:
        return 0.0
    return projection_error(W, P) / W_norm


def svd_guided_init(W: np.ndarray) -> tuple[float, float]:
    """Use SVD of W to get an initial (α, ψ) estimate.

    Strategy:
        - ψ aligns with the angle of the dominant right singular vector
        - α matches the angular spread between top-2 singular directions

    For d_in > 2, we use the first dimension pair's singular structure.
    """
    # Thin SVD — only need top-2 singular vectors
    # W is (d_out, d_in), right singular vectors are (d_in, d_in)
    # We only need V[:, :2]
    try:
        _, S, Vt = np.linalg.svd(W, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.pi / 2, 0.0  # fallback to default 90°

    if len(S) < 2:
        return np.pi / 2, 0.0

    # Use the first two right singular vectors
    v1 = Vt[0, :2] if Vt.shape[1] >= 2 else Vt[0]
    v2 = Vt[1, :2] if Vt.shape[1] >= 2 else Vt[1]

    # ψ = angle of dominant direction
    psi = float(np.arctan2(v1[1] if len(v1) > 1 else 0.0, v1[0]))

    # α = angle between top-2 directions
    cos_angle = np.clip(np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12), -1, 1)
    alpha = float(np.abs(np.arccos(cos_angle)))

    # Clamp α to valid range
    alpha = np.clip(alpha, 0.05, np.pi - 0.05)

    return alpha, psi


def find_optimal_angles(W: np.ndarray, n_axes: int,
                        grid_steps: int = 20) -> tuple[float, float]:
    """Find (α*, ψ*) that minimize projection error for weight matrix W.

    Algorithm:
        1. SVD-guided initialization for (α₀, ψ₀)
        2. Grid search in neighborhood of (α₀, ψ₀)
        3. Return best (α, ψ)

    Args:
        W: Weight matrix, shape (d_out, d_in).
        n_axes: Number of PPT axes.
        grid_steps: Resolution of grid search per dimension.

    Returns:
        (alpha, psi) — optimal angular span and rotation.
    """
    d_in = W.shape[1]

    # SVD-guided starting point
    alpha_init, psi_init = svd_guided_init(W)

    # Grid search around initialization
    # α range: ±40° around init, clamped to (0.05, π-0.05)
    alpha_lo = max(0.05, alpha_init - 0.7)
    alpha_hi = min(np.pi - 0.05, alpha_init + 0.7)
    alphas = np.linspace(alpha_lo, alpha_hi, grid_steps)

    # ψ range: ±45° around init
    psi_lo = psi_init - np.pi / 4
    psi_hi = psi_init + np.pi / 4
    psis = np.linspace(psi_lo, psi_hi, grid_steps)

    best_error = float("inf")
    best_alpha = alpha_init
    best_psi = psi_init

    for alpha in alphas:
        for psi in psis:
            P = build_ppt_projection(d_in, n_axes, alpha, psi)
            err = projection_error(W, P)
            if err < best_error:
                best_error = err
                best_alpha = float(alpha)
                best_psi = float(psi)

    return best_alpha, best_psi


def find_optimal_spec(W: np.ndarray, n_axes: int,
                      grid_steps: int = 20) -> ProjectionSpec:
    """Find optimal ProjectionSpec for a weight matrix.

    Convenience wrapper that returns a full ProjectionSpec.
    """
    d_in = W.shape[1]
    alpha, psi = find_optimal_angles(W, n_axes, grid_steps)
    return ProjectionSpec(d_in=d_in, n_axes=n_axes, alpha=alpha, psi=psi)


def compare_with_fixed_90(W: np.ndarray, n_axes: int) -> dict:
    """Compare adaptive (α*, ψ*) against fixed 90° projection.

    Returns dict with error metrics for both approaches.
    """
    d_in = W.shape[1]

    # Fixed 90°
    P_fixed = build_ppt_projection(d_in, n_axes, alpha=np.pi / 2, psi=0.0)
    err_fixed = relative_projection_error(W, P_fixed)

    # Adaptive
    alpha_opt, psi_opt = find_optimal_angles(W, n_axes)
    P_opt = build_ppt_projection(d_in, n_axes, alpha_opt, psi_opt)
    err_opt = relative_projection_error(W, P_opt)

    return {
        "fixed_90_error": err_fixed,
        "adaptive_error": err_opt,
        "improvement_pct": (err_fixed - err_opt) / max(err_fixed, 1e-12) * 100,
        "optimal_alpha_deg": np.degrees(alpha_opt),
        "optimal_psi_deg": np.degrees(psi_opt),
    }
