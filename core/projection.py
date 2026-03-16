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

"""PPT projection matrix construction.

Implements the core math from the Polygonal Projection Theorem:
- Angular axis partitioning with adaptive span α
- Multi-sector extension for d_in > 2
- Projection matrix P(α, ψ) construction

All projections are fully determined by (d_in, n_axes, alpha, psi),
enabling any verifier to reconstruct P from 4 scalars.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionSpec:
    """Immutable specification for a PPT projection.

    Stores the 4 scalars needed to reconstruct P exactly.
    """
    d_in: int
    n_axes: int
    alpha: float   # angular span in radians
    psi: float     # rotation offset in radians

    def to_dict(self) -> dict:
        return {
            "d_in": self.d_in,
            "n_axes": self.n_axes,
            "alpha": float(self.alpha),
            "psi": float(self.psi),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectionSpec":
        return cls(
            d_in=int(d["d_in"]),
            n_axes=int(d["n_axes"]),
            alpha=float(d["alpha"]),
            psi=float(d["psi"]),
        )


def build_axis_angles(n_axes: int, alpha: float, psi: float = 0.0) -> np.ndarray:
    """Compute PPT axis angles.

    θ_k = ψ + k * α / (n_axes - 1)  for k = 0, ..., n_axes - 1

    Args:
        n_axes: Number of PPT axes (≥ 2).
        alpha: Angular span in radians (0, π).
        psi: Rotation offset in radians.

    Returns:
        Array of shape (n_axes,) with axis angles in radians.
    """
    if n_axes < 2:
        raise ValueError(f"n_axes must be >= 2, got {n_axes}")
    if alpha <= 0 or alpha >= np.pi:
        raise ValueError(f"alpha must be in (0, π), got {alpha}")
    k = np.arange(n_axes, dtype=np.float64)
    return psi + k * (alpha / (n_axes - 1))


def build_unit_vectors(angles: np.ndarray) -> np.ndarray:
    """Build 2D unit direction vectors from angles.

    u_k = [cos(θ_k), sin(θ_k)]^T

    Args:
        angles: Array of shape (n_axes,) with axis angles.

    Returns:
        Array of shape (n_axes, 2) with unit vectors.
    """
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def build_ppt_projection(d_in: int, n_axes: int, alpha: float,
                         psi: float = 0.0) -> np.ndarray:
    """Build the PPT projection matrix P ∈ R^{n_axes × d_in}.

    For d_in = 2: P is simply the unit vectors as rows.
    For d_in > 2: Uses multi-sector extension (Prop A.6).
      - Partition d_in dims into ceil(d_in/2) pairs.
      - Each pair gets a rotated copy of the PPT wedge.
      - Sector rotations are evenly spaced to maximize rank.
      - Rows are normalized to have unit L2 norm.

    The projection is fully determined by (d_in, n_axes, alpha, psi).

    Args:
        d_in: Input dimension.
        n_axes: Number of PPT axes (output dimension).
        alpha: Angular span in radians.
        psi: Rotation offset in radians.

    Returns:
        P: Array of shape (n_axes, d_in), the projection matrix.
    """
    if d_in < 2:
        raise ValueError(f"d_in must be >= 2, got {d_in}")
    if n_axes < 2:
        raise ValueError(f"n_axes must be >= 2, got {n_axes}")
    if n_axes > d_in:
        raise ValueError(f"n_axes ({n_axes}) cannot exceed d_in ({d_in})")

    # PPT axis angles within the wedge
    angles = build_axis_angles(n_axes, alpha, psi)
    U = build_unit_vectors(angles)  # (n_axes, 2)

    if d_in == 2:
        return U

    # Multi-sector extension for d_in > 2
    n_pairs = (d_in + 1) // 2  # ceil(d_in / 2)

    # Sector rotation offsets — evenly spaced to maximize independence
    if n_pairs > 1:
        sector_rotations = np.linspace(0, np.pi, n_pairs, endpoint=False)
    else:
        sector_rotations = np.array([0.0])

    P = np.zeros((n_axes, d_in), dtype=np.float64)

    for pair_idx in range(n_pairs):
        rot = sector_rotations[pair_idx]
        # Rotated unit vectors for this sector
        rotated_angles = angles + rot
        cos_a = np.cos(rotated_angles)
        sin_a = np.sin(rotated_angles)

        dim0 = pair_idx * 2
        dim1 = pair_idx * 2 + 1

        P[:, dim0] = cos_a
        if dim1 < d_in:
            P[:, dim1] = sin_a

    # Normalize each row to unit L2 norm
    row_norms = np.linalg.norm(P, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-12)  # avoid division by zero
    P = P / row_norms

    # When n_axes == d_in (lossless), orthogonalize via QR.
    # This ensures P is orthogonal: P @ P^T = I, so P^{-1} = P^T exactly.
    # QR is deterministic from the same input matrix, so verifiers
    # can reconstruct the same orthogonal P from (alpha, psi).
    if n_axes == d_in:
        Q, _ = np.linalg.qr(P.T)  # Q is (d_in, d_in), orthogonal
        P = Q.T                    # (d_in, d_in), orthogonal rows

    return P


def build_projection_from_spec(spec: ProjectionSpec) -> np.ndarray:
    """Build projection matrix from a ProjectionSpec."""
    return build_ppt_projection(spec.d_in, spec.n_axes, spec.alpha, spec.psi)


def verify_projection_properties(P: np.ndarray, spec: ProjectionSpec,
                                 atol: float = 1e-10) -> dict:
    """Verify that a projection matrix satisfies PPT properties.

    Checks:
        1. Shape is (n_axes, d_in)
        2. Rows have unit norm
        3. Rows are pairwise distinct
        4. Reconstruction from spec produces identical matrix

    Returns:
        Dict with check names → bool.
    """
    results = {}

    # Shape check
    results["shape_correct"] = (
        P.shape == (spec.n_axes, spec.d_in)
    )

    # Unit norm rows
    row_norms = np.linalg.norm(P, axis=1)
    results["rows_unit_norm"] = bool(np.allclose(row_norms, 1.0, atol=atol))

    # Pairwise distinct rows
    n = P.shape[0]
    all_distinct = True
    for i in range(n):
        for j in range(i + 1, n):
            if np.allclose(P[i], P[j], atol=atol):
                all_distinct = False
                break
        if not all_distinct:
            break
    results["rows_pairwise_distinct"] = all_distinct

    # Reconstruction
    P_reconstructed = build_projection_from_spec(spec)
    results["reconstructs_exactly"] = bool(
        np.allclose(P, P_reconstructed, atol=atol)
    )

    return results
