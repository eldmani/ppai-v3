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

"""PPAILinear — Drop-in replacement for nn.Linear with PPT-C compression.

Stores compressed weights W_comp ∈ R^{d_out × n_axes} and the projection
spec (α, ψ, n_axes, d_in). The projection matrix P is reconstructed from
the spec — it is NOT stored as a parameter.

Forward pass:  y = W_comp @ (P @ x) + bias
             = W_comp @ coefficients + bias

With tracing enabled, logs intermediate coefficient vectors.
"""

import weakref

import torch
import torch.nn as nn
import numpy as np
from ..core.projection import build_ppt_projection, ProjectionSpec


class PPAILinear(nn.Module):
    """Linear layer compressed via PPT projection.

    Replaces nn.Linear(d_in, d_out) with:
        1. Project input: c = P @ x       (d_in -> n_axes)
        2. Compressed mul: z = W_comp @ c  (n_axes -> d_out)
        3. Add bias: z = z + b

    P is deterministically reconstructed from (alpha, psi, n_axes, d_in).

    v3: P tensors are cached by spec with bounded size. When the cache
    exceeds max_cache_size, the least-recently-inserted entry is evicted.
    """

    # Class-level cache: (d_in, n_axes, alpha_rounded, psi_rounded) -> tensor
    _P_cache: dict = {}
    _P_cache_order: list = []  # insertion order for bounded eviction
    _P_cache_max_size: int = 64

    @classmethod
    def clear_P_cache(cls):
        """Clear the shared projection matrix cache."""
        cls._P_cache.clear()
        cls._P_cache_order.clear()

    @classmethod
    def set_cache_max_size(cls, max_size: int):
        """Set maximum number of cached projection matrices."""
        cls._P_cache_max_size = max_size

    @property
    def P(self):
        """Shared projection matrix, lazily moved to match weight device."""
        P = PPAILinear._P_cache[self._P_cache_key]
        target = self.weight.device
        if P.device != target:
            P = P.to(target)
            PPAILinear._P_cache[self._P_cache_key] = P
        return P

    def __init__(self, d_in: int, d_out: int, n_axes: int,
                 alpha: float, psi: float, bias: bool = True):
        super().__init__()
        self.spec = ProjectionSpec(d_in=d_in, n_axes=n_axes,
                                   alpha=alpha, psi=psi)
        self.d_out = d_out

        # Build or reuse cached projection matrix (deterministic from spec)
        cache_key = (d_in, n_axes, round(alpha, 10), round(psi, 10))
        if cache_key not in PPAILinear._P_cache:
            # Evict oldest entry if cache is full
            if len(PPAILinear._P_cache) >= PPAILinear._P_cache_max_size:
                oldest = PPAILinear._P_cache_order.pop(0)
                PPAILinear._P_cache.pop(oldest, None)
            P_np = build_ppt_projection(d_in, n_axes, alpha, psi)
            P_tensor = torch.from_numpy(P_np).float()
            PPAILinear._P_cache[cache_key] = P_tensor
            PPAILinear._P_cache_order.append(cache_key)
        self._P_cache_key = cache_key
        # P is NOT a buffer — accessed via property from shared cache
        # This avoids duplication during model.to(device)

        # Compressed weight — this is the trainable/stored parameter
        self.weight = nn.Parameter(torch.empty(d_out, n_axes))
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.bias = None

        # Trace hook — set externally by the trace recorder
        self._trace_hook = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, n_axes: int,
                    alpha: float, psi: float) -> "PPAILinear":
        """Create PPAILinear by compressing an existing nn.Linear.

        When n_axes < d_in (lossy):
            W_comp = W @ P^T  →  W_comp @ (P @ x) ≈ W @ x
        When n_axes == d_in (lossless):
            W_comp = W @ P^{-1}  →  W_comp @ (P @ x) = W @ x  (exact)
        """
        d_out, d_in = linear.weight.shape
        has_bias = linear.bias is not None

        layer = cls(d_in, d_out, n_axes, alpha, psi, bias=has_bias)

        with torch.no_grad():
            W = linear.weight.float()  # (d_out, d_in)
            P = layer.P                # (n_axes, d_in)

            # When n_axes == d_in, P is orthogonal (QR'd in construction)
            # so P^{-1} = P^T, and W @ P^T is exact. Same formula for both.
            W_comp = W @ P.T  # (d_out, n_axes)

            layer.weight.copy_(W_comp)

            if has_bias and linear.bias is not None:
                layer.bias.copy_(linear.bias.float())

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional trace logging.

        Args:
            x: Input tensor of shape (..., d_in).

        Returns:
            Output tensor of shape (..., d_out).
        """
        # Step 1: Project to coefficient space
        coefficients = x @ self.P.T  # (..., n_axes)

        # Step 2: Compressed linear transform
        z = coefficients @ self.weight.T  # (..., d_out)

        # Step 3: Add bias
        if self.bias is not None:
            z = z + self.bias

        # Trace logging
        if self._trace_hook is not None:
            self._trace_hook(
                layer=self,
                input_tensor=x,
                coefficients=coefficients,
                output=z,
            )

        return z

    def extra_repr(self) -> str:
        return (
            f"d_in={self.spec.d_in}, d_out={self.d_out}, "
            f"n_axes={self.spec.n_axes}, "
            f"alpha={self.spec.alpha:.4f}, psi={self.spec.psi:.4f}, "
            f"bias={self.bias is not None}"
        )

    def compression_ratio(self) -> float:
        """Ratio of original parameters to compressed parameters."""
        original = self.spec.d_in * self.d_out
        compressed = self.spec.n_axes * self.d_out
        if self.bias is not None:
            original += self.d_out
            compressed += self.d_out
        return original / compressed
