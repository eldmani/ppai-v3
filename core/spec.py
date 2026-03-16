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

"""Bit-exact activation function specifications.

Every nonlinear function used during PPAI inference is defined here
with a FIXED implementation. Both the inference engine and the verifier
import these same functions, ensuring bit-identical results.

IEEE 754 round-to-nearest-even is assumed throughout.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Scalar specifications (used by verifier — pure Python, no framework deps)
# ---------------------------------------------------------------------------

def relu_scalar(x: float) -> float:
    """ReLU: max(0, x)."""
    return max(0.0, x)


def gelu_approx_scalar(x: float) -> float:
    """GELU (tanh approximation) matching PyTorch's default.

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
    ))


def silu_scalar(x: float) -> float:
    """SiLU / Swish: x * sigmoid(x). Used by Llama/Mistral."""
    return x * (1.0 / (1.0 + math.exp(-x)))


def sigmoid_scalar(x: float) -> float:
    """Logistic sigmoid: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Vector specifications (used by inference engine — numpy)
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU: element-wise max(0, x)."""
    return np.maximum(0.0, x)


def gelu_approx(x: np.ndarray) -> np.ndarray:
    """GELU (tanh approximation)."""
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    ))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax with log-sum-exp stabilization.

    Evaluation order: subtract max, exponentiate, sum, divide.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm used by Llama/Mistral.

    rms_norm(x) = x / sqrt(mean(x^2) + eps) * weight
    """
    ms = np.mean(np.square(x), axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """LayerNorm used by GPT-2.

    layer_norm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


# ---------------------------------------------------------------------------
# Specification registry — maps name → (scalar_fn, vector_fn)
# ---------------------------------------------------------------------------

ACTIVATION_SPECS = {
    "relu": (relu_scalar, relu),
    "gelu_approx": (gelu_approx_scalar, gelu_approx),
    "gelu": (gelu_approx_scalar, gelu_approx),  # alias
    "silu": (silu_scalar, silu),
    "swish": (silu_scalar, silu),  # alias
}

NORM_SPECS = {
    "rms_norm": rms_norm,
    "layer_norm": layer_norm,
}


def get_activation(name: str):
    """Get (scalar_fn, vector_fn) for an activation by name."""
    if name not in ACTIVATION_SPECS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(ACTIVATION_SPECS.keys())}"
        )
    return ACTIVATION_SPECS[name]
