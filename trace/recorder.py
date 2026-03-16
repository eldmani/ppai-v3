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

"""Arithmetic trace recorder for PPAI inference.

Hooks into the forward pass of PPAILinear layers and records
intermediate values (coefficients, outputs) at vector granularity.

The trace can be serialized and independently verified.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional
from contextlib import contextmanager

import torch
import numpy as np


@dataclass
class TraceStep:
    """One step in an arithmetic trace."""
    layer_id: str
    step_type: str             # "project", "matmul", "bias_add"
    input_hash: str            # SHA-256 of input to this step
    output_values: np.ndarray  # The actual output values
    output_hash: str           # SHA-256 of output
    weight_hash: Optional[str] = None  # For matmul steps
    bias_values: Optional[np.ndarray] = None  # For matmul steps with bias
    spec_params: Optional[dict] = None  # α, ψ, n_axes for project steps
    timestamp: float = 0.0


@dataclass
class InferenceTrace:
    """Complete trace of a single forward pass."""
    model_hash: str = ""
    input_hash: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    final_output_hash: str = ""
    total_ops: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def summary(self) -> dict:
        return {
            "model_hash": self.model_hash,
            "input_hash": self.input_hash,
            "final_output_hash": self.final_output_hash,
            "total_steps": len(self.steps),
            "total_ops": self.total_ops,
            "duration_ms": (self.end_time - self.start_time) * 1000,
        }


def _tensor_hash(t: torch.Tensor) -> str:
    """Deterministic hash of a tensor's float32 values."""
    data = t.detach().cpu().float().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _np_hash(a: np.ndarray) -> str:
    """Deterministic hash of a numpy array."""
    return hashlib.sha256(a.astype(np.float32).tobytes()).hexdigest()[:16]


class TraceRecorder:
    """Records arithmetic traces during PPAI model inference.

    Usage:
        recorder = TraceRecorder()
        with recorder.recording(model, input_tensor):
            output = model(input_tensor)
        trace = recorder.get_trace()
    """

    def __init__(self):
        self._trace: Optional[InferenceTrace] = None
        self._hooks: list = []
        self._recording = False

    def _make_hook(self, layer_name: str):
        """Create a trace hook for a specific PPAILinear layer."""
        def hook(layer, input_tensor, coefficients, output):
            if not self._recording or self._trace is None:
                return

            ts = time.time()

            # Step 1: Projection (x → coefficients)
            input_h = _tensor_hash(input_tensor)
            coeff_vals = coefficients.detach().cpu().float().numpy()
            coeff_h = _np_hash(coeff_vals)

            self._trace.steps.append(TraceStep(
                layer_id=layer_name,
                step_type="project",
                input_hash=input_h,
                output_values=coeff_vals,
                output_hash=coeff_h,
                spec_params=layer.spec.to_dict(),
                timestamp=ts,
            ))
            # ops: n_axes * d_in multiplies + (n_axes * (d_in-1)) adds
            n_elem = input_tensor.shape[-1]
            self._trace.total_ops += layer.spec.n_axes * (2 * n_elem - 1)

            # Step 2: Matmul (coefficients → output)
            out_vals = output.detach().cpu().float().numpy()
            out_h = _np_hash(out_vals)
            w_h = _tensor_hash(layer.weight)

            # Include bias info for verifier
            bias_vals = None
            if layer.bias is not None:
                bias_vals = layer.bias.detach().cpu().float().numpy()

            self._trace.steps.append(TraceStep(
                layer_id=layer_name,
                step_type="matmul",
                input_hash=coeff_h,
                output_values=out_vals,
                output_hash=out_h,
                weight_hash=w_h,
                bias_values=bias_vals,
                timestamp=ts,
            ))
            # ops: d_out * n_axes multiplies + d_out * (n_axes-1) adds
            self._trace.total_ops += layer.d_out * (2 * layer.spec.n_axes - 1)

            # Bias add ops (if applicable)
            if layer.bias is not None:
                self._trace.total_ops += layer.d_out

        return hook

    @contextmanager
    def recording(self, model: torch.nn.Module,
                  input_tensor: torch.Tensor,
                  model_hash: str = ""):
        """Context manager that records a trace during model forward pass.

        Args:
            model: PPAI-converted model.
            input_tensor: The input being fed to the model.
            model_hash: Optional hash identifying the model version.
        """
        from ..layers.linear import PPAILinear

        self._trace = InferenceTrace(
            model_hash=model_hash,
            input_hash=_tensor_hash(input_tensor),
            start_time=time.time(),
        )
        self._recording = True

        # Install trace hooks on all PPAILinear layers
        for name, module in model.named_modules():
            if isinstance(module, PPAILinear):
                hook_fn = self._make_hook(name)
                module._trace_hook = hook_fn
                self._hooks.append((module, hook_fn))

        try:
            yield self._trace
        finally:
            self._recording = False
            self._trace.end_time = time.time()

            # Set final output hash from last recorded step
            if self._trace.steps:
                self._trace.final_output_hash = self._trace.steps[-1].output_hash

            # Remove hooks
            for module, _ in self._hooks:
                module._trace_hook = None
            self._hooks.clear()

    def get_trace(self) -> Optional[InferenceTrace]:
        """Get the recorded trace after recording completes."""
        return self._trace
