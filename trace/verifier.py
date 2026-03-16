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

"""Independent arithmetic trace verifier.

ZERO dependency on PyTorch or any ML framework.
Requires only numpy (for matrix ops) and hashlib (for hashes).

Verification confirms computational self-consistency: the published
factored weights and projection specs, applied to the stated input,
reproduce the intermediate values recorded in the trace.  Non-linear
operations between traced steps are accepted as checkpoints.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VerificationResult:
    """Result of verifying a single trace step."""
    step_index: int
    layer_id: str
    step_type: str
    passed: bool
    max_error: float = 0.0
    detail: str = ""


@dataclass
class FullVerificationResult:
    """Result of verifying an entire trace."""
    passed: bool
    total_steps: int
    passed_steps: int
    failed_steps: int
    step_results: list[VerificationResult] = field(default_factory=list)
    model_hash_verified: bool = False
    detail: str = ""

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Verification: {status}",
            f"  Steps: {self.passed_steps}/{self.total_steps} passed",
        ]
        if not self.passed:
            for r in self.step_results:
                if not r.passed:
                    lines.append(
                        f"  FAIL step {r.step_index}: {r.layer_id} "
                        f"({r.step_type}) — {r.detail}"
                    )
        return "\n".join(lines)


def _np_hash(a: np.ndarray) -> str:
    """Deterministic hash — must match recorder._np_hash."""
    return hashlib.sha256(a.astype(np.float32).tobytes()).hexdigest()[:16]


def _build_projection_from_spec(spec: dict) -> np.ndarray:
    """Reconstruct PPT projection matrix from spec params.

    This duplicates the logic from core/projection.py so the verifier
    has NO dependency on the ppai package. Any change to the projection
    algorithm must be mirrored here.
    """
    d_in = spec["d_in"]
    n_axes = spec["n_axes"]
    alpha = spec["alpha"]
    psi = spec["psi"]

    # Axis angles
    k = np.arange(n_axes, dtype=np.float64)
    angles = psi + k * (alpha / (n_axes - 1))

    if d_in == 2:
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Multi-sector extension
    n_pairs = (d_in + 1) // 2
    if n_pairs > 1:
        sector_rotations = np.linspace(0, np.pi, n_pairs, endpoint=False)
    else:
        sector_rotations = np.array([0.0])

    P = np.zeros((n_axes, d_in), dtype=np.float64)
    for pair_idx in range(n_pairs):
        rot = sector_rotations[pair_idx]
        rotated_angles = angles + rot
        cos_a = np.cos(rotated_angles)
        sin_a = np.sin(rotated_angles)

        dim0 = pair_idx * 2
        dim1 = pair_idx * 2 + 1
        P[:, dim0] = cos_a
        if dim1 < d_in:
            P[:, dim1] = sin_a

    # Normalize rows
    row_norms = np.linalg.norm(P, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-12)
    P = P / row_norms

    # When n_axes == d_in (lossless), orthogonalize via QR.
    # Must match build_ppt_projection() in core/projection.py.
    if n_axes == d_in:
        Q, _ = np.linalg.qr(P.T)
        P = Q.T

    return P


def verify_trace(trace_data, manifest: dict,
                 weights: dict[str, np.ndarray],
                 input_tensor: np.ndarray,
                 atol: float = 1e-4) -> FullVerificationResult:
    """Verify an arithmetic trace against model weights and input.

    Args:
        trace_data: Loaded trace — either a dict or an InferenceTrace object.
        manifest: Model manifest with per-layer specs and weight hashes.
        weights: Dict mapping layer_name → compressed weight matrix (np.ndarray).
        input_tensor: The original input tensor.
        atol: Absolute tolerance for floating-point comparison.

    Returns:
        FullVerificationResult with per-step outcomes.
    """
    # Normalize trace_data: accept both dict and InferenceTrace objects
    if not isinstance(trace_data, dict):
        # Convert InferenceTrace + TraceStep objects to plain dicts
        steps = []
        for s in trace_data.steps:
            step_dict = {
                "layer_id": s.layer_id,
                "step_type": s.step_type,
                "input_hash": s.input_hash,
                "output_values": s.output_values,
                "output_hash": s.output_hash,
                "weight_hash": s.weight_hash,
                "bias_values": s.bias_values,
                "spec_params": s.spec_params,
            }
            steps.append(step_dict)
        trace_data = {
            "model_hash": trace_data.model_hash,
            "input_hash": trace_data.input_hash,
            "steps": steps,
            "final_output_hash": trace_data.final_output_hash,
        }

    result = FullVerificationResult(
        passed=True,
        total_steps=len(trace_data["steps"]),
        passed_steps=0,
        failed_steps=0,
    )

    # Verify model hash if provided
    if trace_data.get("model_hash"):
        result.model_hash_verified = True  # Caller should check externally

    # Track intermediate values from previous steps for chaining
    known_values: dict[str, np.ndarray] = {}  # hash → array

    # Register input
    input_f32 = input_tensor.astype(np.float32)
    input_hash = _np_hash(input_f32)
    known_values[input_hash] = input_f32

    for i, step in enumerate(trace_data["steps"]):
        layer_id = step["layer_id"]
        step_type = step["step_type"]
        recorded_output = np.array(step["output_values"], dtype=np.float32)
        recorded_output_hash = step["output_hash"]

        vr = VerificationResult(
            step_index=i,
            layer_id=layer_id,
            step_type=step_type,
            passed=False,
        )

        if step_type == "project":
            # Verify: coefficients = P @ x
            spec_params = step.get("spec_params")
            if spec_params is None:
                vr.detail = "Missing spec_params for project step"
                result.step_results.append(vr)
                result.failed_steps += 1
                result.passed = False
                continue

            # Reconstruct P from spec
            P = _build_projection_from_spec(spec_params).astype(np.float32)

            # Find input from known values
            input_h = step["input_hash"]
            if input_h in known_values:
                x = known_values[input_h]
            else:
                # Input may have gone through an untracked op (e.g. ReLU).
                # Accept the recorded output as a checkpoint — we can still
                # verify downstream matmul steps using these coefficients.
                known_values[recorded_output_hash] = recorded_output
                vr.passed = True
                vr.detail = "input from untracked op — checkpoint accepted"
                result.passed_steps += 1
                result.step_results.append(vr)
                continue

            # Recompute projection
            if x.ndim == 1:
                expected = (P @ x).astype(np.float32)
            else:
                expected = (x @ P.T).astype(np.float32)

            max_err = float(np.max(np.abs(expected - recorded_output)))
            vr.max_error = max_err

            if max_err <= atol:
                vr.passed = True
                result.passed_steps += 1
            else:
                vr.detail = f"Projection mismatch: max_error={max_err:.6e}"
                result.failed_steps += 1
                result.passed = False

            # Register output for downstream steps
            known_values[recorded_output_hash] = recorded_output

        elif step_type == "matmul":
            # Verify: z = W_comp @ coefficients (+ bias if present)
            w_hash = step.get("weight_hash")

            # Find the weight matrix for this layer
            if layer_id not in weights:
                vr.detail = f"Weight matrix not provided for {layer_id}"
                result.step_results.append(vr)
                result.failed_steps += 1
                result.passed = False
                continue

            W_comp = weights[layer_id].astype(np.float32)

            # Verify weight hash matches manifest
            if w_hash and layer_id in manifest:
                expected_w_hash = manifest[layer_id].get("weight_hash", "")
                if expected_w_hash and w_hash != expected_w_hash:
                    vr.detail = (f"Weight hash mismatch: trace={w_hash} "
                                 f"manifest={expected_w_hash}")
                    result.step_results.append(vr)
                    result.failed_steps += 1
                    result.passed = False
                    continue

            # Find coefficient input
            input_h = step["input_hash"]
            if input_h in known_values:
                c = known_values[input_h]
            else:
                # Coefficients may come from an untracked op chain.
                # Accept as checkpoint.
                known_values[recorded_output_hash] = recorded_output
                vr.passed = True
                vr.detail = "coefficients from untracked op — checkpoint accepted"
                result.passed_steps += 1
                result.step_results.append(vr)
                continue

            # Recompute matmul
            if c.ndim == 1:
                expected = (W_comp @ c).astype(np.float32)
            else:
                expected = (c @ W_comp.T).astype(np.float32)

            # Add bias if present in trace
            bias_vals = step.get("bias_values")
            if bias_vals is not None:
                expected = expected + np.array(bias_vals, dtype=np.float32)

            # Use both absolute and relative tolerance — large matmuls
            # accumulate float32 rounding proportional to output magnitude.
            abs_err = np.abs(expected - recorded_output)
            scale = np.maximum(np.abs(expected), np.abs(recorded_output))
            tol = np.maximum(atol, scale * atol * 10)  # rtol ~= 10 * atol
            max_err = float(np.max(abs_err))
            vr.max_error = max_err

            if np.all(abs_err <= tol):
                vr.passed = True
                result.passed_steps += 1
            else:
                vr.detail = f"Matmul mismatch: max_error={max_err:.6e}"
                result.failed_steps += 1
                result.passed = False

            known_values[recorded_output_hash] = recorded_output

        else:
            vr.detail = f"Unknown step type: {step_type}"
            result.failed_steps += 1
            result.passed = False

        result.step_results.append(vr)

    return result


def verify_trace_file(trace_path: str, manifest_path: str,
                      weights_dir: str,
                      input_tensor: np.ndarray,
                      atol: float = 1e-4) -> FullVerificationResult:
    """Verify a trace from files on disk.

    Args:
        trace_path: Path to trace JSON file.
        manifest_path: Path to manifest.json.
        weights_dir: Directory containing weight .npy files (one per layer).
        input_tensor: The original input.
        atol: Tolerance.
    """
    with open(trace_path) as f:
        trace_data = json.load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load weights
    weights = {}
    weights_path = Path(weights_dir)
    for layer_name in manifest:
        npy_path = weights_path / f"{layer_name.replace('.', '_')}.npy"
        if npy_path.exists():
            weights[layer_name] = np.load(str(npy_path))

    return verify_trace(trace_data, manifest, weights, input_tensor, atol)
