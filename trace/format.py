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

"""Trace serialization — save and load arithmetic traces.

Supports two formats:
    - JSON: Human-readable, larger files. Good for debugging.
    - Binary (npz): Compact. Good for production.

v0.3.0 fix: NPZ format now correctly serializes bias_values, which is
required for verifying layers that have bias (e.g., qkv projections in
Qwen-family models).
"""

import json
from pathlib import Path

import numpy as np

from .recorder import InferenceTrace, TraceStep


def save_trace_json(trace: InferenceTrace, path: str):
    """Save trace to a JSON file."""
    data = {
        "version": 1,
        "model_hash": trace.model_hash,
        "input_hash": trace.input_hash,
        "final_output_hash": trace.final_output_hash,
        "total_ops": trace.total_ops,
        "start_time": trace.start_time,
        "end_time": trace.end_time,
        "steps": [],
    }
    for step in trace.steps:
        step_data = {
            "layer_id": step.layer_id,
            "step_type": step.step_type,
            "input_hash": step.input_hash,
            "output_hash": step.output_hash,
            "output_values": step.output_values.tolist(),
            "timestamp": step.timestamp,
        }
        if step.weight_hash:
            step_data["weight_hash"] = step.weight_hash
        if step.bias_values is not None:
            step_data["bias_values"] = step.bias_values.tolist()
        if step.spec_params:
            step_data["spec_params"] = step.spec_params
        data["steps"].append(step_data)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_trace_json(path: str) -> InferenceTrace:
    """Load trace from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    trace = InferenceTrace(
        model_hash=data["model_hash"],
        input_hash=data["input_hash"],
        final_output_hash=data.get("final_output_hash", ""),
        total_ops=data["total_ops"],
        start_time=data["start_time"],
        end_time=data["end_time"],
    )
    for s in data["steps"]:
        bias_vals = None
        if "bias_values" in s and s["bias_values"] is not None:
            bias_vals = np.array(s["bias_values"], dtype=np.float32)
        trace.steps.append(TraceStep(
            layer_id=s["layer_id"],
            step_type=s["step_type"],
            input_hash=s["input_hash"],
            output_hash=s["output_hash"],
            output_values=np.array(s["output_values"], dtype=np.float32),
            weight_hash=s.get("weight_hash"),
            bias_values=bias_vals,
            spec_params=s.get("spec_params"),
            timestamp=s.get("timestamp", 0.0),
        ))

    return trace


def save_trace_npz(trace: InferenceTrace, path: str):
    """Save trace to a compact binary .npz file.

    Stores bias_values alongside output_values for each step that has bias,
    using a `has_bias` flag per step and `step_{i}_bias` arrays.
    """
    metadata = {
        "version": 1,
        "model_hash": trace.model_hash,
        "input_hash": trace.input_hash,
        "final_output_hash": trace.final_output_hash,
        "total_ops": trace.total_ops,
        "start_time": trace.start_time,
        "end_time": trace.end_time,
        "num_steps": len(trace.steps),
    }

    arrays = {"metadata_json": np.array(json.dumps(metadata))}

    # Save step metadata as JSON array and output values as numpy arrays
    step_metadata = []
    for i, step in enumerate(trace.steps):
        has_bias = step.bias_values is not None
        step_metadata.append({
            "layer_id": step.layer_id,
            "step_type": step.step_type,
            "input_hash": step.input_hash,
            "output_hash": step.output_hash,
            "weight_hash": step.weight_hash,
            "has_bias": has_bias,
            "spec_params": step.spec_params,
            "timestamp": step.timestamp,
        })
        arrays[f"step_{i}_output"] = step.output_values.astype(np.float32)
        if has_bias:
            arrays[f"step_{i}_bias"] = step.bias_values.astype(np.float32)

    arrays["steps_json"] = np.array(json.dumps(step_metadata))

    np.savez_compressed(path, **arrays)


def load_trace_npz(path: str) -> InferenceTrace:
    """Load trace from a .npz file."""
    data = np.load(path, allow_pickle=False)

    metadata = json.loads(str(data["metadata_json"]))
    step_metadata = json.loads(str(data["steps_json"]))

    trace = InferenceTrace(
        model_hash=metadata["model_hash"],
        input_hash=metadata["input_hash"],
        final_output_hash=metadata.get("final_output_hash", ""),
        total_ops=metadata["total_ops"],
        start_time=metadata["start_time"],
        end_time=metadata["end_time"],
    )

    for i, s in enumerate(step_metadata):
        bias_vals = None
        if s.get("has_bias", False):
            bias_key = f"step_{i}_bias"
            if bias_key in data:
                bias_vals = data[bias_key]

        trace.steps.append(TraceStep(
            layer_id=s["layer_id"],
            step_type=s["step_type"],
            input_hash=s["input_hash"],
            output_hash=s["output_hash"],
            output_values=data[f"step_{i}_output"],
            weight_hash=s.get("weight_hash"),
            bias_values=bias_vals,
            spec_params=s.get("spec_params"),
            timestamp=s.get("timestamp", 0.0),
        ))

    return trace


def save_trace(trace: InferenceTrace, path: str):
    """Save trace, auto-detecting format from extension."""
    if path.endswith(".json"):
        save_trace_json(trace, path)
    elif path.endswith(".npz"):
        save_trace_npz(trace, path)
    else:
        save_trace_json(trace, path)


def load_trace(path: str) -> InferenceTrace:
    """Load trace, auto-detecting format from extension."""
    if path.endswith(".npz"):
        return load_trace_npz(path)
    return load_trace_json(path)
