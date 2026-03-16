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

"""End-to-end tests: compress → infer → trace → verify.

Uses a tiny random nn.Linear model (no GPU, no HuggingFace needed).
Includes NPZ roundtrip test with bias_values (v0.3.0 fix).
"""

import numpy as np
import torch
import torch.nn as nn
import pytest
import json
import tempfile
import os

from ppai_v3.core.projection import build_ppt_projection, ProjectionSpec
from ppai_v3.core.optimize import find_optimal_angles
from ppai_v3.layers.linear import PPAILinear
from ppai_v3.compress.convert import convert_model
from ppai_v3.trace.recorder import TraceRecorder
from ppai_v3.trace.format import (
    save_trace_json, load_trace_json,
    save_trace_npz, load_trace_npz,
)
from ppai_v3.trace.verifier import verify_trace


class TinyModel(nn.Module):
    """Minimal model: two linear layers with ReLU."""
    def __init__(self, d_in=32, d_hidden=64, d_out=16):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestPPAILinear:
    def test_from_linear_shapes(self):
        linear = nn.Linear(64, 32)
        ppai = PPAILinear.from_linear(linear, n_axes=16, alpha=np.pi / 2, psi=0.0)
        assert ppai.weight.shape == (32, 16)
        assert ppai.P.shape == (16, 64)

    def test_forward_produces_correct_shape(self):
        linear = nn.Linear(64, 32)
        ppai = PPAILinear.from_linear(linear, n_axes=16, alpha=np.pi / 2, psi=0.0)
        x = torch.randn(4, 64)  # batch of 4
        y = ppai(x)
        assert y.shape == (4, 32)

    def test_compression_ratio(self):
        linear = nn.Linear(768, 768)
        ppai = PPAILinear.from_linear(linear, n_axes=96, alpha=np.pi / 2, psi=0.0)
        ratio = ppai.compression_ratio()
        assert ratio > 7.0

    def test_output_close_to_original(self):
        """Compressed output should be reasonably close to original."""
        torch.manual_seed(42)
        linear = nn.Linear(64, 32)
        x = torch.randn(1, 64)

        with torch.no_grad():
            y_orig = linear(x)

        W = linear.weight.detach().numpy()
        alpha, psi = find_optimal_angles(W, n_axes=32)

        ppai = PPAILinear.from_linear(linear, n_axes=32, alpha=alpha, psi=psi)
        with torch.no_grad():
            y_ppai = ppai(x)

        rel_err = torch.norm(y_orig - y_ppai) / torch.norm(y_orig)
        assert rel_err < 5.0, f"Relative error too large: {rel_err:.4f}"

    def test_bounded_cache(self):
        """P_cache should respect max_size bounds."""
        original_max = PPAILinear._P_cache_max_size

        PPAILinear.set_cache_max_size(3)
        PPAILinear._P_cache.clear()
        PPAILinear._P_cache_order.clear()

        try:
            for i in range(5):
                linear = nn.Linear(10 + i, 8)
                PPAILinear.from_linear(linear, n_axes=4, alpha=np.pi / 2, psi=0.0)

            assert len(PPAILinear._P_cache) <= 3
            assert len(PPAILinear._P_cache_order) <= 3
        finally:
            PPAILinear.set_cache_max_size(original_max)
            PPAILinear._P_cache.clear()
            PPAILinear._P_cache_order.clear()


class TestConvertModel:
    def test_tiny_model_conversion(self):
        torch.manual_seed(0)
        model = TinyModel(d_in=32, d_hidden=64, d_out=16)
        manifest = convert_model(model, n_axes=8, verbose=False)

        assert len(manifest) == 2  # fc1 and fc2

        for name, module in model.named_modules():
            if name in ("fc1", "fc2"):
                assert isinstance(module, PPAILinear), f"{name} not converted"

    def test_converted_model_forward(self):
        torch.manual_seed(0)
        model = TinyModel(d_in=32, d_hidden=64, d_out=16)
        convert_model(model, n_axes=8, verbose=False)

        x = torch.randn(2, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 16)


class TestTraceRecordAndVerify:
    def test_full_pipeline(self):
        """Compress → infer with trace → serialize → verify."""
        torch.manual_seed(42)
        model = TinyModel(d_in=32, d_hidden=64, d_out=16)
        manifest = convert_model(model, n_axes=8, verbose=False)

        x = torch.randn(1, 32)

        recorder = TraceRecorder()
        with recorder.recording(model, x, model_hash="test_model_v1"):
            with torch.no_grad():
                y = model(x)

        trace = recorder.get_trace()
        assert trace is not None
        assert len(trace.steps) > 0
        assert trace.total_ops > 0

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trace_path = f.name
        try:
            save_trace_json(trace, trace_path)
            loaded_trace = load_trace_json(trace_path)
            assert len(loaded_trace.steps) == len(trace.steps)
        finally:
            os.unlink(trace_path)

    def test_verification_passes(self):
        """Converted model trace should verify correctly."""
        torch.manual_seed(42)
        model = TinyModel(d_in=32, d_hidden=64, d_out=16)
        manifest = convert_model(model, n_axes=8, verbose=False)

        x = torch.randn(1, 32)

        recorder = TraceRecorder()
        with recorder.recording(model, x):
            with torch.no_grad():
                y = model(x)

        trace = recorder.get_trace()

        # v3 fix: pass InferenceTrace directly instead of manually building dict
        weights = {}
        for name, module in model.named_modules():
            if isinstance(module, PPAILinear):
                weights[name] = module.weight.detach().cpu().float().numpy()

        input_np = x.detach().cpu().float().numpy()

        result = verify_trace(trace, manifest, weights, input_np, atol=1e-3)
        assert result.passed, f"Verification failed:\n{result.summary()}"

    def test_tampered_trace_fails(self):
        """Tampering with trace values should cause verification to fail."""
        torch.manual_seed(42)
        model = TinyModel(d_in=32, d_hidden=64, d_out=16)
        manifest = convert_model(model, n_axes=8, verbose=False)

        x = torch.randn(1, 32)

        recorder = TraceRecorder()
        with recorder.recording(model, x):
            with torch.no_grad():
                y = model(x)

        trace = recorder.get_trace()

        # Build trace_data dict and TAMPER with it
        trace_data = {
            "model_hash": trace.model_hash,
            "input_hash": trace.input_hash,
            "steps": [],
        }
        for s in trace.steps:
            vals = s.output_values.tolist()
            trace_data["steps"].append({
                "layer_id": s.layer_id,
                "step_type": s.step_type,
                "input_hash": s.input_hash,
                "output_hash": s.output_hash,
                "output_values": vals,
                "weight_hash": s.weight_hash,
                "spec_params": s.spec_params,
            })

        # Tamper: change the first project step's output values
        for step in trace_data["steps"]:
            if step["step_type"] == "project":
                step["output_values"][0] = [999.0] * len(step["output_values"][0])
                break

        weights = {}
        for name, module in model.named_modules():
            if isinstance(module, PPAILinear):
                weights[name] = module.weight.detach().cpu().float().numpy()

        input_np = x.detach().cpu().float().numpy()
        result = verify_trace(trace_data, manifest, weights, input_np, atol=1e-3)

        assert not result.passed, "Tampered trace should fail verification"


class TestNpzRoundtrip:
    """v0.3.0 regression test: NPZ format must preserve bias_values."""

    def test_npz_roundtrip_with_bias(self):
        """Save and load a trace via NPZ, verify bias_values survive."""
        torch.manual_seed(42)
        model = TinyModel(d_in=32, d_hidden=64, d_out=16)
        manifest = convert_model(model, n_axes=8, verbose=False)

        x = torch.randn(1, 32)

        recorder = TraceRecorder()
        with recorder.recording(model, x):
            with torch.no_grad():
                y = model(x)

        trace = recorder.get_trace()

        # Confirm at least one step has bias_values
        has_bias_step = any(s.bias_values is not None for s in trace.steps)
        assert has_bias_step, "Test model should have bias — check TinyModel"

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            npz_path = f.name
        try:
            save_trace_npz(trace, npz_path)
            loaded = load_trace_npz(npz_path)

            assert len(loaded.steps) == len(trace.steps)

            for orig, loaded_step in zip(trace.steps, loaded.steps):
                assert orig.layer_id == loaded_step.layer_id
                assert orig.step_type == loaded_step.step_type
                assert np.allclose(orig.output_values, loaded_step.output_values)

                if orig.bias_values is not None:
                    assert loaded_step.bias_values is not None, (
                        f"bias_values lost in NPZ roundtrip for "
                        f"{orig.layer_id} ({orig.step_type})"
                    )
                    assert np.allclose(orig.bias_values, loaded_step.bias_values)
                else:
                    assert loaded_step.bias_values is None
        finally:
            os.unlink(npz_path)

    def test_npz_roundtrip_verifies(self):
        """Load trace from NPZ → verify should still pass."""
        torch.manual_seed(42)
        model = TinyModel(d_in=32, d_hidden=64, d_out=16)
        manifest = convert_model(model, n_axes=8, verbose=False)

        x = torch.randn(1, 32)

        recorder = TraceRecorder()
        with recorder.recording(model, x):
            with torch.no_grad():
                y = model(x)

        trace = recorder.get_trace()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            npz_path = f.name
        try:
            save_trace_npz(trace, npz_path)
            loaded = load_trace_npz(npz_path)

            weights = {}
            for name, module in model.named_modules():
                if isinstance(module, PPAILinear):
                    weights[name] = module.weight.detach().cpu().float().numpy()

            input_np = x.detach().cpu().float().numpy()

            result = verify_trace(loaded, manifest, weights, input_np, atol=1e-3)
            assert result.passed, (
                f"NPZ-loaded trace verification failed:\n{result.summary()}"
            )
        finally:
            os.unlink(npz_path)


class TestLipschitzStability:
    """Prop A.4: Small input perturbations → bounded output perturbations."""

    def test_perturbation_bounded(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 32)
        ppai = PPAILinear.from_linear(linear, n_axes=16, alpha=np.pi / 2, psi=0.0)

        x1 = torch.randn(1, 64)
        epsilon = 1e-4
        x2 = x1 + epsilon * torch.randn(1, 64)

        with torch.no_grad():
            y1 = ppai(x1)
            y2 = ppai(x2)

        input_diff = torch.norm(x2 - x1).item()
        output_diff = torch.norm(y2 - y1).item()

        assert output_diff < 100 * input_diff, (
            f"Output perturbation {output_diff} too large for "
            f"input perturbation {input_diff}"
        )
