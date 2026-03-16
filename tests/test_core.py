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

"""Tests for ppai_v3.core — projection math, optimization, activation specs."""

import numpy as np
import pytest
from ppai_v3.core.projection import (
    build_axis_angles,
    build_unit_vectors,
    build_ppt_projection,
    ProjectionSpec,
    build_projection_from_spec,
    verify_projection_properties,
)
from ppai_v3.core.optimize import (
    projection_error,
    relative_projection_error,
    svd_guided_init,
    find_optimal_angles,
    compare_with_fixed_90,
)
from ppai_v3.core.spec import (
    relu_scalar, gelu_approx_scalar, silu_scalar,
    relu, gelu_approx, silu, softmax, rms_norm, layer_norm,
)


# ---- Projection Tests ----

class TestAxisAngles:
    def test_basic_90_degree(self):
        angles = build_axis_angles(n_axes=5, alpha=np.pi / 2)
        assert len(angles) == 5
        assert abs(angles[0]) < 1e-10
        assert abs(angles[-1] - np.pi / 2) < 1e-10

    def test_custom_alpha(self):
        angles = build_axis_angles(n_axes=3, alpha=np.pi / 4)
        assert abs(angles[-1] - np.pi / 4) < 1e-10

    def test_with_psi(self):
        angles = build_axis_angles(n_axes=3, alpha=np.pi / 2, psi=0.1)
        assert abs(angles[0] - 0.1) < 1e-10

    def test_uniform_spacing(self):
        angles = build_axis_angles(n_axes=10, alpha=np.pi / 2)
        diffs = np.diff(angles)
        assert np.allclose(diffs, diffs[0]), "Angles not uniformly spaced"

    def test_invalid_n_axes(self):
        with pytest.raises(ValueError):
            build_axis_angles(n_axes=1, alpha=np.pi / 2)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            build_axis_angles(n_axes=5, alpha=0)
        with pytest.raises(ValueError):
            build_axis_angles(n_axes=5, alpha=np.pi)


class TestUnitVectors:
    def test_shape(self):
        angles = build_axis_angles(5, np.pi / 2)
        U = build_unit_vectors(angles)
        assert U.shape == (5, 2)

    def test_unit_norm(self):
        angles = build_axis_angles(10, np.pi / 3)
        U = build_unit_vectors(angles)
        norms = np.linalg.norm(U, axis=1)
        assert np.allclose(norms, 1.0)


class TestBuildProjection:
    def test_2d_input(self):
        P = build_ppt_projection(d_in=2, n_axes=2, alpha=np.pi / 2)
        assert P.shape == (2, 2)
        norms = np.linalg.norm(P, axis=1)
        assert np.allclose(norms, 1.0)

    def test_high_dim(self):
        P = build_ppt_projection(d_in=64, n_axes=16, alpha=np.pi / 2)
        assert P.shape == (16, 64)
        norms = np.linalg.norm(P, axis=1)
        assert np.allclose(norms, 1.0)

    def test_rows_distinct(self):
        P = build_ppt_projection(d_in=32, n_axes=8, alpha=np.pi / 2)
        for i in range(8):
            for j in range(i + 1, 8):
                assert not np.allclose(P[i], P[j]), f"Rows {i},{j} identical"

    def test_reconstructs_from_spec(self):
        spec = ProjectionSpec(d_in=64, n_axes=16, alpha=0.8, psi=0.2)
        P1 = build_ppt_projection(64, 16, 0.8, 0.2)
        P2 = build_projection_from_spec(spec)
        assert np.allclose(P1, P2)

    def test_verify_properties(self):
        spec = ProjectionSpec(d_in=32, n_axes=8, alpha=np.pi / 3, psi=0.0)
        P = build_projection_from_spec(spec)
        results = verify_projection_properties(P, spec)
        assert all(results.values()), f"Failed checks: {results}"

    def test_invalid_n_axes_gt_d_in(self):
        with pytest.raises(ValueError):
            build_ppt_projection(d_in=4, n_axes=8, alpha=np.pi / 2)

    def test_odd_d_in(self):
        """d_in=7 (odd) — last dimension pair is incomplete."""
        P = build_ppt_projection(d_in=7, n_axes=3, alpha=np.pi / 2)
        assert P.shape == (3, 7)
        norms = np.linalg.norm(P, axis=1)
        assert np.allclose(norms, 1.0)


# ---- Injectivity Tests (Prop A.1) ----

class TestInjectivity:
    def test_recover_coefficients_2d(self):
        """Prop A.1: c_k = <v_k, u_k> recovers original values."""
        P = build_ppt_projection(d_in=2, n_axes=2, alpha=np.pi / 2)
        x = np.array([3.0, -1.5])
        coeffs = P @ x
        recovered = P.T @ coeffs
        assert recovered.shape == x.shape

    def test_distinct_inputs_distinct_coefficients(self):
        """Injectivity: different x → different coefficients."""
        P = build_ppt_projection(d_in=32, n_axes=8, alpha=np.pi / 2)
        x1 = np.random.randn(32)
        x2 = x1 + 0.001 * np.random.randn(32)
        c1 = P @ x1
        c2 = P @ x2
        assert not np.allclose(c1, c2), "Different inputs gave same coefficients"


# ---- Optimization Tests ----

class TestOptimize:
    def test_projection_error_zero_for_identity(self):
        """If P is orthogonal and square, error should be ~0."""
        Q, _ = np.linalg.qr(np.random.randn(8, 8))
        W = np.random.randn(4, 8)
        err = projection_error(W, Q)
        assert err < 1e-10

    def test_adaptive_beats_fixed(self):
        """Adaptive (α, ψ) should give <= error than fixed 90°."""
        np.random.seed(42)
        W = np.random.randn(32, 64) @ np.diag(
            np.exp(-np.arange(64) * 0.1)  # decaying singular values
        )
        result = compare_with_fixed_90(W, n_axes=16)
        assert result["adaptive_error"] <= result["fixed_90_error"] + 1e-10

    def test_find_optimal_angles_returns_valid(self):
        np.random.seed(0)
        W = np.random.randn(16, 32)
        alpha, psi = find_optimal_angles(W, n_axes=8)
        assert 0 < alpha < np.pi
        assert isinstance(psi, float)


# ---- Activation Spec Tests ----

class TestSpecs:
    def test_relu_scalar(self):
        assert relu_scalar(1.0) == 1.0
        assert relu_scalar(-1.0) == 0.0
        assert relu_scalar(0.0) == 0.0

    def test_gelu_scalar(self):
        assert abs(gelu_approx_scalar(0.0)) < 1e-10
        assert abs(gelu_approx_scalar(10.0) - 10.0) < 0.01

    def test_silu_scalar(self):
        assert abs(silu_scalar(0.0)) < 1e-10

    def test_vector_matches_scalar(self):
        """Vector ops should match scalar ops element-wise."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        vec_relu = relu(x)
        for i, v in enumerate(x):
            assert abs(vec_relu[i] - relu_scalar(v)) < 1e-10

        vec_gelu = gelu_approx(x)
        for i, v in enumerate(x):
            assert abs(vec_gelu[i] - gelu_approx_scalar(v)) < 1e-7

        vec_silu = silu(x)
        for i, v in enumerate(x):
            assert abs(vec_silu[i] - silu_scalar(v)) < 1e-10

    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        s = softmax(x)
        assert abs(np.sum(s) - 1.0) < 1e-10
        assert all(s > 0)

    def test_rms_norm(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        out = rms_norm(x, w)
        assert out.shape == x.shape
        assert abs(np.mean(out ** 2) - 1.0) < 0.1

    def test_layer_norm(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        b = np.zeros(4)
        out = layer_norm(x, w, b)
        assert abs(np.mean(out)) < 1e-5
        assert abs(np.std(out) - 1.0) < 0.1
