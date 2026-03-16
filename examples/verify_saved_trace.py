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

"""Standalone verifier for saved PPAI traces.

Re-verify a previously saved trace without running inference again.
Works with artifacts produced by any PPAI pipeline (GPT-2, Qwen, etc.).

Usage:
    python ppai_v3/examples/verify_saved_trace.py \\
        --trace output/trace_0.npz \\
        --weights output/weights.npz \\
        --manifest output/manifest.json \\
        --input output/input_0.npy

    # Show per-step detail
    python ppai_v3/examples/verify_saved_trace.py \\
        --trace output/trace_0.npz \\
        --weights output/weights.npz \\
        --manifest output/manifest.json \\
        --input output/input_0.npy \\
        --verbose
"""

import argparse
import json
import sys
import time

import numpy as np

from ppai_v3.trace.format import load_trace
from ppai_v3.trace.verifier import verify_trace


def main():
    parser = argparse.ArgumentParser(
        description="PPAI v0.3.0 — Verify a saved inference trace"
    )
    parser.add_argument("--trace", required=True,
                        help="Path to trace file (.npz or .json)")
    parser.add_argument("--weights", required=True,
                        help="Path to weights file (.npz)")
    parser.add_argument("--manifest", required=True,
                        help="Path to manifest file (.json)")
    parser.add_argument("--input", required=True,
                        help="Path to input tensor (.npy)")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Absolute tolerance for verification (default: 1e-4)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-step verification details")
    args = parser.parse_args()

    print("PPAI v0.3.0 — Trace Verification")
    print("=" * 50)

    # Load artifacts
    print(f"\nLoading trace:    {args.trace}")
    trace = load_trace(args.trace)
    print(f"  Steps: {len(trace.steps)}, Ops: {trace.total_ops:,}")

    print(f"Loading weights:  {args.weights}")
    weights = dict(np.load(args.weights))
    print(f"  Arrays: {len(weights)}")

    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f"  Layers: {len(manifest)}")

    print(f"Loading input:    {args.input}")
    input_np = np.load(args.input)
    print(f"  Shape: {input_np.shape}, dtype: {input_np.dtype}")

    # Verify
    print(f"\nRunning verification (atol={args.atol})...")
    t0 = time.time()
    result = verify_trace(trace, manifest, weights, input_np, atol=args.atol)
    elapsed = time.time() - t0

    # Results
    print(f"\n{'=' * 50}")
    print(f"Result: {result.summary()}")
    print(f"Time:   {elapsed:.2f}s")

    if args.verbose and hasattr(result, 'step_results'):
        print(f"\nPer-step details:")
        for i, sr in enumerate(result.step_results):
            status = "PASS" if sr.passed else "FAIL"
            print(f"  [{i:4d}] {status}  {sr.layer_id}  "
                  f"max_err={sr.max_error:.2e}")

    print("=" * 50)
    return result.passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
