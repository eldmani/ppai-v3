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

"""PPAI command-line interface."""

import argparse
import sys
import json


def cmd_convert(args):
    """Convert a HuggingFace model to PPAI format."""
    from .integrations.huggingface import convert_hf_model
    convert_hf_model(
        model_name_or_path=args.model,
        n_axes=args.n_axes,
        output_dir=args.output,
        grid_steps=args.grid_steps,
        verbose=True,
    )


def cmd_verify(args):
    """Verify a trace file against a manifest."""
    import numpy as np
    from .trace.format import load_trace
    from .trace.verifier import verify_trace

    # Load the trace — returns an InferenceTrace which verify_trace accepts directly
    trace = load_trace(args.trace)

    with open(args.manifest) as f:
        manifest = json.load(f)

    # Load weights from a single .npz archive
    weights = dict(np.load(args.weights))

    # Load input
    input_tensor = np.load(args.input_npy)

    result = verify_trace(trace, manifest, weights, input_tensor,
                          atol=args.atol)
    print(result.summary())
    sys.exit(0 if result.passed else 1)


def cmd_inspect(args):
    """Inspect a PPAI model's compression stats."""
    import numpy as np
    from .compress.convert import load_manifest

    manifest = load_manifest(args.model_dir)
    print(f"PPAI Model: {args.model_dir}")
    print(f"Layers: {len(manifest)}")
    print()

    total_orig = 0
    total_comp = 0
    for name, info in manifest.items():
        spec = info["spec"]
        d_out = info["d_out"]
        ratio = info["compression_ratio"]
        err = info["rel_error"]
        orig = spec["d_in"] * d_out
        comp = spec["n_axes"] * d_out
        total_orig += orig
        total_comp += comp
        print(f"  {name}")
        print(f"    {spec['d_in']} -> {spec['n_axes']} axes  "
              f"α={np.degrees(spec['alpha']):.1f}°  "
              f"ψ={np.degrees(spec['psi']):.1f}°  "
              f"err={err:.4f}  ratio={ratio:.1f}x")

    print(f"\nOverall: {total_orig:,} -> {total_comp:,} params "
          f"({total_orig/max(total_comp,1):.1f}x compression)")


def main():
    parser = argparse.ArgumentParser(
        prog="ppai",
        description="PPAI — Verifiable inference via orthogonal projection",
    )
    sub = parser.add_subparsers(dest="command")

    # convert
    p_conv = sub.add_parser("convert", help="Convert HF model to PPAI")
    p_conv.add_argument("model", help="HF model name or path")
    p_conv.add_argument("--n-axes", type=int, default=96)
    p_conv.add_argument("--output", "-o", required=True)
    p_conv.add_argument("--grid-steps", type=int, default=20)

    # verify
    p_ver = sub.add_parser("verify", help="Verify a trace file")
    p_ver.add_argument("trace", help="Trace file (.json or .npz)")
    p_ver.add_argument("--manifest", required=True)
    p_ver.add_argument("--weights", required=True, help="weights.npz file")
    p_ver.add_argument("--input-npy", required=True)
    p_ver.add_argument("--atol", type=float, default=1e-4)

    # inspect
    p_ins = sub.add_parser("inspect", help="Inspect PPAI model")
    p_ins.add_argument("model_dir", help="PPAI model directory")

    args = parser.parse_args()
    if args.command == "convert":
        cmd_convert(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
