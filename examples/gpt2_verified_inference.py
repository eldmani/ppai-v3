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

"""
VERIFIED INFERENCE DEMO — GPT-2
================================
Complete three-phase demo showing the Provider → Verifier flow.

Phase 1 (Provider):  Load compressed GPT-2, run inference, publish trace
Phase 2 (Verifier):  Independently verify the trace using only NumPy
Phase 3 (Verdict):   Report pass/fail

Requirements:
    pip install ppai_v3[hf]

Pre-built GPT-2 models must exist at:
    gpt2-ppai/          (lossy, ~4x compression)
  or
    gpt2-ppai-lossless/  (lossless, exact output)

Run:
    python ppai_v3/examples/gpt2_verified_inference.py
"""

import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from ppai_v3.compress.convert import load_ppai_model, load_manifest
from ppai_v3.trace.recorder import TraceRecorder
from ppai_v3.trace.format import save_trace, load_trace
from ppai_v3.trace.verifier import verify_trace
from ppai_v3.integrations.huggingface import extract_weights_for_verification


def find_model_dir() -> str:
    """Find a pre-built GPT-2 PPAI model in the workspace."""
    candidates = [
        Path(__file__).resolve().parents[2] / "gpt2-ppai-lossless",
        Path(__file__).resolve().parents[2] / "gpt2-ppai",
        Path.cwd() / "gpt2-ppai-lossless",
        Path.cwd() / "gpt2-ppai",
    ]
    for p in candidates:
        if (p / "manifest.json").exists():
            return str(p)
    print("ERROR: No pre-built GPT-2 PPAI model found.")
    print("  Expected one of:")
    for p in candidates:
        print(f"    {p}")
    print("\n  Run first: ppai convert gpt2 --n-axes 0 -o gpt2-ppai-lossless")
    sys.exit(1)


def main():
    model_dir = find_model_dir()
    tmp_dir = tempfile.mkdtemp(prefix="ppai_demo_")
    trace_file = str(Path(tmp_dir) / "trace.json")
    weights_file = str(Path(tmp_dir) / "weights.npz")

    # ════════════════════════════════════════════════════════════
    # PHASE 1: PROVIDER — runs the model, produces answer + trace
    # ════════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 1: PROVIDER (runs the model)")
    print("=" * 60)

    model, manifest = load_ppai_model(model_dir, verbose=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    prompt = "The capital of France is"
    print(f"\n  User prompt: \"{prompt}\"")

    inputs = tokenizer(prompt, return_tensors="pt")

    # Run inference WITH trace recording
    recorder = TraceRecorder()
    with recorder.recording(model, inputs["input_ids"]):
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"], max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    trace = recorder.get_trace()
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"  Model answer: \"{answer}\"")
    print(f"  Trace: {len(trace.steps)} steps, {trace.total_ops:,} scalar ops")

    # Provider publishes: answer + trace + weights
    save_trace(trace, trace_file)
    weights = extract_weights_for_verification(model)
    np.savez(weights_file, **{k: v for k, v in weights.items()})

    print(f"\n  Published:")
    print(f"    - Answer:        \"{answer}\"")
    print(f"    - Trace:         {trace_file} ({len(trace.steps)} steps)")
    print(f"    - Weights:       {weights_file} ({len(weights)} matrices)")
    print(f"    - Manifest:      {model_dir}/manifest.json ({len(manifest)} layers)")

    del model, trace, weights, recorder
    print("\n  [Provider done — model unloaded]")

    # ════════════════════════════════════════════════════════════
    # PHASE 2: VERIFIER — checks the trace (no model needed)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: VERIFIER (independent check, no model needed)")
    print("=" * 60)

    print("\n  Loading published artifacts...")
    loaded_trace = load_trace(trace_file)
    loaded_weights = dict(np.load(weights_file))
    loaded_manifest = load_manifest(model_dir)

    print(f"    Trace:    {len(loaded_trace.steps)} steps")
    print(f"    Weights:  {len(loaded_weights)} matrices")
    print(f"    Manifest: {len(loaded_manifest)} layer specs")

    # Peek inside the trace
    step0 = loaded_trace.steps[0]
    step1 = loaded_trace.steps[1]
    print(f"\n  First trace steps:")
    print(f"    Step 0 (project): {step0.layer_id}")
    print(f"      spec: α={np.degrees(step0.spec_params['alpha']):.1f}°, "
          f"ψ={np.degrees(step0.spec_params['psi']):.1f}°, "
          f"n_axes={step0.spec_params['n_axes']}")
    print(f"    Step 1 (matmul):  {step1.layer_id}")
    print(f"      hash chain: {step1.input_hash == step0.output_hash} "
          f"(step1.input == step0.output)")
    print(f"      has_bias: {step1.bias_values is not None}")

    # Manual check of step 1 (matmul)
    print(f"\n  Manual verification of Step 1:")
    W = loaded_weights[step1.layer_id].astype(np.float32)
    c = step0.output_values.astype(np.float32)
    expected = c @ W.T
    if step1.bias_values is not None:
        expected = expected + np.array(step1.bias_values, dtype=np.float32)
    max_err = float(np.max(np.abs(expected - step1.output_values)))
    print(f"    W_comp: {W.shape}, coefficients: {c.shape}")
    print(f"    Recomputed vs recorded max error: {max_err:.2e}")
    print(f"    → {'MATCH ✓' if max_err < 1e-3 else 'MISMATCH ✗'}")

    # Full automatic verification
    print("\n  Running full automatic verification...")
    t0 = time.time()
    input_np = np.array(tokenizer(prompt)["input_ids"], dtype=np.float32)
    result = verify_trace(loaded_trace, loaded_manifest, loaded_weights, input_np)
    vtime = time.time() - t0

    print(f"    {result.passed_steps}/{result.total_steps} steps PASSED")
    print(f"    Time: {vtime:.2f}s")

    # ════════════════════════════════════════════════════════════
    # PHASE 3: VERDICT
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 3: VERDICT")
    print("=" * 60)
    print(f"\n  Prompt:        \"{prompt}\"")
    print(f"  Answer:        \"{answer}\"")
    print(f"  Trace steps:   {result.total_steps}")
    print(f"  Verified:      {result.passed_steps}/{result.total_steps}")
    status = "VERIFIED ✓" if result.passed else "VERIFICATION FAILED ✗"
    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  {status:^31s}  │")
    print(f"  └─────────────────────────────────┘")
    if result.passed:
        print(f"\n  The answer was provably computed by the stated model")
        print(f"  using the stated weights, verified by arithmetic replay.")
        print(f"  No trust required.\n")
    else:
        print(f"\n  Verification failed. Check trace for details.\n")

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return result.passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
