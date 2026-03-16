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

"""Realistic end-to-end test: patch GPT-2 with PPAI and verify.

Downloads GPT-2 small (124M params), compresses it, runs inference,
records an arithmetic trace, and independently verifies the trace.

This test requires `transformers` to be installed.
Run with: python -m ppai_v3.tests.test_gpt2_realistic
"""

import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ppai_v3.compress.convert import convert_model
from ppai_v3.trace.recorder import TraceRecorder
from ppai_v3.trace.format import save_trace, load_trace
from ppai_v3.trace.verifier import verify_trace
from ppai_v3.integrations.huggingface import extract_weights_for_verification


def main():
    print("=" * 70)
    print("PPAI v0.3.0 Realistic Test: GPT-2 Small (124M params)")
    print("=" * 70)

    # ── Step 1: Load GPT-2 ──────────────────────────────────────────
    print("\n[1/7] Loading GPT-2 from HuggingFace...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model_original = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float32
    )
    model_original.eval()
    total_params = sum(p.numel() for p in model_original.parameters())
    print(f"  Loaded in {time.time() - t0:.1f}s - {total_params:,} parameters")

    # ── Step 2: Run original model ──────────────────────────────────
    print("\n[2/7] Running original GPT-2...")
    prompt = "The key insight of verifiable AI inference is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        orig_logits = model_original(input_ids).logits
        orig_output = model_original.generate(
            input_ids, max_new_tokens=30, do_sample=False
        )
    orig_text = tokenizer.decode(orig_output[0], skip_special_tokens=True)
    print(f"  Prompt:   '{prompt}'")
    print(f"  Output:   '{orig_text}'")

    orig_last_logits = orig_logits[0, -1, :].detach().cpu().numpy()

    # ── Step 3: Compress with PPAI ──────────────────────────────────
    n_axes = 192
    print(f"\n[3/7] Compressing GPT-2 with n_axes={n_axes} (4x compression)...")
    t0 = time.time()
    manifest = convert_model(
        model_original, n_axes=n_axes,
        grid_steps=10,
        skip_patterns=["lm_head"],
        verbose=True,
    )
    compress_time = time.time() - t0
    print(f"\n  Compression done in {compress_time:.1f}s")
    print(f"  Layers converted: {len(manifest)}")

    rel_errors = [v["rel_error"] for v in manifest.values()]
    comp_ratios = [v["compression_ratio"] for v in manifest.values()]
    print(f"  Relative error - mean: {np.mean(rel_errors):.4f}, "
          f"max: {np.max(rel_errors):.4f}, min: {np.min(rel_errors):.4f}")
    print(f"  Compression ratio - mean: {np.mean(comp_ratios):.1f}x")

    # ── Step 4: Run compressed model ────────────────────────────────
    print("\n[4/7] Running compressed GPT-2...")
    with torch.no_grad():
        comp_logits = model_original(input_ids).logits
        comp_output = model_original.generate(
            input_ids, max_new_tokens=30, do_sample=False
        )
    comp_text = tokenizer.decode(comp_output[0], skip_special_tokens=True)
    print(f"  Output:   '{comp_text}'")

    comp_last_logits = comp_logits[0, -1, :].detach().cpu().numpy()
    logit_diff = np.abs(orig_last_logits - comp_last_logits)
    print(f"\n  Logit divergence (last token):")
    print(f"    Mean absolute diff: {np.mean(logit_diff):.4f}")
    print(f"    Max absolute diff:  {np.max(logit_diff):.4f}")
    print(f"    Cosine similarity:  {np.dot(orig_last_logits, comp_last_logits) / (np.linalg.norm(orig_last_logits) * np.linalg.norm(comp_last_logits)):.6f}")

    # ── Step 5: Record arithmetic trace ─────────────────────────────
    print("\n[5/7] Recording arithmetic trace...")
    recorder = TraceRecorder()
    with recorder.recording(model_original, input_ids):
        with torch.no_grad():
            traced_logits = model_original(input_ids).logits
    trace = recorder.get_trace()

    print(f"  Trace steps: {len(trace.steps)}")
    print(f"  Total scalar ops: {trace.total_ops:,}")

    step_types = {}
    for s in trace.steps:
        step_types[s.step_type] = step_types.get(s.step_type, 0) + 1
    print(f"  Step breakdown: {step_types}")

    trace_path = str(__import__("pathlib").Path(__file__).parent / "gpt2_trace.json")
    save_trace(trace, trace_path)
    print(f"  Trace saved to: {trace_path}")

    # ── Step 6: Independent verification ────────────────────────────
    print("\n[6/7] Independently verifying trace (ZERO PyTorch dependency)...")
    t0 = time.time()

    weights = extract_weights_for_verification(model_original)
    print(f"  Weight matrices extracted: {len(weights)}")

    loaded_trace = load_trace(trace_path)
    print(f"  Trace loaded from disk: {len(loaded_trace.steps)} steps")

    input_np = input_ids.detach().cpu().float().numpy().flatten()
    result = verify_trace(loaded_trace, manifest, weights, input_np)
    verify_time = time.time() - t0

    print(f"\n  {result.summary()}")
    print(f"  Verification time: {verify_time:.2f}s")

    # ── Step 7: Summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model:           GPT-2 Small ({total_params:,} params)")
    print(f"  Compression:     {n_axes} axes ({np.mean(comp_ratios):.1f}x avg)")
    print(f"  Mean rel error:  {np.mean(rel_errors):.4f}")
    print(f"  Trace steps:     {len(trace.steps)}")
    print(f"  Scalar ops:      {trace.total_ops:,}")
    print(f"  Verification:    {'PASS' if result.passed else 'FAIL'} "
          f"({result.passed_steps}/{result.total_steps} steps)")
    print(f"  Output match:    {'Yes' if orig_text == comp_text else 'No (expected - lossy compression)'}")
    print("=" * 70)

    return result.passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
