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

"""PPAI v0.3.0 — Qwen 2.5-3B-Instruct Patching Pipeline (GPU).

Sequential pipeline:
  Phase 1: Load original Qwen, run baseline inference, save outputs, unload
  Phase 2: Reload Qwen, convert to PPAI lossless, run inference with trace
  Phase 3: Verification (full or sketch)
  Phase 4: Compare baseline vs PPAI outputs

Validated result: 506/506 PASS, 5/5 prompts IDENTICAL (NVIDIA L4, 24GB).

Usage:
    python ppai_v3/examples/qwen3b/patch_qwen.py
    python ppai_v3/examples/qwen3b/patch_qwen.py --auto-stop
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── CRITICAL: Disable TF32 ──────────────────────────────────────
# Force full float32 precision on GPU matmuls so that PPAI trace
# verification (NumPy, exact float32) matches GPU results.
# Without this, GPU uses 10-bit mantissa (TF32) → trace mismatch.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from ppai_v3.compress.convert import convert_model, save_ppai_model
from ppai_v3.trace.recorder import TraceRecorder
from ppai_v3.trace.format import save_trace, load_trace
from ppai_v3.trace.verifier import verify_trace
from ppai_v3.integrations.huggingface import extract_weights_for_verification
from ppai_v3.layers.linear import PPAILinear


# ── Configuration ──────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "output/qwen-3b-ppai-v3"
BASELINE_DIR = "output/qwen-3b-baseline"

TEST_PROMPTS = [
    "The key insight of verifiable AI inference is",
    "Explain quantum computing in simple terms:",
    "Write a Python function that checks if a number is prime:",
    "The three most important principles of software engineering are",
    "In the year 2030, artificial intelligence will",
]


def detect_device():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        return "cuda"
    print("  WARNING: No GPU detected, using CPU (will be very slow)")
    return "cpu"


def log_memory(tag=""):
    """Print current memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  [GPU mem {tag}] allocated={alloc:.1f}GB reserved={reserved:.1f}GB")
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  [RAM {tag}] used={ram.used/1e9:.1f}GB / {ram.total/1e9:.1f}GB "
              f"({ram.percent}%)")
    except ImportError:
        pass


# ── Phase 1: Baseline ─────────────────────────────────────────

def phase1_baseline(device):
    """Load original model, run inference on test prompts, save results."""
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE (Original Model)")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[1/3] Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {time.time()-t0:.1f}s — {total_params:,} parameters")

    model.to(device)
    log_memory("after load")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[2/3] Running baseline inference on {len(TEST_PROMPTS)} prompts...")
    baseline = {}
    for i, prompt in enumerate(TEST_PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            out_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        last_logits = logits[0, -1, :].detach().cpu().numpy()
        baseline[prompt] = {
            "text": text,
            "last_logits_top5": np.argsort(last_logits)[-5:][::-1].tolist(),
            "last_logits_hash": hash(last_logits.tobytes()),
        }
        print(f"  [{i+1}/{len(TEST_PROMPTS)}] {prompt[:50]}...")
        print(f"    -> {text[len(prompt):len(prompt)+80]}...")

    Path(BASELINE_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{BASELINE_DIR}/baseline_outputs.json", "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\n[3/3] Baseline saved to {BASELINE_DIR}/")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_memory("after unload")

    return baseline


# ── Phase 2: Convert & Infer ──────────────────────────────────

def phase2_convert_and_infer(device):
    """Reload model, convert to PPAI lossless, infer with trace."""
    print("\n" + "=" * 70)
    print("PHASE 2: CONVERT & INFER (PPAI v0.3.0 Lossless)")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    PPAILinear.clear_P_cache()

    print(f"\n[1/6] Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    log_memory("after load (CPU)")

    print(f"\n[2/6] Converting to PPAI lossless (n_axes=0)...")
    t0 = time.time()
    manifest = convert_model(model, n_axes=0, verbose=True)
    convert_time = time.time() - t0
    print(f"  Conversion done in {convert_time:.1f}s")
    print(f"  P cache entries: {len(PPAILinear._P_cache)} unique projections")
    for key, tensor in PPAILinear._P_cache.items():
        d_in = key[0]
        mem_mb = tensor.nelement() * 4 / 1e6
        print(f"    d_in={d_in}: P is {tensor.shape[0]}x{tensor.shape[1]} "
              f"({mem_mb:.0f} MB)")
    log_memory("after conversion (CPU)")

    print(f"\n[3/6] Moving model to {device}...")
    model.to(device)
    log_memory("on GPU")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Trace on first prompt only (trace is expensive)
    print(f"\n[4/6] Running PPAI inference with trace (1 prompt)...")
    trace_prompt = TEST_PROMPTS[0]
    inputs = tokenizer(trace_prompt, return_tensors="pt").to(device)

    recorder = TraceRecorder()
    with recorder.recording(model, inputs["input_ids"],
                            model_hash="qwen-3b-ppai-v3"):
        with torch.no_grad():
            ppai_logits = model(**inputs).logits

    trace = recorder.get_trace()
    print(f"  Recorded {len(trace.steps)} steps, {trace.total_ops:,} ops")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    save_trace(trace, f"{OUTPUT_DIR}/trace_0.npz")
    print(f"  Trace saved to {OUTPUT_DIR}/trace_0.npz")

    # Run generation on ALL prompts for comparison
    print(f"\n[5/6] Running PPAI generation on {len(TEST_PROMPTS)} prompts...")
    ppai_outputs = {}
    for i, prompt in enumerate(TEST_PROMPTS):
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inp).logits
            out_ids = model.generate(
                inp["input_ids"],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        last_logits = logits[0, -1, :].detach().cpu().numpy()
        ppai_outputs[prompt] = {
            "text": text,
            "last_logits_top5": np.argsort(last_logits)[-5:][::-1].tolist(),
        }
        print(f"  [{i+1}/{len(TEST_PROMPTS)}] {prompt[:50]}...")
        print(f"    -> {text[len(prompt):len(prompt)+80]}...")

    with open(f"{OUTPUT_DIR}/ppai_outputs.json", "w") as f:
        json.dump(ppai_outputs, f, indent=2)

    # Extract weights for verification
    print(f"\n[6/6] Extracting weights for verification...")
    weights = extract_weights_for_verification(model)
    np.savez_compressed(f"{OUTPUT_DIR}/weights.npz", **weights)
    print(f"  Saved {len(weights)} weight arrays")

    input_np = inputs["input_ids"].detach().cpu().numpy()
    np.save(f"{OUTPUT_DIR}/input_0.npy", input_np)

    # Save manifest
    with open(f"{OUTPUT_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save PPAI model
    print(f"\n  Saving PPAI model to {OUTPUT_DIR}/...")
    t0 = time.time()
    save_ppai_model(model, manifest, OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Model saved in {time.time()-t0:.1f}s")

    del model, weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_memory("after unload")

    return manifest, ppai_outputs, trace_prompt


# ── Phase 3: Verification ────────────────────────────────────

def phase3_verify(manifest):
    """Run verification on the recorded trace."""
    print("\n" + "=" * 70)
    print("PHASE 3: VERIFICATION")
    print("=" * 70)

    print(f"\n[1/3] Loading trace and weights...")
    trace_data = load_trace(f"{OUTPUT_DIR}/trace_0.npz")
    loaded_weights = dict(np.load(f"{OUTPUT_DIR}/weights.npz"))
    input_np = np.load(f"{OUTPUT_DIR}/input_0.npy")

    print(f"  Trace: {len(trace_data.steps)} steps")
    print(f"  Weights: {len(loaded_weights)} layers")

    print(f"\n[2/3] Running verification...")
    t0 = time.time()
    result = verify_trace(
        trace_data, manifest, loaded_weights, input_np,
        atol=1e-4,
    )
    verify_time = time.time() - t0

    print(f"\n[3/3] Result:")
    print(f"  {result.summary()}")
    print(f"  Time: {verify_time:.2f}s")

    return result


# ── Phase 4: Compare ─────────────────────────────────────────

def phase4_compare(baseline, ppai_outputs):
    """Compare baseline vs PPAI outputs."""
    print("\n" + "=" * 70)
    print("PHASE 4: COMPARISON (Baseline vs PPAI)")
    print("=" * 70)

    identical_count = 0
    total = len(TEST_PROMPTS)

    for prompt in TEST_PROMPTS:
        base_text = baseline[prompt]["text"]
        ppai_text = ppai_outputs[prompt]["text"]
        match = base_text == ppai_text

        base_top5 = baseline[prompt]["last_logits_top5"]
        ppai_top5 = ppai_outputs[prompt]["last_logits_top5"]
        top1_match = base_top5[0] == ppai_top5[0]
        top5_match = base_top5 == ppai_top5

        if match:
            identical_count += 1

        status = "IDENTICAL" if match else "DIFFERENT"
        print(f"\n  Prompt: {prompt[:60]}...")
        print(f"    Text: {status}")
        print(f"    Top-1 token: {'MATCH' if top1_match else 'MISMATCH'}")
        print(f"    Top-5 tokens: {'MATCH' if top5_match else 'MISMATCH'}")
        if not match:
            min_len = min(len(base_text), len(ppai_text))
            diverge_at = min_len
            for j in range(min_len):
                if base_text[j] != ppai_text[j]:
                    diverge_at = j
                    break
            print(f"    Diverges at char {diverge_at}")
            print(f"    Base: ...{base_text[max(0,diverge_at-20):diverge_at+40]}...")
            print(f"    PPAI: ...{ppai_text[max(0,diverge_at-20):diverge_at+40]}...")

    print(f"\n  Summary: {identical_count}/{total} prompts identical")
    return identical_count == total


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PPAI v0.3.0 — Qwen 2.5-3B Patching Pipeline"
    )
    parser.add_argument("--auto-stop", action="store_true",
                        help="Shut down VM when done (cost savings)")
    args = parser.parse_args()

    print("=" * 70)
    print("PPAI v0.3.0 — Qwen Patching Pipeline")
    print(f"Model: {MODEL_NAME}")
    print(f"Mode:  LOSSLESS + Full Verification")
    print("=" * 70)

    device = detect_device()
    log_memory("initial")

    baseline = phase1_baseline(device)
    manifest, ppai_outputs, trace_prompt = phase2_convert_and_infer(device)
    result = phase3_verify(manifest)
    all_identical = phase4_compare(baseline, ppai_outputs)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Layers converted: {len(manifest)}")
    print(f"  Mode:             LOSSLESS")
    print(f"  Verification:     {result.summary().split(chr(10))[0]}")
    print(f"  Output match:     {'ALL IDENTICAL' if all_identical else 'DIFFERENCES FOUND'}")
    print(f"  Saved to:         {OUTPUT_DIR}/")

    if result.passed and all_identical:
        print("\n  >>> SUCCESS: Model patched, verified, outputs identical. <<<")
    elif result.passed:
        print("\n  >>> PARTIAL: Verification passed but outputs differ. <<<")
    else:
        print("\n  >>> WARNING: Verification had failures. Check logs. <<<")

    print("=" * 70)

    if args.auto_stop:
        print("\n  Auto-stop: shutting down VM in 10 seconds...")
        import subprocess
        time.sleep(10)
        subprocess.run(["sudo", "shutdown", "-h", "now"])

    return result.passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
