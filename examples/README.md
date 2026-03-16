# PPAI v0.3.0 — Examples

This directory contains ready-to-run examples demonstrating the full
PPAI pipeline: compress, trace, and verify.

## Examples

### 1. GPT-2 Verified Inference (`gpt2_verified_inference.py`)

Complete three-phase demo on GPT-2 (124M params):
- **Phase 1 — Provider**: Loads compressed GPT-2, runs inference, saves trace
- **Phase 2 — Verifier**: Independently checks the trace using only NumPy
- **Phase 3 — Verdict**: Reports pass/fail

```bash
# Requires: pip install ppai_v3[hf]
python ppai_v3/examples/gpt2_verified_inference.py
```

### 2. Qwen 2.5-3B Pipeline (`qwen3b/`)

Full GPU pipeline for Qwen2.5-3B-Instruct — the same pipeline that
achieved **506/506 PASS, 5/5 prompts IDENTICAL** on an NVIDIA L4.

```bash
# On a GCloud GPU VM (L4 24GB recommended):
bash ppai_v3/examples/qwen3b/setup.sh
python ppai_v3/examples/qwen3b/patch_qwen.py --auto-stop
```

See [qwen3b/REPRODUCE.md](qwen3b/REPRODUCE.md) for step-by-step
instructions including VM creation, upload, and cost estimates.

### 3. Verify a Saved Trace (`verify_saved_trace.py`)

Standalone verifier script — loads a trace from disk, loads weights,
and runs verification. Demonstrates the zero-PyTorch verification path.

```bash
python ppai_v3/examples/verify_saved_trace.py \
    --trace output/trace_0.npz \
    --manifest output/manifest.json \
    --weights output/weights.npz \
    --input output/input_0.npy
```

## Pre-Built Models

The repository includes two pre-built GPT-2 models for immediate use:

| Directory | Mode | Compression | Use Case |
|-----------|------|-------------|----------|
| `gpt2-ppai/` | Lossy (n_axes=192) | ~4x | Quality vs size tradeoff |
| `gpt2-ppai-lossless/` | Lossless (n_axes=d_in) | 1x | Exact output, full auditability |

Both include tokenizer files and can be loaded directly with
`ppai_v3.compress.convert.load_ppai_model()`.
