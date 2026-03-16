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

# PPAI v0.3.0

**Polygonal Projection for Auditable Inference** — a Python library that
factors neural network linear layers using the Polygonal Projection Theorem (PPT)
and produces verifiable arithmetic traces of every inference.

## What It Does

1. **Factor** any PyTorch model’s `nn.Linear` layers into orthogonal
   PPT-projected form (lossless when `n_axes = d_in`).
2. **Record** a full arithmetic trace during inference — every projection
   and matmul step, with SHA-256 hashes.
3. **Verify** that trace independently using *only NumPy* — zero PyTorch
   dependency on the verifier side.  Verification confirms
   **computational self-consistency**: the published weights and
   projection specs, applied to the stated input, reproduce the recorded
   intermediate values.  It does *not* attest model identity (see
   Limitations below).

## Key Results

| Model | Layers | Trace Steps | Verification |
|-------|--------|-------------|--------------|
| GPT-2 (124M) | 72 layers | 144 steps | 144/144 PASS |
| Qwen2.5-3B-Instruct | 253 layers | 506 steps | 506/506 PASS |

Qwen2.5-3B was verified on an NVIDIA L4 GPU with 5 diverse prompts —
all 5 produced **IDENTICAL** outputs between traced and untraced runs.

## Installation

Verifier only (NumPy, no PyTorch):
```bash
pip install ppai-v3
```

Provider (conversion + inference + tracing):
```bash
pip install ppai-v3[provider]
```

With HuggingFace integration:
```bash
pip install ppai-v3[hf]
```

From source:
```bash
git clone <repo-url>
cd ppai_v3
pip install -e ".[all]"
```

## Quick Start

### Compress a HuggingFace Model

```python
from ppai_v3.integrations.huggingface import convert_hf_model

model, manifest = convert_hf_model(
    "gpt2",
    n_axes=192,          # 4x compression
    output_dir="gpt2-ppai",
)
```

### Run Inference with Trace

```python
from ppai_v3.integrations.huggingface import infer_with_trace

output_ids, trace = infer_with_trace(
    model, input_ids,
    trace_path="trace.json",
    max_new_tokens=50,
    do_sample=False,
)
```

### Verify a Trace (Zero PyTorch Dependency)

```python
from ppai_v3.trace.verifier import verify_trace
from ppai_v3.trace.format import load_trace
import numpy as np
import json

trace = load_trace("trace.json")
with open("gpt2-ppai/manifest.json") as f:
    manifest = json.load(f)

# weights: dict of layer_name → np.ndarray
result = verify_trace(trace, manifest, weights, input_np)
print(result.summary())
# Verification: PASS
#   Steps: 144/144 passed
```

### Lossless Mode (n_axes=0)

For exact output preservation (no compression, full auditability):

```python
model, manifest = convert_hf_model("gpt2", n_axes=0, output_dir="gpt2-lossless")
```

### CLI

```bash
# Convert
ppai convert gpt2 --n-axes 192 -o gpt2-ppai/

# Verify
ppai verify trace.npz --manifest gpt2-ppai/manifest.json \
    --weights weights.npz --input-npy input.npy

# Inspect
ppai inspect gpt2-ppai/
```

## Architecture

```
ppai_v3/
├── core/            # PPT projection math, optimization, activation specs
│   ├── projection.py   # build_ppt_projection(), ProjectionSpec
│   ├── optimize.py     # find_optimal_angles(), SVD-guided init
│   └── spec.py         # ReLU, GELU, SiLU, softmax, RMS/LayerNorm
├── layers/          # PyTorch layer implementations
│   └── linear.py       # PPAILinear (drop-in nn.Linear replacement)
├── compress/        # Model conversion pipeline
│   ├── convert.py      # convert_model(), save/load_ppai_model
│   └── calibrate.py    # Post-compression distillation (experimental)
├── trace/           # Arithmetic trace recording & verification
│   ├── recorder.py     # TraceRecorder context manager
│   ├── format.py       # JSON + NPZ serialization (with bias fix)
│   └── verifier.py     # Independent verifier (NumPy only)
├── integrations/    # Framework integrations
│   └── huggingface.py  # HF Transformers helpers
├── cli.py           # Command-line interface
└── tests/           # Test suite (39+ tests)
```

## v0.3.0 Changes

- **Bias serialization fix**: NPZ traces now correctly save and load
  `bias_values` for layers with bias (e.g., QKV projections in Qwen).
  This was the root cause of verification failures on Qwen2.5-3B.
- **Bounded P cache**: `PPAILinear._P_cache` is now bounded (default 64)
  with LRU-style eviction — prevents unbounded memory growth.
- **Proper packaging**: `pyproject.toml`, typed (`py.typed`), versioned.
- **Consistent version**: Single `__version__` source in `__init__.py`.
- **Proper exports**: All subpackages expose their public API via `__init__.py`.
- **CLI fix**: `verify` command passes `InferenceTrace` directly to the
  verifier instead of manually rebuilding a dict.

See [CHANGELOG.md](CHANGELOG.md) for full version history.

## Examples

The `examples/` directory contains complete, runnable demonstrations:

| Example | Description | Requirements |
|---------|-------------|--------------|
| [`gpt2_verified_inference.py`](examples/gpt2_verified_inference.py) | Full Provider→Verifier→Verdict cycle with GPT-2 | CPU, pre-built `gpt2-ppai/` model |
| [`qwen3b/patch_qwen.py`](examples/qwen3b/patch_qwen.py) | 4-phase pipeline: baseline, convert, verify, compare | GPU (L4+), HuggingFace access |
| [`qwen3b/REPRODUCE.md`](examples/qwen3b/REPRODUCE.md) | Step-by-step GCloud reproduction guide | GCloud account |
| [`verify_saved_trace.py`](examples/verify_saved_trace.py) | Re-verify a saved trace (CLI) | CPU only |

```bash
# GPT-2 demo (CPU, ~30 seconds)
python ppai_v3/examples/gpt2_verified_inference.py

# Verify a previously saved trace
python ppai_v3/examples/verify_saved_trace.py \
    --trace output/trace_0.npz \
    --weights output/weights.npz \
    --manifest output/manifest.json \
    --input output/input_0.npy
```

See [`examples/README.md`](examples/README.md) for full details.

## Testing

```bash
# Unit + integration tests (no GPU needed)
pytest ppai_v3/tests/test_core.py ppai_v3/tests/test_pipeline.py -v

# Realistic GPT-2 test (downloads model, ~2 min)
python -m ppai_v3.tests.test_gpt2_realistic
```

## The Polygonal Projection Theorem

The Polygonal Projection Theorem states that for any weight matrix
$W \in \mathbb{R}^{d_{out} \times d_{in}}$ and orthogonal projection
$P \in \mathbb{R}^{d_{in} \times d_{in}}$ constructed via PPT + QR:

$$\hat{y} = W_{comp} \cdot (P \cdot x) = W \cdot x$$

where $W_{comp} = W \cdot P^\top$ (since $P$ is orthogonal, $P^+ = P^\top$).

The projection $P$ is constructed from uniformly-spaced angles in a
polygonal fan, parameterized by sweep angle $\alpha$ and offset $\psi$,
then orthogonalized via QR factorization.

Key properties:
- **Injectivity** (Prop A.1): Different inputs produce different coefficients
- **Determinism** (Prop A.2): Same input always gives same output
- **Step chaining** (Prop A.3): Verifier chains steps via hash-keyed lookup, accepting checkpoints at untracked-op boundaries
- **Lipschitz stability** (Prop A.4): Bounded perturbation propagation
- **Lossless equivalence** (Prop A.6): When $n = d_{in}$, factored output is bit-identical to original

## Limitations

- **Not model-identity attestation.** Verification confirms that the
  published factored weights and projection specs produce the recorded
  trace values.  It does not prove that those weights came from a
  particular original model.
- **Untracked operations.** Non-linear operations (softmax, layer norm,
  activations) between traced linear layers are not individually
  verified.  The verifier accepts their outputs as checkpoints and
  resumes verification at the next traced step.
- **Model hash.** The model hash in the trace is flagged for external
  validation; the verifier does not check it internally.

## License

This software is dual-licensed:

### Open Source — AGPL-3.0

Free to use, modify, and distribute under the terms of the
[GNU Affero General Public License v3.0](LICENSE). Key obligations:
- Derivative works must also be AGPL-3.0
- If you offer this software as a network service, you must provide
  the complete source code to users
- Attribution to the original author is required

### Commercial License

For use in proprietary software, SaaS platforms, or any context where
AGPL-3.0 obligations cannot be met, a separate commercial license is
required.

Contact: **eldhose.mani@hotmail.co.uk**

See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for terms.
