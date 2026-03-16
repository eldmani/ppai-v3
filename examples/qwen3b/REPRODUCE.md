# Reproducing Qwen 2.5-3B-Instruct PPAI Verification

This guide walks through reproducing the PPAI lossless verification of
Qwen 2.5-3B-Instruct on a Google Cloud L4 GPU instance.

**Expected result:** 506/506 layer verifications PASS, 5/5 prompts IDENTICAL.

## Prerequisites

- Google Cloud account with GPU quota (L4 or better)
- `gcloud` CLI installed and authenticated
- The `ppai_v3` package directory

## 1. Create the VM

```bash
gcloud compute instances create ppai-qwen-v3 \
    --zone=us-central1-a \
    --machine-type=g2-standard-16 \
    --accelerator=type=nvidia-l4,count=1 \
    --boot-disk-size=200GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE
```

**Specs:** 16 vCPUs, 64 GB RAM, NVIDIA L4 (24 GB VRAM), ~200 GB disk.
**Cost:** ~$1.50/hr (on-demand). The full pipeline takes ~15–25 minutes.

## 2. Upload the package

```bash
# From your local machine
tar czf ppai_v3.tar.gz ppai_v3/
gcloud compute scp ppai_v3.tar.gz ppai-qwen-v3:~ --zone=us-central1-a
gcloud compute ssh ppai-qwen-v3 --zone=us-central1-a
```

On the VM:
```bash
cd ~
tar xzf ppai_v3.tar.gz
```

## 3. Run setup

```bash
bash ppai_v3/examples/qwen3b/setup.sh
source ~/ppai_v3_env/bin/activate
```

This creates a venv, installs PyTorch (CUDA 12.4), transformers,
and ppai_v3 in editable mode. Takes ~3–5 minutes.

## 4. Run the pipeline

```bash
python3 ppai_v3/examples/qwen3b/patch_qwen.py
```

Or with auto-stop (shuts down VM when done to save costs):
```bash
python3 ppai_v3/examples/qwen3b/patch_qwen.py --auto-stop
```

### What the pipeline does

| Phase | Description | Time |
|-------|-------------|------|
| 1. Baseline | Load original Qwen, run 5 prompts, save outputs | ~2 min |
| 2. Convert | Reload, convert all linear layers to PPAI lossless | ~5 min |
| 3. Verify | Load trace + weights, verify all 506 layer projections | ~5 min |
| 4. Compare | Baseline vs PPAI text output comparison | <1 sec |

## 5. Expected output

```
FINAL SUMMARY
  Model:            Qwen/Qwen2.5-3B-Instruct
  Layers converted: 506
  Mode:             LOSSLESS
  Verification:     506/506 PASS (0 FAIL)
  Output match:     ALL IDENTICAL
```

All 506 linear layers pass verification. All 5 prompts produce
identical text and matching top-5 token predictions.

## 6. Retrieve artifacts

```bash
# From your local machine
gcloud compute scp --recurse \
    ppai-qwen-v3:~/output/qwen-3b-ppai-v3 \
    ./qwen-3b-ppai-v3-artifacts \
    --zone=us-central1-a
```

Artifacts include:
- `trace_0.npz` — Full trace for prompt 0
- `weights.npz` — Extracted weight matrices
- `manifest.json` — Layer conversion manifest
- `ppai_outputs.json` — PPAI inference outputs
- `model.pt` + tokenizer files — The patched model

## 7. Clean up

```bash
gcloud compute instances delete ppai-qwen-v3 --zone=us-central1-a
```

## Troubleshooting

### TF32 precision mismatch
If verification fails on GPU but the code runs, ensure TF32 is disabled.
The pipeline script already sets:
```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

### Out of memory
Qwen 2.5-3B in float32 requires ~12 GB. The L4 (24 GB) has headroom.
If OOM occurs during Phase 2, the sequential load/unload design ensures
only one copy of the model is in memory at a time.

### Verification atol
Default tolerance is `1e-4`. For lossless mode (n_axes=0), exact match
is expected. If you see small failures, check that TF32 is disabled.
