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

"""Model conversion pipeline: patch any HuggingFace model with PPAI layers.

Walks the model's module tree, replaces every nn.Linear with PPAILinear,
finds optimal (α, ψ) per layer, and saves the compression manifest.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from ..core.optimize import find_optimal_angles, relative_projection_error
from ..core.projection import build_ppt_projection, ProjectionSpec
from ..layers.linear import PPAILinear
from .. import __version__


def _set_module(model: nn.Module, target: str, new_module: nn.Module):
    """Replace a submodule by its dotted name path."""
    parts = target.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _weight_hash(tensor: torch.Tensor) -> str:
    """Compute SHA-256 hash of a weight tensor's bytes (truncated to 16 hex chars)."""
    data = tensor.detach().cpu().float().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def convert_model(model: nn.Module, n_axes: int,
                  grid_steps: int = 20,
                  skip_patterns: Optional[list[str]] = None,
                  verbose: bool = True) -> dict:
    """Convert a model's Linear layers to PPAILinear.

    Args:
        model: A PyTorch model (e.g., HuggingFace AutoModel).
        n_axes: Number of PPT axes per layer. Use 0 for lossless mode
                (n_axes=d_in per layer, exact output preservation).
        grid_steps: Grid search resolution for (α, ψ) optimization.
        skip_patterns: Module name patterns to skip (e.g., ["lm_head"]).
        verbose: Print progress.

    Returns:
        manifest: Dict mapping layer names to their ProjectionSpec + metadata.
    """
    lossless = (n_axes == 0)
    skip_patterns = skip_patterns or []
    manifest = {}
    layers_converted = 0
    total_original_params = 0
    total_compressed_params = 0

    # Try to import HuggingFace Conv1D (used by GPT-2, GPT-Neo, etc.)
    _Conv1D = None
    try:
        from transformers.pytorch_utils import Conv1D as _Conv1D
    except ImportError:
        pass

    def _is_linear(module):
        """Check if module is nn.Linear or HuggingFace Conv1D."""
        if isinstance(module, nn.Linear):
            return True
        if _Conv1D is not None and isinstance(module, _Conv1D):
            return True
        return False

    def _get_dims(module):
        """Get (d_out, d_in) regardless of Linear vs Conv1D."""
        if isinstance(module, nn.Linear):
            return module.out_features, module.in_features
        # Conv1D stores weight as (d_in, d_out) — transposed vs nn.Linear
        return module.weight.shape[1], module.weight.shape[0]

    def _to_linear(module):
        """Convert Conv1D to nn.Linear for uniform handling."""
        if isinstance(module, nn.Linear):
            return module
        d_in, d_out = module.weight.shape  # Conv1D: (d_in, d_out)
        linear = nn.Linear(d_in, d_out, bias=module.bias is not None)
        with torch.no_grad():
            linear.weight.copy_(module.weight.T)  # (d_out, d_in)
            if module.bias is not None:
                linear.bias.copy_(module.bias)
        return linear

    # Collect all Linear/Conv1D layers first (can't modify dict during iteration)
    linear_layers = []
    for name, module in model.named_modules():
        if _is_linear(module):
            # Check skip patterns
            if any(pat in name for pat in skip_patterns):
                if verbose:
                    print(f"  SKIP {name} (matches skip pattern)")
                continue
            # Skip layers that are too small to compress (not applicable in lossless mode)
            d_out, d_in = _get_dims(module)
            if not lossless and d_in < n_axes * 2:
                if verbose:
                    print(f"  SKIP {name} (d_in={d_in} < 2*n_axes={2*n_axes})")
                continue
            linear_layers.append((name, module))

    if verbose:
        mode_str = "LOSSLESS (n_axes=d_in)" if lossless else f"n_axes={n_axes}"
        print(f"Converting {len(linear_layers)} layers to PPAI ({mode_str})")

    for name, module in linear_layers:
        # Normalize Conv1D → nn.Linear for uniform handling
        linear = _to_linear(module)
        d_out, d_in = linear.weight.shape

        # In lossless mode, n_axes = d_in for each layer
        layer_n_axes = d_in if lossless else n_axes

        if verbose:
            print(f"  [{layers_converted+1}/{len(linear_layers)}] {name} "
                  f"({d_in} -> {d_out}) ...", end=" ", flush=True)

        if lossless:
            # Lossless: use a fixed angle, no optimization needed
            alpha, psi = np.pi / 2, 0.0
            rel_err = 0.0  # Exact by construction
        else:
            # Find optimal (α, ψ) for this weight matrix
            W_np = linear.weight.detach().cpu().float().numpy()
            alpha, psi = find_optimal_angles(W_np, layer_n_axes, grid_steps)
            P = build_ppt_projection(d_in, layer_n_axes, alpha, psi)
            rel_err = relative_projection_error(W_np, P)

        # Create compressed layer
        ppai_layer = PPAILinear.from_linear(linear, layer_n_axes, alpha, psi)

        # Replace in model
        _set_module(model, name, ppai_layer)

        # Record manifest entry
        spec = ProjectionSpec(d_in=d_in, n_axes=layer_n_axes, alpha=alpha, psi=psi)
        manifest[name] = {
            "spec": spec.to_dict(),
            "d_out": d_out,
            "has_bias": linear.bias is not None,
            "rel_error": float(rel_err),
            "weight_hash": _weight_hash(ppai_layer.weight),
            "compression_ratio": ppai_layer.compression_ratio(),
            "lossless": lossless,
        }

        orig_params = d_in * d_out + (d_out if linear.bias is not None else 0)
        comp_params = layer_n_axes * d_out + (d_out if module.bias is not None else 0)
        total_original_params += orig_params
        total_compressed_params += comp_params
        layers_converted += 1

        if verbose:
            if lossless:
                print(f"LOSSLESS n_axes={layer_n_axes} ratio=1.0x")
            else:
                print(f"alpha={np.degrees(alpha):.1f} psi={np.degrees(psi):.1f} "
                      f"err={rel_err:.4f} ratio={d_in/layer_n_axes:.1f}x")

    if verbose:
        overall_ratio = total_original_params / max(total_compressed_params, 1)
        print(f"\nDone. {layers_converted} layers converted.")
        print(f"  Original params (linear): {total_original_params:,}")
        print(f"  Compressed params:        {total_compressed_params:,}")
        print(f"  Overall compression:      {overall_ratio:.1f}x")

    return manifest


def save_ppai_model(model: nn.Module, manifest: dict, output_dir: str,
                    original_config: Optional[dict] = None):
    """Save a converted PPAI model to disk.

    Saves:
        - model weights (safetensors or pt)
        - manifest.json (per-layer specs + hashes)
        - ppai_config.json (global settings)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save weights
    torch.save(model.state_dict(), out / "model.pt")

    # Save manifest
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save PPAI config
    ppai_config = {
        "version": __version__,
        "spec_version": 1,
        "float_format": "float32",
        "trace_granularity": "vector",
    }
    if original_config:
        ppai_config["original_config"] = original_config
    with open(out / "ppai_config.json", "w") as f:
        json.dump(ppai_config, f, indent=2)


def load_manifest(model_dir: str) -> dict:
    """Load a PPAI manifest from disk."""
    with open(Path(model_dir) / "manifest.json") as f:
        return json.load(f)


def load_ppai_model(model_dir: str, model_name_or_path: str = "gpt2",
                    verbose: bool = True) -> tuple:
    """Load a saved PPAI model from disk.

    Reconstructs the model architecture with PPAILinear layers,
    then loads the saved state dict.

    Args:
        model_dir: Path to saved PPAI model directory.
        model_name_or_path: HF model ID for the base architecture.
        verbose: Print progress.

    Returns:
        (model, manifest) tuple.
    """
    from transformers import AutoModelForCausalLM

    manifest = load_manifest(model_dir)

    if verbose:
        print(f"Loading base architecture: {model_name_or_path}")

    # Create the base model (just the architecture, weights will be overwritten)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float32
    )

    # Replace Linear/Conv1D layers that appear in the manifest with PPAILinear
    for layer_name, info in manifest.items():
        spec = info["spec"]
        d_out = info["d_out"]
        ppai_layer = PPAILinear(
            d_in=spec["d_in"],
            d_out=d_out,
            n_axes=spec["n_axes"],
            alpha=spec["alpha"],
            psi=spec["psi"],
            bias=info.get("has_bias", True),
        )
        _set_module(model, layer_name, ppai_layer)

    # Load saved weights (strict=False to skip P buffers from older saves)
    state_dict = torch.load(Path(model_dir) / "model.pt", map_location="cpu",
                            weights_only=True)
    # Filter out any P buffer keys — P is reconstructed from spec
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.endswith('.P')}
    model.load_state_dict(state_dict)
    model.eval()

    if verbose:
        print(f"Loaded PPAI model: {len(manifest)} compressed layers")

    return model, manifest
