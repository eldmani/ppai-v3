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

"""HuggingFace Transformers integration.

Provides high-level functions to convert, run, and verify HF models.
"""

from typing import Optional
from pathlib import Path

import torch
import numpy as np

from ..compress.convert import convert_model, save_ppai_model, load_manifest
from ..trace.recorder import TraceRecorder
from ..trace.format import save_trace, load_trace


def convert_hf_model(model_name_or_path: str,
                     n_axes: int = 96,
                     output_dir: Optional[str] = None,
                     skip_lm_head: bool = True,
                     grid_steps: int = 20,
                     verbose: bool = True) -> tuple:
    """Convert a HuggingFace model to PPAI format.

    Args:
        model_name_or_path: HF model ID or local path.
        n_axes: Number of PPT axes.
        output_dir: Where to save. Defaults to ./{model_name}-ppai/
        skip_lm_head: Whether to skip the language model head (recommended).
        grid_steps: Optimization grid resolution.
        verbose: Print progress.

    Returns:
        (model, manifest) — the converted model and its manifest.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    if verbose:
        print(f"Loading model: {model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
    )
    model.eval()

    config = AutoConfig.from_pretrained(model_name_or_path)

    skip_patterns = []
    if skip_lm_head:
        skip_patterns.append("lm_head")

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {total_params:,} parameters")

    manifest = convert_model(
        model, n_axes,
        grid_steps=grid_steps,
        skip_patterns=skip_patterns,
        verbose=verbose,
    )

    if output_dir:
        save_ppai_model(model, manifest, output_dir,
                        original_config=config.to_dict())
        # Also save tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            tokenizer.save_pretrained(output_dir)
            if verbose:
                print(f"Tokenizer saved to {output_dir}")
        except Exception:
            pass
        if verbose:
            print(f"PPAI model saved to {output_dir}")

    return model, manifest


def infer_with_trace(model: torch.nn.Module,
                     input_ids: torch.Tensor,
                     model_hash: str = "",
                     trace_path: Optional[str] = None,
                     **generate_kwargs) -> tuple:
    """Run inference with trace recording.

    Args:
        model: PPAI-converted model.
        input_ids: Input token IDs, shape (1, seq_len).
        model_hash: Optional model version hash.
        trace_path: If provided, save trace to this file.
        **generate_kwargs: Passed to model.generate().

    Returns:
        (output_ids, trace) — generated tokens and the arithmetic trace.
    """
    recorder = TraceRecorder()

    with recorder.recording(model, input_ids, model_hash=model_hash):
        with torch.no_grad():
            output_ids = model.generate(input_ids, **generate_kwargs)

    trace = recorder.get_trace()

    if trace_path and trace is not None:
        save_trace(trace, trace_path)

    return output_ids, trace


def extract_weights_for_verification(model: torch.nn.Module) -> dict:
    """Extract compressed weight matrices as numpy arrays for the verifier.

    Returns:
        Dict mapping layer_name → np.ndarray (W_comp).
    """
    from ..layers.linear import PPAILinear

    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, PPAILinear):
            weights[name] = module.weight.detach().cpu().float().numpy()
    return weights
