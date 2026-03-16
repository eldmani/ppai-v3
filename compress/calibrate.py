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

"""Post-compression calibration via knowledge distillation.

Fine-tunes compressed weights W_comp while keeping projection P frozen.
Uses the original model as teacher (soft target distillation).

Note: This module is implemented but not yet validated end-to-end.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..layers.linear import PPAILinear


def calibrate(compressed_model: nn.Module,
              teacher_model: nn.Module,
              dataloader: DataLoader,
              epochs: int = 2,
              lr: float = 1e-4,
              temperature: float = 2.0,
              device: str = "cpu",
              verbose: bool = True) -> dict:
    """Calibrate compressed model weights via distillation from teacher.

    Only W_comp and biases are updated. Projection matrices P are frozen.

    Args:
        compressed_model: PPAI-converted model.
        teacher_model: Original (uncompressed) model.
        dataloader: DataLoader yielding input tensors (e.g., token IDs).
        epochs: Number of calibration epochs.
        lr: Learning rate.
        temperature: Distillation temperature.
        device: "cpu" or "cuda".
        verbose: Print progress.

    Returns:
        Dict with calibration stats (loss history, etc.).
    """
    compressed_model = compressed_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Freeze projection buffers, only train W_comp + bias
    for name, param in compressed_model.named_parameters():
        param.requires_grad = True
    for name, buf in compressed_model.named_buffers():
        # P is registered as a buffer — it's already non-trainable
        pass

    optimizer = torch.optim.AdamW(
        [p for p in compressed_model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader)
    )

    stats = {"losses": [], "epoch_losses": []}

    for epoch in range(epochs):
        compressed_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items()}
            else:
                inputs = batch.to(device)

            # Teacher forward (no grad)
            with torch.no_grad():
                if isinstance(inputs, dict):
                    teacher_out = teacher_model(**inputs)
                else:
                    teacher_out = teacher_model(inputs)
                if hasattr(teacher_out, "logits"):
                    teacher_logits = teacher_out.logits
                else:
                    teacher_logits = teacher_out

            # Student forward
            if isinstance(inputs, dict):
                student_out = compressed_model(**inputs)
            else:
                student_out = compressed_model(inputs)
            if hasattr(student_out, "logits"):
                student_logits = student_out.logits
            else:
                student_logits = student_out

            # KL divergence loss (soft targets)
            loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(compressed_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            stats["losses"].append(batch_loss)
            epoch_loss += batch_loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        stats["epoch_losses"].append(avg_loss)

        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    return stats
