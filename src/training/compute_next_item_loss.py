"""Compute next-item cross-entropy loss ignoring right-padded target positions."""

import torch
import torch.nn.functional as F


def compute_next_item_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """Compute cross-entropy over valid next-item targets while excluding pad_id."""
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3D [batch, seq_len, vocab], got {tuple(logits.shape)}")
    if target_ids.ndim != 2 or target_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"target_ids shape {tuple(target_ids.shape)} does not match logits {tuple(logits.shape[:2])}."
        )
    if (target_ids < 0).any() or (target_ids >= logits.size(-1)).any():
        raise ValueError("target_ids contains out-of-bounds class indices.")
    if not (target_ids != pad_id).any():
        raise ValueError("target_ids must contain at least 1 non-padding target.")

    vocab_size = logits.size(-1)
    safe_logits = logits.clone()
    safe_logits[..., pad_id] = -1e9

    return F.cross_entropy(
        safe_logits.reshape(-1, vocab_size),
        target_ids.reshape(-1).to(dtype=torch.long),
        ignore_index=pad_id,
    )
