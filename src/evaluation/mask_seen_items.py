"""Mask padding and seen history items while keeping held-out target items rankable."""

from collections.abc import Sequence
import torch


def mask_seen_items(
    scores: torch.Tensor,
    seen_histories: Sequence[Sequence[int]] | torch.Tensor,
    target_items: Sequence[int] | torch.Tensor | None = None,
    pad_id: int = 0,
) -> torch.Tensor:
    """Set scores for pad_id and seen history items to -inf while preserving target_items."""
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D [batch_size, vocab_size], got shape {tuple(scores.shape)}")

    batch_size, vocab_size = scores.shape
    if pad_id < 0 or pad_id >= vocab_size:
        raise ValueError(f"pad_id={pad_id} out of bounds for vocab_size={vocab_size}")

    work_dtype = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
    if not scores.is_floating_point():
        work_dtype = torch.float32

    original = scores.to(dtype=work_dtype)
    masked = original.clone()
    masked[:, pad_id] = -torch.inf

    if isinstance(seen_histories, torch.Tensor):
        if seen_histories.size(0) != batch_size:
            raise ValueError("seen_histories batch dimension does not match scores.")
        history_rows = seen_histories.tolist()
    else:
        if len(seen_histories) != batch_size:
            raise ValueError("seen_histories length does not match scores batch size.")
        history_rows = seen_histories

    for row_idx, history in enumerate(history_rows):
        valid_seen = [int(item_id) for item_id in history if 0 <= int(item_id) < vocab_size and int(item_id) != pad_id]
        if valid_seen:
            seen_tensor = torch.tensor(valid_seen, dtype=torch.long, device=masked.device)
            masked[row_idx, seen_tensor] = -torch.inf

    if target_items is not None:
        target_list = target_items.tolist() if isinstance(target_items, torch.Tensor) else list(target_items)
        if len(target_list) != batch_size:
            raise ValueError("target_items length does not match scores batch size.")
        for row_idx, target in enumerate(target_list):
            target_id = int(target)
            if target_id == pad_id or target_id < 0 or target_id >= vocab_size:
                raise ValueError(f"Invalid target item ID {target_id} at row {row_idx}.")
            masked[row_idx, target_id] = original[row_idx, target_id]

    return masked
