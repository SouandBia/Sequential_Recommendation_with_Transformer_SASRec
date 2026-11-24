"""Score full-catalog next-item logits from last valid sequence positions in mini-batches."""

import torch
from torch import nn


def score_split(
    model: nn.Module,
    input_ids: torch.Tensor,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Compute [num_users, num_items + 1] full-catalog scores on CPU in evaluation mode."""
    if input_ids.ndim != 2 or input_ids.size(0) == 0:
        raise ValueError("input_ids must be a non-empty 2D tensor [num_users, max_seq_len].")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    target_device = torch.device(device)
    model.eval()
    score_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, input_ids.size(0), batch_size):
            batch_x = input_ids[start : start + batch_size].to(target_device, non_blocking=True)
            logits = model(batch_x, all_positions=False)
            score_chunks.append(logits.detach().to(dtype=torch.float32, device="cpu"))

    return torch.cat(score_chunks, dim=0)
