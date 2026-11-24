"""Run a single training epoch over padded next-item batches with VRAM tracking."""

from collections.abc import Callable
import time
import torch
from torch import nn


def train_epoch(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    batch_size: int = 128,
    pad_id: int = 0,
    device: str | torch.device = "cpu",
    grad_clip: float = 1.0,
    use_amp: bool = False,
) -> dict[str, float]:
    """Train model for one epoch over shuffled mini-batches and return loss, time, and VRAM."""
    if input_ids.ndim != 2 or target_ids.shape != input_ids.shape:
        raise ValueError("input_ids and target_ids must be 2D tensors with identical shapes.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    target_device = torch.device(device)
    amp_enabled = bool(use_amp and target_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    model.train()
    num_samples = input_ids.size(0)
    permutation = torch.randperm(num_samples)

    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)

    total_loss = 0.0
    total_valid_tokens = 0
    start_time = time.perf_counter()

    for start in range(0, num_samples, batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch_x = input_ids[batch_indices].to(target_device, non_blocking=True)
        batch_y = target_ids[batch_indices].to(target_device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=target_device.type, enabled=amp_enabled):
            logits = model(batch_x, all_positions=True)
            loss = loss_fn(logits, batch_y, pad_id)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Encountered non-finite training loss: {loss.item()}")

        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        valid_tokens = int((batch_y != pad_id).sum().item())
        total_loss += float(loss.item()) * valid_tokens
        total_valid_tokens += valid_tokens

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        peak_vram_mb = float(torch.cuda.max_memory_allocated(target_device)) / (1024.0 ** 2)
    else:
        peak_vram_mb = 0.0

    elapsed_sec = time.perf_counter() - start_time
    avg_loss = total_loss / max(total_valid_tokens, 1)

    return {
        "train_loss": avg_loss,
        "epoch_sec": elapsed_sec,
        "peak_vram_mb": peak_vram_mb,
    }
