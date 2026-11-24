"""Fit the Most-Popular baseline strictly from training split interaction counts."""

from collections.abc import Mapping, Sequence
import torch


def fit_popularity(
    train_sequences: Mapping[int, Sequence[int]],
    num_items: int,
    pad_id: int = 0,
) -> dict:
    """Count global item frequencies from train_sequences only (zero val/test leakage)."""
    if not train_sequences:
        raise ValueError("train_sequences must not be empty.")
    if num_items <= 0:
        raise ValueError(f"num_items must be > 0, got {num_items}")
    if pad_id != 0:
        raise ValueError(f"CONTRACT.md strictly reserves pad_id=0, got {pad_id}")

    counts = torch.zeros(num_items + 1, dtype=torch.float32)
    total_train_events = 0

    for user_id, seq in train_sequences.items():
        for raw_item in seq:
            item_id = int(raw_item)
            if item_id == pad_id:
                continue
            if item_id < 0 or item_id > num_items:
                raise ValueError(
                    f"User {user_id} has item_id={item_id} outside [1, {num_items}]."
                )
            counts[item_id] += 1.0
            total_train_events += 1

    if total_train_events == 0:
        raise ValueError("train_sequences contained zero valid non-padding interactions.")

    scores = counts.clone()
    scores[pad_id] = -1e9

    return {
        "counts": counts,
        "scores": scores,
        "total_train_events": total_train_events,
    }
