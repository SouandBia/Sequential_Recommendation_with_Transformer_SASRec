"""Right-pad and truncate item sequences for training and inference."""

from collections.abc import Sequence
import torch


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    max_seq_len: int = 100,
    pad_id: int = 0,
    build_targets: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Truncate to most recent valid items and right-pad with pad_id=0.

    When build_targets=False: returns input_ids of shape [B, max_seq_len].
    When build_targets=True: extracts next-item pairs within each sequence and returns
    (input_ids, target_ids), both right-padded to [B, max_seq_len].
    """
    if not sequences:
        raise ValueError("sequences must not be empty.")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
    if pad_id != 0:
        raise ValueError(f"CONTRACT.md strictly reserves pad_id=0, got {pad_id}")

    batch_size = len(sequences)
    input_tensor = torch.full((batch_size, max_seq_len), fill_value=pad_id, dtype=torch.long)

    if not build_targets:
        for row_idx, seq in enumerate(sequences):
            valid = [int(item_id) for item_id in seq if int(item_id) != pad_id]
            if not valid:
                raise ValueError(f"Sequence at index {row_idx} contains no valid items.")
            if any(item_id < 0 for item_id in valid):
                raise ValueError(f"Sequence at index {row_idx} contains negative item ID.")
            recent = valid[-max_seq_len:]
            input_tensor[row_idx, : len(recent)] = torch.tensor(recent, dtype=torch.long)
        return input_tensor

    target_tensor = torch.full((batch_size, max_seq_len), fill_value=pad_id, dtype=torch.long)
    for row_idx, seq in enumerate(sequences):
        valid = [int(item_id) for item_id in seq if int(item_id) != pad_id]
        if len(valid) < 2:
            raise ValueError(
                f"Sequence at index {row_idx} has {len(valid)} valid items; next-item targets require >= 2."
            )
        if any(item_id < 0 for item_id in valid):
            raise ValueError(f"Sequence at index {row_idx} contains negative item ID.")
        window = valid[-(max_seq_len + 1) :]
        inp = window[:-1]
        tgt = window[1:]
        input_tensor[row_idx, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        target_tensor[row_idx, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)

    return input_tensor, target_tensor
