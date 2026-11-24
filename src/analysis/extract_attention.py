"""Extract sequence-level causal attention diagnostics from a trained SASRec model."""

import torch
from torch import nn


def extract_attention(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int = 1,
    head_idx: int = 0,
    pad_id: int = 0,
) -> dict:
    """Extract causal attention weights for one sequence at a chosen layer and head.

    Diagnostic caveat: attention weights reflect position-to-position relationships
    inside the sequence and do NOT constitute causal proof for a recommended movie.
    """
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError(f"input_ids must have shape [1, seq_len], got {tuple(input_ids.shape)}")

    valid_mask = input_ids[0] != pad_id
    valid_len = int(valid_mask.sum().item())
    if valid_len <= 0:
        raise ValueError("input_ids must contain at least 1 valid non-padding item.")

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        _, attentions = model(input_ids.to(device), all_positions=False, return_attention=True)

    if layer_idx < 0 or layer_idx >= len(attentions):
        raise ValueError(f"layer_idx={layer_idx} out of range for {len(attentions)} layers.")

    layer_attn = attentions[layer_idx]
    if head_idx < 0 or head_idx >= layer_attn.size(1):
        raise ValueError(f"head_idx={head_idx} out of range for {layer_attn.size(1)} heads.")

    matrix = layer_attn[0, head_idx, :valid_len, :valid_len].detach().cpu().to(torch.float32)
    last_query_weights = matrix[valid_len - 1].tolist()

    return {
        "layer": int(layer_idx),
        "head": int(head_idx),
        "valid_len": valid_len,
        "matrix": [[round(float(v), 6) for v in row] for row in matrix.tolist()],
        "last_query_weights": [round(float(v), 6) for v in last_query_weights],
        "caveat": (
            "Sequence-level attention weights illustrate position-to-position weighting "
            "within the chosen layer/head and do not prove causal attribution for a specific recommendation."
        ),
    }
