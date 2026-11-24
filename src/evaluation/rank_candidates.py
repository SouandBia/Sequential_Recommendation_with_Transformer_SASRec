"""Rank full-catalog candidates with deterministic ascending item_id tie-breaking."""

from collections.abc import Sequence
import torch


def rank_candidates(
    masked_scores: torch.Tensor,
    target_items: Sequence[int] | torch.Tensor | None = None,
    top_k: int = 10,
    pad_id: int = 0,
) -> dict[str, torch.Tensor | None]:
    """Compute top-K candidate IDs/scores and optional 1-indexed target ranks.

    Tie-breaking rule (CONTRACT.md Section 4):
    Higher score wins; exact score ties are broken deterministically by smaller item_id.
    pad_id is always forced to -inf so it can never appear as a valid candidate.
    """
    if masked_scores.ndim != 2:
        raise ValueError(f"masked_scores must be 2D, got shape {tuple(masked_scores.shape)}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    batch_size, vocab_size = masked_scores.shape
    num_valid_catalog = vocab_size - 1
    if num_valid_catalog <= 0:
        raise ValueError("vocab_size must contain at least 1 valid item beyond pad_id.")

    scores = masked_scores.clone()
    scores[:, pad_id] = -torch.inf

    effective_k = min(top_k, num_valid_catalog)
    # Stable sort of (-score) preserves ascending column index (smaller item_id wins ties).
    sorted_indices = torch.argsort(-scores, dim=-1, stable=True)
    top_k_ids = sorted_indices[:, :effective_k]
    top_k_scores = torch.gather(scores, dim=1, index=top_k_ids)

    ranks: torch.Tensor | None = None
    if target_items is not None:
        targets = (
            target_items.to(device=scores.device, dtype=torch.long)
            if isinstance(target_items, torch.Tensor)
            else torch.tensor(list(target_items), dtype=torch.long, device=scores.device)
        )
        if targets.shape != (batch_size,):
            raise ValueError(f"target_items must have shape ({batch_size},), got {tuple(targets.shape)}")
        if ((targets == pad_id) | (targets < 0) | (targets >= vocab_size)).any():
            raise ValueError("target_items contains pad_id or out-of-range item ID.")

        target_scores = scores.gather(1, targets.unsqueeze(1))
        if not torch.isfinite(target_scores).all():
            raise ValueError("One or more target items have non-finite (-inf/NaN) masked scores.")

        col_ids = torch.arange(vocab_size, device=scores.device).unsqueeze(0)
        finite_mask = torch.isfinite(scores) & (col_ids != pad_id)

        strictly_higher = (finite_mask & (scores > target_scores)).sum(dim=1)
        tied_smaller_id = (finite_mask & (scores == target_scores) & (col_ids < targets.unsqueeze(1))).sum(dim=1)
        ranks = 1 + strictly_higher + tied_smaller_id

    return {
        "top_k_ids": top_k_ids,
        "top_k_scores": top_k_scores,
        "ranks": ranks,
    }
