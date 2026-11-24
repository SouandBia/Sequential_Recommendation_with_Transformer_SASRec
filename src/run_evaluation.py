"""Orchestrator for full-catalog candidate masking, ranking, and HR@K/NDCG@K evaluation."""

from collections.abc import Sequence
import torch

from src.evaluation.compute_hr_ndcg import compute_hr_ndcg
from src.evaluation.mask_seen_items import mask_seen_items
from src.evaluation.rank_candidates import rank_candidates


def run_evaluation(
    scores: torch.Tensor,
    seen_histories: Sequence[Sequence[int]] | torch.Tensor,
    target_items: Sequence[int] | torch.Tensor,
    top_k: int = 10,
    pad_id: int = 0,
) -> dict:
    """Mask seen history items, rank all valid catalog items, and compute HR@K and NDCG@K."""
    masked = mask_seen_items(
        scores=scores,
        seen_histories=seen_histories,
        target_items=target_items,
        pad_id=pad_id,
    )
    ranked = rank_candidates(
        masked_scores=masked,
        target_items=target_items,
        top_k=top_k,
        pad_id=pad_id,
    )
    metrics = compute_hr_ndcg(ranks=ranked["ranks"], top_k=top_k)
    return {
        **metrics,
        "ranks": ranked["ranks"],
        "top_k_ids": ranked["top_k_ids"],
        "top_k_scores": ranked["top_k_scores"],
    }
