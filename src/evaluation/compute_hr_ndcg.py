"""Compute Hit Rate (HR@K) and Normalized Discounted Cumulative Gain (NDCG@K)."""

from collections.abc import Sequence
import torch


def compute_hr_ndcg(
    ranks: Sequence[int] | torch.Tensor,
    top_k: int = 10,
) -> dict[str, float]:
    """Compute HR@K and NDCG@K from 1-indexed held-out target ranks."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    rank_tensor = (
        ranks.detach().to(dtype=torch.float64, device="cpu")
        if isinstance(ranks, torch.Tensor)
        else torch.tensor(list(ranks), dtype=torch.float64)
    )
    if rank_tensor.ndim != 1 or rank_tensor.numel() == 0:
        raise ValueError("ranks must be a non-empty 1D sequence of 1-indexed ranks.")
    if (rank_tensor < 1).any() or not torch.isfinite(rank_tensor).all():
        raise ValueError("All ranks must be finite integers >= 1.")

    hits = rank_tensor <= float(top_k)
    hr = float(hits.to(torch.float64).mean().item())
    gains = torch.where(
        hits,
        1.0 / torch.log2(rank_tensor + 1.0),
        torch.zeros_like(rank_tensor),
    )
    ndcg = float(gains.mean().item())

    return {
        f"HR@{top_k}": hr,
        f"NDCG@{top_k}": ndcg,
    }
