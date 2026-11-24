"""Compute full-catalog HR@K and NDCG@K across user history-length quantile groups."""

from collections.abc import Mapping, Sequence
import math
import numpy as np


def _cohort_labels(num_quantiles: int) -> list[str]:
    if num_quantiles == 3:
        return ["Short", "Medium", "Long"]
    return [f"Q{idx + 1}" for idx in range(num_quantiles)]


def group_by_length(
    user_lengths: Sequence[int],
    ranks_by_model: Mapping[str, Sequence[int]],
    top_k: int = 10,
    num_quantiles: int = 3,
) -> list[dict]:
    """Partition users into balanced length-quantile cohorts and evaluate every model per cohort."""
    num_users = len(user_lengths)
    if num_users == 0:
        raise ValueError("user_lengths must not be empty.")
    if num_quantiles <= 0 or num_quantiles > num_users:
        raise ValueError(f"num_quantiles={num_quantiles} invalid for {num_users} users.")
    if not ranks_by_model:
        raise ValueError("ranks_by_model must not be empty.")

    for model_name, ranks in ranks_by_model.items():
        if len(ranks) != num_users:
            raise ValueError(
                f"Model {model_name!r} has {len(ranks)} ranks, expected {num_users}."
            )

    order = sorted(range(num_users), key=lambda idx: (int(user_lengths[idx]), idx))
    chunks = np.array_split(np.asarray(order, dtype=np.int64), num_quantiles)
    labels = _cohort_labels(num_quantiles)

    rows: list[dict] = []
    for cohort_idx, (label, indices) in enumerate(zip(labels, chunks), start=1):
        idx_list = indices.tolist()
        cohort_lens = [int(user_lengths[i]) for i in idx_list]
        min_len = min(cohort_lens)
        max_len = max(cohort_lens)
        median_len = float(np.median(cohort_lens))

        for model_name, ranks in ranks_by_model.items():
            cohort_ranks = [int(ranks[i]) for i in idx_list]
            hits = [1.0 if r <= top_k else 0.0 for r in cohort_ranks]
            ndcgs = [1.0 / math.log2(r + 1.0) if r <= top_k else 0.0 for r in cohort_ranks]
            rows.append(
                {
                    "cohort_index": cohort_idx,
                    "cohort_label": label,
                    "length_range": f"{min_len}-{max_len}",
                    "min_len": min_len,
                    "max_len": max_len,
                    "median_len": round(median_len, 1),
                    "user_count": len(idx_list),
                    "model": model_name,
                    f"HR@{top_k}": round(sum(hits) / len(hits), 6),
                    f"NDCG@{top_k}": round(sum(ndcgs) / len(ndcgs), 6),
                    "median_rank": float(np.median(cohort_ranks)),
                    "mean_rank": round(float(np.mean(cohort_ranks)), 2),
                }
            )
    return rows
