"""Unit tests for full-catalog candidate masking, tie-breaking, HR@10, and NDCG@10."""

import math
import pytest
import torch

from src.evaluation.compute_hr_ndcg import compute_hr_ndcg
from src.evaluation.mask_seen_items import mask_seen_items
from src.evaluation.rank_candidates import rank_candidates
from src.run_evaluation import run_evaluation


def test_hand_computed_hr_and_ndcg_at_10() -> None:
    # 4 users with 1-indexed ranks: 1, 3, 10, 11
    # Hits@10: [1, 1, 1, 0] -> HR@10 = 3 / 4 = 0.75
    # NDCG@10: (1/log2(2) + 1/log2(4) + 1/log2(11) + 0) / 4
    ranks = [1, 3, 10, 11]
    expected_hr = 0.75
    expected_ndcg = (1.0 + 0.5 + (1.0 / math.log2(11.0)) + 0.0) / 4.0

    metrics = compute_hr_ndcg(ranks=ranks, top_k=10)
    assert metrics["HR@10"] == pytest.approx(expected_hr, abs=1e-9)
    assert metrics["NDCG@10"] == pytest.approx(expected_ndcg, abs=1e-9)


def test_mask_seen_items_excludes_history_and_pad_but_keeps_target() -> None:
    # Vocab size = 6 (0=pad, 1..5 valid items)
    scores = torch.tensor(
        [
            [999.0, 10.0, 8.0, 5.0, 4.0, 2.0],  # User 0: seen [1, 2], target=3 -> rank 1!
            [999.0, 9.0, 7.0, 6.0, 5.0, 1.0],   # User 1: repeat target=2 in seen [1, 2], target=2 must stay rankable!
        ],
        dtype=torch.float32,
    )
    seen = [[1, 2, 0], [1, 2, 0]]
    targets = [3, 2]

    result = run_evaluation(scores=scores, seen_histories=seen, target_items=targets, top_k=3, pad_id=0)
    # User 0: pad=0 masked, 1 & 2 masked -> remaining [3 (5.0), 4 (4.0), 5 (2.0)] -> target 3 is rank 1
    # User 1: pad=0 masked, 1 masked, 2 restored (7.0) -> remaining [2 (7.0), 3 (6.0), 4 (5.0), 5 (1.0)] -> target 2 is rank 1
    assert result["ranks"].tolist() == [1, 1]
    assert result["HR@3"] == pytest.approx(1.0)
    assert result["NDCG@3"] == pytest.approx(1.0)
    assert (result["top_k_ids"] != 0).all()


def test_deterministic_tie_breaking_by_smaller_item_id() -> None:
    # Items 1..5 all have identical score 3.5; pad_id=0 has 100.0
    scores = torch.tensor([[100.0, 3.5, 3.5, 3.5, 3.5, 3.5]], dtype=torch.float32)
    masked = mask_seen_items(scores=scores, seen_histories=[[2]], target_items=[4], pad_id=0)

    # Valid unmasked items are 1, 3, 4, 5 all tied at 3.5 -> sorted order [1, 3, 4, 5], so item 4 is rank 3
    ranked_first = rank_candidates(masked_scores=masked, target_items=[4], top_k=4, pad_id=0)
    ranked_second = rank_candidates(masked_scores=masked, target_items=[4], top_k=4, pad_id=0)

    assert ranked_first["top_k_ids"].tolist() == [[1, 3, 4, 5]]
    assert ranked_first["ranks"].tolist() == [3]
    assert torch.equal(ranked_first["top_k_ids"], ranked_second["top_k_ids"])
    assert torch.equal(ranked_first["ranks"], ranked_second["ranks"])
