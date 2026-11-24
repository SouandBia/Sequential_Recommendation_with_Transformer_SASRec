"""Collect concrete, diverse representative error cases from held-out test predictions."""

from collections import Counter
from collections.abc import Mapping, Sequence


def _format_movie(item_id: int, item2meta: Mapping[int, dict]) -> dict:
    meta = item2meta.get(int(item_id), {"title": f"Item {item_id}", "genres": [], "year": None})
    return {
        "item_id": int(item_id),
        "title": str(meta["title"]),
        "genres": list(meta.get("genres", [])),
        "year": meta.get("year"),
    }


def _build_case_record(
    case_type: str,
    user_idx: int,
    user_ids: Sequence[int],
    test_histories: Sequence[Sequence[int]],
    test_targets: Sequence[int],
    ranks_by_model: Mapping[str, Sequence[int]],
    top_k_ids_by_model: Mapping[str, Sequence[Sequence[int]]],
    top_k_scores_by_model: Mapping[str, Sequence[Sequence[float]]],
    item2meta: Mapping[int, dict],
    train_counts: Mapping[int, int] | None,
) -> dict:
    history = list(test_histories[user_idx])
    target_id = int(test_targets[user_idx])
    recent_items = [_format_movie(iid, item2meta) for iid in history[-8:]]
    target_meta = _format_movie(target_id, item2meta)

    recent_genres = Counter(g for item in recent_items[-5:] for g in item["genres"])
    dominant_recent_genres = [g for g, _c in recent_genres.most_common(3)]

    model_predictions: dict[str, dict] = {}
    for model_name in ranks_by_model.keys():
        top_ids = list(top_k_ids_by_model[model_name][user_idx])[:5]
        top_scores = list(top_k_scores_by_model[model_name][user_idx])[:5]
        model_predictions[model_name] = {
            "target_rank": int(ranks_by_model[model_name][user_idx]),
            "top_5": [
                {
                    **_format_movie(iid, item2meta),
                    "score": round(float(score), 4),
                }
                for iid, score in zip(top_ids, top_scores)
            ],
        }

    return {
        "case_type": case_type,
        "user_id": int(user_ids[user_idx]),
        "history_length": len(history),
        "recent_8_history": recent_items,
        "dominant_recent_5_genres": dominant_recent_genres,
        "target_movie": target_meta,
        "target_train_frequency": int(train_counts.get(target_id, 0)) if train_counts else None,
        "model_predictions": model_predictions,
    }


def collect_error_cases(
    user_ids: Sequence[int],
    test_histories: Sequence[Sequence[int]],
    test_targets: Sequence[int],
    ranks_by_model: Mapping[str, Sequence[int]],
    top_k_ids_by_model: Mapping[str, Sequence[Sequence[int]]],
    top_k_scores_by_model: Mapping[str, Sequence[Sequence[float]]],
    item2meta: Mapping[int, dict],
    train_counts: Mapping[int, int] | None = None,
    max_cases: int = 3,
) -> list[dict]:
    """Select diverse test failure cases: SASRec-miss/GRU-hit, genre shift, and long-tail/truncation."""
    num_users = len(user_ids)
    if num_users == 0:
        raise ValueError("user_ids must not be empty.")
    if max_cases <= 0:
        raise ValueError(f"max_cases must be positive, got {max_cases}")

    sas_ranks = ranks_by_model["sasrec"]
    gru_ranks = ranks_by_model["gru4rec"]
    pop_ranks = ranks_by_model["popularity"]

    chosen_indices: list[tuple[str, int]] = []
    used: set[int] = set()

    # Case 1: GRU4Rec hits Top-10 while SASRec misses clearly (rank > 25)
    c1_candidates = [
        i
        for i in range(num_users)
        if int(gru_ranks[i]) <= 10 and int(sas_ranks[i]) >= 30 and 25 <= len(test_histories[i]) <= 120
    ]
    if c1_candidates:
        c1_candidates.sort(key=lambda i: (int(gru_ranks[i]), -min(int(sas_ranks[i]), 300), i))
        chosen_indices.append(("sasrec_miss_gru4rec_hit", c1_candidates[0]))
        used.add(c1_candidates[0])

    # Case 2: Abrupt genre shift relative to last 5 items -> both SASRec & GRU4Rec miss Top-10
    c2_candidates: list[tuple[int, int]] = []
    for i in range(num_users):
        if i in used or int(sas_ranks[i]) <= 20 or int(gru_ranks[i]) <= 20:
            continue
        hist = test_histories[i]
        if len(hist) < 8:
            continue
        recent_genre_counts = Counter(
            g for iid in hist[-5:] for g in item2meta.get(int(iid), {}).get("genres", [])
        )
        target_genres = set(item2meta.get(int(test_targets[i]), {}).get("genres", []))
        if not recent_genre_counts or not target_genres:
            continue
        top_genre, top_count = recent_genre_counts.most_common(1)[0]
        if top_count >= 4 and target_genres.isdisjoint(recent_genre_counts.keys()):
            c2_candidates.append((-top_count, i))
    if c2_candidates:
        c2_candidates.sort(key=lambda pair: (pair[0], int(sas_ranks[pair[1]]), pair[1]))
        chosen_indices.append(("abrupt_genre_shift_both_miss", c2_candidates[0][1]))
        used.add(c2_candidates[0][1])

    # Case 3: Long-history (>150) with low-frequency niche target missed by all three models
    c3_candidates: list[tuple[int, int]] = []
    for i in range(num_users):
        if i in used:
            continue
        if len(test_histories[i]) > 150 and int(sas_ranks[i]) > 50 and int(gru_ranks[i]) > 50 and int(pop_ranks[i]) > 100:
            freq = int(train_counts.get(int(test_targets[i]), 999999)) if train_counts else 0
            if 5 <= freq <= 80 or train_counts is None:
                c3_candidates.append((freq, i))
    if c3_candidates:
        c3_candidates.sort(key=lambda pair: (pair[0], -len(test_histories[pair[1]]), pair[1]))
        chosen_indices.append(("long_history_niche_target_miss", c3_candidates[0][1]))
        used.add(c3_candidates[0][1])

    # Fallback fill if fewer than max_cases matched (e.g., small synthetic datasets)
    if len(chosen_indices) < max_cases:
        remaining = [i for i in range(num_users) if i not in used and int(sas_ranks[i]) > 10]
        if not remaining:
            remaining = [i for i in range(num_users) if i not in used]
        remaining.sort(key=lambda i: (-int(sas_ranks[i]), i))
        for idx in remaining:
            if len(chosen_indices) >= max_cases:
                break
            chosen_indices.append(("general_sasrec_miss", idx))
            used.add(idx)

    return [
        _build_case_record(
            case_type=case_type,
            user_idx=user_idx,
            user_ids=user_ids,
            test_histories=test_histories,
            test_targets=test_targets,
            ranks_by_model=ranks_by_model,
            top_k_ids_by_model=top_k_ids_by_model,
            top_k_scores_by_model=top_k_scores_by_model,
            item2meta=item2meta,
            train_counts=train_counts,
        )
        for case_type, user_idx in chosen_indices[:max_cases]
    ]
