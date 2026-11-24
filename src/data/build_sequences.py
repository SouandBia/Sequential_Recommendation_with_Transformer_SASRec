"""Build chronologically ordered, contiguously indexed user interaction sequences.

Data caveats (CONTRACT.md Section 3):
- MovieLens-1M timestamps reflect when ratings were recorded, NOT verified watch times.
- Equal timestamps for the same user are broken deterministically by
  (timestamp, raw_movie_id, original_line_index).
- Item ID 0 is strictly reserved for padding; rated movies map contiguously to 1..N.
"""

from collections import Counter, defaultdict
import hashlib
import json
import statistics


def _compute_mapping_version(item_id_to_raw: dict[int, int]) -> str:
    canonical = json.dumps(
        {str(k): int(v) for k, v in sorted(item_id_to_raw.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"ml1m_v1_{digest}"


def _summarize_lengths(lengths: list[int]) -> dict[str, float | int]:
    ordered = sorted(lengths)
    n = len(ordered)
    p25_idx = int(0.25 * (n - 1))
    p75_idx = int(0.75 * (n - 1))
    p90_idx = int(0.90 * (n - 1))
    return {
        "min": int(ordered[0]),
        "p25": int(ordered[p25_idx]),
        "median": float(statistics.median(ordered)),
        "mean": round(float(statistics.fmean(ordered)), 4),
        "p75": int(ordered[p75_idx]),
        "p90": int(ordered[p90_idx]),
        "max": int(ordered[-1]),
    }


def build_sequences(
    ratings: list[tuple[int, int, int, int, int]],
    movies: dict[int, dict],
    min_interactions: int = 5,
) -> dict:
    """Index MovieLens interactions into contiguous 1..N items and chronological user sequences."""
    if min_interactions < 3:
        raise ValueError("min_interactions must be >= 3 to support Leave-One-Out train/val/test.")

    per_user_raw: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    dropped_missing_movie = 0
    rating_counter: Counter[int] = Counter()

    for raw_user_id, raw_movie_id, rating, timestamp, line_index in ratings:
        if raw_movie_id not in movies:
            dropped_missing_movie += 1
            continue
        per_user_raw[raw_user_id].append((timestamp, raw_movie_id, line_index, rating))

    deduped_users: dict[int, list[int]] = {}
    dropped_duplicate_user_item = 0
    dropped_short_users = 0
    users_with_timestamp_ties = 0
    tied_interaction_steps = 0
    valid_interactions = 0

    for raw_user_id in sorted(per_user_raw.keys()):
        # Deterministic tie-breaking: (timestamp, raw_movie_id, line_index)
        events = sorted(per_user_raw[raw_user_id], key=lambda row: (row[0], row[1], row[2]))

        seen_movies: set[int] = set()
        clean_events: list[tuple[int, int, int]] = []
        for timestamp, raw_movie_id, _line_idx, rating in events:
            if raw_movie_id in seen_movies:
                dropped_duplicate_user_item += 1
                continue
            seen_movies.add(raw_movie_id)
            clean_events.append((timestamp, raw_movie_id, rating))

        if len(clean_events) < min_interactions:
            dropped_short_users += 1
            continue

        ties_for_user = sum(
            1 for idx in range(1, len(clean_events)) if clean_events[idx][0] == clean_events[idx - 1][0]
        )
        if ties_for_user > 0:
            users_with_timestamp_ties += 1
            tied_interaction_steps += ties_for_user

        for _ts, _mid, rating in clean_events:
            rating_counter[rating] += 1

        deduped_users[raw_user_id] = [raw_mid for _ts, raw_mid, _rating in clean_events]
        valid_interactions += len(clean_events)

    if not deduped_users:
        raise ValueError("No users satisfied min_interactions after filtering.")

    rated_raw_movies = sorted({raw_mid for seq in deduped_users.values() for raw_mid in seq})
    raw_to_item_id = {raw_mid: idx for idx, raw_mid in enumerate(rated_raw_movies, start=1)}
    item_id_to_raw = {idx: raw_mid for raw_mid, idx in raw_to_item_id.items()}
    item2meta = {idx: dict(movies[raw_mid]) for idx, raw_mid in item_id_to_raw.items()}

    user_sequences: dict[int, list[int]] = {}
    seq_lengths: list[int] = []
    for internal_uid, raw_uid in enumerate(sorted(deduped_users.keys()), start=1):
        mapped_seq = [raw_to_item_id[raw_mid] for raw_mid in deduped_users[raw_uid]]
        user_sequences[internal_uid] = mapped_seq
        seq_lengths.append(len(mapped_seq))

    num_users = len(user_sequences)
    num_items = len(raw_to_item_id)
    density = valid_interactions / float(num_users * num_items)

    stats = {
        "raw_interactions": len(ratings),
        "valid_interactions": valid_interactions,
        "catalog_movies_total": len(movies),
        "rated_movies_retained": num_items,
        "unrated_movies_excluded": len(movies) - num_items,
        "dropped_missing_movie_interactions": dropped_missing_movie,
        "dropped_duplicate_user_item_interactions": dropped_duplicate_user_item,
        "dropped_short_users": dropped_short_users,
        "users_with_timestamp_ties": users_with_timestamp_ties,
        "tied_timestamp_steps": tied_interaction_steps,
        "tie_break_policy": "(timestamp, raw_movie_id, original_line_index)",
        "density": round(density, 6),
        "sparsity": round(1.0 - density, 6),
        "rating_counts": {int(k): int(rating_counter[k]) for k in sorted(rating_counter.keys())},
        "sequence_lengths": _summarize_lengths(seq_lengths),
        "all_user_lengths": seq_lengths,
    }

    return {
        "user_sequences": user_sequences,
        "num_users": num_users,
        "num_items": num_items,
        "item2meta": item2meta,
        "raw_to_item_id": raw_to_item_id,
        "item_id_to_raw": item_id_to_raw,
        "mapping_version": _compute_mapping_version(item_id_to_raw),
        "stats": stats,
    }
