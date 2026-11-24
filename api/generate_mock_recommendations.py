"""Generate deterministic Week 1 mock recommendations matching CONTRACT.md schema."""


def _score_candidate(item_id: int, meta: dict, seed_genres: set[str], salt: int) -> float:
    overlap = len(seed_genres.intersection(meta.get("genres", [])))
    pseudo = ((item_id * 1103515245 + salt * 12345) & 0x7FFFFFFF) / float(0x7FFFFFFF)
    return round(1.5 * overlap + pseudo, 4)


def _rank_top_k(
    candidates: list[int],
    catalog: dict[int, dict],
    seed_genres: set[str],
    top_k: int,
    salt: int,
) -> list[dict]:
    scored = [
        (
            -_score_candidate(item_id, catalog[item_id], seed_genres, salt),
            item_id,
            _score_candidate(item_id, catalog[item_id], seed_genres, salt),
        )
        for item_id in candidates
    ]
    scored.sort(key=lambda row: (row[0], row[1]))
    results: list[dict] = []
    for _neg_score, item_id, raw_score in scored[:top_k]:
        meta = catalog[item_id]
        results.append(
            {
                "item_id": item_id,
                "title": meta["title"],
                "genres": list(meta["genres"]),
                "score": raw_score,
            }
        )
    return results


def generate_mock_recommendations(
    history_ids: list[int],
    catalog: dict[int, dict],
    top_k: int = 10,
) -> dict:
    """Build deterministic mock Top-K lists for SASRec, GRU4Rec, and Most-Popular."""
    seen = set(history_ids)
    candidates = [item_id for item_id in sorted(catalog.keys()) if item_id > 0 and item_id not in seen]

    last_genres = set(catalog[history_ids[-1]].get("genres", []))
    all_genres = {g for hid in history_ids for g in catalog[hid].get("genres", [])}

    seq_len = min(len(history_ids), 100)
    denom = float(seq_len * (seq_len + 1)) / 2.0
    weights = [round((idx + 1) / denom, 4) for idx in range(seq_len)]

    return {
        "history_ids": list(history_ids),
        "sasrec_recommendations": _rank_top_k(candidates, catalog, last_genres, top_k, salt=17),
        "gru4rec_recommendations": _rank_top_k(candidates, catalog, all_genres, top_k, salt=53),
        "popular_recommendations": _rank_top_k(candidates, catalog, set(), top_k, salt=97),
        "attention": {
            "layer": 1,
            "head": 0,
            "weights": weights,
        },
    }
