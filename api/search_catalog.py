"""Search MovieLens-1M catalog items by case-insensitive title substring."""


def search_catalog(
    catalog: dict[int, dict],
    query: str = "",
    limit: int = 20,
) -> list[dict]:
    """Return up to limit catalog entries matching query ordered by item_id."""
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    needle = query.strip().lower()
    matches: list[dict] = []
    for item_id in sorted(catalog.keys()):
        meta = catalog[item_id]
        if needle and needle not in meta["title"].lower():
            continue
        matches.append(
            {
                "item_id": item_id,
                "title": meta["title"],
                "genres": list(meta["genres"]),
                "year": meta["year"],
            }
        )
        if len(matches) >= limit:
            break
    return matches
