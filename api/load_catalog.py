"""Load MovieLens-1M internal item catalog metadata for the FastAPI service."""

import json
from pathlib import Path

_FALLBACK_CATALOG: dict[int, dict] = {
    1: {"title": "Toy Story (1995)", "genres": ["Animation", "Children's", "Comedy"], "year": 1995},
    2: {"title": "Jumanji (1995)", "genres": ["Adventure", "Children's", "Fantasy"], "year": 1995},
    3: {"title": "Grumpier Old Men (1995)", "genres": ["Comedy", "Romance"], "year": 1995},
    4: {"title": "Waiting to Exhale (1995)", "genres": ["Comedy", "Drama"], "year": 1995},
    5: {"title": "Father of the Bride Part II (1995)", "genres": ["Comedy"], "year": 1995},
    6: {"title": "Heat (1995)", "genres": ["Action", "Crime", "Thriller"], "year": 1995},
    7: {"title": "Sabrina (1995)", "genres": ["Comedy", "Romance"], "year": 1995},
    8: {"title": "Tom and Huck (1995)", "genres": ["Adventure", "Children's"], "year": 1995},
    9: {"title": "Sudden Death (1995)", "genres": ["Action"], "year": 1995},
    10: {"title": "GoldenEye (1995)", "genres": ["Action", "Adventure", "Thriller"], "year": 1995},
    11: {"title": "American President, The (1995)", "genres": ["Comedy", "Drama", "Romance"], "year": 1995},
    12: {"title": "Dracula: Dead and Loving It (1995)", "genres": ["Comedy", "Horror"], "year": 1995},
    13: {"title": "Balto (1995)", "genres": ["Animation", "Children's"], "year": 1995},
    14: {"title": "Nixon (1995)", "genres": ["Drama"], "year": 1995},
    15: {"title": "Cutthroat Island (1995)", "genres": ["Action", "Adventure", "Romance"], "year": 1995},
}


def load_catalog(item2title_path: str | Path = "data/item2title.json") -> dict[int, dict]:
    """Load internal item_id (1..N) -> {title, genres, year} from item2title.json."""
    path = Path(item2title_path)
    if not path.is_file():
        return {item_id: dict(meta) for item_id, meta in _FALLBACK_CATALOG.items()}

    with path.open("r", encoding="utf-8") as handle:
        raw_map = json.load(handle)

    catalog: dict[int, dict] = {}
    for key, meta in raw_map.items():
        item_id = int(key)
        if item_id <= 0:
            continue
        catalog[item_id] = {
            "title": str(meta["title"]),
            "genres": list(meta.get("genres", [])),
            "year": meta.get("year"),
        }
    return catalog
