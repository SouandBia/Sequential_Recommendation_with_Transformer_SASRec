"""Persist processed MovieLens-1M splits and item metadata artifacts."""

import json
from pathlib import Path
import pickle


def save_dataset(
    sequence_bundle: dict,
    splits: dict[str, dict],
    processed_path: str | Path,
    item2title_path: str | Path,
) -> dict:
    """Save data/ml1m_processed.pkl and data/item2title.json matching CONTRACT.md."""
    payload = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "num_users": int(sequence_bundle["num_users"]),
        "num_items": int(sequence_bundle["num_items"]),
        "item2meta": sequence_bundle["item2meta"],
        "raw_to_item_id": sequence_bundle["raw_to_item_id"],
        "item_id_to_raw": sequence_bundle["item_id_to_raw"],
        "mapping_version": str(sequence_bundle["mapping_version"]),
        "stats": sequence_bundle["stats"],
    }

    pkl_file = Path(processed_path)
    pkl_file.parent.mkdir(parents=True, exist_ok=True)
    with pkl_file.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    json_file = Path(item2title_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    item2title = {
        str(item_id): {
            "title": meta["title"],
            "genres": list(meta["genres"]),
            "year": meta["year"],
        }
        for item_id, meta in sorted(sequence_bundle["item2meta"].items())
    }
    with json_file.open("w", encoding="utf-8") as handle:
        json.dump(item2title, handle, ensure_ascii=False, indent=2)

    return payload
