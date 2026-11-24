"""Orchestrator for MovieLens-1M preprocessing and Leave-One-Out dataset creation."""

import json
from pathlib import Path

from src.data.build_sequences import build_sequences
from src.data.load_movies import load_movies
from src.data.load_ratings import load_ratings
from src.data.save_dataset import save_dataset
from src.data.split_leave_one_out import split_leave_one_out


def prepare_data(config_path: str | Path = "configs/default.json") -> dict:
    """Run the full MovieLens-1M data preparation pipeline in order."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])

    ratings = load_ratings(raw_dir / "ratings.dat")
    movies = load_movies(raw_dir / "movies.dat")
    sequence_bundle = build_sequences(
        ratings=ratings,
        movies=movies,
        min_interactions=int(data_cfg["min_user_interactions"]),
    )
    splits = split_leave_one_out(sequence_bundle["user_sequences"])
    dataset = save_dataset(
        sequence_bundle=sequence_bundle,
        splits=splits,
        processed_path=data_cfg["processed_path"],
        item2title_path=data_cfg["item2title_path"],
    )
    return dataset


if __name__ == "__main__":
    result = prepare_data()
    print(
        json.dumps(
            {
                "num_users": result["num_users"],
                "num_items": result["num_items"],
                "mapping_version": result["mapping_version"],
                "stats": {k: v for k, v in result["stats"].items() if k != "all_user_lengths"},
            },
            indent=2,
        )
    )
