"""Unit tests for MovieLens-1M data loading, indexing, and Leave-One-Out splitting."""

import json
from pathlib import Path
import pickle
import pytest

from src.data.build_sequences import build_sequences
from src.data.load_movies import load_movies
from src.data.load_ratings import load_ratings
from src.data.save_dataset import save_dataset
from src.data.split_leave_one_out import split_leave_one_out


def test_load_and_build_sequences_tie_breaking_and_filtering(tmp_path: Path) -> None:
    movies_file = tmp_path / "movies.dat"
    ratings_file = tmp_path / "ratings.dat"

    # Movie 999 is unrated and must not appear in contiguous 1..N mapping.
    movies_file.write_text(
        "\n".join(
            [
                "10::Alpha (1995)::Action|Sci-Fi",
                "20::Beta (1996)::Comedy",
                "30::Gamma (1997)::Drama",
                "40::Delta (1998)::Thriller",
                "50::Epsilon (1999)::Animation|Children's",
                "60::Zeta (2000)::Romance",
                "999::Unrated Film (2001)::Horror",
            ]
        ),
        encoding="latin-1",
    )

    # User 1: 6 valid interactions, with tied timestamp=200 on raw movies 40 and 20,
    # plus a duplicate (user 1, movie 10) and missing movie 777 that must be dropped.
    # User 2: only 4 interactions (< min_interactions=5), must be excluded.
    # User 3: 5 interactions in reverse file order to verify chronological sorting.
    ratings_file.write_text(
        "\n".join(
            [
                "1::10::5::100",
                "1::10::4::150",  # duplicate user-movie -> dropped
                "1::777::3::160",  # missing from movies.dat -> dropped
                "1::40::4::200",  # tied timestamp=200 with movie 20; 20 < 40 so 20 comes first
                "1::20::5::200",
                "1::30::3::300",
                "1::50::4::400",
                "1::60::5::500",
                "2::10::5::10",
                "2::20::4::20",
                "2::30::3::30",
                "2::40::5::40",
                "3::60::5::900",
                "3::50::4::800",
                "3::30::3::700",
                "3::20::4::600",
                "3::10::5::500",
            ]
        ),
        encoding="latin-1",
    )

    ratings = load_ratings(ratings_file)
    movies = load_movies(movies_file)
    bundle = build_sequences(ratings=ratings, movies=movies, min_interactions=5)

    assert bundle["num_users"] == 2
    assert bundle["num_items"] == 6
    assert 0 not in bundle["item2meta"]
    assert set(bundle["item2meta"].keys()) == {1, 2, 3, 4, 5, 6}
    assert 999 not in bundle["raw_to_item_id"]

    # Raw movie IDs [10, 20, 30, 40, 50, 60] map deterministically to [1, 2, 3, 4, 5, 6]
    seq_u1 = bundle["user_sequences"][1]
    seq_u2 = bundle["user_sequences"][2]
    assert seq_u1 == [1, 2, 4, 3, 5, 6]
    assert seq_u2 == [1, 2, 3, 5, 6]

    splits = split_leave_one_out(bundle["user_sequences"])
    assert splits["train"][1] == [1, 2, 4, 3]
    assert splits["val"][1] == 5
    assert splits["test"][1] == 6

    assert splits["train"][2] == [1, 2, 3]
    assert splits["val"][2] == 5
    assert splits["test"][2] == 6

    # Zero target leakage: neither val nor test targets appear in train prefix
    for uid in (1, 2):
        assert splits["val"][uid] not in splits["train"][uid]
        assert splits["test"][uid] not in splits["train"][uid]

    processed_path = tmp_path / "ml1m_processed.pkl"
    item2title_path = tmp_path / "item2title.json"
    saved = save_dataset(bundle, splits, processed_path, item2title_path)

    loaded_pkl = pickle.loads(processed_path.read_bytes())
    loaded_json = json.loads(item2title_path.read_text(encoding="utf-8"))

    required_keys = {
        "train",
        "val",
        "test",
        "num_users",
        "num_items",
        "item2meta",
        "raw_to_item_id",
        "item_id_to_raw",
        "mapping_version",
    }
    assert required_keys.issubset(loaded_pkl.keys())
    assert loaded_pkl["mapping_version"] == saved["mapping_version"]
    assert "0" not in loaded_json
    assert len(loaded_json) == 6
    assert loaded_json["1"]["year"] == 1995


def test_split_leave_one_out_rejects_short_or_padded_sequences() -> None:
    with pytest.raises(ValueError):
        split_leave_one_out({1: [1, 2]})

    with pytest.raises(ValueError):
        split_leave_one_out({1: [1, 0, 2, 3, 4]})
