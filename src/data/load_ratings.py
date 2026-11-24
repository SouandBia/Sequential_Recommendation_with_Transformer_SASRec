"""Load raw MovieLens-1M ratings records with line indices for tie-breaking."""

from pathlib import Path


def _parse_rating_line(line: str, line_index: int) -> tuple[int, int, int, int, int]:
    parts = line.rstrip("\r\n").split("::")
    if len(parts) != 4:
        raise ValueError(f"Invalid ratings format at line {line_index + 1}: {line!r}")

    raw_user_id = int(parts[0])
    raw_movie_id = int(parts[1])
    rating = int(float(parts[2]))
    timestamp = int(parts[3])

    if raw_user_id <= 0 or raw_movie_id <= 0:
        raise ValueError(f"Non-positive ID at line {line_index + 1}: {line!r}")
    if rating < 1 or rating > 5:
        raise ValueError(f"Rating out of [1, 5] range at line {line_index + 1}: {rating}")
    if timestamp < 0:
        raise ValueError(f"Negative timestamp at line {line_index + 1}: {timestamp}")

    return (raw_user_id, raw_movie_id, rating, timestamp, line_index)


def load_ratings(ratings_path: str | Path) -> list[tuple[int, int, int, int, int]]:
    """Read MovieLens-1M ratings.dat into (user_id, movie_id, rating, timestamp, line_idx)."""
    path = Path(ratings_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ratings file not found: {path}")

    records: list[tuple[int, int, int, int, int]] = []
    with path.open("r", encoding="latin-1") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            records.append(_parse_rating_line(line, line_index))

    if not records:
        raise ValueError(f"Ratings file is empty: {path}")

    return records
