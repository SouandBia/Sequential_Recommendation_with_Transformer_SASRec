"""Load raw MovieLens-1M movie metadata (title, genres, release year)."""

from pathlib import Path
import re

_YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def _extract_year(title: str) -> int | None:
    match = _YEAR_PATTERN.search(title)
    return int(match.group(1)) if match else None


def _parse_movie_line(line: str, line_index: int) -> tuple[int, dict]:
    parts = line.rstrip("\r\n").split("::")
    if len(parts) != 3:
        raise ValueError(f"Invalid movies format at line {line_index + 1}: {line!r}")

    raw_movie_id = int(parts[0])
    if raw_movie_id <= 0:
        raise ValueError(f"Non-positive MovieID at line {line_index + 1}: {raw_movie_id}")

    title = parts[1].strip()
    if not title:
        raise ValueError(f"Empty movie title at line {line_index + 1}")

    genres_raw = parts[2].strip()
    genres = [g.strip() for g in genres_raw.split("|") if g.strip() and g.strip() != "(no genres listed)"]
    year = _extract_year(title)

    return raw_movie_id, {"title": title, "genres": genres, "year": year}


def load_movies(movies_path: str | Path) -> dict[int, dict]:
    """Read MovieLens-1M movies.dat into raw_movie_id -> {title, genres, year}."""
    path = Path(movies_path)
    if not path.is_file():
        raise FileNotFoundError(f"Movies file not found: {path}")

    movies: dict[int, dict] = {}
    with path.open("r", encoding="latin-1") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            raw_movie_id, meta = _parse_movie_line(line, line_index)
            if raw_movie_id in movies:
                raise ValueError(f"Duplicate MovieID {raw_movie_id} in {path}")
            movies[raw_movie_id] = meta

    if not movies:
        raise ValueError(f"Movies file is empty: {path}")

    return movies
