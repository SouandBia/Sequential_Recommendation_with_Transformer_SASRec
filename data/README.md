# MovieLens-1M Dataset Guide

> **Important:** Do **not** commit raw MovieLens-1M files (`*.dat`, `*.zip`) or generated `.pkl` files to a public GitHub repository. They are excluded in `.gitignore`.

## 1. Download Official Dataset

1. Download **MovieLens-1M** from GroupLens:
   - Official page: https://grouplens.org/datasets/movielens/1m/
   - Direct archive: https://files.grouplens.org/datasets/movielens/ml-1m.zip
2. Extract the archive so the files sit directly inside `data/ml-1m/`:
   ```text
   data/ml-1m/
   ├── ratings.dat   # UserID::MovieID::Rating::Timestamp (~1,000,209 rows)
   ├── movies.dat    # MovieID::Title::Genres (~3,883 movies; ~3,706 rated)
   ├── users.dat     # UserID::Gender::Age::Occupation::Zip-code (6,040 users)
   └── README
   ```

## 2. Generated Processed Artifacts

Running the preprocessing pipeline produces:
- `data/ml1m_processed.pkl` — Leave-One-Out splits (`train`, `val`, `test`), counts (`num_users`, `num_items`), metadata (`item2meta`), ID mappings, and `mapping_version`.
- `data/item2title.json` — Clean lookup table of internal item IDs (`1..N`) to movie titles, genres, and release years for `api/` and `web/`.

See [`../CONTRACT.md`](../CONTRACT.md) for the exact schema, padding conventions (`0 = padding`, `1..N = valid items`), minimum user length (`>= 5`), and deterministic timestamp tie-breaking rules.
