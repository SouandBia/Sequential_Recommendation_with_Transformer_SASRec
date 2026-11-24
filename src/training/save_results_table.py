"""Write structured experiment metrics rows to a CSV file."""

from collections.abc import Sequence
import csv
from pathlib import Path


def save_results_table(
    rows: Sequence[dict],
    csv_path: str | Path,
) -> Path:
    """Persist a non-empty sequence of result dictionaries to CSV."""
    if not rows:
        raise ValueError("rows must not be empty.")

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return out_path
