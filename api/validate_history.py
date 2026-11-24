"""Validate incoming recommendation request fields against CONTRACT.md rules."""


def validate_history(
    history_ids: list[int],
    catalog: dict[int, dict],
    top_k: int = 10,
) -> None:
    """Validate non-empty history_ids in 1..N (excluding pad_id=0) and positive top_k."""
    if not history_ids:
        raise ValueError("history_ids must contain at least 1 valid MovieLens-1M item ID.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    for idx, item_id in enumerate(history_ids):
        if item_id == 0:
            raise ValueError(f"history_ids[{idx}] uses reserved padding ID 0.")
        if item_id not in catalog:
            raise ValueError(f"history_ids[{idx}]={item_id} is not a valid catalog item ID.")
