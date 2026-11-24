"""Split chronological user sequences using the Leave-One-Out protocol."""


def split_leave_one_out(user_sequences: dict[int, list[int]]) -> dict[str, dict]:
    """Split each sequence S^u = (s_1..s_n) into train (1..n-2), val (n-1), test (n)."""
    if not user_sequences:
        raise ValueError("user_sequences must not be empty.")

    train: dict[int, list[int]] = {}
    val: dict[int, int] = {}
    test: dict[int, int] = {}

    for user_id, seq in user_sequences.items():
        if len(seq) < 3:
            raise ValueError(
                f"User {user_id} has {len(seq)} interactions; Leave-One-Out requires >= 3."
            )
        if any(item_id <= 0 for item_id in seq):
            raise ValueError(f"User {user_id} sequence contains non-positive/padding item ID.")

        train[user_id] = list(seq[:-2])
        val[user_id] = int(seq[-2])
        test[user_id] = int(seq[-1])

    return {"train": train, "val": val, "test": test}
