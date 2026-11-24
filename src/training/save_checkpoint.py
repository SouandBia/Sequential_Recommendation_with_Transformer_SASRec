"""Save model checkpoints adhering strictly to CONTRACT.md Section 6 schema."""

from pathlib import Path
import torch


def save_checkpoint(
    checkpoint_path: str | Path,
    model_state_dict: dict,
    model_type: str,
    config: dict,
    seed: int,
    mapping_version: str,
    num_items: int,
    val_metrics: dict[str, float],
    extra: dict | None = None,
) -> Path:
    """Persist a verified .pt checkpoint with metadata, mapping version, and val metrics."""
    allowed_types = {"sasrec", "gru4rec", "popularity"}
    if model_type not in allowed_types:
        raise ValueError(f"model_type must be one of {sorted(allowed_types)}, got {model_type!r}")
    if num_items <= 0:
        raise ValueError(f"num_items must be > 0, got {num_items}")
    if not mapping_version:
        raise ValueError("mapping_version must be a non-empty string.")
    for required_metric in ("HR@10", "NDCG@10"):
        if required_metric not in val_metrics:
            raise ValueError(f"val_metrics missing required key {required_metric!r}.")

    payload = {
        "model_state_dict": model_state_dict,
        "model_type": model_type,
        "config": dict(config),
        "seed": int(seed),
        "mapping_version": str(mapping_version),
        "num_items": int(num_items),
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
    }
    if extra:
        payload["extra"] = dict(extra)

    out_path = Path(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return out_path
