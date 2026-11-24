"""Load and verify trained SASRec, GRU4Rec, and Most-Popular checkpoints for FastAPI."""

from collections.abc import Callable
from pathlib import Path
import torch


def load_models(
    checkpoints_dir: str | Path,
    sasrec_builder: Callable[..., torch.nn.Module],
    gru4rec_builder: Callable[..., torch.nn.Module],
    expected_num_items: int | None = None,
    expected_mapping_version: str | None = None,
    device: str = "cpu",
) -> dict | None:
    """Load verified checkpoints and validate shared mapping_version and num_items."""
    ckpt_root = Path(checkpoints_dir)
    paths = {
        "popularity": ckpt_root / "popularity_best.pt",
        "gru4rec": ckpt_root / "gru4rec_best.pt",
        "sasrec": ckpt_root / "sasrec_best.pt",
    }
    if not all(path.is_file() for path in paths.values()):
        return None

    loaded = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }

    mapping_versions = {str(ckpt["mapping_version"]) for ckpt in loaded.values()}
    if len(mapping_versions) != 1:
        raise ValueError(f"Mismatched mapping_version across checkpoints: {sorted(mapping_versions)}")
    mapping_version = next(iter(mapping_versions))
    if expected_mapping_version and mapping_version != expected_mapping_version:
        raise ValueError(
            f"Checkpoint mapping_version={mapping_version!r} != expected {expected_mapping_version!r}."
        )

    item_counts = {int(ckpt["num_items"]) for ckpt in loaded.values()}
    if len(item_counts) != 1:
        raise ValueError(f"Mismatched num_items across checkpoints: {sorted(item_counts)}")
    num_items = next(iter(item_counts))
    if expected_num_items is not None and num_items != expected_num_items:
        raise ValueError(f"Checkpoint num_items={num_items} != catalog size {expected_num_items}.")

    target_device = torch.device(device)

    sas_cfg = loaded["sasrec"]["config"]
    sas_model = sasrec_builder(
        num_items=num_items,
        d_model=int(sas_cfg["d_model"]),
        num_blocks=int(sas_cfg["num_blocks"]),
        num_heads=int(sas_cfg["num_heads"]),
        dropout=float(sas_cfg["dropout"]),
        max_seq_len=int(sas_cfg["max_seq_len"]),
        pad_id=0,
    )
    sas_model.load_state_dict(loaded["sasrec"]["model_state_dict"])
    sas_model.to(target_device).eval()

    gru_cfg = loaded["gru4rec"]["config"]
    gru_model = gru4rec_builder(
        num_items=num_items,
        d_model=int(gru_cfg["d_model"]),
        num_layers=int(gru_cfg["num_layers"]),
        dropout=float(gru_cfg["dropout"]),
        max_seq_len=int(gru_cfg["max_seq_len"]),
        pad_id=0,
    )
    gru_model.load_state_dict(loaded["gru4rec"]["model_state_dict"])
    gru_model.to(target_device).eval()

    pop_scores = loaded["popularity"]["model_state_dict"]["scores"].to(dtype=torch.float32, device="cpu")

    return {
        "sasrec": sas_model,
        "gru4rec": gru_model,
        "popularity_scores": pop_scores,
        "num_items": num_items,
        "mapping_version": mapping_version,
        "max_seq_len": int(sas_cfg["max_seq_len"]),
        "device": str(target_device),
        "val_metrics": {name: ckpt["val_metrics"] for name, ckpt in loaded.items()},
    }
