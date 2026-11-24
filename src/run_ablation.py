"""Orchestrator for post-MVP SASRec sequence-length ablation (L in {50, 100, 200})."""

import argparse
import json
from pathlib import Path
import pickle
import torch

from src.analysis.plot_ablation_results import plot_ablation_results
from src.evaluation.compute_hr_ndcg import compute_hr_ndcg
from src.evaluation.mask_seen_items import mask_seen_items
from src.evaluation.rank_candidates import rank_candidates
from src.models.build_sasrec import build_sasrec
from src.training.compute_next_item_loss import compute_next_item_loss
from src.training.pad_sequences import pad_sequences
from src.training.save_results_table import save_results_table
from src.training.score_split import score_split
from src.training.set_seed import set_seed
from src.training.train_epoch import train_epoch


def run_ablation(
    config_path: str | Path = "configs/default.json",
    seq_lengths: tuple[int, ...] = (50, 100, 200),
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 60,
    patience: int = 10,
    grad_clip: float = 1.0,
) -> list[dict]:
    """Compare SASRec validation accuracy and RTX 4060 VRAM across max_seq_len values."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    seed = int(cfg["seed"])
    pad_id = int(cfg["data"]["pad_id"])
    top_k = int(cfg["evaluation"]["top_k"])
    sas_cfg = cfg["models"]["sasrec"]
    results_dir = Path(cfg["paths"]["results_dir"])
    report_figures_dir = Path(cfg["paths"]["report_dir"]) / "figures"

    with Path(cfg["data"]["processed_path"]).open("rb") as handle:
        dataset = pickle.load(handle)

    user_ids = sorted(dataset["train"].keys())
    num_items = int(dataset["num_items"])
    train_seqs = [dataset["train"][uid] for uid in user_ids]
    val_targets = [int(dataset["val"][uid]) for uid in user_ids]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ablation_rows: list[dict] = []
    for seq_len in seq_lengths:
        set_seed(seed)
        train_x, train_y = pad_sequences(train_seqs, max_seq_len=seq_len, pad_id=pad_id, build_targets=True)
        val_x = pad_sequences(train_seqs, max_seq_len=seq_len, pad_id=pad_id, build_targets=False)

        model = build_sasrec(
            num_items=num_items,
            d_model=int(sas_cfg["d_model"]),
            num_blocks=int(sas_cfg["num_blocks"]),
            num_heads=int(sas_cfg["num_heads"]),
            dropout=float(sas_cfg["dropout"]),
            max_seq_len=seq_len,
            pad_id=pad_id,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_ndcg = -1.0
        best_hr = -1.0
        best_epoch = 0
        no_improve = 0
        peak_vram_mb = 0.0
        total_train_sec = 0.0

        for epoch in range(1, max_epochs + 1):
            stats = train_epoch(
                model=model,
                input_ids=train_x,
                target_ids=train_y,
                optimizer=optimizer,
                loss_fn=compute_next_item_loss,
                batch_size=batch_size,
                pad_id=pad_id,
                device=device,
                grad_clip=grad_clip,
                use_amp=False,
            )
            total_train_sec += stats["epoch_sec"]
            peak_vram_mb = max(peak_vram_mb, stats["peak_vram_mb"])

            val_scores = score_split(model=model, input_ids=val_x, batch_size=256, device=device)
            masked = mask_seen_items(val_scores, train_seqs, val_targets, pad_id=pad_id)
            ranked = rank_candidates(masked, val_targets, top_k=top_k, pad_id=pad_id)
            metrics = compute_hr_ndcg(ranked["ranks"], top_k=top_k)

            if metrics[f"NDCG@{top_k}"] > best_ndcg:
                best_ndcg = metrics[f"NDCG@{top_k}"]
                best_hr = metrics[f"HR@{top_k}"]
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        ablation_rows.append(
            {
                "model": "sasrec",
                "max_seq_len": int(seq_len),
                "num_blocks": int(sas_cfg["num_blocks"]),
                "d_model": int(sas_cfg["d_model"]),
                "seed": seed,
                "best_epoch": best_epoch,
                "val_HR@10": round(best_hr, 6),
                "val_NDCG@10": round(best_ndcg, 6),
                "train_time_sec": round(total_train_sec, 2),
                "peak_vram_mb": round(peak_vram_mb, 2),
            }
        )

    save_results_table(ablation_rows, results_dir / "ablation_seq_len.csv")
    plot_ablation_results(
        ablation_rows=ablation_rows,
        output_paths=[
            results_dir / "figures" / "ablation_seq_len.png",
            report_figures_dir / "ablation_seq_len.png",
        ],
    )
    return ablation_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SASRec max_seq_len ablation.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    rows = run_ablation(config_path=args.config, max_epochs=args.max_epochs, patience=args.patience)
    print(json.dumps(rows, indent=2))
