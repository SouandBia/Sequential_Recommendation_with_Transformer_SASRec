"""Orchestrator for Week 2 training, validation selection, and single-pass test evaluation."""

import argparse
import copy
import json
from pathlib import Path
import pickle
import time
import torch

from src.analysis.plot_training_curves import plot_training_curves
from src.evaluation.compute_hr_ndcg import compute_hr_ndcg
from src.evaluation.mask_seen_items import mask_seen_items
from src.evaluation.rank_candidates import rank_candidates
from src.models.build_gru4rec import build_gru4rec
from src.models.build_sasrec import build_sasrec
from src.models.fit_popularity import fit_popularity
from src.training.compute_next_item_loss import compute_next_item_loss
from src.training.pad_sequences import pad_sequences
from src.training.save_checkpoint import save_checkpoint
from src.training.save_results_table import save_results_table
from src.training.score_split import score_split
from src.training.set_seed import set_seed
from src.training.train_epoch import train_epoch


def _evaluate_scores(
    scores: torch.Tensor,
    seen_histories: list[list[int]],
    target_items: list[int],
    top_k: int,
    pad_id: int,
) -> dict[str, float]:
    masked = mask_seen_items(
        scores=scores,
        seen_histories=seen_histories,
        target_items=target_items,
        pad_id=pad_id,
    )
    ranked = rank_candidates(
        masked_scores=masked,
        target_items=target_items,
        top_k=top_k,
        pad_id=pad_id,
    )
    return compute_hr_ndcg(ranks=ranked["ranks"], top_k=top_k)


def _train_neural_model(
    model_type: str,
    model: torch.nn.Module,
    model_cfg: dict,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_seen: list[list[int]],
    val_targets: list[int],
    test_x: torch.Tensor,
    test_seen: list[list[int]],
    test_targets: list[int],
    num_items: int,
    mapping_version: str,
    seed: int,
    top_k: int,
    pad_id: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    grad_clip: float,
    use_amp: bool,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[dict, list[dict]]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_ndcg = -1.0
    best_val_metrics: dict[str, float] = {}
    best_state_dict: dict | None = None
    best_epoch = 0
    epochs_without_improve = 0
    peak_vram_mb = 0.0
    total_train_sec = 0.0
    curve_rows: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        epoch_stats = train_epoch(
            model=model,
            input_ids=train_x,
            target_ids=train_y,
            optimizer=optimizer,
            loss_fn=compute_next_item_loss,
            batch_size=batch_size,
            pad_id=pad_id,
            device=device,
            grad_clip=grad_clip,
            use_amp=use_amp,
        )
        total_train_sec += epoch_stats["epoch_sec"]
        peak_vram_mb = max(peak_vram_mb, epoch_stats["peak_vram_mb"])

        val_scores = score_split(model=model, input_ids=val_x, batch_size=256, device=device)
        val_metrics = _evaluate_scores(val_scores, val_seen, val_targets, top_k=top_k, pad_id=pad_id)

        curve_rows.append(
            {
                "model": model_type,
                "epoch": epoch,
                "train_loss": round(epoch_stats["train_loss"], 6),
                "HR@10": round(val_metrics[f"HR@{top_k}"], 6),
                "NDCG@10": round(val_metrics[f"NDCG@{top_k}"], 6),
                "epoch_sec": round(epoch_stats["epoch_sec"], 3),
                "peak_vram_mb": round(epoch_stats["peak_vram_mb"], 2),
            }
        )

        current_ndcg = val_metrics[f"NDCG@{top_k}"]
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            best_state_dict = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    assert best_state_dict is not None
    full_cfg = {
        **model_cfg,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "patience": patience,
        "grad_clip": grad_clip,
        "use_amp": use_amp,
    }
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model_state_dict=best_state_dict,
        model_type=model_type,
        config=full_cfg,
        seed=seed,
        mapping_version=mapping_version,
        num_items=num_items,
        val_metrics=best_val_metrics,
        extra={
            "best_epoch": best_epoch,
            "epochs_run": len(curve_rows),
            "total_train_sec": round(total_train_sec, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
        },
    )

    # Single-pass held-out test evaluation strictly using the validation-selected checkpoint
    model.load_state_dict(best_state_dict)
    model = model.to(device)
    t0 = time.perf_counter()
    test_scores = score_split(model=model, input_ids=test_x, batch_size=256, device=device)
    test_metrics = _evaluate_scores(test_scores, test_seen, test_targets, top_k=top_k, pad_id=pad_id)
    test_eval_sec = time.perf_counter() - t0

    summary_row = {
        "model": model_type,
        "seed": seed,
        "max_seq_len": int(model_cfg["max_seq_len"]),
        "d_model": int(model_cfg["d_model"]),
        "best_epoch": best_epoch,
        "epochs_run": len(curve_rows),
        "val_HR@10": round(best_val_metrics[f"HR@{top_k}"], 6),
        "val_NDCG@10": round(best_val_metrics[f"NDCG@{top_k}"], 6),
        "test_HR@10": round(test_metrics[f"HR@{top_k}"], 6),
        "test_NDCG@10": round(test_metrics[f"NDCG@{top_k}"], 6),
        "train_time_sec": round(total_train_sec, 2),
        "test_eval_sec": round(test_eval_sec, 3),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "mapping_version": mapping_version,
    }
    return summary_row, curve_rows


def run_training(
    config_path: str | Path = "configs/default.json",
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 60,
    patience: int = 10,
    grad_clip: float = 1.0,
    use_amp: bool = False,
) -> dict:
    """Train Most-Popular, GRU4Rec, and SASRec and record full-catalog Val/Test results."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    seed = int(cfg["seed"])
    pad_id = int(cfg["data"]["pad_id"])
    max_seq_len = int(cfg["data"]["max_seq_len"])
    top_k = int(cfg["evaluation"]["top_k"])
    checkpoints_dir = Path(cfg["paths"]["checkpoints_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    report_figures_dir = Path(cfg["paths"]["report_dir"]) / "figures"

    with Path(cfg["data"]["processed_path"]).open("rb") as handle:
        dataset = pickle.load(handle)

    user_ids = sorted(dataset["train"].keys())
    num_items = int(dataset["num_items"])
    mapping_version = str(dataset["mapping_version"])

    train_seqs = [dataset["train"][uid] for uid in user_ids]
    val_targets = [int(dataset["val"][uid]) for uid in user_ids]
    test_seen = [dataset["train"][uid] + [int(dataset["val"][uid])] for uid in user_ids]
    test_targets = [int(dataset["test"][uid]) for uid in user_ids]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Most-Popular baseline (fit strictly on train interactions)
    set_seed(seed)
    t_pop = time.perf_counter()
    pop_fit = fit_popularity(dataset["train"], num_items=num_items, pad_id=pad_id)
    pop_train_sec = time.perf_counter() - t_pop
    pop_scores_matrix = pop_fit["scores"].unsqueeze(0).expand(len(user_ids), -1)

    pop_val_metrics = _evaluate_scores(
        pop_scores_matrix, train_seqs, val_targets, top_k=top_k, pad_id=pad_id
    )
    save_checkpoint(
        checkpoint_path=checkpoints_dir / "popularity_best.pt",
        model_state_dict={"counts": pop_fit["counts"], "scores": pop_fit["scores"]},
        model_type="popularity",
        config=cfg["models"]["popularity"],
        seed=seed,
        mapping_version=mapping_version,
        num_items=num_items,
        val_metrics=pop_val_metrics,
        extra={"total_train_events": pop_fit["total_train_events"]},
    )

    t_pop_eval = time.perf_counter()
    pop_test_metrics = _evaluate_scores(
        pop_scores_matrix, test_seen, test_targets, top_k=top_k, pad_id=pad_id
    )
    pop_test_eval_sec = time.perf_counter() - t_pop_eval

    pop_row = {
        "model": "popularity",
        "seed": seed,
        "max_seq_len": 0,
        "d_model": 0,
        "best_epoch": 0,
        "epochs_run": 0,
        "val_HR@10": round(pop_val_metrics[f"HR@{top_k}"], 6),
        "val_NDCG@10": round(pop_val_metrics[f"NDCG@{top_k}"], 6),
        "test_HR@10": round(pop_test_metrics[f"HR@{top_k}"], 6),
        "test_NDCG@10": round(pop_test_metrics[f"NDCG@{top_k}"], 6),
        "train_time_sec": round(pop_train_sec, 4),
        "test_eval_sec": round(pop_test_eval_sec, 3),
        "peak_vram_mb": 0.0,
        "mapping_version": mapping_version,
    }

    # Shared right-padded tensors for neural models
    train_x, train_y = pad_sequences(train_seqs, max_seq_len=max_seq_len, pad_id=pad_id, build_targets=True)
    val_x = pad_sequences(train_seqs, max_seq_len=max_seq_len, pad_id=pad_id, build_targets=False)
    test_x = pad_sequences(test_seen, max_seq_len=max_seq_len, pad_id=pad_id, build_targets=False)

    # 2. GRU4Rec baseline
    set_seed(seed)
    gru_cfg = cfg["models"]["gru4rec"]
    gru_model = build_gru4rec(
        num_items=num_items,
        d_model=int(gru_cfg["d_model"]),
        num_layers=int(gru_cfg["num_layers"]),
        dropout=float(gru_cfg["dropout"]),
        max_seq_len=int(gru_cfg["max_seq_len"]),
        pad_id=pad_id,
    )
    gru_row, gru_curves = _train_neural_model(
        model_type="gru4rec",
        model=gru_model,
        model_cfg=gru_cfg,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_seen=train_seqs,
        val_targets=val_targets,
        test_x=test_x,
        test_seen=test_seen,
        test_targets=test_targets,
        num_items=num_items,
        mapping_version=mapping_version,
        seed=seed,
        top_k=top_k,
        pad_id=pad_id,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        grad_clip=grad_clip,
        use_amp=use_amp,
        device=device,
        checkpoint_path=checkpoints_dir / "gru4rec_best.pt",
    )

    # 3. SASRec main model
    set_seed(seed)
    sas_cfg = cfg["models"]["sasrec"]
    sas_model = build_sasrec(
        num_items=num_items,
        d_model=int(sas_cfg["d_model"]),
        num_blocks=int(sas_cfg["num_blocks"]),
        num_heads=int(sas_cfg["num_heads"]),
        dropout=float(sas_cfg["dropout"]),
        max_seq_len=int(sas_cfg["max_seq_len"]),
        pad_id=pad_id,
    )
    sas_row, sas_curves = _train_neural_model(
        model_type="sasrec",
        model=sas_model,
        model_cfg=sas_cfg,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_seen=train_seqs,
        val_targets=val_targets,
        test_x=test_x,
        test_seen=test_seen,
        test_targets=test_targets,
        num_items=num_items,
        mapping_version=mapping_version,
        seed=seed,
        top_k=top_k,
        pad_id=pad_id,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        grad_clip=grad_clip,
        use_amp=use_amp,
        device=device,
        checkpoint_path=checkpoints_dir / "sasrec_best.pt",
    )

    summary_rows = [pop_row, gru_row, sas_row]
    all_curves = gru_curves + sas_curves

    save_results_table(summary_rows, results_dir / "baselines.csv")
    save_results_table(all_curves, results_dir / "training_curves.csv")
    plot_training_curves(
        curves_by_model={"gru4rec": gru_curves, "sasrec": sas_curves},
        popularity_val_metrics=pop_val_metrics,
        output_paths=[
            results_dir / "figures" / "validation_curves.png",
            report_figures_dir / "validation_curves.png",
        ],
    )

    return {
        "device": str(device),
        "results": summary_rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Most-Popular, GRU4Rec, and SASRec.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true")
    args = parser.parse_args()

    output = run_training(
        config_path=args.config,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        use_amp=args.use_amp,
    )
    print(json.dumps(output, indent=2))
