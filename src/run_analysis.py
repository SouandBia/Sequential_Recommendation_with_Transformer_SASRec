"""Orchestrator for Week 3 length-quantile analysis, 3 error cases, and attention diagnostics."""

import json
from pathlib import Path
import pickle
import torch

from src.analysis.collect_error_cases import collect_error_cases
from src.analysis.extract_attention import extract_attention
from src.analysis.group_by_length import group_by_length
from src.analysis.plot_attention_heatmap import plot_attention_heatmap
from src.analysis.plot_length_groups import plot_length_groups
from src.evaluation.mask_seen_items import mask_seen_items
from src.evaluation.rank_candidates import rank_candidates
from src.models.build_gru4rec import build_gru4rec
from src.models.build_sasrec import build_sasrec
from src.training.pad_sequences import pad_sequences
from src.training.save_results_table import save_results_table
from src.training.score_split import score_split


def _load_verified_checkpoint(path: Path, expected_mapping_version: str, expected_num_items: int) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["mapping_version"] != expected_mapping_version:
        raise ValueError(
            f"Checkpoint {path} mapping_version={ckpt['mapping_version']!r} "
            f"does not match dataset {expected_mapping_version!r}."
        )
    if int(ckpt["num_items"]) != expected_num_items:
        raise ValueError(
            f"Checkpoint {path} num_items={ckpt['num_items']} != {expected_num_items}."
        )
    return ckpt


def run_analysis(config_path: str | Path = "configs/default.json") -> dict:
    """Run Week 3 error analysis and optional attention diagnostics on saved checkpoints."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

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
    item2meta = dataset["item2meta"]

    test_histories = [dataset["train"][uid] + [int(dataset["val"][uid])] for uid in user_ids]
    test_targets = [int(dataset["test"][uid]) for uid in user_ids]
    user_lengths = [len(hist) for hist in test_histories]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pop_ckpt = _load_verified_checkpoint(checkpoints_dir / "popularity_best.pt", mapping_version, num_items)
    gru_ckpt = _load_verified_checkpoint(checkpoints_dir / "gru4rec_best.pt", mapping_version, num_items)
    sas_ckpt = _load_verified_checkpoint(checkpoints_dir / "sasrec_best.pt", mapping_version, num_items)

    pop_counts_tensor = pop_ckpt["model_state_dict"]["counts"]
    train_counts = {idx: int(pop_counts_tensor[idx].item()) for idx in range(1, num_items + 1)}

    gru_cfg = gru_ckpt["config"]
    gru_model = build_gru4rec(
        num_items=num_items,
        d_model=int(gru_cfg["d_model"]),
        num_layers=int(gru_cfg["num_layers"]),
        dropout=float(gru_cfg["dropout"]),
        max_seq_len=int(gru_cfg["max_seq_len"]),
        pad_id=pad_id,
    )
    gru_model.load_state_dict(gru_ckpt["model_state_dict"])
    gru_model = gru_model.to(device)

    sas_cfg = sas_ckpt["config"]
    sas_model = build_sasrec(
        num_items=num_items,
        d_model=int(sas_cfg["d_model"]),
        num_blocks=int(sas_cfg["num_blocks"]),
        num_heads=int(sas_cfg["num_heads"]),
        dropout=float(sas_cfg["dropout"]),
        max_seq_len=int(sas_cfg["max_seq_len"]),
        pad_id=pad_id,
    )
    sas_model.load_state_dict(sas_ckpt["model_state_dict"])
    sas_model = sas_model.to(device)

    test_x = pad_sequences(test_histories, max_seq_len=max_seq_len, pad_id=pad_id, build_targets=False)

    raw_scores_by_model = {
        "popularity": pop_ckpt["model_state_dict"]["scores"].unsqueeze(0).expand(len(user_ids), -1),
        "gru4rec": score_split(gru_model, test_x, batch_size=256, device=device),
        "sasrec": score_split(sas_model, test_x, batch_size=256, device=device),
    }

    ranks_by_model: dict[str, list[int]] = {}
    top_k_ids_by_model: dict[str, list[list[int]]] = {}
    top_k_scores_by_model: dict[str, list[list[float]]] = {}

    for model_name, raw_scores in raw_scores_by_model.items():
        masked = mask_seen_items(raw_scores, test_histories, test_targets, pad_id=pad_id)
        ranked = rank_candidates(masked, test_targets, top_k=top_k, pad_id=pad_id)
        ranks_by_model[model_name] = ranked["ranks"].tolist()
        top_k_ids_by_model[model_name] = ranked["top_k_ids"].tolist()
        top_k_scores_by_model[model_name] = ranked["top_k_scores"].tolist()

    group_rows = group_by_length(
        user_lengths=user_lengths,
        ranks_by_model=ranks_by_model,
        top_k=top_k,
        num_quantiles=3,
    )
    save_results_table(group_rows, results_dir / "length_quantiles.csv")
    plot_length_groups(
        group_rows=group_rows,
        output_paths=[
            results_dir / "figures" / "length_quantiles.png",
            report_figures_dir / "length_quantiles.png",
        ],
    )

    error_cases = collect_error_cases(
        user_ids=user_ids,
        test_histories=test_histories,
        test_targets=test_targets,
        ranks_by_model=ranks_by_model,
        top_k_ids_by_model=top_k_ids_by_model,
        top_k_scores_by_model=top_k_scores_by_model,
        item2meta=item2meta,
        train_counts=train_counts,
        max_cases=3,
    )
    error_cases_path = results_dir / "error_cases.json"
    error_cases_path.write_text(json.dumps(error_cases, indent=2), encoding="utf-8")

    # Optional sequence-level attention diagnostic on a real SASRec hit user (len 12..25)
    sample_idx = next(
        (
            i
            for i in range(len(user_ids))
            if ranks_by_model["sasrec"][i] == 1 and 12 <= len(test_histories[i]) <= 24
        ),
        0,
    )
    sample_history = test_histories[sample_idx][-max_seq_len:]
    sample_x = pad_sequences([sample_history], max_seq_len=max_seq_len, pad_id=pad_id, build_targets=False)
    attention_diag = extract_attention(
        model=sas_model,
        input_ids=sample_x,
        layer_idx=1,
        head_idx=0,
        pad_id=pad_id,
    )
    sample_titles = [item2meta[iid]["title"] for iid in sample_history]
    plot_attention_heatmap(
        attention_matrix=attention_diag["matrix"],
        item_titles=sample_titles,
        layer_idx=1,
        head_idx=0,
        max_display_steps=12,
        output_paths=[
            results_dir / "figures" / "attention_heatmap.png",
            report_figures_dir / "attention_heatmap.png",
        ],
    )
    attention_sample = {
        "user_id": int(user_ids[sample_idx]),
        "target_movie": item2meta[test_targets[sample_idx]],
        "sasrec_rank": int(ranks_by_model["sasrec"][sample_idx]),
        "history_titles": sample_titles,
        "layer": attention_diag["layer"],
        "head": attention_diag["head"],
        "last_query_weights": attention_diag["last_query_weights"],
        "caveat": attention_diag["caveat"],
    }
    (results_dir / "attention_sample.json").write_text(json.dumps(attention_sample, indent=2), encoding="utf-8")

    return {
        "length_quantiles": group_rows,
        "error_cases": error_cases,
        "attention_sample": attention_sample,
    }


if __name__ == "__main__":
    summary = run_analysis()
    print(json.dumps(summary, indent=2))
