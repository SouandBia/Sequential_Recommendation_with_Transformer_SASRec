"""Run real PyTorch inference for SASRec, GRU4Rec, and Most-Popular on an ordered history."""

import torch


def _mask_and_rank(
    raw_scores: torch.Tensor,
    seen_ids: set[int],
    catalog: dict[int, dict],
    top_k: int,
    pad_id: int,
) -> list[dict]:
    scores = raw_scores.detach().to(dtype=torch.float32, device="cpu").clone()
    scores[pad_id] = -torch.inf
    vocab_size = scores.size(0)
    for item_id in seen_ids:
        if 0 <= item_id < vocab_size:
            scores[item_id] = -torch.inf

    effective_k = min(top_k, max(vocab_size - 1 - len(seen_ids), 0))
    if effective_k <= 0:
        return []

    sorted_ids = torch.argsort(-scores, dim=-1, stable=True)[:effective_k].tolist()
    results: list[dict] = []
    for item_id in sorted_ids:
        raw_val = float(scores[item_id].item())
        if not torch.isfinite(torch.tensor(raw_val)):
            continue
        meta = catalog[item_id]
        results.append(
            {
                "item_id": int(item_id),
                "title": str(meta["title"]),
                "genres": list(meta.get("genres", [])),
                "score": round(raw_val, 4),
            }
        )
    return results


def predict_recommendations(
    history_ids: list[int],
    models_bundle: dict,
    catalog: dict[int, dict],
    top_k: int = 10,
    pad_id: int = 0,
    attention_layer: int = 1,
    attention_head: int = 0,
) -> dict:
    """Compute real Top-K recommendations with raw model scores and optional SASRec attention."""
    max_seq_len = int(models_bundle["max_seq_len"])
    recent_history = [int(iid) for iid in history_ids[-max_seq_len:]]
    valid_len = len(recent_history)
    seen_ids = set(int(iid) for iid in history_ids)

    device = torch.device(models_bundle["device"])
    input_tensor = torch.tensor([recent_history], dtype=torch.long, device=device)

    sas_model = models_bundle["sasrec"]
    gru_model = models_bundle["gru4rec"]
    pop_scores = models_bundle["popularity_scores"]

    with torch.no_grad():
        sas_logits, attentions = sas_model(input_tensor, all_positions=False, return_attention=True)
        gru_logits = gru_model(input_tensor, all_positions=False)

    layer_idx = min(attention_layer, len(attentions) - 1)
    head_idx = min(attention_head, attentions[layer_idx].size(1) - 1)
    last_query_weights = (
        attentions[layer_idx][0, head_idx, valid_len - 1, :valid_len]
        .detach()
        .cpu()
        .to(torch.float32)
        .tolist()
    )

    return {
        "history_ids": list(history_ids),
        "sasrec_recommendations": _mask_and_rank(sas_logits[0], seen_ids, catalog, top_k, pad_id),
        "gru4rec_recommendations": _mask_and_rank(gru_logits[0], seen_ids, catalog, top_k, pad_id),
        "popular_recommendations": _mask_and_rank(pop_scores, seen_ids, catalog, top_k, pad_id),
        "attention": {
            "layer": int(layer_idx),
            "head": int(head_idx),
            "weights": [round(float(w), 6) for w in last_query_weights],
        },
    }
