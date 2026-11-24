"""Unit tests for Most-Popular, GRU4Rec, training epoch, checkpointing, and reproducibility."""

from pathlib import Path
import pytest
import torch

from src.models.build_gru4rec import build_gru4rec
from src.models.fit_popularity import fit_popularity
from src.run_evaluation import run_evaluation
from src.training.compute_next_item_loss import compute_next_item_loss
from src.training.pad_sequences import pad_sequences
from src.training.save_checkpoint import save_checkpoint
from src.training.score_split import score_split
from src.training.set_seed import set_seed
from src.training.train_epoch import train_epoch


def test_fit_popularity_uses_train_only_without_leakage() -> None:
    # Item 5 appears ONLY in val/test and never in train; its count must stay 0.
    train_sequences = {
        1: [1, 2, 3],
        2: [2, 3, 4],
        3: [2, 1, 4],
    }
    pop = fit_popularity(train_sequences=train_sequences, num_items=5, pad_id=0)

    assert pop["counts"].tolist() == [0.0, 2.0, 3.0, 2.0, 2.0, 0.0]
    assert pop["scores"][0].item() < -1e8
    assert pop["total_train_events"] == 9

    # Evaluate user 1 (seen [1, 2, 3]) targeting item 4 -> unmasked items are 4 (count=2) and 5 (count=0) -> rank 1
    scores_matrix = pop["scores"].unsqueeze(0)
    eval_out = run_evaluation(scores=scores_matrix, seen_histories=[[1, 2, 3]], target_items=[4], top_k=2)
    assert eval_out["ranks"].tolist() == [1]
    assert eval_out["HR@2"] == pytest.approx(1.0)


def test_gru4rec_finite_gradients_padding_invariance_and_last_position() -> None:
    set_seed(42)
    model = build_gru4rec(num_items=30, d_model=32, num_layers=1, dropout=0.0, max_seq_len=10)
    model.train()

    seqs = [[2, 5, 9], [1, 4, 7, 12, 18]]
    x, y = pad_sequences(seqs, max_seq_len=10, pad_id=0, build_targets=True)
    logits = model(x, all_positions=True)
    loss = compute_next_item_loss(logits, y, pad_id=0)

    assert torch.isfinite(loss).item()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(param.grad).all().item(), f"Non-finite grad in {name}"
    assert torch.all(model.item_emb.weight.grad[0] == 0.0).item()

    # Right-padding invariance & last valid position selection in eval mode
    model.eval()
    unpadded = torch.tensor([[4, 8, 15]], dtype=torch.long)
    padded = torch.tensor([[4, 8, 15, 0, 0, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        seq_u = model(unpadded, all_positions=True)
        seq_p = model(padded, all_positions=True)
        next_u = model(unpadded, all_positions=False)
        next_p = model(padded, all_positions=False)

    assert torch.allclose(seq_u, seq_p[:, :3, :], atol=1e-6)
    assert torch.allclose(next_u, next_p, atol=1e-6)
    assert torch.allclose(next_p[0], seq_p[0, 2], atol=1e-6)

    with pytest.raises(ValueError):
        model(torch.zeros((1, 4), dtype=torch.long))


def test_train_epoch_score_split_checkpoint_and_reproducibility(tmp_path: Path) -> None:
    raw_seqs = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]]
    x, y = pad_sequences(raw_seqs, max_seq_len=6, pad_id=0, build_targets=True)

    def _run_once() -> tuple[float, float, torch.Tensor, dict]:
        set_seed(42)
        m = build_gru4rec(num_items=10, d_model=16, num_layers=1, dropout=0.0, max_seq_len=6)
        opt = torch.optim.Adam(m.parameters(), lr=5e-2)
        s1 = train_epoch(m, x, y, opt, compute_next_item_loss, batch_size=2, pad_id=0, device="cpu")
        s2 = train_epoch(m, x, y, opt, compute_next_item_loss, batch_size=2, pad_id=0, device="cpu")
        scores = score_split(m, x, batch_size=2, device="cpu")
        return s1["train_loss"], s2["train_loss"], scores, m.state_dict()

    loss1_a, loss2_a, scores_a, state_a = _run_once()
    loss1_b, loss2_b, scores_b, _ = _run_once()

    assert loss2_a < loss1_a
    assert loss1_a == pytest.approx(loss1_b, abs=1e-6)
    assert torch.allclose(scores_a, scores_b, atol=1e-6)

    ckpt_path = save_checkpoint(
        checkpoint_path=tmp_path / "gru4rec_best.pt",
        model_state_dict=state_a,
        model_type="gru4rec",
        config={"d_model": 16, "num_layers": 1},
        seed=42,
        mapping_version="test_v1",
        num_items=10,
        val_metrics={"HR@10": 0.5, "NDCG@10": 0.3},
    )
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model_type", "config", "seed", "mapping_version", "num_items", "val_metrics"):
        assert key in loaded
    assert loaded["mapping_version"] == "test_v1"
    assert loaded["seed"] == 42
