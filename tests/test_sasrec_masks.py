"""Unit tests for pure-PyTorch SASRec right-padding, causal masking, and finite gradients."""

import pytest
import torch

from src.models.build_sasrec import build_sasrec
from src.training.compute_next_item_loss import compute_next_item_loss
from src.training.pad_sequences import pad_sequences
from src.training.set_seed import set_seed


def test_forward_backward_finite_on_short_right_padded_batches() -> None:
    set_seed(42)
    model = build_sasrec(num_items=50, d_model=64, num_blocks=2, num_heads=2, dropout=0.2, max_seq_len=16)
    model.train()

    raw_train_sequences = [
        [3, 14],
        [7, 11, 25, 42],
        [1, 2, 3, 4, 5, 6, 7],
    ]
    input_ids, target_ids = pad_sequences(raw_train_sequences, max_seq_len=16, pad_id=0, build_targets=True)
    logits = model(input_ids, all_positions=True)
    loss = compute_next_item_loss(logits=logits, target_ids=target_ids, pad_id=0)

    assert torch.isfinite(loss).item()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all().item(), f"Non-finite gradient in {name}"

    # Padding embedding row 0 and its gradient must stay strictly zero
    assert torch.all(model.item_emb.weight[0] == 0.0).item()
    assert torch.all(model.item_emb.weight.grad[0] == 0.0).item()


def test_zero_future_token_leakage() -> None:
    set_seed(42)
    model = build_sasrec(num_items=40, d_model=64, num_blocks=2, num_heads=2, dropout=0.0, max_seq_len=10)
    model.eval()

    seq_a = torch.tensor([[5, 12, 19, 23, 31, 0, 0]], dtype=torch.long)
    seq_b = torch.tensor([[5, 12, 19, 39, 2, 0, 0]], dtype=torch.long)  # Changed tokens at indices 3 and 4

    with torch.no_grad():
        logits_a = model(seq_a, all_positions=True)
        logits_b = model(seq_b, all_positions=True)

    # Positions 0, 1, 2 MUST be completely unaffected by changes at positions 3, 4
    assert torch.allclose(logits_a[:, :3, :], logits_b[:, :3, :], atol=1e-6, rtol=0.0)
    assert not torch.allclose(logits_a[:, 3, :], logits_b[:, 3, :], atol=1e-4)


def test_right_padding_invariance_and_attention_mask() -> None:
    set_seed(42)
    model = build_sasrec(num_items=40, d_model=64, num_blocks=2, num_heads=2, dropout=0.0, max_seq_len=12)
    model.eval()

    unpadded = torch.tensor([[12, 25, 33]], dtype=torch.long)
    padded_short = torch.tensor([[12, 25, 33, 0, 0]], dtype=torch.long)
    padded_long = torch.tensor([[12, 25, 33, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        seq_u, attn_u = model(unpadded, all_positions=True, return_attention=True)
        seq_s, attn_s = model(padded_short, all_positions=True, return_attention=True)
        seq_l, attn_l = model(padded_long, all_positions=True, return_attention=True)

        next_u = model(unpadded, all_positions=False)
        next_s = model(padded_short, all_positions=False)
        next_l = model(padded_long, all_positions=False)

    assert torch.allclose(seq_u, seq_s[:, :3, :], atol=1e-6, rtol=0.0)
    assert torch.allclose(seq_u, seq_l[:, :3, :], atol=1e-6, rtol=0.0)
    assert torch.allclose(next_u, next_s, atol=1e-6, rtol=0.0)
    assert torch.allclose(next_u, next_l, atol=1e-6, rtol=0.0)

    for layer_idx in range(len(attn_u)):
        assert torch.allclose(attn_u[layer_idx], attn_l[layer_idx][:, :, :3, :3], atol=1e-6, rtol=0.0)
        # Valid queries 0..2 must assign strictly 0 weight to padded keys 3..9
        assert torch.all(attn_l[layer_idx][:, :, :3, 3:] == 0.0).item()
        # Padded queries 3..9 must also be zeroed out
        assert torch.all(attn_l[layer_idx][:, :, 3:, :] == 0.0).item()


def test_last_valid_position_selection_and_all_pad_rejection() -> None:
    set_seed(42)
    model = build_sasrec(num_items=30, d_model=64, num_blocks=2, num_heads=2, dropout=0.0, max_seq_len=8)
    model.eval()

    batch = pad_sequences(
        sequences=[
            [9],                  # last valid index = 0
            [4, 15, 22],          # last valid index = 2
            [1, 3, 5, 7, 11, 13], # last valid index = 5
        ],
        max_seq_len=8,
        pad_id=0,
        build_targets=False,
    )

    with torch.no_grad():
        seq_logits = model(batch, all_positions=True)
        last_logits = model(batch, all_positions=False)

    assert torch.allclose(last_logits[0], seq_logits[0, 0], atol=1e-6)
    assert torch.allclose(last_logits[1], seq_logits[1, 2], atol=1e-6)
    assert torch.allclose(last_logits[2], seq_logits[2, 5], atol=1e-6)

    with pytest.raises(ValueError):
        model(torch.zeros((1, 4), dtype=torch.long))
