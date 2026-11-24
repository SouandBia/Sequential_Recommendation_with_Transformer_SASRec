"""Build a pure-PyTorch GRU4Rec sequential recommendation baseline."""

import torch
from torch import nn


class _GRU4Rec(nn.Module):
    """Recurrent sequential recommender using GRU with right-padding safety."""

    def __init__(
        self,
        num_items: int,
        d_model: int,
        num_layers: int,
        dropout: float,
        max_seq_len: int,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dropout = nn.Dropout(dropout)
        self.final_ln = nn.LayerNorm(d_model, eps=1e-8)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[self.pad_id].zero_()
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _validate_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D [batch_size, seq_len], got {tuple(input_ids.shape)}")
        batch_size, seq_len = input_ids.shape
        if batch_size == 0 or seq_len == 0:
            raise ValueError("input_ids must have non-zero batch_size and seq_len.")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )
        if ((input_ids < 0) | (input_ids > self.num_items)).any():
            raise ValueError(f"input_ids contains IDs outside [0, {self.num_items}].")

        valid_mask = input_ids != self.pad_id
        if not valid_mask.any(dim=1).all():
            raise ValueError("Every input sequence must contain at least 1 valid non-padding item.")
        return valid_mask

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        valid_mask = self._validate_inputs(input_ids)
        mask_float = valid_mask.unsqueeze(-1).to(dtype=self.item_emb.weight.dtype)

        x = self.item_emb(input_ids)
        x = self.emb_dropout(x) * mask_float
        gru_out, _ = self.gru(x)
        hidden = self.final_ln(self.out_dropout(gru_out)) * mask_float
        return hidden

    def select_last_valid_hidden(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        last_indices = torch.where(input_ids != self.pad_id, positions, -1).max(dim=1).values
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        return hidden[batch_indices, last_indices], last_indices

    def _score_items(self, hidden_state: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(hidden_state, self.item_emb.weight.transpose(0, 1))
        logits = logits.clone()
        logits[..., self.pad_id] = -1e9
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        all_positions: bool = True,
    ) -> torch.Tensor:
        hidden = self.encode(input_ids)
        if all_positions:
            return self._score_items(hidden)
        last_hidden, _ = self.select_last_valid_hidden(hidden, input_ids)
        return self._score_items(last_hidden)


def build_gru4rec(
    num_items: int,
    d_model: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    max_seq_len: int = 100,
    pad_id: int = 0,
) -> nn.Module:
    """Instantiate a pure-PyTorch GRU4Rec baseline following CONTRACT.md defaults."""
    if num_items <= 0:
        raise ValueError(f"num_items must be > 0, got {num_items}")
    if d_model <= 0:
        raise ValueError(f"d_model must be > 0, got {d_model}")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
    if pad_id != 0:
        raise ValueError(f"CONTRACT.md strictly reserves pad_id=0, got {pad_id}")

    return _GRU4Rec(
        num_items=num_items,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
    )
