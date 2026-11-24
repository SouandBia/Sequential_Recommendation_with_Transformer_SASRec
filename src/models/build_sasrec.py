"""Build a pure-PyTorch SASRec model with right-padding and causal mask safety."""

import math
import torch
from torch import nn


class _CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention safe against right-padding and NaN gradients."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal_allow = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        )
        key_allow = valid_mask[:, None, None, :]
        allow = causal_allow[None, None, :, :] & key_allow

        # Prevent all-masked softmax rows (which cause NaN forward/backward) if any query precedes valid tokens.
        row_has_valid = allow.any(dim=-1, keepdim=True)
        diag_fallback = torch.eye(seq_len, dtype=torch.bool, device=x.device)[None, None, :, :]
        safe_allow = torch.where(row_has_valid, allow, diag_fallback)

        masked_scores = scores.masked_fill(~safe_allow, -torch.inf)
        attn_weights = torch.softmax(masked_scores, dim=-1)

        query_valid = (valid_mask[:, None, :, None] & row_has_valid).to(dtype=attn_weights.dtype)
        attn_weights = attn_weights * query_valid

        dropped_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(dropped_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(context)
        out = out * valid_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out, attn_weights


class _PointwiseFeedForward(nn.Module):
    """Position-wise feed-forward network (Kang & McAuley, 2018)."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class _SASRecBlock(nn.Module):
    """Pre-LayerNorm Transformer block with causal self-attention and pointwise FFN."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-8)
        self.attn = _CausalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-8)
        self.ffn = _PointwiseFeedForward(d_model=d_model, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_float = valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        attn_out, attn_weights = self.attn(self.ln1(x), valid_mask)
        x = (x + self.drop1(attn_out)) * mask_float
        ffn_out = self.ffn(self.ln2(x))
        x = (x + self.drop2(ffn_out)) * mask_float
        return x, attn_weights


class _SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model in pure PyTorch."""

    def __init__(
        self,
        num_items: int,
        d_model: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        max_seq_len: int,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                _SASRecBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model, eps=1e-8)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[self.pad_id].zero_()

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

    def encode(
        self,
        input_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        valid_mask = self._validate_inputs(input_ids)
        _batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.item_emb(input_ids) * math.sqrt(self.d_model) + self.pos_emb(positions)
        x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = self.emb_dropout(x)
        x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)

        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn_weights = block(x, valid_mask)
            if return_attention:
                attentions.append(attn_weights)

        hidden = self.final_ln(x) * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        return hidden, attentions

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
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden, attentions = self.encode(input_ids, return_attention=return_attention)
        if all_positions:
            logits = self._score_items(hidden)
        else:
            last_hidden, _ = self.select_last_valid_hidden(hidden, input_ids)
            logits = self._score_items(last_hidden)

        if return_attention:
            return logits, attentions
        return logits


def build_sasrec(
    num_items: int,
    d_model: int = 64,
    num_blocks: int = 2,
    num_heads: int = 2,
    dropout: float = 0.2,
    max_seq_len: int = 100,
    pad_id: int = 0,
) -> nn.Module:
    """Instantiate a pure-PyTorch SASRec model following CONTRACT.md defaults."""
    if num_items <= 0:
        raise ValueError(f"num_items must be > 0, got {num_items}")
    if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be positive and divisible by num_heads ({num_heads}).")
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
    if pad_id != 0:
        raise ValueError(f"CONTRACT.md strictly reserves pad_id=0, got {pad_id}")

    return _SASRec(
        num_items=num_items,
        d_model=d_model,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
    )
