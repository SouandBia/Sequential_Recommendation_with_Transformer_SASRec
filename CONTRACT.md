# Shared Technical Contract — Team 39 (Project 8: SASRec)

> Source of truth agreed by **Student A** (Models & Experiments) and **Student B** (Data, Evaluation & Demo).  
> **Do not change any contract field below without explicit agreement from both teammates.**

---

## 1. Scope Boundaries

| Priority | Item | Notes |
| --- | --- | --- |
| **MUST-HAVE (MVP)** | MovieLens-1M + Leave-One-Out pipeline | `ratings.dat`, `movies.dat`, stable timestamp ordering, min 5 valid interactions/user |
| **MUST-HAVE (MVP)** | Baselines: Most-Popular & GRU4Rec | Popularity from train split only; GRU4Rec in pure PyTorch; shared splits/IDs/candidates |
| **MUST-HAVE (MVP)** | Main Model: SASRec in pure PyTorch | Causal self-attention, learned item + positional embeddings, right-padding, padding-safe attention, FFN, residual, dropout, LayerNorm |
| **MUST-HAVE (MVP)** | Full-Catalog Evaluation & Error Analysis | HR@10, NDCG@10 on all valid candidates; length-quantile groups; 3 concrete error cases |
| **MUST-HAVE (MVP)** | Local Demo: FastAPI + React (Stitch UI) | Ordered real MovieLens-1M titles -> Top-10 from SASRec, GRU4Rec, Popularity with raw scores |
| **MUST-HAVE (MVP)** | Course Deliverables | 15–20 page report (7 sections), 5 slides, tested GitHub repo + source ZIP |
| **SHOULD-HAVE (Post-MVP)** | Ablation (`L ∈ {50, 100, 200}` or `b ∈ {1, 2, 3}`) | Only after MVP is complete and stable |
| **SHOULD-HAVE (Post-MVP)** | Attention Heatmap | Optional sequence-level diagnostic for a chosen layer/head (not causal proof) |
| **OUT OF SCOPE** | BERT4Rec, Amazon Reviews, ML-20M/25M, Cloud Deploy | Explicitly excluded |

> Completion means executing all required experiments and deliverables rigorously—**not** guaranteeing that SASRec beats every baseline.

---

## 2. Item IDs, Padding & Sequence Rules

- **Padding ID**: `0` is strictly reserved for padding.
- **Valid Item IDs**: Contiguous integers `1..N`, where `N` is determined during preprocessing from rated movies in MovieLens-1M.
- **Default Max Sequence Length**: `L = 100` (see `configs/default.json`).
- **Truncation**: Sequences longer than `L` keep the **most recent `L` valid items**.
- **Padding Side**: **Right-padding** with `0` (e.g., `[12, 45, 301, 0, 0, ...]`).
- **Valid Input Requirement**: Every sequence fed to a model must contain **at least 1 valid item** (`item_id > 0`).
- **Prediction Position**: Always read the representation at the **last valid (non-padding) position**; never predict from a padding query state.
- **Default Seed**: `42`.

---

## 3. Processed Dataset Schema (`data/ml1m_processed.pkl`)

Python `dict` with keys:

```python
{
    "train": dict[int, list[int]],   # user_id -> ordered item_ids [s_1, ..., s_{n-2}]
    "val": dict[int, int],           # user_id -> target item s_{n-1}
    "test": dict[int, int],          # user_id -> target item s_n
    "num_users": int,
    "num_items": int,                # N (valid items are 1..N; 0 is padding)
    "item2meta": dict[int, dict],    # item_id -> {"title": str, "genres": list[str], "year": int | None}
    "raw_to_item_id": dict[int, int],# original MovieLens MovieID -> internal 1..N
    "item_id_to_raw": dict[int, int],# internal 1..N -> original MovieLens MovieID
    "mapping_version": str,          # hash/version string stored in checkpoints for validation
}
```

Companion file: `data/item2title.json` mapping stringified internal item ID (`"1"`..`"N"`) to metadata (`title`, `genres`, `year`) for the API and React UI.

**Data caveats (must be documented in code & report):**
- Rating timestamps reflect when a user submitted a rating, **not** verified watch times.
- Deterministic tie-breaking rule is required whenever two interactions for the same user share the same timestamp (e.g., sort by `(timestamp, original_line_index)` or `(timestamp, raw_movie_id)` and document it).

---

## 4. Leave-One-Out Split & Evaluation Metrics

For each user's chronologically sorted sequence $S^u = (s_1, \dots, s_n)$ with $n \ge 5$:

- **Train**: $(s_1, \dots, s_{n-2})$ — next-item supervision targets come strictly from within this prefix.
- **Validation**: Predict $s_{n-1}$ given history $(s_1, \dots, s_{n-2})$. Used for hyperparameter tuning and checkpoint selection (`NDCG@10`).
- **Test**: Predict $s_n$ given history $(s_1, \dots, s_{n-1})$. Evaluated **once** after final model selection. Never tune on test results.

### Full-Catalog Ranking Protocol (Primary)
- Candidate set = all valid items `1..N`, excluding padding `0` and items already seen in the user's input history, **while explicitly keeping the held-out target item** in the candidate pool.
- Fixed deterministic tie-breaking policy (e.g., break equal scores consistently by `item_id`).
- All three models (`SASRec`, `GRU4Rec`, `Most-Popular`) must be evaluated on the exact same user set, item-ID mapping, and candidate rules.
- Optional `sampled-100` (1 positive + 100 negatives) may be added post-MVP for reference, but must **never** be compared directly against full-catalog scores.

### Metric Formulas
$$\text{HR}@10 = \frac{1}{|U|} \sum_{u \in U} \mathbb{I}(\text{rank}_u \le 10)$$

$$\text{NDCG}@10 = \frac{1}{|U|} \sum_{u \in U} \frac{\mathbb{I}(\text{rank}_u \le 10)}{\log_2(\text{rank}_u + 1)}$$

where $\text{rank}_u \ge 1$ is the 1-indexed rank of user $u$'s held-out target item among valid candidates.

---

## 5. FastAPI Inference Contract (`POST /api/recommend`)

### Request
```json
{
  "history_ids": [12, 45, 301],
  "top_k": 10
}
```
- `history_ids`: ordered non-empty list of valid internal MovieLens-1M item IDs (`1..N`).
- `top_k`: positive integer (default `10`).

### Response
```json
{
  "history_ids": [12, 45, 301],
  "sasrec_recommendations": [
    {
      "item_id": 88,
      "title": "Toy Story (1995)",
      "genres": ["Animation", "Children's", "Comedy"],
      "score": 1.24
    }
  ],
  "gru4rec_recommendations": [],
  "popular_recommendations": [],
  "attention": {
    "layer": 1,
    "head": 0,
    "weights": []
  }
}
```
- `attention` may be `null` (optional sequence-level diagnostic; does **not** prove causal attribution for a specific recommended movie).
- `score` is the **raw model score** (logit/count)—the UI must **never** label it as a confidence percentage or probability.

---

## 6. Checkpoint, Memory & Loss Contract

### Checkpoint Payload (`checkpoints/*.pt`)
Every saved checkpoint must include:
- `model_state_dict`
- `model_type` (`"sasrec"` | `"gru4rec"` | `"popularity"`)
- `config` (exact architecture & training hyperparameters)
- `seed`
- `mapping_version` & `num_items`
- `val_metrics` (`{"HR@10": ..., "NDCG@10": ...}`)

### Loss & RTX 4060 (8 GB VRAM) Strategy
- Start with next-item **cross-entropy** over valid target positions (`ignore_index=0`).
- If full-catalog logits across all timesteps exceed 8 GB VRAM:
  1. Reduce batch size or sequence length `L`,
  2. Enable mixed precision (AMP), or
  3. Compute logits/loss in chunks.
- If switching to a sampled loss (e.g., BCE with negative sampling), record the exact change in config and report—do not assume one loss automatically improves NDCG.

---

## 7. Mandatory Verification Gates

1. **Data & Metrics (`tests/`)**:
   - Synthetic sequences verify exact hand-computed `HR@10` and `NDCG@10`.
   - Zero target leakage from `val`/`test` into `train` or `popularity` counts.
   - Seen history items are masked out while target item remains rankable; `pad_id=0` is always excluded.
2. **SASRec Mask & Padding (`tests/`)**:
   - Forward/backward pass produces finite loss and finite gradients on short right-padded batches.
   - Modifying any future valid token `t' > t` produces zero change at valid output position `t` (no future leakage).
   - Changing padding length / right-padding does not alter valid attention outputs.
   - Last valid hidden state is selected accurately for variable-length histories.
