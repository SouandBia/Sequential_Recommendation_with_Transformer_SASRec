# FastAPI Backend (`api/`)

Owner: **Student B** (with model loader hooks aligned with **Student A**'s checkpoints).

## Responsibilities
- **Week 1 (Mock Mode)**: Expose `POST /api/recommend` (plus catalog search endpoint such as `GET /api/movies`) returning valid schema responses matching [`../CONTRACT.md`](../CONTRACT.md) so the React UI can integrate immediately.
- **Week 3 (Real Inference)**: Load verified PyTorch checkpoints (`checkpoints/sasrec_best.pt`, `checkpoints/gru4rec_best.pt`, popularity stats) and `data/item2title.json` at startup, validate `mapping_version`, and return Top-10 recommendations with raw model scores (and optional `attention` weights or `null`).

## API Contract Summary
- Endpoint: `POST /api/recommend`
- Request: `{"history_ids": [12, 45, 301], "top_k": 10}`
- Response: `history_ids`, `sasrec_recommendations`, `gru4rec_recommendations`, `popular_recommendations`, `attention` (`object | null`).
- Validate that `history_ids` is non-empty and all IDs exist in `1..N`.
- Measure and report real local latency; do not claim arbitrary latency guarantees.
