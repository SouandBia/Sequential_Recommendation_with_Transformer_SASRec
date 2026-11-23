# Test Suite (`tests/`)

Run all tests from the repository root:

```powershell
pytest
```

## Required Verification Tests (per Execution Plan & `CONTRACT.md`)

1. **Data & Leave-One-Out Split (M1 — Student B)**:
   - Valid item IDs are strictly `1..N` with `0` reserved for padding.
   - Chronological order with deterministic timestamp tie-breaking.
   - No `val` or `test` target leaks into `train` interactions or Most-Popular counts.
2. **Full-Catalog Metrics (M1 — Student B)**:
   - Synthetic ranking cases match hand-computed `HR@10` and `NDCG@10`.
   - Seen history items are excluded while the held-out target item remains rankable.
   - Padding ID `0` is excluded and tie-breaking is deterministic.
3. **SASRec Right-Padding & Causal Mask Safety (M3 — Student A)**:
   - Forward & backward pass yield finite loss and finite gradients on short right-padded batches.
   - Modifying any future valid token does not change earlier valid outputs (no future leakage).
   - Changing right-padding does not alter valid attention outputs.
   - Last valid hidden state is selected accurately for variable-length sequences.
4. **API Contract (M4 — Student B)**:
   - `POST /api/recommend` validates invalid/empty histories and matches the agreed response schema.
