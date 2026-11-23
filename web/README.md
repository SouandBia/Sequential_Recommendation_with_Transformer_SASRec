# Stitch React Frontend (`web/`)

Owner: **Student B**.

## Responsibilities
- Built with React (UI design generated via Stitch; Stitch is UI-only and performs **no** AI inference).
- Connects to the local FastAPI backend (`http://127.0.0.1:8000/api/recommend`).
- Core UI requirements:
  1. Search and add **real MovieLens-1M movie titles** in explicit user-chosen chronological order.
  2. Display loading and error states cleanly (including empty-history validation).
  3. Render side-by-side **Top-10 recommendation lists** for **SASRec**, **GRU4Rec**, and **Most-Popular**.
  4. Display **raw model scores**—never label them as confidence percentages or probabilities.
  5. Optional post-MVP: visualize sequence-level attention weights if `attention` is non-null, with a clear caveat that weights describe position relationships rather than causal proof.

## Fallback Plan (Risk #3)
If complex React UI integration slips past Week 3, keep a minimal React/HTML client against the existing FastAPI endpoint (or a minimal Streamlit fallback using the exact same checkpoints).
