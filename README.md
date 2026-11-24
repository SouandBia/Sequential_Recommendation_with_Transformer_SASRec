# Team 39 — Project 8: Sequential Recommendation with Transformer (SASRec)

Next-item sequential recommendation on **MovieLens-1M**, comparing a pure-PyTorch **SASRec** (causal self-attention) against **GRU4Rec** and **Most-Popular** baselines under a shared Leave-One-Out full-catalog evaluation protocol, with a local **FastAPI + React (Stitch UI)** interactive demo.

- **Execution Plan**: [`Team39_Project8_SASRec_Execution_Plan_Tieng_Viet.docx`](Team39_Project8_SASRec_Execution_Plan_Tieng_Viet.docx)
- **Shared Technical Contract**: [`CONTRACT.md`](CONTRACT.md)
- **Shared Config Defaults**: [`configs/default.json`](configs/default.json)

---

## Team Division

| Member | Primary Ownership | Report Sections |
| --- | --- | --- |
| **Student A** | Pure-PyTorch SASRec & GRU4Rec, right-padding/causal mask tests, loss & VRAM benchmarking (RTX 4060), checkpoint selection (val NDCG@10), evaluation & error analysis | Sections 3, 5 (+ co-author 4, 7) |
| **Student B** | MovieLens-1M preprocessing & Leave-One-Out pipeline, full-catalog evaluator & unit tests, Most-Popular baseline, movie metadata, FastAPI backend (mock -> real), Stitch React UI | Sections 1, 2, 6 (+ co-author 4, 7) |

---

## Repository Structure

```text
CS467/
├── configs/
│   └── default.json          # Locked contract defaults (seed, L=100, d=64, 2 blocks, 2 heads)
├── data/
│   ├── ml-1m/                # Unpacked raw MovieLens-1M files (gitignored)
│   └── README.md             # Official download instructions & processed schema
├── src/                      # One-responsibility-per-file Python modules + orchestrators
│   ├── data/                 # Loading, validation, Leave-One-Out splitting
│   ├── models/               # SASRec, GRU4Rec, Most-Popular
│   ├── training/             # Seeding, batching, loss, epoch loop, checkpoints
│   ├── evaluation/           # Full-catalog scoring, HR@10 & NDCG@10
│   ├── analysis/             # Length-quantile buckets, error cases, optional attention
│   └── README.md             # Module design & mapping to plan milestones
├── tests/                    # Synthetic metrics, no-leakage, causal mask & API contract tests
├── checkpoints/              # Saved .pt checkpoints (weights + mapping_version + config + seed)
├── api/                      # FastAPI service exposing POST /api/recommend
├── web/                      # React frontend (Stitch UI) calling FastAPI
├── results/
│   └── figures/              # Metrics CSVs (baselines.csv, etc.) & plots
├── report/
│   ├── figures/              # Report diagrams & screenshots
│   ├── slides/               # 5-slide final presentation
│   └── outline.md            # 15–20 page 7-section report structure
├── CONTRACT.md               # Agreed data, evaluation, model & API contract
├── requirements.txt          # Pinned Python 3.10 dependencies
└── pyproject.toml            # Project & pytest settings
```

---

## Local Setup (Python 3.10 + RTX 4060 8 GB VRAM)

### 1. Python Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
- **With RTX 4060 GPU (Windows, CUDA 13.0 wheel):**
  ```powershell
  pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
  pip install -r requirements.txt
  ```
- **Standard / CPU-only or fresh-clone verification:**
  ```powershell
  pip install -r requirements.txt
  ```

Verify PyTorch & GPU visibility:
```powershell
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

### 3. Download MovieLens-1M
Follow [`data/README.md`](data/README.md) to download `ml-1m.zip` from GroupLens and unpack it into `data/ml-1m/`. Do **not** commit raw MovieLens files to GitHub.

### 4. Run Tests
```powershell
pytest
```

---

## 48-Hour Kickoff Checklist

- [x] **Task 1 (Both | Hours 1–6)**: Initialize repository structure (`data/`, `src/`, `tests/`, `checkpoints/`, `api/`, `web/`, `results/`, `report/`), pin Python dependencies (`requirements.txt`), and lock shared data/API/model contract (`CONTRACT.md`, `configs/default.json`).
- [x] **Task 2 (Student B | Hours 6–48)**: Download MovieLens-1M from GroupLens; build chronological Leave-One-Out preprocessing (`data/ml1m_processed.pkl`, `data/item2title.json`); implement & unit-test full-catalog HR@10/NDCG@10 on synthetic data; expose mock `POST /api/recommend` in `api/` for immediate React integration.
- [x] **Task 3 (Student A | Hours 6–48)**: Implement pure-PyTorch SASRec (`d=64`, 2 blocks, 2 heads, `dropout=0.2`, right-padding, `L=100`); verify finite forward/backward pass, last-valid-position selection, padding invariance, and zero future token leakage via synthetic tests.

---

## 4-Week Milestones

| Milestone | Owner | Core Deliverable |
| --- | --- | --- |
| **M1** (Week 1) | Student B | MovieLens-1M preprocessing + Leave-One-Out + full-catalog HR@10/NDCG@10 tests + mock FastAPI |
| **M2** (Week 2) | Student A + B | Most-Popular & GRU4Rec baselines evaluated reproducibly -> `results/baselines.csv` |
| **M3** (Week 2) | Student A | Pure-PyTorch SASRec verified & trained (`checkpoints/sasrec_best.pt` chosen by val NDCG@10) |
| **M4** (Week 3) | Both | Length-quantile & 3-case error analysis + end-to-end local FastAPI + Stitch React demo |
| **M5** (Week 4) | Both | 15–20 page report (7 sections), 5 slides, fresh-clone test & final GitHub + ZIP submission |
