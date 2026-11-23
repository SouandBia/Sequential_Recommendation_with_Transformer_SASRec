# `src/` Architecture — One Responsibility Per File

All Python code under `src/` follows the **One Responsibility Per File** rule:

- Every module performs **one job**, named in `verb_noun.py` format.
- Each module exposes **one main public function** with the same name as the file (private `_helpers` allowed only for that single job).
- Single-job modules do **not** import other single-job modules.
- Separate **orchestrator** files (`src/prepare_data.py`, `src/run_training.py`, `src/run_evaluation.py`, `src/run_analysis.py`) import and chain single-job modules in order without embedding processing logic of their own.
- No catch-all `utils.py` or `helpers.py` files.

## Planned Mapping from Execution Plan

| Execution Plan Concept | Single-Job Modules | Orchestrator |
| --- | --- | --- |
| `dataset` (M1 — Student B) | `src/data/load_ratings.py`, `src/data/load_movies.py`, `src/data/build_sequences.py`, `src/data/split_leave_one_out.py`, `src/data/save_dataset.py` | `src/prepare_data.py` |
| `metrics` (M1 — Student B) | `src/evaluation/mask_seen_items.py`, `src/evaluation/rank_candidates.py`, `src/evaluation/compute_hr_ndcg.py` | `src/run_evaluation.py` |
| `baselines` (M2 — Student A & B) | `src/models/fit_popularity.py`, `src/models/build_gru4rec.py` | `src/run_training.py` |
| `sasrec` (M3 — Student A) | `src/models/build_sasrec.py` (with right-padding & causal attention mask safety) | `src/run_training.py` |
| `train` (M2–M3 — Student A) | `src/training/set_seed.py`, `src/training/pad_sequences.py`, `src/training/compute_next_item_loss.py`, `src/training/train_epoch.py`, `src/training/save_checkpoint.py` | `src/run_training.py` |
| Error Analysis (M4 — Student A) | `src/analysis/group_by_length.py`, `src/analysis/collect_error_cases.py`, `src/analysis/extract_attention.py` (optional) | `src/run_analysis.py` |
