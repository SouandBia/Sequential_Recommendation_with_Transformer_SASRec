"""Unit and integration tests for Week 3 error analysis, attention diagnostics, and real API."""

from pathlib import Path
import time
from fastapi.testclient import TestClient
import pytest
import torch

from api.create_app import create_app
from api.load_models import load_models
from src.analysis.collect_error_cases import collect_error_cases
from src.analysis.extract_attention import extract_attention
from src.analysis.group_by_length import group_by_length
from src.models.build_gru4rec import build_gru4rec
from src.models.build_sasrec import build_sasrec


def test_group_by_length_cohorts_and_metrics() -> None:
    lengths = [10, 12, 50, 60, 200, 250]
    ranks = {
        "popularity": [20, 30, 15, 25, 1, 5],
        "gru4rec": [1, 12, 2, 11, 3, 20],
        "sasrec": [1, 5, 1, 2, 1, 15],
    }
    rows = group_by_length(user_lengths=lengths, ranks_by_model=ranks, top_k=10, num_quantiles=3)
    assert len(rows) == 9
    short_sas = next(r for r in rows if r["cohort_label"] == "Short" and r["model"] == "sasrec")
    assert short_sas["user_count"] == 2
    assert short_sas["HR@10"] == pytest.approx(1.0)


def test_collect_error_cases_and_extract_attention() -> None:
    item2meta = {
        i: {"title": f"Movie {i} (1995)", "genres": ["Action"] if i <= 5 else ["Drama"], "year": 1995}
        for i in range(1, 12)
    }
    cases = collect_error_cases(
        user_ids=[1, 2, 3],
        test_histories=[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 1, 2, 3], [1] * 160],
        test_targets=[6, 7, 8],
        ranks_by_model={"popularity": [120, 150, 200], "gru4rec": [3, 40, 80], "sasrec": [50, 60, 90]},
        top_k_ids_by_model={"popularity": [[1, 2]] * 3, "gru4rec": [[6, 2]] * 3, "sasrec": [[2, 3]] * 3},
        top_k_scores_by_model={"popularity": [[10.0, 9.0]] * 3, "gru4rec": [[2.0, 1.0]] * 3, "sasrec": [[1.5, 1.2]] * 3},
        item2meta=item2meta,
        train_counts={6: 50, 7: 30, 8: 15},
        max_cases=3,
    )
    assert len(cases) == 3
    assert all("target_movie" in c and "model_predictions" in c for c in cases)

    model = build_sasrec(num_items=20, d_model=32, num_blocks=2, num_heads=2, dropout=0.0, max_seq_len=8)
    x = torch.tensor([[2, 4, 6, 8, 0, 0]], dtype=torch.long)
    attn = extract_attention(model=model, input_ids=x, layer_idx=1, head_idx=0, pad_id=0)
    assert attn["valid_len"] == 4
    assert len(attn["last_query_weights"]) == 4
    assert sum(attn["last_query_weights"]) == pytest.approx(1.0, abs=1e-4)


def test_real_checkpoints_api_end_to_end_and_mapping_validation() -> None:
    if not (Path("checkpoints/sasrec_best.pt").is_file() and Path("data/item2title.json").is_file()):
        pytest.skip("Trained checkpoints or item2title.json not present.")

    # Mismatched mapping_version must be rejected
    with pytest.raises(ValueError, match="mapping_version"):
        load_models(
            checkpoints_dir="checkpoints",
            sasrec_builder=build_sasrec,
            gru4rec_builder=build_gru4rec,
            expected_mapping_version="wrong_version_xyz",
            device="cpu",
        )

    client = TestClient(create_app(item2title_path="data/item2title.json", checkpoints_dir="checkpoints", device="cpu"))
    health = client.get("/api/health")
    assert health.status_code == 200
    health_data = health.json()
    assert health_data["mode"] == "real"
    assert health_data["catalog_size"] == 3706
    assert health_data["mapping_version"] is not None

    history = [1, 2, 6, 10]
    t0 = time.perf_counter()
    res = client.post("/api/recommend", json={"history_ids": history, "top_k": 10})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert res.status_code == 200
    assert elapsed_ms > 0.0

    body = res.json()
    assert body["history_ids"] == history
    for key in ("sasrec_recommendations", "gru4rec_recommendations", "popular_recommendations"):
        recs = body[key]
        assert len(recs) == 10
        ids = [r["item_id"] for r in recs]
        assert 0 not in ids
        assert set(ids).isdisjoint(set(history))
        assert all(isinstance(r["score"], float | int) for r in recs)

    assert body["attention"] is not None
    assert len(body["attention"]["weights"]) == len(history)
    assert sum(body["attention"]["weights"]) == pytest.approx(1.0, abs=1e-3)
