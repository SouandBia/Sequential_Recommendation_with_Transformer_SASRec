"""Unit tests for the FastAPI /api/recommend and /api/movies contract."""

import json
from pathlib import Path
from fastapi.testclient import TestClient

from api.create_app import create_app


def test_recommend_and_movies_contract_and_cors(tmp_path: Path) -> None:
    catalog_file = tmp_path / "item2title.json"
    catalog = {
        str(idx): {
            "title": f"Movie {idx} (199{idx % 10})",
            "genres": ["Action"] if idx % 2 == 0 else ["Comedy", "Drama"],
            "year": 1990 + (idx % 10),
        }
        for idx in range(1, 26)
    }
    catalog_file.write_text(json.dumps(catalog), encoding="utf-8")

    client = TestClient(create_app(item2title_path=catalog_file))

    # CORS preflight simulating browser React app call
    preflight = client.options(
        "/api/recommend",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers.get("access-control-allow-origin") == "*"

    movies_res = client.get("/api/movies", params={"q": "Movie 1", "limit": 5})
    assert movies_res.status_code == 200
    assert len(movies_res.json()["items"]) == 5

    rec_res = client.post("/api/recommend", json={"history_ids": [2, 5, 10], "top_k": 10})
    assert rec_res.status_code == 200
    body = rec_res.json()

    assert body["history_ids"] == [2, 5, 10]
    for key in ("sasrec_recommendations", "gru4rec_recommendations", "popular_recommendations"):
        recs = body[key]
        assert len(recs) == 10
        returned_ids = [item["item_id"] for item in recs]
        assert 0 not in returned_ids
        assert set(returned_ids).isdisjoint({2, 5, 10})
        for item in recs:
            assert set(item.keys()) == {"item_id", "title", "genres", "score"}
            assert isinstance(item["score"], float | int)

    assert body["attention"] is None or set(body["attention"].keys()) == {"layer", "head", "weights"}


def test_recommend_rejects_invalid_histories(tmp_path: Path) -> None:
    catalog_file = tmp_path / "item2title.json"
    catalog_file.write_text(
        json.dumps({"1": {"title": "Toy Story (1995)", "genres": ["Animation"], "year": 1995}}),
        encoding="utf-8",
    )
    client = TestClient(create_app(item2title_path=catalog_file))

    assert client.post("/api/recommend", json={"history_ids": [], "top_k": 10}).status_code == 422
    assert client.post("/api/recommend", json={"history_ids": [0], "top_k": 10}).status_code == 422
    assert client.post("/api/recommend", json={"history_ids": [9999], "top_k": 10}).status_code == 422
    assert client.post("/api/recommend", json={"history_ids": [1], "top_k": 0}).status_code == 422
