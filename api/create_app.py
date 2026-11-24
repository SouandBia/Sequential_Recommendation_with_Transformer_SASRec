"""FastAPI application orchestrator exposing /api/recommend and /api/movies."""

from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.generate_mock_recommendations import generate_mock_recommendations
from api.load_catalog import load_catalog
from api.search_catalog import search_catalog
from api.validate_history import validate_history


class _RecommendRequest(BaseModel):
    history_ids: list[int] = Field(..., description="Ordered MovieLens-1M internal item IDs (1..N).")
    top_k: int = Field(default=10, description="Number of recommendations to return.")


def create_app(item2title_path: str | Path = "data/item2title.json") -> FastAPI:
    """Create and wire the FastAPI recommendation service (Week 1 Mock Mode)."""
    catalog = load_catalog(item2title_path)

    service = FastAPI(
        title="Team 39 — SASRec MovieLens-1M Recommendation API",
        version="0.1.0",
    )
    service.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @service.get("/", include_in_schema=False)
    def root() -> FileResponse:
        ui_file = Path("web/index.html")
        if not ui_file.is_file():
            raise HTTPException(status_code=404, detail="web/index.html not found.")
        return FileResponse(ui_file)

    @service.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "mode": "mock",
            "catalog_size": len(catalog),
        }

    @service.get("/api/movies")
    def movies(
        q: str = Query(default="", description="Case-insensitive title search query."),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        return {"items": search_catalog(catalog=catalog, query=q, limit=limit)}

    @service.post("/api/recommend")
    def recommend(request: _RecommendRequest) -> dict:
        try:
            validate_history(
                history_ids=request.history_ids,
                catalog=catalog,
                top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return generate_mock_recommendations(
            history_ids=request.history_ids,
            catalog=catalog,
            top_k=request.top_k,
        )

    return service


app = create_app()
