"""FastAPI application orchestrator exposing /api/recommend and /api/movies."""

from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.generate_mock_recommendations import generate_mock_recommendations
from api.load_catalog import load_catalog
from api.load_models import load_models
from api.predict_recommendations import predict_recommendations
from api.search_catalog import search_catalog
from api.validate_history import validate_history
from src.models.build_gru4rec import build_gru4rec
from src.models.build_sasrec import build_sasrec


class _RecommendRequest(BaseModel):
    history_ids: list[int] = Field(..., description="Ordered MovieLens-1M internal item IDs (1..N).")
    top_k: int = Field(default=10, description="Number of recommendations to return.")


def create_app(
    item2title_path: str | Path = "data/item2title.json",
    checkpoints_dir: str | Path | None = "checkpoints",
    device: str = "cpu",
) -> FastAPI:
    """Create and wire the FastAPI recommendation service (real checkpoints or mock fallback)."""
    catalog = load_catalog(item2title_path)
    models_bundle = (
        load_models(
            checkpoints_dir=checkpoints_dir,
            sasrec_builder=build_sasrec,
            gru4rec_builder=build_gru4rec,
            expected_num_items=len(catalog),
            device=device,
        )
        if checkpoints_dir is not None
        else None
    )
    mode = "real" if models_bundle is not None else "mock"

    service = FastAPI(
        title="Team 39 — SASRec MovieLens-1M Recommendation API",
        version="1.0.0",
    )
    service.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    web_dist_assets = Path("web/dist/assets")
    if web_dist_assets.is_dir():
        service.mount("/assets", StaticFiles(directory=web_dist_assets), name="assets")

    @service.get("/", include_in_schema=False)
    def root() -> FileResponse:
        built_ui = Path("web/dist/index.html")
        fallback_ui = Path("web/index.html")
        target = built_ui if built_ui.is_file() else fallback_ui
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Frontend index.html not found.")
        return FileResponse(target)

    @service.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "mode": mode,
            "catalog_size": len(catalog),
            "mapping_version": models_bundle["mapping_version"] if models_bundle else None,
            "device": models_bundle["device"] if models_bundle else "none",
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

        if models_bundle is not None:
            return predict_recommendations(
                history_ids=request.history_ids,
                models_bundle=models_bundle,
                catalog=catalog,
                top_k=request.top_k,
            )

        return generate_mock_recommendations(
            history_ids=request.history_ids,
            catalog=catalog,
            top_k=request.top_k,
        )

    return service


app = create_app()
