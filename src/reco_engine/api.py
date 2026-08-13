"""REST API layer.

Exposes the recommendation engine over HTTP so it can actually be
deployed and called by other services, instead of only being usable
as a library imported into a script.

Run locally:
    uvicorn reco_engine.api:app --reload

Endpoints:
    GET  /health
    GET  /recommendations/{user_id}?k=5
    POST /interactions   {"user_id": "...", "item_id": "..."}
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .logging_config import get_logger
from .persistence import JSONFileRepository
from .service import RecommendationService

logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="Recommendation Engine API",
    version="0.2.0",
    description="Serves candidate generation, scoring, and ranking over HTTP.",
)

_repo = JSONFileRepository(settings.data_dir, settings.history_file, settings.features_file)
_service = RecommendationService(_repo)


class RecommendationItem(BaseModel):
    item_id: str
    score: float
    explanation: str


class InteractionRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    item_id: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/recommendations/{user_id}", response_model=List[RecommendationItem])
def get_recommendations(user_id: str, k: Optional[int] = None) -> List[RecommendationItem]:
    if k is not None and k <= 0:
        raise HTTPException(status_code=400, detail="k must be a positive integer")
    results = _service.get_recommendations(user_id, k)
    return [RecommendationItem(item_id=r.item_id, score=r.score, explanation=r.explanation)
            for r in results]


@app.post("/interactions", status_code=204)
def record_interaction(payload: InteractionRequest) -> None:
    _service.record_interaction(payload.user_id, payload.item_id)
    logger.info("Interaction recorded via API: user=%s item=%s", payload.user_id, payload.item_id)
