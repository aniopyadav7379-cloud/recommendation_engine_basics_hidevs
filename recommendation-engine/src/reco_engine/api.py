"""REST API layer.

Exposes the recommendation engine over HTTP so it can actually be
deployed and called by other services, instead of only being usable
as a library imported into a script.

Run locally:
    uvicorn reco_engine.api:app --reload

Endpoints:
    GET  /health          Back-compat liveness check (kept for existing callers)
    GET  /health/live      Liveness: is the process up and able to respond at all
    GET  /health/ready      Readiness: can the service actually serve traffic
                             (i.e. is the persistence layer reachable)
    GET  /recommendations/{user_id}?k=5
    POST /interactions   {"user_id": "...", "item_id": "..."}

All request validation (id shape, `k` bounds, required fields) is
expressed declaratively through Pydantic/FastAPI constraints rather
than hand-rolled `if` checks, so every validation failure is reported
through the same mechanism and the same response envelope:

    {"error": {"type": "<machine-readable-code>", "message": "<human text>"}}

Internal exception details/tracebacks are never included in a response
body — they're logged server-side instead.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Path, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .exceptions import RecoEngineError
from .logging_config import get_logger
from .persistence import JSONFileRepository
from .service import RecommendationService

logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="Recommendation Engine API",
    version="0.3.0",
    description="Serves candidate generation, scoring, and ranking over HTTP.",
)

_repo = JSONFileRepository(settings.data_dir, settings.history_file, settings.features_file)
_service = RecommendationService(_repo)

# Identifiers are restricted to a conservative, URL- and JSON-safe
# character set. This isn't about "sanitizing for security" (the
# repository never interpolates these into a query or a shell command)
# — it's about rejecting obviously-malformed input (empty strings,
# whitespace-only ids, stray slashes/control characters) declaratively,
# before it ever reaches the recommendation pipeline.
_ID_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"

UserIdPath = Annotated[
    str,
    Path(..., pattern=_ID_PATTERN, description="Target user id (letters, digits, '.', '_', '-'; 1-64 chars)"),
]
KQuery = Annotated[
    int | None,
    Query(gt=0, le=settings.max_k, description="Number of recommendations to return"),
]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RecommendationItem(BaseModel):
    item_id: str
    score: float
    explanation: str


class InteractionRequest(BaseModel):
    user_id: str = Field(..., pattern=_ID_PATTERN, description="1-64 chars: letters, digits, '.', '_', '-'")
    item_id: str = Field(..., pattern=_ID_PATTERN, description="1-64 chars: letters, digits, '.', '_', '-'")


class ErrorBody(BaseModel):
    type: str
    message: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody


# ---------------------------------------------------------------------------
# Request timing + logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    """Log every request's method, path, status code, and duration.

    Never logs request/response bodies (which could contain user data)
    — only routing metadata, which is safe and genuinely useful for
    diagnosing slow endpoints or elevated error rates.
    """
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.2fms)",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Centralized error handling — never leak internals, always one envelope shape
# ---------------------------------------------------------------------------
def _error_response(status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"type": error_type, "message": message}})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Pydantic's own field-level messages are safe to surface (they
    # describe the request the caller sent, not internal state).
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(part) for part in first.get("loc", []) if part != "body")
    detail = first.get("msg", "Invalid request")
    message = f"{field}: {detail}" if field else detail
    logger.info("Validation error on %s %s: %s", request.method, request.url.path, message)
    return _error_response(status.HTTP_422_UNPROCESSABLE_CONTENT, "validation_error", message)


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    return _error_response(exc.status_code, "request_error", str(exc.detail))


@app.exception_handler(RecoEngineError)
async def handle_engine_error(request: Request, exc: RecoEngineError) -> JSONResponse:
    logger.warning("Engine error on %s %s: %s", request.method, request.url.path, exc)
    return _error_response(status.HTTP_400_BAD_REQUEST, "engine_error", str(exc))


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # The one place a bare Exception handler is correct: this is the
    # last line of defense. Full detail (including traceback) goes to
    # the server log only; the client gets a generic, safe message.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "An internal error occurred."
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Back-compat endpoint, unchanged. Prefer /health/live and
    /health/ready for new integrations (e.g. Kubernetes probes)."""
    return {"status": "ok"}


@app.get("/health/live")
def health_live() -> dict:
    """Liveness: process is up and can handle a request at all. Never
    touches storage — a slow disk should fail readiness, not liveness."""
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready() -> JSONResponse:
    """Readiness: can this instance actually serve traffic right now.

    Exercises the persistence layer with a cheap, read-only call; if
    that fails, the instance is up but not ready and shouldn't receive
    traffic yet (or anymore).
    """
    try:
        _repo.get_all_user_history()
    except Exception as exc:  # noqa: BLE001 - any storage failure means "not ready"
        logger.warning("Readiness check failed: %s", exc)
        return _error_response(status.HTTP_503_SERVICE_UNAVAILABLE, "not_ready", "Storage is not reachable.")
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})


# ---------------------------------------------------------------------------
# Recommendations / interactions
# ---------------------------------------------------------------------------
@app.get(
    "/recommendations/{user_id}",
    response_model=list[RecommendationItem],
    responses={422: {"model": ErrorEnvelope}},
)
def get_recommendations(user_id: UserIdPath, k: KQuery = None) -> list[RecommendationItem]:
    results = _service.get_recommendations(user_id, k)
    return [
        RecommendationItem(item_id=r.item_id, score=r.score, explanation=r.explanation)
        for r in results
    ]


@app.post("/interactions", status_code=204, responses={422: {"model": ErrorEnvelope}})
def record_interaction(payload: InteractionRequest) -> None:
    _service.record_interaction(payload.user_id, payload.item_id)
    logger.info("Interaction recorded via API: user=%s item=%s", payload.user_id, payload.item_id)
