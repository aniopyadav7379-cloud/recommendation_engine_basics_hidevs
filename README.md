# Recommendation Engine

A modular recommendation engine: similarity metrics, candidate
generation, weighted scoring/ranking, and offline evaluation — wired
into a service layer, a REST API, a CLI, and a durable persistence
layer.

## Architecture

```
                    ┌─────────────────┐
                    │   Repository     │  (persistence.py)
                    │ InMemory | JSON  │
                    └────────┬─────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      │                      │                       │
┌─────▼──────┐      ┌────────▼────────┐     ┌────────▼────────┐
│ Similarity │◄─────┤ CandidateGen     │     │ Scorer & Ranker  │
│ Calculator │      │ (Strategy pattern)│────►│ (weighted fusion)│
└────────────┘      └──────────────────┘     └────────┬─────────┘
                                                        │
                                              ┌─────────▼─────────┐
                                              │ RecommendationSvc  │
                                              │  (service.py)      │
                                              └────┬───────────┬───┘
                                                   │           │
                                          ┌────────▼──┐   ┌────▼─────┐
                                          │  REST API  │   │   CLI    │
                                          │ (api.py)   │   │ (cli.py) │
                                          └────────────┘   └──────────┘
```

- **`similarity.py`** — cosine, Jaccard, Pearson, and adjusted-cosine
  (rater-bias-corrected) similarity. Jaccard is memoized since the
  same set pairs get compared repeatedly during candidate scoring.
- **`persistence.py`** — a `Repository` ABC decouples every other
  component from storage. `InMemoryRepository` for tests/experiments,
  `JSONFileRepository` for durable, atomically-written disk storage.
  Swapping in a SQL/NoSQL-backed repository later means implementing
  the same interface — nothing upstream changes.
- **`candidate_gen.py`** — Strategy pattern: `CollaborativeStrategy`,
  `ContentBasedStrategy`, `PopularityStrategy` each implement
  `CandidateStrategy.generate()`. `CandidateGenerator` composes them
  and blends via round-robin interleaving (`hybrid_candidates`), with
  cold-start users automatically falling back to popularity.
- **`scorer.py`** — pluggable weighted scoring (`add_scorer`), with
  per-scorer failure isolation (one broken signal doesn't kill a
  ranking pass) and a human-readable explanation per result.
- **`evaluator.py`** — precision@k, recall@k, NDCG@k (position-aware),
  and `evaluate_all` for batch evaluation, skipping users with no
  ground truth rather than counting them as zero.
- **`service.py`** — the orchestration layer: wires generation →
  scoring → ranking behind one call, and owns the *personalized*
  relevance function (built from each user's own tag profile — this
  is what makes recommendations differ per user instead of collapsing
  to the same global ranking).
- **`api.py`** — FastAPI layer exposing `GET /recommendations/{user_id}`,
  `POST /interactions`, `GET /health`.
- **`cli.py`** — `reco-engine recommend <user>` / `record <user> <item>`
  for local/batch use without running a server.
- **`config.py`** — all tunables (weights, cache paths, log level) are
  environment-variable driven (`RECO_*`), not hard-coded.
- **`exceptions.py`** — a small typed error hierarchy so callers can
  catch exactly what they care about.
- **`logging_config.py`** — structured `logging` output instead of
  `print`, level controlled via `RECO_LOG_LEVEL`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the `reco-engine` CLI entry point
```

## Running things

```bash
# Run the demo pipeline against seed data in data/
python run_demo.py

# CLI
python -m reco_engine.cli recommend alice --k 5
python -m reco_engine.cli record alice job_9

# API (http://localhost:8000/docs for interactive Swagger UI)
uvicorn reco_engine.api:app --reload

# Tests
pytest --cov=reco_engine --cov-report=term-missing
```

## Docker

```bash
docker build -t reco-engine .
docker run -p 8000:8000 reco-engine
```

## Configuration

All settings are overridable via environment variables (see
`config.py` for the full list), e.g.:

```bash
RECO_MAX_CANDIDATES=50 RECO_WEIGHT_RELEVANCE=0.6 RECO_LOG_LEVEL=DEBUG \
  python -m reco_engine.cli recommend alice
```

## Known limitations / next steps

This is still an in-process, single-instance system — the honest
scope of what "add production readiness to an algorithmic prototype"
can mean without introducing a real database or cloud infra:

- `JSONFileRepository` is fine for a single process; concurrent
  writers would need a real database (Postgres/Redis) or file locking.
- Recency/popularity signals are passed in as plain dicts rather than
  computed by a feature-store/batch pipeline.
- No auth/rate-limiting on the API — add a reverse proxy or
  `fastapi`-level auth middleware before exposing this publicly.
- No horizontal scaling story yet (would need the repository swapped
  for a shared backend first).
