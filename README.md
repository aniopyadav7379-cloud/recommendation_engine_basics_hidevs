# Recommendation Engine — Production-Ready Recommendation Engine

A modular, dependency-light recommendation engine: candidate generation
(collaborative + content-based + popularity), pluggable weighted scoring
with honest explanations, offline evaluation (precision/recall/NDCG@k), a
REST API, and a CLI — built around the Repository and Strategy design
patterns so storage and generation approaches can evolve independently of
everything else.

## Contents

- [Architecture](#architecture)
- [Recommendation pipeline](#recommendation-pipeline)
- [Personalization](#personalization)
- [Cold-start handling](#cold-start-handling)
- [API](#api)
- [Configuration](#configuration)
- [Installation](#installation)
- [Running](#running)
- [Testing](#testing)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Security](#security)
- [Performance](#performance)
- [Compatibility notes](#compatibility-notes)
- [Future improvements](#future-improvements)

## Architecture

```
src/reco_engine/
├── similarity.py       # Component 1: cosine / Jaccard / Pearson / adjusted-cosine
├── candidate_gen.py     # Component 2: Strategy pattern over Repository
├── scorer.py              # Component 3: weighted scoring + honest explanations
├── evaluator.py             # Component 4: precision@k / recall@k / NDCG@k
├── persistence.py             # Repository interface + InMemory / JSONFile impls
├── service.py                   # Orchestrator facade + in-process caching
├── config.py                      # Env-driven, validated Settings
├── exceptions.py                    # Exception hierarchy
├── logging_config.py                  # Shared stdlib logging setup
├── api.py                               # FastAPI REST layer
└── cli.py                                 # argparse CLI
```

Two design patterns hold the codebase together:

- **Repository pattern** (`persistence.py`) — every component reads user
  history and item features through the abstract `Repository` interface,
  never through raw dicts or files directly. `InMemoryRepository` (tests,
  quick experiments) and `JSONFileRepository` (durable, atomic writes) are
  the two implementations provided; swapping in a SQL/NoSQL-backed one
  later means implementing this one interface — nothing upstream changes.
- **Strategy pattern** (`candidate_gen.py`) — each candidate-generation
  approach (`CollaborativeStrategy`, `ContentBasedStrategy`,
  `PopularityStrategy`) is an independent class implementing
  `CandidateStrategy.generate()`. Adding a new approach (e.g. a trained
  embedding-model lookup) means writing one new class and registering it;
  `CandidateGenerator` itself never needs to change.

`RecommendationService` is the single orchestrator every caller (the API,
the CLI, `run_demo.py`, or a batch job) depends on. It wires candidate
generation → scoring → ranking behind one method call and owns the
service-level in-process cache described in [Performance](#performance).

## Recommendation pipeline

For a single `get_recommendations(user_id, k)` call:

1. **Candidate generation** (`CandidateGenerator.hybrid_candidates`) —
   runs three strategies and interleaves their results round-robin so no
   single strategy dominates the pool, always excluding items the user
   has already interacted with:
   - *Collaborative filtering*: find users whose interaction history
     overlaps this user's (Jaccard similarity over item sets), and
     surface items those similar users engaged with.
   - *Content-based filtering*: build a tag profile from the union of
     tags across the user's own history, and surface items whose tags
     overlap it most.
   - *Popularity*: the globally most-interacted-with items, used both as
     a blended signal and as the universal cold-start fallback.
2. **Scoring** (`RecommendationScorer.rank_candidates`) — each candidate
   is scored by every registered scoring function (by default:
   `relevance`, `recency`, `popularity`), combined via configurable
   weights, and given a plain-language explanation built only from the
   signals that meaningfully contributed (see
   [Personalization](#personalization)).
3. **Ranking** — candidates are deduplicated, sorted by score descending,
   and truncated to the requested `k`.
4. **Evaluation** (offline, separate from the request path) —
   `RecommendationEvaluator` computes precision@k, recall@k, and NDCG@k
   against known ground truth for measuring pipeline quality over time.

## Personalization

The `relevance` scorer is what makes recommendations depend on `user_id`
at all: it computes the Jaccard overlap between a user's tag profile (the
union of tags across every item in their history) and each candidate
item's tags. Two users with an overlapping candidate pool but different
histories get different rankings because their profiles differ.

Explanations are generated from the *actual* weighted contribution of
each scoring signal — never a canned template:

```
job_2: score=0.693 (69%)  Recommended based on relevance match 67%, recency match 90%, popularity match 60%.
```

A signal only appears in the explanation if its contribution
(`raw_score × weight`) clears a small threshold; a recency score of
`0.02` that barely moved the final number is correctly omitted rather
than padding the explanation with noise. When nothing clears the
threshold (e.g. a cold-start user with zero relevance signal), the
explanation says so honestly instead of inventing a personalized reason:
`"No strong personalization signal for this item; shown as a general
suggestion."`

## Cold-start handling

A user with no interaction history has no possible collaborative or
content-based signal — `CollaborativeStrategy` and `ContentBasedStrategy`
both detect this and fall back to `PopularityStrategy` automatically, so
`get_recommendations` still returns a sensible, non-empty list instead of
an error or nothing at all. `run_demo.py` demonstrates this explicitly
with a synthetic brand-new user.

## API

```
GET  /health          Back-compat liveness check (unchanged; kept for existing callers)
GET  /health/live      Liveness: process is up and can respond at all
GET  /health/ready      Readiness: persistence layer is actually reachable
GET  /recommendations/{user_id}?k=5
POST /interactions   {"user_id": "...", "item_id": "..."}
```

- **`/health` vs `/health/live` vs `/health/ready`** — `/health` is kept
  byte-for-byte compatible for existing integrations. The new pair
  follows standard Kubernetes-style probe semantics: liveness never
  touches storage (a slow disk should trigger a readiness failure, not a
  container restart), readiness does a cheap read against the repository
  and returns `503` if it fails.
- **Validation** — `user_id` (path) and `user_id`/`item_id` (request
  body) are validated declaratively via a Pydantic/FastAPI pattern
  constraint (`^[A-Za-z0-9_.-]{1,64}$`); `k` is validated via a FastAPI
  `Query` constraint (`gt=0`, capped at the configured `RECO_MAX_K`).
  Every validation failure returns the same envelope shape with status
  `422`.
- **Errors** — every error response, regardless of cause, has the shape:

  ```json
  {"error": {"type": "validation_error", "message": "query.k: Input should be greater than 0"}}
  ```

  Unhandled exceptions are caught by a final handler that logs the full
  traceback server-side and returns a generic `500` — internal details
  are never included in a response body.
- **Request logging** — a middleware logs `method`, `path`, `status_code`,
  and duration in milliseconds for every request. Bodies are never
  logged.

### Example

```bash
curl http://localhost:8000/recommendations/alice?k=3

curl -X POST http://localhost:8000/interactions \
     -H "Content-Type: application/json" \
     -d '{"user_id": "alice", "item_id": "job_9"}'
```

## Configuration

Every tunable is an environment variable prefixed `RECO_`, loaded once per
process via `get_settings()` (`functools.lru_cache`). All pre-existing
variable names and defaults are unchanged; one variable was added
(`RECO_MAX_K`).

| Variable | Default | Notes |
|---|---|---|
| `RECO_MAX_CANDIDATES` | `20` | Must be > 0 |
| `RECO_WEIGHT_RELEVANCE` | `0.5` | Must be ≥ 0 |
| `RECO_WEIGHT_RECENCY` | `0.2` | Must be ≥ 0 |
| `RECO_WEIGHT_POPULARITY` | `0.3` | Must be ≥ 0; at least one weight must be > 0 |
| `RECO_DEFAULT_TOP_K` | `10` | Must be > 0 and ≤ `RECO_MAX_K` |
| `RECO_MAX_K` | `100` | **New.** Upper bound the API will accept for `k` |
| `RECO_DATA_DIR` | `data` | Directory for the JSON persistence files |
| `RECO_HISTORY_FILE` | `user_history.json` | Filename within `RECO_DATA_DIR` |
| `RECO_FEATURES_FILE` | `item_features.json` | Filename within `RECO_DATA_DIR` |
| `RECO_LOG_LEVEL` | `INFO` | Case-insensitive; normalized to uppercase; must be a valid level |
| `RECO_API_HOST` | `0.0.0.0` | |
| `RECO_API_PORT` | `8000` | Must be between 1 and 65535 |

All values are validated eagerly in `Settings.__post_init__`, so a bad
value fails loudly at startup with a specific message
(`ConfigurationError`) instead of surfacing later as a confusing runtime
bug.

## Installation

Requires **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running

```bash
# API (http://localhost:8000)
uvicorn reco_engine.api:app --reload

# CLI
python -m reco_engine.cli recommend alice --k 5
python -m reco_engine.cli record alice job_9

# Demo: personalized recs, scores, explanations, evaluation, cold-start
python run_demo.py
```

## Testing

```bash
python -m compileall src tests   # fast syntax check
pytest -q                          # 104 tests
pytest --cov=reco_engine --cov-report=term-missing
ruff check src tests
```

Test coverage by area:

- **Similarity** — cosine, Jaccard, Pearson, adjusted-cosine, including
  edge cases (empty/zero/mismatched-length input).
- **Candidate generation** — every strategy plus their popularity
  fallback for cold-start users.
- **Scoring** — weighting, deduplication, failing-scorer isolation, and a
  dedicated explanation-quality suite (only meaningful signals are named;
  honest fallback text when there's no signal at all).
- **Evaluation** — precision/recall/NDCG@k edge cases; multi-user
  averaging that skips (not zero-scores) users without ground truth.
- **Configuration** — every validation rule (weights, `top_k`/`max_k`
  bounds, port range, log level normalization) and confirmation that
  every pre-existing `RECO_*` variable is still honored.
- **Persistence** — round-trip durability, atomic writes producing valid
  JSON, and cold (missing-file) startup.
- **Service-level caching** — the user-profile cache reflects a new
  interaction immediately (never stale), is scoped per-user (an
  interaction for one user doesn't touch another's cached profile), and
  the generator's popularity cache is correctly invalidated too.
- **API** — `/health`, `/health/live`, `/health/ready` (including a
  simulated storage failure), invalid `user_id`/`item_id`/`k`, the error
  envelope shape, and that an interaction actually changes a subsequent
  recommendation call.
- **CLI** — `recommend` and `record` subcommands, including the
  no-candidates and missing-subcommand paths.

## Docker

Multi-stage build (slim Python base, non-root user, port 8000):

```bash
docker build -t reco-engine .
docker run -p 8000:8000 reco-engine
```

The container's `HEALTHCHECK` calls `/health/live` (liveness) rather than
the deeper `/health`, matching the liveness/readiness split in the API —
a container shouldn't be restarted just because its storage volume is
briefly unavailable.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`, across Python
**3.10, 3.11, and 3.12**:

1. Install dependencies (with pip caching keyed on `requirements.txt` /
   `pyproject.toml`)
2. `ruff check` (lint)
3. `python -m compileall` (fast syntax gate)
4. `pytest --cov=reco_engine --cov-report=term-missing --cov-report=xml`
   (coverage uploaded as a build artifact per Python version)
5. Import-check `reco_engine.api:app` to catch wiring/startup regressions
   that unit tests alone might miss

## Security

- No hardcoded secrets anywhere — all environment-specific configuration
  goes through `RECO_*` environment variables.
- All caller-supplied identifiers (`user_id`, `item_id`) are validated
  against a conservative allow-list pattern before reaching the
  recommendation pipeline.
- `k` is bounded (`RECO_MAX_K`) so a request can't force the service to
  rank and return an unbounded number of items.
- Unhandled exceptions are always caught by a final handler: full details
  (including the traceback) go to the server log only; the HTTP response
  body always contains a generic, safe message.
- Atomic file writes (write-to-temp-then-rename) in `JSONFileRepository`
  mean a crash mid-write can't corrupt the on-disk history.

## Performance

- **`SimilarityCalculator.jaccard_similarity`** is memoized
  (`functools.lru_cache`, keyed on sorted tuples so set ordering never
  causes a spurious cache miss) — the same pair of tag sets is compared
  repeatedly across candidates and users, so this cache meaningfully cuts
  repeat work.
- **Per-user tag profile caching** (`RecommendationService`) — the
  original implementation rebuilt a user's tag profile (a union over
  every historical item's tags) from scratch for *every single candidate
  scored*, i.e. `O(candidates × history size)` per request. It's now
  built once per user and cached, invalidated automatically the instant
  `record_interaction` changes that user's history — there is no code
  path that can serve a stale profile.
- **Popularity ranking caching** (`CandidateGenerator`) — the global
  popularity ranking doesn't depend on `user_id` at all, so recomputing
  it via a full history scan on every single `hybrid_candidates` call
  (for every user) was pure waste. It's now cached after first
  computation and invalidated by the same `record_interaction` call that
  changes the underlying data.
- Both caches are plain dicts with explicit invalidation hooks — no TTLs,
  no external cache dependency, nothing that could silently serve stale
  data for longer than "until the next interaction."

## Compatibility notes

- **FastAPI/Starlette/httpx/AnyIO** — `httpx` was replaced with `httpx2`
  (its drop-in, currently-maintained successor) as the dev/test
  dependency for `starlette.testclient.TestClient`, removing the
  `StarletteDeprecationWarning` that `httpx` triggered. The one remaining
  warning (`anyio.abc.BlockingPortal` alias deprecation) originates
  *inside* Starlette's own `testclient` module, not from any call site in
  this codebase — there is no application-level fix for it, so it's
  explicitly and narrowly silenced in `pyproject.toml`
  (`[tool.pytest.ini_options].filterwarnings`), scoped to that exact
  message and module so no other deprecation warning is masked.
  `HTTP_422_UNPROCESSABLE_ENTITY` was also replaced with FastAPI's
  current alias, `HTTP_422_UNPROCESSABLE_CONTENT`.
- **`k` validation status code changed from 400 → 422.** Previously,
  invalid `k` was checked with a hand-written `if` and raised a manual
  `400`. It's now expressed as a declarative FastAPI `Query` constraint
  (the literal ask: "modernize FastAPI validation with Pydantic"), which
  reports through the same mechanism — and therefore the same status
  code, `422` — as every other request-shape validation failure (missing
  fields, malformed ids). This is a deliberate consistency improvement,
  not an oversight; the previous behavior is preserved in spirit
  (invalid `k` is still rejected before reaching the pipeline) but the
  status code is now uniform across all validation failures.
- **`get_recommendations(user_id, k=0)`** now genuinely means "return
  zero recommendations" instead of silently falling back to
  `default_top_k` (the previous `k or default` check treated `0` and
  `None` identically). The API rejects `k=0` before it would ever reach
  this code path, so this only changes direct library usage.
- Everything else — the similarity formulas, candidate-generation logic,
  scoring/ranking shape, evaluation formulas, and the `Repository`/
  `Settings` public shapes — is unchanged.

## Future improvements

- Swap `JSONFileRepository` for a real database-backed `Repository`
  implementation (the interface is already storage-agnostic) if
  concurrent writers or dataset size ever exceed what flat files handle
  comfortably.
- Move `recency`/`popularity` signal computation to an actual scheduled
  batch job / feature store, as already called out in `service.py` —
  they're currently supplied in-process for simplicity.
- Add authentication/authorization to the API if it's ever exposed
  outside a trusted network — it currently has none, by design, to stay
  dependency-light for this stage of the project.
- Add a matrix-factorization or embedding-based candidate strategy
  alongside the existing three; the Strategy pattern in
  `candidate_gen.py` was built specifically so this requires no changes
  to `CandidateGenerator` itself.
