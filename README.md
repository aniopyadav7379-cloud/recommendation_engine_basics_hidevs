# Day 29: Recommendation Engine Components

A foundational, in-memory toolkit for building a recommendation engine.
Four independent modules cover similarity, candidate generation,
scoring/ranking, and evaluation — the algorithmic core that tomorrow's
full system will be built on top of.

No database, no external services, no runtime dependencies. Everything
runs on plain Python dictionaries.

## Components

| Module | Class | Purpose |
|---|---|---|
| `similarity.py` | `SimilarityCalculator` | Cosine similarity, Jaccard similarity, Pearson correlation |
| `candidate_gen.py` | `CandidateGenerator` | Collaborative, content-based, popularity, and hybrid candidate pools |
| `scorer.py` | `RecommendationScorer` | Weighted, pluggable scoring and ranking of candidates |
| `evaluator.py` | `RecommendationEvaluator` | precision@k, recall@k, NDCG@k, and aggregate evaluation |

### `SimilarityCalculator`

```python
from similarity import SimilarityCalculator

sc = SimilarityCalculator()
sc.cosine_similarity([1, 2, 3], [1, 2, 4])      # user/item vectors -> float in [-1, 1]
sc.jaccard_similarity({"python", "sql"}, {"python"})  # tag/skill sets -> float in [0, 1]
sc.pearson_correlation([5, 3, 4], [4, 2, 5])    # rating patterns -> float in [-1, 1]
```

Edge cases return a sensible default instead of raising: empty or
zero-magnitude vectors, empty sets, constant rating series all
resolve to `0.0` (or `1.0` for two empty sets under Jaccard, since
they're vacuously identical). Mismatched vector/series lengths raise
`ValueError`.

### `CandidateGenerator`

```python
from candidate_gen import CandidateGenerator

gen = CandidateGenerator(
    user_item_history={"alice": ["book1", "book2"], "bob": ["book1", "book2", "book3"]},
    item_tags={"book1": {"scifi"}, "book2": {"scifi", "adventure"}, "book3": {"scifi", "mystery"}},
    item_popularity={"book1": 100, "book2": 80, "book3": 40},
)

gen.collaborative_candidates("alice")   # items liked by similar users
gen.content_based_candidates("alice")   # items with overlapping tags
gen.popularity_candidates()             # most popular overall
gen.hybrid_candidates("alice")          # interleaved combination, de-duplicated
```

Every strategy falls back to `popularity_candidates()` for cold-start
users (no history, or no similar peers found). Results are capped at
`CandidateGenerator.MAX_LIMIT` (50) regardless of the requested limit.

### `RecommendationScorer`

```python
from scorer import RecommendationScorer

scorer = RecommendationScorer()
scorer.add_scorer("popularity", lambda user_id, item_id, ctx: ctx["pop"].get(item_id, 0) / 100, weight=1.0)
scorer.add_scorer("relevance", my_relevance_fn, weight=2.0)

scorer.calculate_score("alice", "book1", context={"pop": {...}})
scorer.rank_candidates("alice", candidates, context={"pop": {...}}, limit=10)
```

A scoring function is any callable `(user_id, item_id, context) -> float`.
Scores are combined as a weighted average, clamped to `[0, 1]`. If a
scorer raises an exception it's skipped rather than failing the whole
ranking. Each result includes a `breakdown` per factor and a one-line
`explanation` naming the strongest contributor.

### `RecommendationEvaluator`

```python
from evaluator import RecommendationEvaluator

RecommendationEvaluator.precision_at_k(recommendations, relevant_items, k=10)
RecommendationEvaluator.recall_at_k(recommendations, relevant_items, k=10)
RecommendationEvaluator.ndcg_at_k(recommendations, relevant_items, k=10)
RecommendationEvaluator.evaluate_all(recommendations_dict, ground_truth_dict, k=10)
```

`evaluate_all` averages all three metrics across users, skipping any
user missing from `ground_truth_dict` rather than counting them as a
zero score.

## Installation

```bash
pip install -r requirements.txt
```

The library modules themselves (`similarity.py`, `candidate_gen.py`,
`scorer.py`, `evaluator.py`) have zero runtime dependencies — only the
standard library. `requirements.txt` pins `pytest` for the test suite.

## Running the tests

```bash
python3 -m pytest
```

54 test cases across five files in `tests/`: one per module, plus
`test_integration.py`, which chains all four components together the
way the full system will tomorrow — generate candidates, score and
rank them, then evaluate the ranking against a ground-truth set.
`pytest.ini` puts the project root on `sys.path` so tests can import
the top-level modules directly, no packaging step required.

## Project structure

```
day29_project/
├── similarity.py       # Component 1: similarity metrics
├── candidate_gen.py    # Component 2: candidate generation strategies
├── scorer.py            # Component 3: weighted scoring and ranking
├── evaluator.py          # Component 4: evaluation metrics
├── requirements.txt
├── pytest.ini
└── tests/
    ├── test_similarity.py
    ├── test_candidate_gen.py
    ├── test_scorer.py
    ├── test_evaluator.py
    └── test_integration.py
```

## Status / what's intentionally out of scope

This is the algorithmic core only — built to sit under a real system,
not to be one. Not included yet, by design:

- No database or persistence layer (dictionaries stand in for storage)
- No API layer or web framework
- No containerization (Dockerfile) or CI configuration
- No packaging (`pyproject.toml`) — not installable as a library yet

Those come once there's a concrete system to wire this into.
