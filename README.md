# Mini Recommendation Engine

A simple, working recommendation engine with four components — the
same building blocks behind how platforms like Netflix and Amazon
suggest items. Runs entirely on plain Python dictionaries and sets,
no database or external libraries required.

## How it works, end to end

```
user preferences  --->  Candidate Generator  --->  Scorer  --->  top picks
                          (narrows catalog          (ranks them)
                           down to relevant                          |
                           items)                                    v
                                                              Evaluator
                                                      (checks precision against
                                                       what the user actually liked)
```

## The four components

### 1. `similarity.py` — Similarity Calculator

Measures how similar two things are:

- **`cosine_similarity(vec1, vec2)`** — compares two numeric vectors
  (e.g. rating patterns). Returns a value roughly in `[0, 1]`.
- **`jaccard_similarity(set1, set2)`** — compares two sets (e.g. movie
  genres, tags). Returns `intersection / union`, a value in `[0, 1]`.

```python
from similarity import SimilarityCalculator

sim = SimilarityCalculator()
sim.cosine_similarity([1, 2, 3], [1, 2, 4])
sim.jaccard_similarity({"sci-fi", "action"}, {"sci-fi", "drama"})
```

Handles empty vectors/sets and zero vectors without crashing.

### 2. `candidate_gen.py` — Candidate Generator

Takes a catalog of items (each described by a set of tags) and a
user's preferences (a set of tags they like), and returns the items
worth considering — ranked by how well their tags overlap with the
user's preferences (using Jaccard similarity under the hood).

```python
from candidate_gen import CandidateGenerator

catalog = {"movie1": {"action", "sci-fi"}, "movie2": {"romance"}}
gen = CandidateGenerator(catalog)
gen.find_candidates({"action", "sci-fi"}, limit=10)
# -> ["movie1"]
```

Returns `[]` for an empty catalog, empty preferences, or no matches —
never crashes on missing data.

### 3. `scorer.py` — Scorer

Takes the candidates and ranks them, combining two signals:

- **relevance** (70% weight) — tag overlap with user preferences
- **rating** (30% weight) — the item's average rating, if known

Returns the top N picks as `(item_id, score)` pairs, best first.

```python
from scorer import Scorer

scorer = Scorer(catalog, item_ratings={"movie1": 4.5})
scorer.rank_candidates(["movie1", "movie2"], {"action", "sci-fi"}, top_n=3)
# -> [("movie1", 0.79), ...]
```

An empty candidate list returns `[]`; an item missing from the catalog
or ratings just scores low instead of raising an error.

### 4. `evaluator.py` — Evaluator

Checks how good a set of recommendations actually was, using
precision: of the items recommended, what fraction did the user
actually want?

```python
from evaluator import Evaluator

ev = Evaluator()
ev.precision(recommended_items=["movie1", "movie2"], relevant_items=["movie1"])
# -> 0.5
ev.precision_at_k(recommended_items, relevant_items, k=3)
```

Returns `0.0` (rather than crashing) when there are no recommendations
or no known relevant items to compare against.

## Running it

```bash
python3 demo.py   # see all four components work together on sample movie data
python3 test.py   # run all tests
```

Each module can also be run on its own (`python3 similarity.py`, etc.)
to see its individual test cases.

## Project structure

```
recommendation_engine/
├── similarity.py     # Component 1: cosine + Jaccard similarity
├── candidate_gen.py  # Component 2: finds items matching preferences
├── scorer.py         # Component 3: ranks candidates, returns top picks
├── evaluator.py       # Component 4: precision-based evaluation
├── demo.py            # wires all four together on sample data
└── test.py            # tests for every component + the full pipeline
```

## What this doesn't do (on purpose)

This is a learning-focused mini system, not a production one: no
database, no API, no real user data — just enough to see how
similarity, candidate generation, scoring, and evaluation fit
together. From here, the natural next steps are collaborative
filtering with real user-item interaction data, more evaluation
metrics (recall, NDCG), and persisting the catalog somewhere other
than a Python dictionary.
