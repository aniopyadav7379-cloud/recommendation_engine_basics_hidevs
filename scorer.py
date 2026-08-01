"""
scorer.py — Component 3: Scorer & Ranker

Combines one or more weighted scoring functions into a single score
per (user, item) pair, then ranks candidates by that score.

A "scorer" is any callable with the signature:

    scorer_fn(user_id, item_id, context) -> float in [0, 1]

`context` is a plain dict the caller supplies with whatever the
scoring functions need (e.g. {"item_popularity": {...}, "item_tags": {...}}).
Each scorer function is expected to clamp/normalize its own output;
RecommendationScorer additionally clamps defensively so one bad
scorer can't blow up the combined score.
"""


from typing import Any, Callable, Dict, List, Optional

UserId = str
ItemId = str
Context = Dict[str, Any]
ScorerFn = Callable[[UserId, ItemId, Context], float]


class RecommendationScorer:
    """Registers weighted scoring functions and combines them into a
    single relevance score per item, with a short human-readable
    explanation of what drove the score.
    """

    def __init__(self) -> None:
        # name -> {"fn": callable, "weight": float}
        self._scorers: Dict[str, Dict[str, Any]] = {}

    def add_scorer(self, name: str, function: ScorerFn, weight: float = 1.0) -> None:
        """Register a scoring function under `name` with a relative weight.

        Weights don't need to sum to 1 — calculate_score normalizes
        by the total weight of scorers that actually ran.
        """
        if not callable(function):
            raise TypeError("add_scorer: function must be callable")
        if weight < 0:
            raise ValueError("add_scorer: weight must be non-negative")
        self._scorers[name] = {"fn": function, "weight": weight}

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def calculate_score(
        self, user_id: UserId, item_id: ItemId, context: Optional[Context] = None
    ) -> Dict[str, Any]:
        """Score a single item for a user as a weighted average of all
        registered scorers. Returns a dict:

            {
                "score": float in [0, 1],
                "breakdown": {name: raw_score, ...},
                "explanation": "human readable summary"
            }

        A scorer that raises an exception is skipped (not fatal) so
        one broken factor doesn't take down the whole ranking; it's
        omitted from the breakdown and excluded from the weight total.
        """
        context = context or {}

        if not self._scorers:
            return {"score": 0.0, "breakdown": {}, "explanation": "no scorers registered"}

        breakdown = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for name, spec in self._scorers.items():
            try:
                raw = spec["fn"](user_id, item_id, context)
                raw = self._clamp(float(raw))
            except Exception:
                continue
            breakdown[name] = raw
            weighted_sum += raw * spec["weight"]
            total_weight += spec["weight"]

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        explanation = self._build_explanation(breakdown)

        return {
            "score": round(final_score, 4),
            "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "explanation": explanation,
        }

    @staticmethod
    def _build_explanation(breakdown: Dict[str, float]) -> str:
        if not breakdown:
            return "No scoring factors available"
        # Explain using the single strongest contributing factor.
        top_factor = max(breakdown.items(), key=lambda pair: pair[1])
        name, value = top_factor
        return f"Recommended primarily due to '{name}' score of {round(value, 2)}"

    def rank_candidates(
        self,
        user_id: UserId,
        candidates: List[ItemId],
        context: Optional[Context] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Score every candidate item and return the top `limit`,
        highest-scored first.

        Each entry: {"item_id": ..., "score": ..., "breakdown": ..., "explanation": ...}
        """
        if not candidates:
            return []

        results = []
        for item_id in candidates:
            scored = self.calculate_score(user_id, item_id, context)
            results.append({"item_id": item_id, **scored})

        results.sort(key=lambda entry: entry["score"], reverse=True)
        limit = max(1, limit)
        return results[:limit]


# ---------------------------------------------------------------------------
# Simple standalone test cases
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scorer = RecommendationScorer()

    # Relevance: how well item tags match a fixed "user interest" set in context
    def relevance_scorer(user_id, item_id, context):
        user_tags = context.get("user_tags", set())
        item_tags = context.get("item_tags", {}).get(item_id, set())
        if not user_tags or not item_tags:
            return 0.0
        overlap = user_tags & item_tags
        return len(overlap) / len(user_tags | item_tags)

    # Popularity: normalized by the max popularity present in context
    def popularity_scorer(user_id, item_id, context):
        pop = context.get("item_popularity", {})
        if not pop:
            return 0.0
        max_pop = max(pop.values()) or 1
        return pop.get(item_id, 0) / max_pop

    # Recency: 1.0 for items flagged "new", else 0.3
    def recency_scorer(user_id, item_id, context):
        new_items = context.get("new_items", set())
        return 1.0 if item_id in new_items else 0.3

    scorer.add_scorer("relevance", relevance_scorer, weight=2.0)
    scorer.add_scorer("popularity", popularity_scorer, weight=1.0)
    scorer.add_scorer("recency", recency_scorer, weight=0.5)

    context = {
        "user_tags": {"python", "backend"},
        "item_tags": {
            "i1": {"python", "backend"},
            "i2": {"design"},
            "i3": {"python", "ml"},
        },
        "item_popularity": {"i1": 50, "i2": 10, "i3": 30},
        "new_items": {"i3"},
    }

    single = scorer.calculate_score("u1", "i1", context)
    assert 0.0 <= single["score"] <= 1.0
    assert "relevance" in single["breakdown"]

    ranked = scorer.rank_candidates("u1", ["i1", "i2", "i3"], context, limit=2)
    assert len(ranked) == 2
    assert ranked[0]["item_id"] == "i1"  # best tag + popularity match

    empty_scorer = RecommendationScorer()
    assert empty_scorer.calculate_score("u1", "i1")["score"] == 0.0
    assert empty_scorer.rank_candidates("u1", []) == []

    print("scorer.py: all inline tests passed")
