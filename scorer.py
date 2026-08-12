"""Component 3: Scorer & Ranker.

Combines weighted scoring functions into a final ranking with a short
explanation for each recommended item.
"""

from dataclasses import dataclass


@dataclass
class ScoredItem:
    item_id: str
    score: float
    explanation: str


class RecommendationScorer:
    """Registers scoring functions and produces weighted, ranked results."""

    def __init__(self):
        self._scorers = []  # list of (name, function, weight)

    def add_scorer(self, name, function, weight):
        """Register a scoring function: fn(user_id, item_id, context) -> [0,1]."""
        if weight < 0:
            raise ValueError("weight must be non-negative")
        self._scorers.append((name, function, weight))

    def calculate_score(self, user_id, item_id, context=None):
        """Compute the weighted combined score for one item.
        Returns (score, explanation). A failing scorer is skipped, not fatal."""
        context = context or {}
        contributions = []

        for name, fn, weight in self._scorers:
            try:
                raw = float(fn(user_id, item_id, context))
            except Exception:
                continue
            raw = max(0.0, min(1.0, raw))
            contributions.append((name, raw, weight))

        total_weight = sum(w for _, _, w in contributions)
        if total_weight == 0:
            return 0.0, "No applicable scoring signals"

        score = sum(raw * w for _, raw, w in contributions) / total_weight
        top = sorted(contributions, key=lambda c: c[1] * c[2], reverse=True)[:2]
        explanation = "Recommended due to: " + ", ".join(
            f"{name} ({raw:.2f})" for name, raw, _ in top
        )
        return max(0.0, min(1.0, score)), explanation

    def rank_candidates(self, user_id, candidates, context=None, limit=10):
        """Score every candidate (deduplicated) and return the top `limit`."""
        seen, results = set(), []
        for item_id in candidates:
            if item_id in seen:
                continue
            seen.add(item_id)
            score, explanation = self.calculate_score(user_id, item_id, context)
            results.append(ScoredItem(item_id, score, explanation))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
