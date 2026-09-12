"""reco_engine: a small, dependency-light recommendation engine toolkit.

Public surface:
    SimilarityCalculator, CandidateGenerator, RecommendationScorer,
    RecommendationEvaluator, RecommendationService, get_settings
"""

from __future__ import annotations

from .candidate_gen import CandidateGenerator
from .config import get_settings
from .evaluator import RecommendationEvaluator
from .scorer import RecommendationScorer, ScoredItem
from .service import RecommendationService
from .similarity import SimilarityCalculator

__all__ = [
    "SimilarityCalculator",
    "CandidateGenerator",
    "RecommendationScorer",
    "ScoredItem",
    "RecommendationEvaluator",
    "RecommendationService",
    "get_settings",
]

__version__ = "0.3.0"
