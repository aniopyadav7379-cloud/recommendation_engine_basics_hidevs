"""reco_engine: a small, dependency-light recommendation engine toolkit.

Public surface:
    SimilarityCalculator, CandidateGenerator, RecommendationScorer,
    RecommendationEvaluator, RecommendationService, get_settings
"""

from .similarity import SimilarityCalculator
from .candidate_gen import CandidateGenerator
from .scorer import RecommendationScorer, ScoredItem
from .evaluator import RecommendationEvaluator
from .service import RecommendationService
from .config import get_settings

__all__ = [
    "SimilarityCalculator",
    "CandidateGenerator",
    "RecommendationScorer",
    "ScoredItem",
    "RecommendationEvaluator",
    "RecommendationService",
    "get_settings",
]

__version__ = "0.2.0"
