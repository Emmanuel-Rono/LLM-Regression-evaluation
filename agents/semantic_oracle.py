from typing import Dict
from evaluations.evaluation_metric import semantic_similarity

class SemanticOracle:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def evaluate(self, baseline: str, candidate: str) -> Dict:
        score = semantic_similarity(baseline, candidate)

        return {
            "metric": "semantic_similarity",
            "score": score,
            "threshold": self.threshold,
            "passed": score >= self.threshold,
        }
