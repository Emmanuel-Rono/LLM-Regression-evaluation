from typing import Dict


class ReasoningOracle:
    def __init__(self, min_markers: int):
        self.min_markers = min_markers
        self._markers = ["because", "therefore", "thus", "for example"]

    def evaluate(self, response: str) -> Dict:
        score = sum(1 for m in self._markers if m in response.lower())

        return {
            "metric": "reasoning_markers",
            "score": score,
            "threshold": self.min_markers,
            "passed": score >= self.min_markers,
        }
