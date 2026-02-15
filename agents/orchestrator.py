from typing import Dict, List


class Orchestrator:
    def __init__(self, semantic_oracle, reasoning_oracle, safety_oracle):
        self.semantic_oracle = semantic_oracle
        self.reasoning_oracle = reasoning_oracle
        self.safety_oracle = safety_oracle

    def evaluate(self, baseline: str, candidate: str) -> Dict:
        semantic = self.semantic_oracle.evaluate(baseline, candidate)
        reasoning = self.reasoning_oracle.evaluate(candidate)
        safety = self.safety_oracle.evaluate(candidate)

        failures: List[str] = []

        if not semantic["passed"]:
            failures.append("semantic")

        if not reasoning["passed"]:
            failures.append("reasoning")

        if not safety["passed"]:
            failures.append("safety")

        return {
            "semantic": semantic,
            "reasoning": reasoning,
            "safety": safety,
            "regression_detected": len(failures) > 0,
            "failures": failures,
        }
