import yaml
import json
from datetime import datetime
from pathlib import Path

from agents.test_agent import TestAgent
from agents.semantic_oracle import SemanticOracle
from agents.reasoning_oracle import ReasoningOracle
from agents.safety_oracle import SafetyOracle
from agents.orchestrator import Orchestrator


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    models = load_yaml("config/models.yaml")["models"]
    prompts = load_yaml("config/prompts.yaml")["prompts"]
    settings = load_yaml("config/settings.yaml")

    baseline = next(m for m in models if m["role"] == "baseline")
    candidates = [m for m in models if m["role"] == "candidate"]

    semantic_oracle = SemanticOracle(settings["semantic"]["threshold"])
    reasoning_oracle = ReasoningOracle(settings["reasoning"]["min_markers"])
    safety_oracle = SafetyOracle()

    orchestrator = Orchestrator(
        semantic_oracle,
        reasoning_oracle,
        safety_oracle,
    )

    baseline_agent = TestAgent(baseline["id"])

    baseline_outputs = {
        p["id"]: baseline_agent.run(p["text"])
        for p in prompts
    }

    results = []

    for candidate in candidates:
        agent = TestAgent(candidate["id"])

        for p in prompts:
            candidate_output = agent.run(p["text"])
            decision = orchestrator.evaluate(
                baseline_outputs[p["id"]],
                candidate_output,
            )

            results.append({
                "prompt_id": p["id"],
                "candidate": candidate["alias"],
                **decision,
            })

    report = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "baseline": baseline["alias"],
        "results": results,
    }

    Path("reports").mkdir(exist_ok=True)
    report_path = Path("reports") / f"run_{report['run_id']}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
