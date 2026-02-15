import yaml
from agents.orchestrator import Orchestrator
from agents.semantic_oracle import SemanticOracle
from agents.reasoning_oracle import ReasoningOracle
from agents.safety_oracle import SafetyOracle


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def test_regression_logic():
    settings = load_yaml("config/settings.yaml")

    semantic = SemanticOracle(settings["semantic"]["threshold"])
    reasoning = ReasoningOracle(settings["reasoning"]["min_markers"])
    safety = SafetyOracle()

    orchestrator = Orchestrator(semantic, reasoning, safety)

    baseline = "HTTP 401 means unauthorized."
    candidate = "HTTP 401 indicates the user is not authenticated."

    result = orchestrator.evaluate(baseline, candidate)

    assert isinstance(result["regression_detected"], bool)
