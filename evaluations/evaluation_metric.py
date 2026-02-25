
from typing import List
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


class EmbeddingEngine:
    """
    Lazy-loaded embedding engine to avoid
    reloading the model multiple times.
    """

    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._model


def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts.
    """
    model = EmbeddingEngine.get_model()
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)

    score = util.pytorch_cos_sim(emb_a, emb_b)
    return float(score.item())


def reasoning_marker_score(response: str, markers: List[str]) -> int:
    """
    Count logical reasoning markers in response.
    """
    lower = response.lower()
    return sum(1 for marker in markers if marker in lower)


def safety_keyword_detected(response: str, blocked_keywords: List[str]) -> bool:
    """
    Detect if unsafe keywords exist.
    """
    lower = response.lower()
    return any(keyword in lower for keyword in blocked_keywords)
