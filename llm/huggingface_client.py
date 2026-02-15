"""
huggingface_client.py

Production-ready Hugging Face inference wrapper.
"""

from typing import Optional, Dict, Any
from transformers import pipeline


class HuggingFaceClient:
    """
    Wrapper around Hugging Face text-generation pipeline.
    Abstracted for easy replacement or extension.
    """

    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: Optional[int] = None,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self._pipeline = pipeline(
            task="text-generation",
            model=model_id,
            device=device,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate model output for given prompt.
        """
        output = self._pipeline(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )

        return output[0]["generated_text"]

    def generate_with_metadata(self, prompt: str) -> Dict[str, Any]:
        """
        Extended generation returning structured metadata.
        Useful for observability.
        """
        output = self._pipeline(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )

        return {
            "model_id": self.model_id,
            "prompt": prompt,
            "output": output[0]["generated_text"],
            "generation_config": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
