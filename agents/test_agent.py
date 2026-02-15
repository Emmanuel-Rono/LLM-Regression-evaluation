from llm.hf_client import HFClient

class TestAgent:
    def __init__(self, model_id: str):
        self.client = HFClient(model_id)

    def run(self, prompt: str) -> str:
        return self.client.generate(prompt)
