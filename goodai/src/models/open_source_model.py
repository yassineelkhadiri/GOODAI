import os
import requests

from dotenv import load_dotenv
from typing import List, Dict

from goodai.src.models.base_model import BaseModel

load_dotenv(".env")


class OpenSourceModel(BaseModel):

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.api_key = os.getenv("HUGGING_FACE_API_KEY")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def query_llm(self, message: str) -> str:
        """Query an open source model and retrieve its response.

        Args:
            message: User message/question.

        Returns: LLM response.
        """
        raw_response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "inputs": self.prompt_template.format(message),
                "parameters": {"return_full_text": False},
            },
        ).json()
        return self.format_response(raw_response)

    def format_response(self, raw_response: List[Dict]) -> str:
        """Retrieve the response of the LLM from its raw response."""
        generated_text = str(raw_response[0].get("generated_text", ""))
        response = generated_text.strip().split("\n")[0]
        response = response.replace("AI: ", "")
        return response
