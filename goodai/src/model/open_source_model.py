import os
import requests

from dotenv import load_dotenv

from goodai.src.model.base_model import BaseModel

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
                "inputs": message,
                "parameters": {"return_full_text": False},
            },
        ).json()
        response = raw_response[0].get("generated_text")
        return response
