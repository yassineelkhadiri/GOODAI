import os
from dotenv import load_dotenv
from openai import OpenAI

from goodai.src.model.base_model import BaseModel

load_dotenv(".env")


class OpenAIModel(BaseModel):

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def query_llm(self, message: str) -> str:
        return ""
