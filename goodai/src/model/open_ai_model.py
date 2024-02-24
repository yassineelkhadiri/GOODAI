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
        """Query an OpenAI model and retrieve its response.

        Args:
            message: User message/question.

        Returns: LLM response.
        """
        messages = [{"user": message}]
        raw_response = self.client.chat.completions.create(
            model=self.model_name, messages=messages  # type:ignore
        )
        formatted_response = str(raw_response.choices[0].message.content)
        return formatted_response
