import os
import re
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

    def format_prompt(
        self, message: str, additional_informations: Dict[str, List[str]]
    ) -> str:
        """Prepare the prompt of the LLM.

        Args:
            message: user question/interaction.
            additional_informations: informations to augment the prompt.

        Returns: Formatted prompt.
        """
        prompt_content = [
            self.prompts["system"],
            self.prompts["user_input"].format(message),
        ]
        recent_memories = additional_informations.get("recent_memories", [])
        relevant_memories = additional_informations.get("relevant_memories", [])
        if recent_memories and relevant_memories:
            prompt_content.insert(
                1, self.prompts["recent_memories"].format(recent_memories)
            )
            prompt_content.insert(
                2, self.prompts["relevant_memories"].format(relevant_memories)
            )
        if recent_memories and not relevant_memories:
            prompt_content.insert(
                1, self.prompts["recent_memories"].format(recent_memories)
            )
        if not recent_memories and relevant_memories:
            prompt_content.insert(
                1, self.prompts["relevant_memories"].format(relevant_memories)
            )

        return "\n".join(prompt_content)

    def query_llm(
        self, message: str, recent_memories: List[str], relevant_memories: List[str]
    ) -> str:
        """Query an open source model and retrieve its response.

        Args:
            message: User message/question.
            recent_memories: collection of most recent interactions.
            relevant_memories: collection of relevent interactions to the current message.

        Returns: LLM response.
        """
        memories = {
            "recent_memories": recent_memories,
            "relevant_memories": relevant_memories,
        }
        raw_response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "inputs": self.format_prompt(
                    message=message, additional_informations=memories
                ),
                "parameters": {"return_full_text": False},
            },
        ).json()
        return OpenSourceModel.format_response(raw_response)

    @staticmethod
    def extract_response(generated_text: str) -> str:
        response_match = re.search(r"response:\s*(.*)", generated_text, re.IGNORECASE)
        if response_match:
            return response_match.group(1).strip()
        return generated_text

    @staticmethod
    def remove_prefixes(response: str) -> str:
        prefixes = ["AI:", "Assistant:", "Answer:", "Response:"]
        for prefix in prefixes:
            response = response.replace(prefix, "").strip()
        return response

    @staticmethod
    def extract_based_response(response: str) -> str:
        if response.startswith("Based") and "should be:" in response:
            return response.split("should be:")[1].strip()
        if response.startswith("Based") and "would be:" in response:
            return response.split("would be:")[1].strip()
        return response

    @staticmethod
    def format_response(raw_response: List[Dict]) -> str:
        """Retrieve the response of the LLM from its raw response."""
        generated_text = str(raw_response[0].get("generated_text", ""))
        response = OpenSourceModel.extract_response(generated_text)
        response = OpenSourceModel.remove_prefixes(response)
        response = OpenSourceModel.extract_based_response(response)
        return response
