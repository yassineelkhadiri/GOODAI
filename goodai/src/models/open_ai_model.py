import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict

from goodai.src.models.base_model import BaseModel

load_dotenv(".env")


class OpenAIModel(BaseModel):

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def format_prompt(
        self, message: str, additional_informations: Dict[str, List[str]]
    ) -> str:
        """Prepare the prompt of the LLM.

        Args:
            message: user question/interaction.
            additional_informations: informations to augment the prompt.

        Returns: Formatted prompt.
        """
        prompt_content = [self.prompts["user_input"].format(message)]
        recent_memories = additional_informations.get("recent_memories", [])
        relevant_memories = additional_informations.get("relevant_memories", [])
        if recent_memories and relevant_memories:
            prompt_content.insert(
                0, self.prompts["recent_memories"].format(recent_memories)
            )
            prompt_content.insert(
                1, self.prompts["relevant_memories"].format(relevant_memories)
            )
        if recent_memories and not relevant_memories:
            prompt_content.insert(
                0, self.prompts["recent_memories"].format(recent_memories)
            )
        if not recent_memories and relevant_memories:
            prompt_content.insert(
                0, self.prompts["relevant_memories"].format(relevant_memories)
            )

        return "\n".join(prompt_content)

    def query_llm(
        self, message: str, recent_memories: List[str], relevant_memories: List[str]
    ) -> str:
        """Query an Open AI model and retrieve its response.

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
        messages = [
            {"system": self.prompts["system"]},
            {
                "user": self.format_prompt(
                    message=message, additional_informations=memories
                )
            },
        ]
        raw_response = self.client.chat.completions.create(
            model=self.model_name, messages=messages  # type:ignore
        )
        formatted_response = str(raw_response.choices[0].message.content)
        return formatted_response
