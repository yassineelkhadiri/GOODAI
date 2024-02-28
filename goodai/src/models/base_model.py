from abc import ABC, abstractmethod
from typing import List, Dict


class BaseModel(ABC):
    """Base model implementation."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.prompts = {
            "system": "You are an AI chatbot designed to interact with a user and help them with their requests. Users may interact with you to provide information or ask you about something they already told you. Your goal is to help the user and interact with them kindly. Please respond only based on the input you receive from the user. If you don't know the answer to a user question, simply reply with 'I don't know.'.",  # noqa:E501
            "basic": "And this is the user input: {}",
            "recent_memories": "These are recent interactions the user had with you: {}",  # noqa: E501
            "relevant_memories": "These are the most relevant interactions to the provided user input: {}",  # noqa: E501
        }

    @abstractmethod
    def format_prompt(
        self, message: str, additional_information: Dict[str, List[str]]
    ) -> str:
        pass

    @abstractmethod
    def query_llm(
        self, message: str, recent_memories: List[str], relevant_memories: List[str]
    ) -> str:
        pass
