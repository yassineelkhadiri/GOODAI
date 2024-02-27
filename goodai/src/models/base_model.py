from abc import ABC, abstractmethod
from typing import List, Dict


class BaseModel(ABC):
    """Base model implementation."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.prompts = {
            "system": "You are an AI chatbot designed to interact with a user and help with them with their requests. Users may interact with you to provide information or ask you about something they already told you, your goal is to help the user and interact with him kindly.",  # noqa:E501
            "basic": "User input: {}",
            "recent_memories": "The following are the recent interactions the user had with you: {}",  # noqa: E501
            "relevant_memories": "The following are the relevant interactions the user had with you: {}",  # noqa: E501
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
