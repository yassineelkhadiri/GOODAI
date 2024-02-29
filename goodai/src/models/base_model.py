from abc import ABC, abstractmethod
from typing import List, Dict

from goodai.src.models.prompt import (
    SYSTEM_PROMPT,
    RECENT_MEMORIES_PROMPT,
    RELEVANT_MEMORIES_PROMPT,
    USER_INPUT_PROMPT,
)


class BaseModel(ABC):
    """Base model implementation."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.prompts = {
            "system": SYSTEM_PROMPT,
            "user_input": USER_INPUT_PROMPT,
            "recent_memories": RECENT_MEMORIES_PROMPT,
            "relevant_memories": RELEVANT_MEMORIES_PROMPT,
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
