from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base model implementation."""

    def __init__(self, model_name) -> None:
        self.model_name = model_name

    @abstractmethod
    def query_llm(self, message: str) -> str:
        pass
