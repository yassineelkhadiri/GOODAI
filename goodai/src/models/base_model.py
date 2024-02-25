from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base model implementation."""

    def __init__(self, model_name) -> None:
        self.model_name = model_name

    @property
    def prompt_template(self):
        return """
        You are an AI chatbot designed to interact with a user
        and help with them with their requests.
        Users may instruct or query you about information you
        have already received to aid them in their daily tasks.
        User input: {}
        """

    @abstractmethod
    def query_llm(self, message: str) -> str:
        pass
