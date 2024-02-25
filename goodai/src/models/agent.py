from typing import Union

from goodai.src.models import OpenAIModel, OpenSourceModel
from goodai.src.memory.memory import MemoryManager, Memory


class Agent:
    """
    Class representing an agent.
    An agent has two main attributes:
        - An LLM to interact with.
        - A memory manage to manage the memories
            stored localy and in the vector database.
    """

    def __init__(self, model: Union[OpenAIModel, OpenSourceModel]) -> None:
        self.model = model
        self.memory_manager = MemoryManager()

    def interact(self, message: str) -> str:
        """
        Forward user query to the LLM and store the memory in the buffer.
        Perform a check on whether the memory buffer is full
        to push the changes to the distant database.

        Args:
            message: User's input

        Returns: LLM response.
        """
        memory = Memory(user_input=message)
        self.memory_manager.save_memory(memory)
        llm_reponse = self.model.query_llm(message)
        self.memory_manager.manage()
        return llm_reponse
