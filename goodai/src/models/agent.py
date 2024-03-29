from typing import Union

from goodai.src.models import OpenAIModel, OpenSourceModel
from goodai.src.memory import MemoryManager


class Agent:
    """
    Class representing an agent.
    An agent has two main attributes:
        - An LLM to interact with.
        - A memory manage to manage the memories
            that handles the interactions.
    """

    def __init__(self, model: Union[OpenAIModel, OpenSourceModel]) -> None:
        self.model = model
        self.memory_manager = MemoryManager()

    def __del__(self) -> None:
        del self.memory_manager

    def interact(self, message: str) -> str:
        """
        Forward user query to the LLM and store the memory in the buffer.
        Perform a check on whether the memory buffer is full
        to push the changes to the session database.

        Args:
            message: User's input

        Returns: LLM response.
        """
        recent_memories, relevant_memories = self.memory_manager.manage(message)
        recent_memories_content = [memory.user_input for memory in recent_memories]
        relevant_memories_content = [memory.user_input for memory in relevant_memories]
        llm_reponse = self.model.query_llm(
            message, recent_memories_content, relevant_memories_content
        )
        # This is added to remove any memories from the llm response.
        for element in set(recent_memories_content + relevant_memories_content):
            if element in llm_reponse:
                llm_reponse = llm_reponse.replace(element, "")
        return llm_reponse.strip()

    def new_session(self) -> None:
        """Starts a new session with the agent."""
        self.memory_manager.clear_session()
