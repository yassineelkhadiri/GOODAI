import uuid
import logging

from typing import List, Dict, Union, Tuple

from goodai.src.memory.conversation_db import ConversationDatabase
from goodai.src.memory.memory import Memory, MEMORY_TYPE

from goodai.src.models.tokenizer import Tokenizer

logger = logging.getLogger()


class MemoryManager:
    """Class responsible for managing memories."""

    def __init__(self) -> None:
        self.conversation_database = ConversationDatabase()
        self.memory_buffer: List[Memory] = []
        self.tokenizer = Tokenizer()

    def __del__(self) -> None:
        """
        This is added to insure that any memories left
        in the buffer are uploaded to the Pinecone Database.
        """
        self.insert_to_database()

    @property
    def buffer_is_full(self) -> bool:
        """Checks if the buffer is full."""
        return len(self.memory_buffer) > 20

    def save_memory(self, new_memory: Memory) -> None:
        """Create a memory from user input and save it in the local buffer"""
        for memory in self.memory_buffer:
            if new_memory == memory:
                logger.warn("Duplicated memory found.")
                new_memory.memory_type = MEMORY_TYPE.DUPLICATED
        self.memory_buffer.append(new_memory)

    def insert_to_database(self) -> None:
        """
        Prepares the memories present in the buffers
        and save it in the Pinecone vector database.

        Note: Policy for duplicated memories: ignored.
        """
        vectors = []
        for memory in self.memory_buffer:
            if memory.memory_type != MEMORY_TYPE.DUPLICATED:
                vectors.append(
                    {
                        "id": str(uuid.uuid4()),
                        "values": memory.encoded_user_input,
                        "metadata": {
                            "user_input": memory.user_input,
                            "memory_type": memory.memory_type,
                            "timestamp": memory.timestamp,
                            "expiration": memory.expiration,
                        },
                    }
                )
        self.conversation_database.upsert_conversations(vectors)

    def manage(self, user_input: str) -> Tuple[List[Memory], List[Memory]]:
        """
        This method is responsible for managing memories on each interaction made.

        This includes:
            - Adding new memory to local buffer.
            - Freeing up the local buffer if it's full (maximum of 20 memories).
            - Fetching the latest 5 memories from the local buffer.
            - Fetching the top 5 most related memories stored in the Pinecone database.

        Args:
            user_input: user input to the LLM.

        Returns:
            5 recent memories in the buffer.
            5 relevant memories stored in the vector database sorted by timestamp.
        """
        encoded_user_input = self.tokenizer.encode(user_input)
        new_memory = Memory(user_input, encoded_user_input)
        self.save_memory(new_memory)
        if self.buffer_is_full:
            logger.info("Clearing local memory buffer.")
            self.insert_to_database()
            self.memory_buffer = []

        top_5_most_recent_memories = self.memory_buffer[-5:]

        top_5_related_memories_in_raw_format = (
            self.conversation_database.retrieve_related_memories(
                encoded_user_input.tolist(), top_k=5
            )
        )

        top_5_related_memories = sorted(
            self._process_memories(top_5_related_memories_in_raw_format),
            key=lambda x: x.timestamp,
        )
        return top_5_most_recent_memories[:-1], top_5_related_memories

    def fetch_session(self) -> None:
        """Recall the latest session of the agent."""
        latest_memories_in_raw_format = (
            self.conversation_database.fetch_latest_5_memories()
        )
        return self._process_memories(latest_memories_in_raw_format)

    def clear_session(self) -> None:
        """Clears the conversation database of the memory manager."""
        self.conversation_database.clear_records()
        self.memory_buffer = []

    def _process_memories(
        self, memories_in_raw_format: Dict
    ) -> Union[List[Memory], List]:
        """Parse memories in raw response."""
        if len(memories_in_raw_format.get("matches", [])) == 0:
            return []
        return [
            Memory.from_dict(memory) for memory in memories_in_raw_format["matches"]
        ]
