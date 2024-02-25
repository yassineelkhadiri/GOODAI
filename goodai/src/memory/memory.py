import uuid
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, List

from goodai.src.memory.conversation_db import ConversationDatabase
from goodai.src.memory.utils import MEMORY_TYPE

from goodai.src.models.tokenizer import Tokenizer

logger = logging.getLogger()


@dataclass
class Memory:
    """Base class to represent a memory."""

    user_input: str
    memory_type: str = MEMORY_TYPE.NEW
    timestamp: Any = datetime.now()
    expiration: Any = timestamp + timedelta(days=2 * 30)

    def __repr__(self) -> str:
        return f"Memory(content={self.user_input}, memory_type={self.memory_type}, timestamp={self.timestamp}, expiration={self.expiration})"  # noqa:E501


class MemoryManager:
    """Class responsible for managing memories."""

    def __init__(self) -> None:
        self.conversation_database = ConversationDatabase()
        self.memory_buffer: List[Memory] = []
        self.tokenizer = Tokenizer()

    def save_memory(self, memory: Memory) -> None:
        """Save a memory in the local buffer"""
        self.memory_buffer.append(memory)

    def insert_to_database(self) -> None:
        """
        Prepares the memories present in the buffers
        and save it in the Pinecone vector database.
        """
        vectors = []
        for memory in self.memory_buffer:
            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": self.tokenizer.encode(memory.user_input),
                    "metadata": {
                        "user_input": memory.user_input,
                        "memory_type": memory.memory_type,
                        "timestamp": memory.timestamp,
                        "expiration": memory.expiration,
                    },
                }
            )
        self.conversation_database.upsert_conversations(vectors)

    @property
    def buffer_is_full(self) -> bool:
        """Checks if the buffer is full."""
        return len(self.memory_buffer) > 2

    def manage(self) -> None:
        """Manage memories."""
        if self.buffer_is_full:
            logger.info("Clearing local memory buffer.")
            self.insert_to_database()
            self.memory_buffer = []
