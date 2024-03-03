import logging

from typing import List, Dict, Union, Tuple

from goodai.src.memory.conversation_db import SessionDatabase
from goodai.src.memory.memory import Memory

from goodai.src.models.tokenizer import Tokenizer

logger = logging.getLogger()


class MemoryManager:
    """Class responsible for managing memories."""

    def __init__(self) -> None:
        self.session_database = SessionDatabase()
        self.tokenizer = Tokenizer()
        self.memory_buffer: List[Memory] = self._fetch_session()

    def __del__(self) -> None:
        """
        This is added to insure that any memories left
        in the buffer are saved in the session database.
        """
        try:
            self._insert_to_session_database()
        except Exception:
            pass

    def manage(self, user_input: str) -> Tuple[List[Memory], List[Memory]]:
        """
        This method is responsible for managing memories on each interaction made.

        This includes:
            - Adding new memory to local buffer.
            - Fetching the latest 10 memories from the local buffer.
            - Fetching the top 10 most related memories stored in the session database.

        Args:
            user_input: user input to the LLM.

        Returns:
            10 recent memories in the buffer.
            10 relevant memories sorted by timestamp.
        """
        encoded_user_input = self.tokenizer.encode(user_input)
        new_memory = Memory(user_input, encoded_user_input)
        self._save_memory(new_memory)

        top_10_most_recent_memories = self.memory_buffer[-10:-1]

        top_10_related_memories_in_raw_format = (
            self.session_database.fetch_most_relevant_memories(
                encoded_memory=encoded_user_input, number_of_records=10
            )
        )

        top_10_related_memories = sorted(
            self._process_memories_in_list(top_10_related_memories_in_raw_format),
            key=lambda x: x.timestamp,
        )
        return top_10_most_recent_memories, top_10_related_memories

    def clear_session(self) -> None:
        """Clears session databases of the memory manager."""
        self.session_database.clear_database()
        self.memory_buffer = []

    def _save_memory(self, new_memory: Memory) -> None:
        """
        Create a memory from user input and
        check if the new memory is a duplicate
        if the new memory is a duplicate then
        the older memory is  discarded
        otherwise it is saved in the local buffer

        Args:
            new_memory: new memory to be saved to the buffer.
        """
        if not self.memory_buffer:
            self.memory_buffer.append(new_memory)
        else:
            for memory in self.memory_buffer:
                if new_memory == memory:
                    logger.warn("Duplicated memory found.")
                    self.memory_buffer.remove(memory)
            self.memory_buffer.append(new_memory)

    def _insert_to_session_database(self) -> None:
        """
        Prepares the memories present in the buffer
        and save it in the sqlite session database.

        Note: Policy for duplicated memories: ignored.
        """
        memories_as_str = [
            [
                memory.user_input,
                str(memory.encoded_user_input.tolist()),
                memory.memory_type,
                memory.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                memory.expiration.strftime("%Y-%m-%d %H:%M:%S.%f"),
            ]
            for memory in self.memory_buffer
        ]
        self.session_database.insert_memories(memories_as_str)

    def _fetch_session(self) -> Union[List[Memory], List]:
        """Recall the latest session of the agent."""
        latest_memories_in_raw_format = (
            self.session_database.fetch_most_recent_memories()
        )
        return self._process_memories_in_list(latest_memories_in_raw_format)

    def _process_memories_in_dict(
        self, memories_in_raw_format: Dict
    ) -> Union[List[Memory], List]:
        """Parse memories in raw response."""
        if len(memories_in_raw_format.get("matches", [])) == 0:
            return []
        return [
            Memory.from_dict(memory) for memory in memories_in_raw_format["matches"]
        ]

    def _process_memories_in_list(
        self, memories_in_raw_format: List[List[str]]
    ) -> Union[List[Memory], List]:
        """Parse memories in raw response."""
        if len(memories_in_raw_format) == 0:
            return []
        return [Memory.from_list(memory) for memory in memories_in_raw_format]
