import logging
import numpy as np

from typing import List, Dict, Union, Tuple
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity

from goodai.src.memory.conversation_db import SessionDatabase
from goodai.src.memory.memory import Memory, MEMORY_TYPE
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

        This includes
            - Fetching the latest 5 memories.
            - Fetching the top 5 most related memories.
            - Adding new memory to local buffer.

        Args:
            user_input: user input to the LLM.

        Returns:
            5 recent memories in the buffer.
            5 relevant memories sorted by timestamp.
        """
        encoded_user_input = self.tokenizer.encode(user_input)

        top_5_most_recent_memories = self.memory_buffer[-5:]
        top_5_related_memories = self._get_relevant_memories(encoded_user_input)

        new_timestamp = datetime.now()
        new_memory = Memory(
            user_input=user_input,
            encoded_user_input=encoded_user_input,
            memory_type=MEMORY_TYPE.NEW,
            timestamp=new_timestamp,
            expiration=new_timestamp + timedelta(days=2 * 30),
        )
        self._save_memory(new_memory)

        return top_5_most_recent_memories, top_5_related_memories

    def clear_session(self) -> None:
        """Clears session databases of the memory manager."""
        self.session_database.clear_database()
        self.memory_buffer = []

    def _get_relevant_memories(
        self, encoded_user_input: np.ndarray, number_of_records: int = 5
    ) -> List[Memory]:
        """
        This method is responsible for fetching memories
        present in the memory buffer that are relevant
        to a given user input based on vector similarity.

        Args:
            encoded_user_input: User input encoded using the tokenizer.
            number_of_records: Number of memories to return, Defaults to 5.
        """
        if not self.memory_buffer:
            return []
        else:
            encoded_memories = [
                memory.encoded_user_input for memory in self.memory_buffer
            ]
            reshaped_encoded_memory = encoded_user_input.reshape(1, -1)

            similarities = np.squeeze(
                cosine_similarity(reshaped_encoded_memory, encoded_memories)
            )
            if np.ndim(similarities) == 0:
                similarities = [similarities]
            else:
                similarities = similarities.tolist()
            pairs = [
                (similarity, index)
                for similarity, index in zip(similarities, range(len(similarities)))
            ]
            pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
            top_pairs = []
            for _, index in pairs_sorted:
                top_pairs.append(index)
                if len(top_pairs) >= number_of_records:
                    break
            return [self.memory_buffer[index] for index in top_pairs]

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
        latest_memories_in_raw_format = self.session_database.get_all_memories()
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
