import logging
from typing import Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import spacy


FORMAT = "[%(asctime)s]:[%(levelname)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

SIMILARITY_THRESHOLD = 0.9
nlp = spacy.load("en_core_web_md")


class MEMORY_TYPE:
    DUPLICATED = "duplicated"
    CONTRADICTORY = "contradictory"
    TEMPORARY = "temporary"
    EPISODIC = "episodic"
    NEW = "new"


@dataclass
class Memory:
    """Base class to represent a memory."""

    user_input: str
    memory_type: str = MEMORY_TYPE.NEW
    timestamp: Any = datetime.now()
    expiration: Any = timestamp + timedelta(days=2 * 30)

    def __eq__(self, other: object) -> bool:
        """
        Check if a given memory is the same as the current instance
        in order to determine if a memory is dupliacted or not.

        Args:
            other: Memory to compare with.

        Raises:
            ValueError: if the 'other' object is not of type Memory.

        Returns: True if the memory is duplicated, False otherwise.
        """
        if not isinstance(other, Memory):
            raise ValueError("Operation not supported.")
        else:
            current_memory_doc = nlp(self.user_input)
            other_memory_doc = nlp(other.user_input)
            similarity = current_memory_doc.similarity(other_memory_doc)
            return similarity >= SIMILARITY_THRESHOLD

    def __repr__(self) -> str:
        return f"Memory(content={self.user_input}, memory_type={self.memory_type}, timestamp={self.timestamp}, expiration={self.expiration})"  # noqa:E501
