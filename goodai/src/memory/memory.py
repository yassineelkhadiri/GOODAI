from collections import Counter
import logging
import spacy

from typing import Any, Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


FORMAT = "[%(asctime)s]:[%(name)s]:[%(levelname)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

SIMILARITY_THRESHOLD = 0.8
nlp = spacy.load("en_core_web_md")


class MEMORY_TYPE:
    DUPLICATE = "duplicate"
    CONTRADICTORY = "contradictory"
    TEMPORARY = "temporary"
    EPISODIC = "episodic"
    NEW = "new"


@dataclass
class Memory:
    """Base class to represent a memory."""

    user_input: str
    encoded_user_input: np.ndarray
    memory_type: str = MEMORY_TYPE.NEW
    timestamp: Any = datetime.now()
    expiration: Any = timestamp + timedelta(days=2 * 30)

    @property
    def is_duplicate(self) -> bool:
        """Checks if a memory is duplicated."""
        return self.memory_type == MEMORY_TYPE.DUPLICATE

    def __eq__(self, other: object) -> bool:
        """
        Check if a given memory is the same as the current instance
        in order to determine if a memory is dupliacted or not
        by weighing the context similary and the token similarity.

        Args:
            other: Memory to compare with.

        Raises:
            ValueError: if the 'other' object is not of type Memory.

        Returns: True if the memory is duplicated, False otherwise.
        """
        if not isinstance(other, Memory):
            raise ValueError("Operation not supported.")
        else:
            current_tokens = self.user_input.split()
            other_tokens = other.user_input.split()
            intersection_count = sum(
                (Counter(current_tokens) & Counter(other_tokens)).values()
            )
            current_memory_doc = nlp(self.user_input)
            other_memory_doc = nlp(other.user_input)
            context_similarity = current_memory_doc.similarity(other_memory_doc)

            token_similarity = intersection_count / max(
                len(current_tokens), len(other_tokens)
            )
            weighted_similarity = 0.6 * context_similarity + 0.4 * token_similarity
            return weighted_similarity >= SIMILARITY_THRESHOLD

    def __repr__(self) -> str:
        return f"Memory(content={self.user_input}, memory_type={self.memory_type}, timestamp={self.timestamp}, expiration={self.expiration})"  # noqa:E501

    @classmethod
    def from_dict(cls, data: Dict) -> "Memory":
        """Creates an instance from a dict."""
        return cls(
            user_input=data["metadata"]["user_input"],
            encoded_user_input=np.array(data["values"]),
            memory_type=data["metadata"]["memory_type"],
            timestamp=datetime.strptime(
                data["metadata"]["timestamp"], "%Y-%m-%d %H:%M:%S.%f"
            ),
            expiration=datetime.strptime(
                data["metadata"]["expiration"], "%Y-%m-%d %H:%M:%S.%f"
            ),
        )

    @classmethod
    def from_list(cls, data: List[str]) -> "Memory":
        """Creates an instance from a list."""
        return cls(
            user_input=data[1],
            encoded_user_input=np.array(eval(data[2])),
            memory_type=data[3],
            timestamp=datetime.strptime(data[4], "%Y-%m-%d %H:%M:%S.%f"),
            expiration=datetime.strptime(data[5], "%Y-%m-%d %H:%M:%S.%f"),
        )
