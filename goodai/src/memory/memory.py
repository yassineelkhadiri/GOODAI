import logging
from typing import Any
from dataclasses import dataclass
from datetime import datetime, timedelta

FORMAT = "[%(asctime)s]:[%(levelname)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


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

    def __repr__(self) -> str:
        return f"Memory(content={self.user_input}, memory_type={self.memory_type}, timestamp={self.timestamp}, expiration={self.expiration})"  # noqa:E501
