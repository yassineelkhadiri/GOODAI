import logging

FORMAT = "[%(asctime)s]:[%(levelname)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class MEMORY_TYPE:
    DUPLICATED = "duplicated"
    CONTRADICTORY = "contradictory"
    TEMPORARY = "temporary"
    EPISODIC = "episodic"
    NEW = "new"
