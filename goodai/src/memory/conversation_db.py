import os
import logging
import sqlite3

from sqlite3 import Connection, Cursor
from typing import Any, Dict, List, Tuple

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv(".env")
logger = logging.getLogger()


_INDEX_METRIC = "cosine"
_INDEX_DIMENSION = 768
_SPECS = ServerlessSpec(cloud="aws", region="us-west-2")


class ConversationDatabase:
    """
    THIS IMPLEMENTATION IS OBSELETE BUT KEPT 
    FOR FURTHER INVESTIGATION ON HOW TO OPTIMIZE IT.
    """
    """A class representation of a database storing all records."""

    def __init__(self, index_name: str = "conversations") -> None:
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.pinecone_client = Pinecone(api_key=self.api_key)
        if index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=_INDEX_DIMENSION,
                metric=_INDEX_METRIC,
                spec=_SPECS,
            )

        self.pinecone_index = self.pinecone_client.Index(self.index_name)
        logger.info("Connection to vector database established.")

    def upsert_conversations(self, vectors: List[Dict]):
        """Insert a list of vectors in the Pinecone vector database."""
        self.pinecone_index.upsert(vectors)
        logger.info("Records uploaded to vector database.")

    def retrieve_related_memories(
        self, encoded_memory_vector: List[float], top_k: int = 5
    ) -> Dict:
        """Retrieve examples that are most likely related to the current memory.

        Args:
            encoded_memory_vector: Encoded memory content.
            top_k: Number of examples to fetch. Defaults to 5.

        """
        results = self.pinecone_index.query(
            vector=encoded_memory_vector,
            top_k=top_k,
            include_values=True,
            include_metadata=True,
        )
        return results

    def clear_records(self) -> None:
        """Delete all records in the current index."""
        self.pinecone_client.delete_index(self.index_name)
        self.pinecone_client.create_index(
            name=self.index_name,
            dimension=_INDEX_DIMENSION,
            metric=_INDEX_METRIC,
            spec=_SPECS,
        )
        self.pinecone_index = self.pinecone_client.Index(self.index_name)
        logger.info("Pinecone records cleared.")


class SessionDatabase:
    """
    A simple sqlite database to store the latest records
    to be used to retrieve the latest interactions.
    """

    CACHE_FOLDER = "cache"
    DATABASE_NAME = "session.db"

    def __init__(self):
        self.database_file_path = os.path.join(
            os.path.dirname(__file__), self.CACHE_FOLDER, self.DATABASE_NAME
        )
        if os.path.exists(self.database_file_path):
            self.connection = sqlite3.connect(self.database_file_path)
            self.cursor = self.connection.cursor()
        else:
            self.connection, self.cursor = self.create_database()
        logger.info("connection to session database established.")

    def create_database(self) -> Tuple[Connection, Cursor]:
        """Creates the database file in the cache direcotry."""
        os.makedirs(
            os.path.join(os.path.dirname(__file__), self.CACHE_FOLDER), exist_ok=True
        )
        with sqlite3.connect(self.database_file_path) as conn:
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT,
                    encoded_user_input TEXT,
                    memory_type TEXT,
                    timestamp TEXT,
                    expiration TEXT
                )
            """
            conn.execute(create_table_sql)
            conn.commit()
            return conn, conn.cursor()

    def insert_memories(self, memories: List[List[str]]) -> None:
        """Insert memories into the database."""
        try:
            for memory in memories:
                self.cursor.execute(
                    "INSERT INTO memories (user_input, encoded_user_input, memory_type, timestamp, expiration) VALUES (?, ?, ?, ?, ?)",  # noqa:E501
                    (
                        memory[0],
                        memory[1],
                        memory[2],
                        memory[3],
                        memory[4],
                    ),
                )
            self.connection.commit()
            logger.info("Memories saved in local session database")

        except sqlite3.Error:
            logger.error("Error inserting memories to the session database")

    def fetch_most_recent_memories(self, num_records: int = 5) -> List[Any]:
        """Fetch the most recent memories from the database."""
        try:
            self.cursor.execute(
                f"SELECT * FROM memories ORDER BY timestamp DESC LIMIT {num_records}"
            )
            recent_memories = self.cursor.fetchall()
            return recent_memories
        except sqlite3.Error:
            logger.error(
                "Error fetching most recent memories from the session database."
            )
            return []

    def clear_database(self) -> None:
        """Delete all rows from the memories table."""
        try:
            self.cursor.execute("DELETE FROM memories")
            self.connection.commit()
            logger.info("Cleared session database.")

        except sqlite3.Error:
            logger.error("Error deleting rows from session database.")
