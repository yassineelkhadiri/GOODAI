import os
import logging
from typing import Dict, List

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv(".env")
logger = logging.getLogger()


_INDEX_METRIC = "cosine"
_INDEX_DIMENSION = 768
_SPECS = ServerlessSpec(cloud="aws", region="us-west-2")


class ConversationDatabase:
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
        logger.info("Connection to database established.")

    def upsert_conversations(self, vectors: List[Dict]):
        """Insert a list of vectors in the Pinecone vector database."""
        self.pinecone_index.upsert(vectors)

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
            include_values=False,
            include_metadata=True,
        )
        return results

    def fetch_latest_5_memories(self) -> Dict:
        """Collect the latest memories stored in the index."""
        # TODO: implement logic for fetching the last 5 records.
        return {}

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
