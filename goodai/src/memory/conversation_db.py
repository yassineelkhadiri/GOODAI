import os
import logging
from typing import Dict, List

from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv(".env")
logger = logging.getLogger()


class ConversationDatabase:
    def __init__(self, index_name: str = "conversations") -> None:
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.pinecone_client = Pinecone(api_key=self.api_key)
        self.pinecone_index = self.pinecone_client.Index(self.index_name)
        logger.info("Connection to database established.")

    def upsert_conversations(self, vectors: List[Dict]):
        """Inser a list of vectors in the Pinecone vector database."""
        self.pinecone_index.upsert(vectors)

    # def find_similar_conversations(self, query_embedding, top_k=5):
    #     results = pinecone.query(self.index_name, query_embedding, top_k=top_k)
    #     return results.ids
