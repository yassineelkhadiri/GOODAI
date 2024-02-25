import numpy as np
from sentence_transformers import SentenceTransformer


class Tokenizer:
    """Class used to tokenize user input and produce embeddings."""

    def __init__(
        self, tokenizer_name: str = "sentence-transformers/all-mpnet-base-v2"
    ) -> None:
        self.model = SentenceTransformer(tokenizer_name, device="cpu")

    def encode(self, input: str) -> np.ndarray:
        """Encoder a given string to vectors."""
        return self.model.encode(input)
