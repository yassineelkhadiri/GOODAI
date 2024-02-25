import pytest
import numpy as np
from goodai.src.models.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer()


@pytest.mark.parametrize(
    "input, expected_type, expected_shape",
    [
        ("This is a test sentence.", np.ndarray, (768,)),
        ("", np.ndarray, (768,)),
    ],
)
def test_encode(input, expected_type, expected_shape, tokenizer: Tokenizer):
    encoded_vector = tokenizer.encode(input)

    assert isinstance(encoded_vector, expected_type)
    assert encoded_vector.shape == expected_shape

