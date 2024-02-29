import pytest
from unittest.mock import patch, MagicMock

from goodai.src.models import OpenSourceModel


@pytest.fixture
def open_source_model():
    return OpenSourceModel("test_model")


@patch("requests.post")
def test_query_llm(mock_post, open_source_model: OpenSourceModel):
    mock_response = MagicMock()
    mock_response.json.return_value = [{"generated_text": "Test reply"}]
    mock_post.return_value = mock_response
    response = open_source_model.query_llm("Test message", [], [])

    mock_post.assert_called_once_with(
        open_source_model.api_url,
        headers=open_source_model.headers,
        json={
            "inputs": open_source_model.format_prompt("Test message", {}),
            "parameters": {"return_full_text": False},
        },
    )
    assert response == "Test reply"
