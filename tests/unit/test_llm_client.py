import pytest
from unittest.mock import MagicMock, patch

def test_llm_generate():
    with patch("generation.llm_client.OpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test answer"
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        from generation.llm_client import LLMClient
        client = LLMClient()
        answer = client.generate("What is AI?", "AI is artificial intelligence.")
        assert answer == "Test answer"
