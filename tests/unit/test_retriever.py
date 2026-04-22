import pytest
from unittest.mock import MagicMock, patch

def test_retriever_returns_results():
    with patch("retrieval.retriever.chromadb") as mock_chroma:
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["chunk 1", "chunk 2"]],
            "distances": [[0.1, 0.2]]
        }
        mock_chroma.HttpClient.return_value.get_or_create_collection.return_value = mock_collection
        from retrieval.retriever import Retriever
        retriever = Retriever()
        retriever.collection = mock_collection
        results = retriever.retrieve("test query")
        assert len(results) == 2
        assert "text" in results[0]
        assert "score" in results[0]
