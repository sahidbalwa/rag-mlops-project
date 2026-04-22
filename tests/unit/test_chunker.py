import pytest
from ingestion.text_chunker import TextChunker

def test_chunker_splits_text():
    chunker = TextChunker()
    text = "This is a test. " * 100
    chunks = chunker.chunk(text, {"source": "test.txt"})
    assert len(chunks) > 1
    assert "text" in chunks[0]
    assert "chunk_id" in chunks[0]

def test_chunker_metadata():
    chunker = TextChunker()
    chunks = chunker.chunk("Hello world.", {"source": "doc.pdf"})
    assert chunks[0]["metadata"]["source"] == "doc.pdf"
