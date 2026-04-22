"""
Tests for the ingestion pipeline components.
=== FILE: tests/test_ingestion.py ===
"""

from ingestion.text_chunker import TextChunker
from langchain_core.documents import Document

def test_chunking_logic():
    """
    Tests if the TextChunker correctly splits a document into chunks.
    """
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    
    # Create a dummy large document
    long_text = " ".join(["word"] * 100)  # 100 words, each plus space is 5 chars -> 500 characters
    dummy_doc = Document(page_content=long_text, metadata={"source": "dummy.txt"})
    
    chunks = chunker.chunk_document([dummy_doc])
    
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    
    # Check if metadata propogated
    for chunk in chunks:
        assert chunk.metadata["source"] == "dummy.txt"
