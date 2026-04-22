"""
Module for chunking text into smaller pieces for embedding and retrieval.
=== FILE: ingestion/text_chunker.py ===
"""

import time
import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TextChunker:
    """
    Handles text chunking using both fixed-size window and sentence-aware strategies.
    Supports integration with LangChain's text splitters.
    """

    def __init__(self, strategy: str = "fixed", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the text chunker.

        Args:
            strategy (str): 'fixed' for fixed-size character chunking or 'sentence' for NLP-based chunking.
            chunk_size (int): Max size of each chunk.
            chunk_overlap (int): Overlap between chunks to prevent losing context.
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if self.strategy == "sentence":
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                nltk.download('punkt_tab')
            logger.info("Initializing NLTKTextSplitter for sentence-aware chunking.")
            self.splitter = NLTKTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        else:
            logger.info("Initializing RecursiveCharacterTextSplitter for fixed-size chunking.")
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )

    def chunk_document(self, document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits the text of a loaded document into chunks.

        Args:
            document_data (Dict[str, Any]): Dictionary containing 'text' and 'metadata'.

        Returns:
            List[Dict[str, Any]]: List of dictionary chunks, each having 'text' and 'metadata'.
        """
        start_time = time.time()
        text = document_data.get("text", "")
        metadata = document_data.get("metadata", {})

        logger.info(f"Chunking document '{metadata.get('source', 'unknown')}' using strategy: {self.strategy}")
        try:
            chunks = self.splitter.split_text(text)
        except Exception as e:
            logger.error(f"Error during chunking: {e}")
            raise

        chunk_docs = []
        for i, chunk_text in enumerate(chunks):
            # We copy original metadata and inject chunk specific info
            chunk_meta = metadata.copy()
            chunk_meta["chunk_id"] = i
            chunk_docs.append({
                "text": chunk_text,
                "metadata": chunk_meta
            })

        elapsed = time.time() - start_time
        logger.info(f"Chunking completed in {elapsed:.3f} seconds. Created {len(chunk_docs)} chunks.")
        
        return chunk_docs
