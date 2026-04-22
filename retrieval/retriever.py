"""
Module for querying the vector store and retrieving relevant documents.
=== FILE: retrieval/retriever.py ===
"""

import os
import time
import logging
from typing import List, Tuple
from langchain_core.documents import Document
from ingestion.vector_store_uploader import VectorStoreUploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Retriever:
    """
    Handles similarity search against the vector store to fetch relevant context.
    """

    def __init__(self, backend: str = None, top_k: int = 5):
        """
        Initializes the Retriever.

        Args:
            backend (str): 'chroma' or 'pinecone'.
            top_k (int): Number of top documents to retrieve.
        """
        self.top_k = top_k
        self.uploader = VectorStoreUploader(backend=backend)
        self.vector_store = self.uploader.vector_store
        
        logger.info(f"Initialized Retriever with backend={self.uploader.backend}, top_k={self.top_k}")

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """
        Retrieves the top_k most similar documents for the given query.

        Args:
            query (str): The question or search term.
            top_k (int, optional): Overrides the default top_k.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing the Document and its similarity score.
        """
        start_time = time.time()
        k = top_k or self.top_k
        
        logger.info(f"Retrieving top {k} documents for query: '{query}'")
        
        try:
            # similarity_search_with_score returns List[Tuple[Document, float]]
            results = self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise

        elapsed = time.time() - start_time
        logger.info(f"Retrieved {len(results)} documents in {elapsed:.3f} seconds.")
        
        return results
