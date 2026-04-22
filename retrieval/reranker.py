"""
Module for reranking retrieved documents using Cohere Rerank API.
=== FILE: retrieval/reranker.py ===
"""

import os
import time
import logging
from typing import List, Tuple
from langchain_core.documents import Document

try:
    import cohere
except ImportError:
    cohere = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Reranker:
    """
    Reranks retrieved documents to improve relevance using an external API like Cohere.
    Falls back to original ordering if API key isn't provided.
    """

    def __init__(self, top_n: int = 3):
        """
        Initializes the Reranker.

        Args:
            top_n (int): Final number of documents to retain after reranking.
        """
        self.top_n = top_n
        self.api_key = os.getenv("COHERE_API_KEY")
        
        if self.api_key and cohere:
            self.client = cohere.Client(self.api_key)
            logger.info("Cohere Reranker initialized successfully.")
        else:
            self.client = None
            logger.warning("COHERE_API_KEY not set or cohere package missing. Reranking will be skipped.")

    def rerank(self, query: str, docs_with_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Reranks the documents based on query relevance.

        Args:
            query (str): The search query.
            docs_with_scores (List[Tuple[Document, float]]): The originally retrieved documents.

        Returns:
            List[Tuple[Document, float]]: The reordered top_n documents. 
        """
        if not docs_with_scores:
            return []

        # If there is no client, execute fallback strategy: just return the top_n items as is.
        if not self.client:
            logger.info("Using fallback reranker (returning original top_n).")
            return docs_with_scores[:self.top_n]

        start_time = time.time()
        logger.info(f"Reranking {len(docs_with_scores)} documents via Cohere.")

        try:
            # Extract plain text strings for the Cohere API
            texts = [doc.page_content for doc, _ in docs_with_scores]
            
            response = self.client.rerank(
                model='rerank-english-v3.0',
                query=query,
                documents=texts,
                top_n=self.top_n
            )

            # Reconstruct the results list with updated indices and scores from Cohere
            reranked_results = []
            for result in response.results:
                original_doc = docs_with_scores[result.index][0]
                reranked_score = result.relevance_score
                reranked_results.append((original_doc, reranked_score))
                
            elapsed = time.time() - start_time
            logger.info(f"Reranking completed in {elapsed:.3f} seconds.")
            return reranked_results

        except Exception as e:
            logger.error(f"Error during Cohere reranking: {e}. Falling back to original ordering.")
            return docs_with_scores[:self.top_n]
