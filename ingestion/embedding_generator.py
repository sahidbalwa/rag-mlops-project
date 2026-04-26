"""
Module for generating embeddings from text chunks.
=== FILE: ingestion/embedding_generator.py ===
"""

import os
import time
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import langchain embedding providers
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Handles the generation of embeddings using various providers.
    Automatically picks provider based on EMBEDDING_PROVIDER env variable.
    """

    def __init__(self, provider: str = None, model_name: str = None):
        """
        Initializes the embedding generator and connects to the underlying model.

        Args:
            provider (str): 'huggingface' or 'openai'. Defaults to reading from env.
            model_name (str): The specific model name to use. 
        """
        self.provider = provider or os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
        self.model_name = model_name

        logger.info(f"Initializing embedding generator with provider: {self.provider}")
        try:
            if self.provider == "openai":
                self.model_name = self.model_name or "text-embedding-3-small"
                self.embeddings = OpenAIEmbeddings(model=self.model_name)
            elif self.provider == "huggingface":
                self.model_name = self.model_name or "sentence-transformers/all-MiniLM-L6-v2"
                self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            else:
                raise ValueError(f"Unknown embedding provider: {self.provider}")
        except Exception as e:
            logger.error(f"Error initializing embedding provider '{self.provider}': {e}")
            raise

    def embed_documents(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates and attaches embeddings to a list of text chunks.

        Args:
            chunks (List[Dict[str, Any]]): List of chunk dictionaries containing 'text'.

        Returns:
            List[Dict[str, Any]]: The original chunks, now with an 'embedding' key added.
        """
        start_time = time.time()
        
        texts = [chunk["text"] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks using {self.provider} ({self.model_name})")
        
        try:
            # Generate embeddings list in batch
            embeddings_list = self.embeddings.embed_documents(texts)
            
            # Attach embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings_list[i]
                
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            raise

        elapsed = time.time() - start_time
        logger.info(f"Successfully generated {len(texts)} embeddings in {elapsed:.3f} seconds.")
        
        return chunks
