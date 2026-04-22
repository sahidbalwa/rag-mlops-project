"""
Module for uploading embeddings and documents into a vector store.
=== FILE: ingestion/vector_store_uploader.py ===
"""

import os
import time
import logging
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain Vectorstore integrations
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Pinecone
from langchain_core.documents import Document
from pinecone import Pinecone as PineconeClient

# Local module imports
from ingestion.embedding_generator import EmbeddingGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VectorStoreUploader:
    """
    Handles the upload of chunked texts and their embeddings to a chosen Vector Store
    We support Chroma (local) and Pinecone (cloud).
    """

    def __init__(self, backend: str = None, collection_name: str = "rag_collection"):
        """
        Initializes the Vector Store uploader.

        Args:
            backend (str): 'chroma' or 'pinecone'. Defaults to reading from VECTOR_STORE_BACKEND.
            collection_name (str): Name of the collection or index to store documents.
        """
        self.backend = backend or os.getenv("VECTOR_STORE_BACKEND", "chroma").lower()
        self.collection_name = collection_name
        
        # We need the embedding generator to pass the embedding function to the vector stores
        self.embedding_generator = EmbeddingGenerator()
        
        logger.info(f"Initializing VectorStoreUploader with backend: {self.backend}")

        try:
            if self.backend == "chroma":
                # Initialize local ChromaDB client
                persist_dir = os.path.join(os.getcwd(), "data", "chroma_db")
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_generator.embeddings,
                    persist_directory=persist_dir
                )
            elif self.backend == "pinecone":
                # Initialize cloud Pinecone client
                api_key = os.getenv("PINECONE_API_KEY")
                if not api_key:
                    raise ValueError("PINECONE_API_KEY environment variable is not set.")
                
                # Setup Pinecone using the official client logic for Langchain
                pc = PineconeClient(api_key=api_key)
                
                index_name = os.getenv("PINECONE_INDEX_NAME", self.collection_name)
                # Langchain Pinecone wrapper
                self.vector_store = Pinecone.from_existing_index(
                    index_name=index_name,
                    embedding=self.embedding_generator.embeddings
                )
            else:
                raise ValueError(f"Unknown vector store backend: {self.backend}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store backend '{self.backend}': {e}")
            raise

    def upload(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Uploads a list of document chunks to the configured vector store.

        Args:
            chunks (List[Dict[str, Any]]): List containing 'text', 'metadata', and optionally 'embedding'.

        Returns:
            List[str]: List of unique IDs assigned to the inserted chunks.
        """
        start_time = time.time()
        
        if not chunks:
            logger.warning("No chunks provided for upload.")
            return []

        # Convert simple dictionaries back into Langchain Document objects
        documents = []
        ids = []
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            doc = Document(page_content=chunk["text"], metadata=chunk.get("metadata", {}))
            documents.append(doc)
            ids.append(doc_id)

        logger.info(f"Uploading {len(documents)} documents to {self.backend}...")

        try:
            # The add_documents call will compute embeddings again if they aren't pre-computed.
            # Usually, langchain vector stores handle generating embeddings internally via the 
            # passed `embedding_function`, so we just pass the Document objects.
            self.vector_store.add_documents(documents, ids=ids)
        except Exception as e:
            logger.error(f"Error while uploading documents: {e}")
            raise

        elapsed = time.time() - start_time
        logger.info(f"Successfully uploaded {len(ids)} documents in {elapsed:.3f} seconds.")

        return ids
