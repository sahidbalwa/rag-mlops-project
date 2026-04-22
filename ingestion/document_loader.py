"""
Module for loading documents from various formats (PDF, TXT, DOCX) into text format.
=== FILE: ingestion/document_loader.py ===
"""

import os
import time
import logging
from typing import List, Dict, Any
import fitz  # PyMuPDF

# We wrap docx in a try/except to gracefully fail if python-docx is not installed
try:
    from docx import Document
except ImportError:
    Document = None

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Handles loading of documents of different formats and extracting text.
    Supports PDF (using PyMuPDF), TXT, and DOCX formats.
    """

    def __init__(self, supported_formats: List[str] = None):
        """
        Initializes the document loader.

        Args:
            supported_formats (List[str]): List of supported extensions (e.g., ["pdf", "txt", "docx"])
        """
        self.supported_formats = supported_formats or ["pdf", "txt", "docx"]

    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Loads a document given its path and returns its text content and metadata.

        Args:
            file_path (str): The local path to the file.

        Returns:
            Dict[str, Any]: A dictionary containing 'text' and 'metadata'.
        """
        start_time = time.time()
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.split(".")[-1].lower()
        if ext not in self.supported_formats:
            logger.error(f"Unsupported file format: {ext}")
            raise ValueError(f"Unsupported file format: {ext}")

        logger.info(f"Loading document from: {file_path}")

        text = ""
        try:
            if ext == "pdf":
                text = self._load_pdf(file_path)
            elif ext == "txt":
                text = self._load_txt(file_path)
            elif ext == "docx":
                text = self._load_docx(file_path)
        except Exception as e:
            logger.error(f"Error while loading {file_path}: {e}")
            raise

        elapsed = time.time() - start_time
        logger.info(f"Successfully loaded {file_path} in {elapsed:.3f} seconds.")

        return {
            "text": text,
            "metadata": {
                "source": file_path,
                "type": ext,
                "loaded_at": time.time()
            }
        }

    def _load_pdf(self, file_path: str) -> str:
        """
        Extracts text from a PDF file using PyMuPDF.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text.
        """
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text

    def _load_txt(self, file_path: str) -> str:
        """
        Reads text from a TXT file.

        Args:
            file_path (str): Path to the TXT file.

        Returns:
            str: Extracted text.
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _load_docx(self, file_path: str) -> str:
        """
        Extracts text from a DOCX file.

        Args:
            file_path (str): Path to the DOCX file.

        Returns:
            str: Extracted text.
        """
        if Document is None:
            logger.error("python-docx is not installed. Extracted text may be unavailable.")
            raise ImportError("python-docx is required to read .docx files.")
            
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
