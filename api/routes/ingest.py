"""
Route for uploading and ingesting documents into the RAG pipeline.
=== FILE: api/routes/ingest.py ===
"""

import os
import time
import shutil
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas.response import IngestResponse
from ingestion.document_loader import DocumentLoader
from ingestion.text_chunker import TextChunker
from ingestion.vector_store_uploader import VectorStoreUploader

router = APIRouter()
logger = logging.getLogger(__name__)

# Basic allowed extensions
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}
MAX_FILE_SIZE_MB = 10

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Accepts a multipart file upload, runs the ingestion pipeline, and stores embeddings.
    """
    start_time = time.time()
    
    # Validate extension
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}")
        
    temp_dir = os.path.join(os.getcwd(), "tmp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save file to disk
        file.file.seek(0, 2)
        file_size_mb = file.file.tell() / (1024 * 1024)
        file.file.seek(0)
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=400, detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB.")
            
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Pipeline execution
        logger.info(f"Starting ingestion pipeline for {file.filename}")
        
        # 1. Load document
        loader = DocumentLoader(supported_formats=list(ALLOWED_EXTENSIONS))
        doc_data = loader.load_document(temp_path)
        
        # 2. Chunk text
        chunker = TextChunker()
        chunks = chunker.chunk_document(doc_data)
        
        # 3. Upload to vector store (Embedding is handled here indirectly via the uploader)
        uploader = VectorStoreUploader()
        chunk_ids = uploader.upload(chunks)
        
        process_time = time.time() - start_time
        
        return IngestResponse(
            filename=file.filename,
            chunk_count=len(chunk_ids),
            processing_time_sec=process_time
        )
        
    except Exception as e:
        logger.error(f"Failed to ingest document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
