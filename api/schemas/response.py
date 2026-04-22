"""
Module defining API response schemas using Pydantic v2.
=== FILE: api/schemas/response.py ===
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str

class IngestResponse(BaseModel):
    """
    Schema for document ingestion response.
    """
    filename: str = Field(..., description="Name of the file processed.")
    chunk_count: int = Field(..., description="Number of text chunks generated and stored.")
    processing_time_sec: float = Field(..., description="Time taken to process in seconds.")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "report_2023.pdf",
                "chunk_count": 42,
                "processing_time_sec": 3.45
            }
        }
    )

class SourceDocument(BaseModel):
    """
    Schema representing a source reference in the unified response.
    """
    source: str
    relevance_score: float
    content_snippet: str

class QueryResponse(BaseModel):
    """
    Schema for the RAG evaluation and QA response.
    """
    answer: str = Field(..., description="The generated answer from the LLM.")
    has_answer: bool = Field(..., description="True if the LLM states it found the answer.")
    word_count: int = Field(..., description="Word count of the returned answer.")
    sources: List[SourceDocument] = Field(default_factory=list, description="List of source contexts used.")
    processing_time_sec: float = Field(..., description="E2E processing time for the query.")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "The core features are A, B, and C.",
                "has_answer": True,
                "word_count": 8,
                "sources": [
                    {
                        "source": "manual.pdf",
                        "relevance_score": 0.98,
                        "content_snippet": "The core features of the system include A, B, and C."
                    }
                ],
                "processing_time_sec": 1.25
            }
        }
    )
