"""
Module defining API request schemas using Pydantic v2.
=== FILE: api/schemas/request.py ===
"""

from pydantic import BaseModel, ConfigDict, Field

class QueryRequest(BaseModel):
    """
    Schema for the main Q&A query request.
    """
    question: str = Field(..., description="The user's question to ask the RAG pipeline.")
    top_k: int = Field(default=5, ge=1, le=20, description="Override the number of retrieved documents.")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What are the core features of the product?",
                "top_k": 5
            }
        }
    )
