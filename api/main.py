"""
FastAPI application entrypoint.
=== FILE: api/main.py ===
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.logging import RequestLoggingMiddleware
from api.routes import health, ingest, query
from mlops.retraining import trigger

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for startup and shutdown execution.
    """
    logger.info("Initializing RAG MLOps Pipeline Application...")
    # Add any explicit DB connections or model preloadings here
    yield
    logger.info("Shutting down RAG MLOps Pipeline Application...")
    # Add cleanup tasks here

app = FastAPI(
    title="RAG MLOps Pipeline API",
    description="Production-grade API for Document Q&A RAG application",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(query.router, tags=["Query"])
app.include_router(trigger.router, tags=["MLOps"])
