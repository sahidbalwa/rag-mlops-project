"""
Module for enabling LangSmith tracing for LangChain objects.
=== FILE: mlops/tracking/langsmith_tracer.py ===
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LangSmithTracer:
    """
    Ensures that LangSmith tracing is correctly activated so that LangChain 
    chains report deeper trace sequences and latencies.
    """
    
    @staticmethod
    def setup_tracing():
        """
        Checks environment variables to verify that LangSmith tracing can operate.
        Normally LangSmith works implicitly via the LANGCHAIN_TRACING_V2 env var,
        so this acts as a verification/hook layer.
        """
        tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
        api_key = os.getenv("LANGCHAIN_API_KEY")
        project_name = os.getenv("LANGCHAIN_PROJECT", "rag-mlops-pipeline")
        
        if tracing_v2 == "true" and api_key and api_key != "your_langsmith_api_key_here":
            logger.info(f"LangSmith Tracing v2 is ENABLED for project: {project_name}")
            return True
        else:
            logger.warning("LangSmith Tracing is DISABLED or missing valid API Key.")
            return False
