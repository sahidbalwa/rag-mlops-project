"""
Module and API route for triggering retraining or ingestion workflows.
=== FILE: mlops/retraining/trigger.py ===
"""

import os
import requests
import logging
from fastapi import APIRouter, HTTPException

router = APIRouter()
logger = logging.getLogger(__name__)

# Usually defined in .env
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")

@router.post("/trigger-ingestion")
async def trigger_ingestion_dag(reason: str = "drift_detected"):
    """
    Webhook endpoint to kick off the Airflow ingestion DAG when drift is detected.
    """
    logger.info(f"Received trigger for ingestion DAG. Reason: {reason}")
    
    # Airflow REST API endpoint to trigger a DAG run
    dag_id = "rag_ingestion_dag"
    api_endpoint = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    
    payload = {
        "conf": {"trigger_reason": reason}
    }
    
    try:
        response = requests.post(
            api_endpoint,
            json=payload,
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info("Successfully triggered Airflow ingestion DAG.")
            return {"status": "success", "message": "DAG triggered successfully."}
        else:
            logger.error(f"Failed to trigger DAG: {response.text}")
            raise HTTPException(status_code=502, detail=f"Airflow API Error: {response.text}")
            
    except Exception as e:
        logger.error(f"Error communicating with Airflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))
