"""
Module for tracking experimental parameters, models, and scores via MLflow.
=== FILE: mlops/tracking/mlflow_tracker.py ===
"""

import os
import mlflow
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MLflowTracker:
    """
    Handles logging of LLM generation runs to a remote MLflow Tracking Server.
    """

    def __init__(self, experiment_name: str = "rag-experiments"):
        """
        Initializes the MLflow tracker.
        
        Args:
            experiment_name (str): Experiment name in MLflow.
        """
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", experiment_name)
        
        # Keep track if start_run is active
        self.active_run = None
        
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Connected to MLflow Tracking server at {self.tracking_uri}")
        except Exception as e:
            logger.warning(f"Could not initialize MLflow. Runs will not be logged. Error: {e}")

    def start_run(self, run_name: str = None) -> None:
        """
        Starts an MLflow run.
        """
        if mlflow.active_run() is None:
            self.active_run = mlflow.start_run(run_name=run_name)
            logger.info(f"Started MLflow run '{run_name}'.")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Logs generation and retrieval parameters.
        """
        try:
            if mlflow.active_run():
                mlflow.log_params(params)
        except Exception as e:
            logger.error(f"Error logging params to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Logs RAG evaluation metrics or raw retrieval scores.
        """
        try:
            if mlflow.active_run():
                mlflow.log_metrics(metrics)
        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {e}")

    def end_run(self) -> None:
        """
        Ends the current MLflow run.
        """
        if mlflow.active_run():
            mlflow.end_run()
            self.active_run = None
            logger.info("Ended MLflow run.")
