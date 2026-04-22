"""
Module for detecting data drift in embeddings.
=== FILE: mlops/monitoring/drift_detector.py ===
"""

import logging
import pandas as pd
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    Report = None
    DataDriftPreset = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EmbeddingDriftDetector:
    """
    Computes statistical data drift on embedding vectors to detect if incoming
    queries or new documents distribution diverged from the reference baseline.
    """

    def __init__(self, reference_data: pd.DataFrame):
        """
        Initializes the drift detector with a baseline matrix.
        
        Args:
            reference_data (pd.DataFrame): The baseline embeddings dataframe.
        """
        self.reference_data = reference_data
        
        if Report is None:
            logger.warning("Evidently package is missing. Drift detection is disabled.")

    def detect_drift(self, current_data: pd.DataFrame) -> bool:
        """
        Evaluates drift between the reference and current distributions.
        
        Args:
            current_data (pd.DataFrame): The current batch of embeddings.
            
        Returns:
            bool: True if significant drift is detected, False otherwise.
        """
        if Report is None:
            return False

        logger.info("Computing Data Drift using Evidently...")
        
        try:
            # We treat embeddings as numerical columns
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(reference_data=self.reference_data, current_data=current_data)
            
            # Parse evidently's JSON output for the dataset drift flag
            result = data_drift_report.as_dict()
            drift_detected = result["metrics"][0]["result"]["dataset_drift"]
            
            logger.info(f"Drift Analysis Complete. Drift Detected: {drift_detected}")
            
            return drift_detected
        except Exception as e:
            logger.error(f"Error computing drift: {e}")
            return False
