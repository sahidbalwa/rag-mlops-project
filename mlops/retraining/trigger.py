"""
trigger.py
Auto-triggers retraining when drift is detected.
"""
from mlops.monitoring.drift_detector import DriftDetector
import pandas as pd


class RetrainingTrigger:
    def __init__(self):
        self.detector = DriftDetector()

    def check_and_trigger(self, reference_df: pd.DataFrame, current_df: pd.DataFrame):
        drift_detected = self.detector.detect(reference_df, current_df)
        if drift_detected:
            print("⚠️  Drift detected! Triggering retraining pipeline...")
            self.trigger_retraining()
        else:
            print("✅ No drift detected. Pipeline healthy.")

    def trigger_retraining(self):
        # Hook into Airflow DAG or run pipeline directly
        print("🔄 Retraining pipeline triggered.")
