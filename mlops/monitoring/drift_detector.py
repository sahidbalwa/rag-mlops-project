"""
drift_detector.py
Detects data/embedding drift using Evidently AI.
"""
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


class DriftDetector:
    def __init__(self):
        self.threshold = float(os.getenv("DRIFT_THRESHOLD", 0.15))
        self.report_dir = os.getenv("EVIDENTLY_REPORT_DIR", "./reports/evidently")

    def detect(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report.save_html(f"{self.report_dir}/drift_report.html")
        result = report.as_dict()
        drift_score = result["metrics"][0]["result"]["dataset_drift"]
        print(f"[Drift] Score: {drift_score} | Threshold: {self.threshold}")
        return drift_score > self.threshold
