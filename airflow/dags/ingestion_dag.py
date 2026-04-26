"""
ingestion_dag.py
Airflow DAG: Daily document ingestion pipeline.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


def run_ingestion():
    print("Running document ingestion pipeline...")


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="daily_ingestion_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["rag", "ingestion"],
) as dag:
    ingest_task = PythonOperator(
        task_id="run_ingestion",
        python_callable=run_ingestion,
    )
