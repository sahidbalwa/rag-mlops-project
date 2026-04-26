"""
evaluation_dag.py
Airflow DAG: Weekly RAGAS evaluation.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


def run_evaluation():
    print("Running RAGAS evaluation pipeline...")


with DAG(
    dag_id="weekly_evaluation_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["rag", "evaluation"],
) as dag:
    eval_task = PythonOperator(
        task_id="run_evaluation",
        python_callable=run_evaluation,
    )
