"""
Airflow DAG for running scheduled evaluations on the RAG pipeline.
=== FILE: orchestration/dags/evaluation_dag.py ===
"""

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'rag_evaluation_dag',
    default_args=default_args,
    description='Weekly offline evaluation of RAG pipeline performance',
    schedule_interval='@weekly',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['rag', 'evaluation'],
)

def run_offline_evaluation(**kwargs):
    """
    Connects to the evaluation module to trigger RAGAS evaluation on 
    a sample dataset and logs metrics to MLflow.
    """
    print("Starting weekly RAG evaluation process...")
    # Delay importing evaluation modules to prevent Airflow DAG parse timeouts 
    # if those modules are heavy.
    try:
        from evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator()
        
        # We would typically pull a golden eval dataset or production logs here
        ds_path = os.getenv("EVAL_DATASET_PATH", "data/eval_set.csv")
        
        if not os.path.exists(ds_path):
            print(f"Evaluation dataset {ds_path} not found. Skipping.")
            return
            
        print("Running RAGAS evaluator...")
        results = evaluator.evaluate_dataset(ds_path)
        print(f"Evaluation complete. Results: {results}")

    except ImportError as e:
        print(f"Failed to load evaluation modules: {e}")
    except Exception as e:
        print(f"Error during offline evaluation: {e}")

eval_task = PythonOperator(
    task_id='weekly_rag_evaluation',
    python_callable=run_offline_evaluation,
    provide_context=True,
    dag=dag,
)

eval_task
