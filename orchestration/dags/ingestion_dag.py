"""
Airflow DAG for batch document ingestion into the vector store.
=== FILE: orchestration/dags/ingestion_dag.py ===
"""

import os
import glob
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# We do dynamic path insertion so Airflow can find our local modules 
# Alternatively, this assumes the pipeline root is in PYTHONPATH
from ingestion.document_loader import DocumentLoader
from ingestion.text_chunker import TextChunker
from ingestion.vector_store_uploader import VectorStoreUploader

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'rag_ingestion_dag',
    default_args=default_args,
    description='Automated batch ingestion of new documents',
    schedule_interval=timedelta(days=1), # Runs daily or when triggered manually
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['rag', 'ingestion'],
)

def scan_and_ingest(**kwargs):
    """
    Scans a designated watch folder for new documents and processes them.
    """
    watch_folder = os.getenv("INGESTION_WATCH_FOLDER", "/app/data/new_docs")
    if not os.path.exists(watch_folder):
        print(f"Watch folder {watch_folder} does not exist. Skipping ingestion.")
        return

    # Find all supported files
    supported_exts = ["*.pdf", "*.txt", "*.docx"]
    files_to_process = []
    for ext in supported_exts:
        files_to_process.extend(glob.glob(os.path.join(watch_folder, ext)))

    if not files_to_process:
        print("No new documents found for ingestion.")
        return

    loader = DocumentLoader()
    chunker = TextChunker()
    uploader = VectorStoreUploader()

    for file_path in files_to_process:
        print(f"Processing: {file_path}")
        try:
            doc_data = loader.load_document(file_path)
            chunks = chunker.chunk_document(doc_data)
            chunk_ids = uploader.upload(chunks)
            print(f"Uploaded {len(chunk_ids)} chunks for {file_path}.")
            
            # Optionally move the processed file to an 'archive' folder
            archive_folder = os.path.join(os.path.dirname(watch_folder), "archived_docs")
            os.makedirs(archive_folder, exist_ok=True)
            os.rename(file_path, os.path.join(archive_folder, os.path.basename(file_path)))
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

ingest_task = PythonOperator(
    task_id='batch_ingest_documents',
    python_callable=scan_and_ingest,
    provide_context=True,
    dag=dag,
)

ingest_task
