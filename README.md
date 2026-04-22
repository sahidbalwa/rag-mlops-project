# Enterprise RAG MLOps Pipeline

A production-ready Enterprise Retrieval-Augmented Generation (RAG) system built with best-in-class MLOps practices.

## Overview

This repository demonstrates an end-to-end RAG architecture that connects a robust backend (built on FastAPI) with a Streamlit frontend, automated orchestration via Airflow, evaluation using Ragas, tracking via MLflow/LangSmith, and deployment using Docker compose.

### Core Architecture

- **Data Ingestion**: Processes PDFs, TXT, and DOCX files. Text is chunked with sliding windows and embedded into a Vector Database (e.g. Chroma).
- **Retrieval Pipeline**: Implements advanced retrieval strategies including semantic search followed by a cross-encoder Reranking step to surface the most relevant context.
- **LLM Generation**: Connects to OpenAI or HuggingFace models using LangChain, relying on version-controlled prompt templates.
- **Backend API**: A FastAPI service exposing endpoints for health-checks, document ingestion, and question-answering with built-in telemetry.
- **Frontend App**: A Streamlit interface to quickly query the knowledge base and interactively upload new files.
- **MLOps & Evaluation**:
  - **Tracking**: Logs parameters and metrics to an MLflow Tracking Server.
  - **Tracing**: LangSmith integration to trace LLM calls.
  - **Drift & Monitoring**: Evidently integration to detect data drift on incoming embeddings. Prometheus metrics collection for operational profiling.
  - **Offline Evaluation**: Integrates with Ragas to analyze precision, recall, faithfulness, and answer relevancy based on a golden dataset.

## Project Structure

```
├── api/                  # FastAPI backend routes and schemas
├── configs/              # YAML configurations for prompts and models
├── data/                 # Raw and processed documents (mapped as volume)
├── docker/               # Dockerfiles and Compose configurations
├── evaluation/           # Ragas integration for offline eval
├── frontend/             # Streamlit application UI elements
├── ingestion/            # Loaders, chunkers, and DB integrators
├── mlops/                # Tracking, tracing, monitoring, and triggering tools
├── orchestration/        # Airflow DAGs for batch ingestion and eval scheduling
├── retrieval/            # Core RAG retrieval and reranking models
├── generation/           # LLM clients and response parsing
├── tests/                # Unit tests for components
└── README.md             # This file
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- OpenAI API Key (or generic HuggingFace configuration in `.env`)

### Environment Setup

1. Copy the example environment variables:
   ```bash
   cp .env.example .env
   ```
2. Fill your Open AI keys and configure the MLflow/Airflow connections in `.env`.

### Running Locally with Docker

To bring up the entire stack (FastAPI Backend, Streamlit Frontend, MLflow server, Prometheus):

```bash
cd docker
docker-compose up --build
```

- **Frontend UI**: http://localhost:8501
- **FastAPI Specs**: http://localhost:8000/docs
- **MLflow Tracking**: http://localhost:5000
- **Prometheus Metrics**: http://localhost:9090

### Local Development Setup

If you prefer to run locally without Docker:

```bash
# Setup Virtual Env
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run the Frontend
streamlit run frontend/app.py
```

## Workflows

### 1. Ingestion Workflow
- Upload via the Streamlit Sidebar OR POST to `/ingest`.
- The document is parsed, chunked, embedded, and stored in the vector store.
- **Orchestrated via Airflow**: `rag_ingestion_dag` can scan a watch folder daily.

### 2. Query Workflow
- Enter a query in the Streamlit UI OR POST to `/query`.
- System retrieves Top-K contexts, reranks them using a cross-encoder, processes through the LLM, and formats the response complete with source citations.
- MLflow logs the run internally, and Prometheus updates counters.

### 3. Evaluation & Retraining Workflow
- **Drift Detected**: When drift is noted via Evidently, a webhook is fired to `api/trigger-ingestion` initiating an Airflow pipeline.
- **Weekly Eval**: `rag_evaluation_dag` runs offline evaluations against the golden dataset using Ragas.

## CI/CD 

GitHub Actions are configured in `.github/workflows/ci.yml` which will spin up ubuntu-latest boxes, lint with `flake8`, and run `pytest` upon PRs hitting the `master` repository.
