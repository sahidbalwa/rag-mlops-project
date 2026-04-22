"""
metrics_collector.py
Collects and exposes Prometheus metrics.
"""
from prometheus_client import Counter, Histogram, start_http_server

REQUEST_COUNT = Counter("rag_requests_total", "Total query requests")
LATENCY = Histogram("rag_latency_seconds", "Query latency in seconds")
RETRIEVAL_SCORE = Histogram("rag_retrieval_score", "Top retrieval score")

def start_metrics_server(port: int = 8090):
    start_http_server(port)
    print(f"[Prometheus] Metrics server started on :{port}")
