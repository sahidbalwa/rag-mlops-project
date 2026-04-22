"""
Module for operational metric collection using Prometheus.
=== FILE: mlops/monitoring/metrics_collector.py ===
"""

from prometheus_client import Counter, Histogram

# Metric Definitions
# 1. Total Requests Counter
REQUESTS_TOTAL = Counter(
    "rag_requests_total", 
    "Total number of requests received by the RAG API.",
    ["endpoint"]
)

# 2. End-to-End Latency
E2E_LATENCY = Histogram(
    "rag_e2e_latency_seconds",
    "End-to-End processing latency for query requests",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# 3. Retrieval Latency
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Time taken to retrieve context from the Vector Store",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0]
)

# 4. LLM Generation Latency
LLM_LATENCY = Histogram(
    "rag_llm_latency_seconds",
    "Time taken for the LLM to generate a response",
    buckets=[0.5, 1.0, 2.0, 4.0, 8.0, 15.0]
)

# 5. Cache Hits Counter
CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Total number of queries served from cache instead of LLM generation."
)
