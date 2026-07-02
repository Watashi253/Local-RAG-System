# Enterprise Multi-Tenant Local RAG Platform

A high-performance, production-hardened Retrieval-Augmented Generation (RAG) platform built completely locally for total data privacy. The system features advanced semantic retrieval optimization pipelines isolated inside a containerized environment.

## Core Architecture Specifications
* **Vector Engine:** Qdrant Cluster (isolated collection spaces mapped to session states).
* **Local Inference:** Ollama orchestrating `llama3.2` execution parameters.
* **Retrieval Strategy:** Reciprocal Rank Fusion (RRF) hybrid search backed by a Cross-Encoder Reranker.
* **Text Processing:** Percentile-based `SemanticChunker` execution limits.

## Rapid Deployment Orchestration

1. **Verify Environment Variables:**
   ```bash
   cp .env.example .env