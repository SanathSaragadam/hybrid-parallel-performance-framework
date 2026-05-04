# System Architecture

## High-Level Pipeline

PM100 / F-DATA Dataset
↓
Preprocessing + Feature Selection
↓
Chunk Partitioning
↓
NOTEARS on Each Chunk
↓
Local DAG Generation
↓
Edge Stability Analysis
↓
Global DAG Merge
↓
Filtered DAG / Causal Chains
↓
Scaling Experiments
↓
Final Interpretation

## Core Modules
- Data Loader
- Feature Selector
- Chunk Partitioner
- NOTEARS Discovery Engine
- Local DAG Generator
- Global DAG Merger
- Visualization Engine
- Scaling Analyzer

## Main Outputs
- local DAGs
- global DAG
- filtered DAG
- causal chain graph
- scaling plots
