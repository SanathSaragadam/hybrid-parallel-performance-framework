# Scalable NOTEARS-Based Causal Discovery Framework for HPC Performance and Energy Analytics

This project builds a scalable causal discovery framework for structured performance datasets.

The system is designed to:
- preprocess real-world HPC datasets
- partition data into chunks
- run NOTEARS-based causal discovery on each chunk
- generate local DAGs
- merge local DAGs into a stable global causal graph
- analyze runtime, energy, and resource relationships
- evaluate scaling behavior across dataset size and chunk size

## Dataset Direction
- Prototype / earlier experiments: California Housing, Insurance
- Main project dataset: PM100 or F-DATA from Zenodo
- Scaling dataset: HPC / performance-oriented large datasets

## Core Workflow
1. Dataset preprocessing and feature selection
2. Chunk partitioning
3. NOTEARS causal discovery on each chunk
4. Local DAG generation
5. Stability-aware global DAG merge
6. Validation and interpretation
7. Scaling experiments
