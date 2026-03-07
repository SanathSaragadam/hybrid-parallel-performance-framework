# Project Charter (Week 1)

## Project Title
Hybrid MPI+OpenMP Framework for Distributed Causal Graph Construction on Regression Data

## Objective
Design and evaluate a hybrid parallel framework using MPI and OpenMP to analyze
large-scale regression datasets and construct causal graphs.

The system partitions a dataset into multiple chunks across distributed processes.
Each chunk independently builds a local causal Directed Acyclic Graph (DAG) based
on regression relationships. These local DAGs are then merged into a global causal
graph representing the overall dependencies in the dataset.

The framework will support multiple execution models, data partitioning strategies,
and I/O methods, while also collecting detailed performance metrics to analyze
scalability and system efficiency.

## Research Questions
1. How does distributing regression data into chunks affect the accuracy and
   consistency of local causal DAG construction?

2. How do different DAG merging strategies influence the structure and stability
   of the global causal graph?

3. How do execution models (static partition vs dynamic workpool) affect
   scalability and load balancing in distributed causal graph construction?

4. How do dataset partitioning strategies affect communication overhead
   and parallel efficiency?

## Performance Metrics
- Wall clock time
- Compute / Communication / I/O time breakdown
- Speedup
- Parallel efficiency
- Efficiency loss (1 - efficiency)
- Throughput (chunks processed per second)
- Memory usage
- (Optional) Cache miss rate via PAPI

## Experimental Variables
- MPI ranks
- OpenMP threads
- Dataset size
- Chunk size
- Execution model
- Partition strategy
- DAG merging strategy
- I/O strategy
