# Project Charter (Week 1)

## Project Title
Hybrid MPI+OpenMP Performance Analytics Framework

## Objective
Design and evaluate a hybrid parallel framework using MPI and OpenMP.
The system will support multiple execution models, domain decomposition strategies,
and I/O strategies, and will be instrumented for detailed performance analysis.

## Research Questions
1. How do execution models (static vs workpool) affect scalability and efficiency?
2. How do domain decompositions (block vs cyclic vs block-cyclic) affect communication overhead?
3. How do I/O strategies (rank0 + broadcast vs MPI-IO) affect overall performance?

## Performance Metrics
- Wall clock time
- Compute / Communication / I/O time breakdown
- Speedup
- Parallel efficiency
- Efficiency loss (1 - efficiency)
- Throughput (tasks per second)
- Memory usage
- (Optional) Cache miss rate via PAPI

## Experimental Variables
- MPI ranks
- OpenMP threads
- Input size (N)
- Batch size / number of tasks
- Execution model
- Decomposition strategy
- I/O strategy