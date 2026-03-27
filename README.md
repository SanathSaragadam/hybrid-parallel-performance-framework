# hybrid-parallel-performance-framework
# Hybrid-Parallel-Causal-Modeling-Framework-for-Regression-Data-Analytics (Cloud-Integrated)

## Goal

Build a hybrid MPI+OpenMP framework that:

partitions a regression dataset into chunks

builds local causal DAGs from each chunk

merges local DAGs into a global causal graph

evaluates performance, scalability, and merge behavior



## Key Deliverables
- Task graph + execution model (workpool / producer-consumer)
- Hybrid parallelism: MPI + OpenMP
- Performance measurement: timing + Caliper/TAU/HPCToolkit
- Analysis: correctness, compute/comm/I/O breakdown, scalability metrics, comparisons across models/decompositions/I/O
- Code + report + presentation

## Repo Structure
- `src/` MPI+OpenMP application
- `scripts/` experiment runners + parsing + plotting
- `docs/` design notes + weekly updates
- `results/` generated results (ignored); `results/sample/` stores tiny example outputs
- `terraform/` AWS infra (S3 + optional DynamoDB/Lambda)
- `dashboard/` visualization UI
- `report/`, `presentation/` final deliverables

## Visualization Features
- Correlation heatmap for dataset analysis
- Chunk-level edge distribution analysis
- Global DAG edge frequency visualization
