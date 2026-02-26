# hybrid-parallel-performance-framework
# Hybrid MPI+OpenMP Performance Analytics Framework (Cloud-Integrated)

## Goal
Build and evaluate a hybrid parallel framework (MPI + OpenMP) with multiple execution models,
domain decompositions, and I/O strategies, instrumented with profiling tools (Caliper, optional PAPI).
Publish experiment results via a cloud pipeline (AWS S3 + dashboard) with CI/CD + IaC.

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
