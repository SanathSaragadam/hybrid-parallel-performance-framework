# Task Graph (DAG)

## Tasks
T0: Read/Generate Input (I/O)
T1: Create Task List (batch items or tiles)
T2: Distribute Tasks (MPI comm)
T3: Compute Tasks (OpenMP compute)
T4: Reduce Results (MPI comm)
T5: Write Output + Metrics (I/O)

## Dependencies (DAG)
T0 → T1 → T2 → T3 → T4 → T5
