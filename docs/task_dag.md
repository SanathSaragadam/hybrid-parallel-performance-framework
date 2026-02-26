# Task Graph (DAG)

T0: Read/Generate Input (I/O)
T1: Create Task List (batch items / tiles)
T2: Distribute Tasks (MPI communication)
T3: Compute Tasks (OpenMP parallel region)
T4: Reduce Results (MPI communication)
T5: Write Output + Metrics (I/O)

## Dependencies (DAG)
T0 → T1 → T2 → T3 → T4 → T5
