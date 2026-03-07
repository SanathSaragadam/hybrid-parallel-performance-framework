# Task Graph (DAG)

## Tasks

T0: Load regression dataset (I/O)

T1: Partition dataset into chunks

T2: Construct local causal DAG per chunk using regression relationships

T3: Extract dependency edges between variables

T4: Merge local DAGs into a global causal graph

T5: Export causal graph and performance metrics

## Dependencies (DAG)

T0 → T1 → T2 → T3 → T4 → T5
