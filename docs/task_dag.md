# Task Graph (DAG)

## Tasks
T0: Read regression dataset
T1: Partition dataset into chunks
T2: Build local causal DAG per chunk
T3: Extract edge weights / dependencies
T4: Merge local DAGs into global DAG
T5: Validate global graph + export metrics

## Dependencies (DAG)
T0 → T1 → T2 → T3 → T4 → T5
