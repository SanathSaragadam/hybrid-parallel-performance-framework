# Project Flowchart

```mermaid
flowchart TD
    A[PM100 / F-DATA Dataset] --> B[Preprocessing + Feature Selection]
    B --> C[Chunk Partitioning]
    C --> D[NOTEARS on Each Chunk]
    D --> E[Local DAGs]
    E --> F[Edge Stability Analysis]
    F --> G[Global DAG Merge]
    G --> H[Filtered DAG / Causal Chains]
    H --> I[Scaling Experiments]
    I --> J[Final Results + Presentation]
