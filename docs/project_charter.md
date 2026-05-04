
# Project Charter

## Project Title
Scalable NOTEARS-Based Causal Discovery Framework for HPC Performance and Energy Analytics

## Objective
Design and implement a scalable causal discovery framework using NOTEARS on large structured datasets.

The system will:
- preprocess HPC-oriented datasets
- partition the dataset into chunks
- run NOTEARS on each chunk to identify local DAGs
- merge local DAGs into a stable global DAG
- analyze causal relationships among runtime, energy, power, thread count, and other system metrics
- evaluate scalability with respect to dataset size and chunk size

## Main Dataset Direction
- PM100 dataset from Zenodo
- F-DATA dataset from Zenodo

## Research Questions
1. Which variables causally affect runtime?
2. Which variables causally affect energy and power usage?
3. How stable are causal edges across chunks?
4. How does NOTEARS behave under chunk-based execution?
5. How does runtime change with dataset size and chunk size?

## Deliverables
- preprocessing pipeline
- NOTEARS-based local DAG discovery
- chunk-based local DAG generation
- global DAG merge
- filtered DAG view
- causal chain visualization
- runtime scaling analysis
- chunk-size scaling analysis
- final report and presentation

## Planned Output Graphs
- full causal graph
- filtered causal graph
- causal chain explanation graph
- local chunk DAGs
- runtime vs dataset size
- runtime vs chunk size
- validation and interpretation visuals
