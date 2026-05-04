#!/bin/bash

echo "=============================================="
echo " PM100 NOTEARS Causal Discovery Pipeline"
echo "=============================================="

echo ""
echo "Step 1: Inspecting HPC dataset..."
python scripts/notears/inspect_hpc_dataset.py

echo ""
echo "Step 2: Running single NOTEARS DAG..."
python scripts/notears/run_notears_single.py

echo ""
echo "Step 3: Running chunked NOTEARS local DAGs..."
python scripts/notears/run_notears_chunks.py

echo ""
echo "Step 4: Merging local DAGs into stable global DAG..."
python scripts/notears/merge_notears_dags.py

echo ""
echo "Step 5: Running final comparison and causal chain analysis..."
python scripts/notears/final_notears_analysis_pack.py

echo ""
echo "=============================================="
echo " Pipeline completed successfully."
echo " Results saved in results/notears/"
echo "=============================================="
